"""Single-writer, hash-chained journal for read-only shadow decisions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, cast

from tradingbot.health import write_health
from tradingbot.shadow.bundle import ShadowBundle, ShadowBundleError

SHADOW_JOURNAL_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ZERO_HASH = "0" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ShadowJournal:
    """Durably record public-data decisions and reject concurrent writers."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        bundle: ShadowBundle,
    ) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise ShadowBundleError(
                "shadow run ID must use only letters, digits, '.', '_' and '-'"
            )
        self.run_id = run_id
        self.root = Path(root).expanduser().resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock: BinaryIO | None = None
        self._locked = False
        self._acquire_lock()
        try:
            self.run_manifest_path = self.root / "run.json"
            expected_manifest: dict[str, object] = {
                "shadow_journal_schema_version": SHADOW_JOURNAL_SCHEMA_VERSION,
                "run_id": run_id,
                "bundle_id": bundle.bundle_id,
                "bundle_fingerprint": bundle.bundle_fingerprint,
                "scope": {
                    "bybit_access": "public-read-only",
                    "order_submission": False,
                    "trading_credentials_allowed": False,
                },
            }
            if self.run_manifest_path.exists():
                try:
                    loaded: object = json.loads(self.run_manifest_path.read_bytes())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ShadowBundleError("shadow run manifest is unreadable") from exc
                if loaded != expected_manifest:
                    raise ShadowBundleError(
                        "existing shadow run belongs to another bundle or scope"
                    )
            else:
                rendered = json.dumps(
                    expected_manifest,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
                with self.run_manifest_path.open(
                    "x", encoding="utf-8", newline="\n"
                ) as target:
                    target.write(rendered)
                    target.flush()
                    os.fsync(target.fileno())
            self.run_manifest_sha256 = _sha256_file(self.run_manifest_path)
            self.events = self._validate_existing_events()
            self.sequence = len(self.events)
            self.previous_hash = (
                _ZERO_HASH
                if not self.events
                else str(self.events[-1]["event_hash"])
            )
            self.last_recorded_at_ns = (
                0
                if not self.events
                else int(self.events[-1]["recorded_at_ns"])
            )
        except BaseException:
            self.close()
            raise

    def _acquire_lock(self) -> None:
        lock_path = self.root / ".writer.lock"
        lock = lock_path.open("a+b")
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if sys.platform == "win32":  # pragma: no cover - platform-specific
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on deployment Linux
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock.close()
            raise ShadowBundleError(
                f"another writer already owns shadow run {self.run_id}"
            ) from exc
        self._lock = lock
        self._locked = True

    def _validate_existing_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        expected_sequence = 1
        previous = _ZERO_HASH
        previous_recorded_at_ns = 0
        for path in sorted(self.root.glob("events-*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            raise ShadowBundleError(
                                f"empty event line in {path.name}:{line_number}"
                            )
                        parsed: object = json.loads(line)
                        if not isinstance(parsed, dict):
                            raise ShadowBundleError(
                                f"invalid event object in {path.name}:{line_number}"
                            )
                        event = cast(dict[str, Any], parsed)
                        stored_hash = event.get("event_hash")
                        unsigned = {
                            key: value for key, value in event.items() if key != "event_hash"
                        }
                        if (
                            event.get("shadow_journal_schema_version")
                            != SHADOW_JOURNAL_SCHEMA_VERSION
                            or event.get("sequence") != expected_sequence
                            or event.get("previous_hash") != previous
                            or event.get("run_manifest_sha256")
                            != self.run_manifest_sha256
                            or stored_hash != _sha256_json(unsigned)
                        ):
                            raise ShadowBundleError(
                                f"shadow journal chain failed at {path.name}:{line_number}"
                            )
                        recorded_at_ns = event.get("recorded_at_ns")
                        if isinstance(recorded_at_ns, bool) or not isinstance(
                            recorded_at_ns, int
                        ) or recorded_at_ns < previous_recorded_at_ns:
                            raise ShadowBundleError(
                                f"shadow journal timestamp is invalid at "
                                f"{path.name}:{line_number}"
                            )
                        event_day = datetime.fromtimestamp(
                            recorded_at_ns / 1_000_000_000, tz=UTC
                        ).date().isoformat()
                        if path.name != f"events-{event_day}.jsonl":
                            raise ShadowBundleError(
                                f"shadow journal event is stored under the wrong UTC day: "
                                f"{path.name}:{line_number}"
                            )
                        events.append(event)
                        expected_sequence += 1
                        previous = str(stored_hash)
                        previous_recorded_at_ns = recorded_at_ns
            except json.JSONDecodeError as exc:
                raise ShadowBundleError(f"shadow journal JSON is corrupt: {path}") from exc
        return events

    def append(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        recorded_at_ns: int | None = None,
    ) -> dict[str, Any]:
        if not event_type or not self._locked:
            raise ShadowBundleError("shadow journal is closed or event type is empty")
        now_ns = time.time_ns() if recorded_at_ns is None else recorded_at_ns
        now_ns = max(now_ns, self.last_recorded_at_ns)
        if now_ns <= 0:
            raise ShadowBundleError("shadow event timestamp must be positive")
        unsigned: dict[str, object] = {
            "shadow_journal_schema_version": SHADOW_JOURNAL_SCHEMA_VERSION,
            "run_manifest_sha256": self.run_manifest_sha256,
            "sequence": self.sequence + 1,
            "recorded_at_ns": now_ns,
            "event_type": event_type,
            "previous_hash": self.previous_hash,
            "payload": payload,
        }
        event: dict[str, Any] = {**unsigned, "event_hash": _sha256_json(unsigned)}
        day = datetime.fromtimestamp(now_ns / 1_000_000_000, tz=UTC).date().isoformat()
        path = self.root / f"events-{day}.jsonl"
        encoded = (_canonical_json(event) + "\n").encode("utf-8")
        with path.open("ab", buffering=0) as target:
            target.write(encoded)
            os.fsync(target.fileno())
        self.events.append(event)
        self.sequence += 1
        self.previous_hash = str(event["event_hash"])
        self.last_recorded_at_ns = now_ns
        return event

    def write_health(self, payload: dict[str, object]) -> None:
        write_health(
            self.root / "health.json",
            {
                "shadow_journal_schema_version": SHADOW_JOURNAL_SCHEMA_VERSION,
                "run_id": self.run_id,
                "journal_sequence": self.sequence,
                "journal_head_hash": self.previous_hash,
                **payload,
            },
        )

    def close(self) -> None:
        if self._lock is None:
            return
        if self._locked:
            try:
                self._lock.seek(0)
                if sys.platform == "win32":  # pragma: no cover - platform-specific
                    import msvcrt

                    msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised on deployment Linux
                    import fcntl

                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        self._lock.close()
        self._lock = None
        self._locked = False

    def __enter__(self) -> ShadowJournal:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
