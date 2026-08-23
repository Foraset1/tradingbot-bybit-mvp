"""Immutable daily Parquet archive and verified raw-retention operations.

The collector keeps writing the current UTC partition while this module audits and
archives a fully elapsed day.  A day manifest is the commit marker: orphaned audit or
canonical outputs are harmless and can be reused, while retention only trusts a
committed day whose canonical files pass their full integrity validation.

``plan_raw_retention`` produces an exact dry-run plan.  ``apply_raw_retention`` is a
separate, explicit operation which accepts that saved plan, requires its fingerprint as
an operator confirmation, repeats every archive/raw integrity check under the archive
lock, and only then unlinks the exact approved files.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from tradingbot.data.audit import AUDIT_REPORT_SCHEMA_VERSION, audit_dataset
from tradingbot.data.canonical import (
    AuditedInputFile,
    DatasetBuildError,
    build_canonical_dataset,
    load_audit_input_manifest,
    validate_canonical_dataset,
)
from tradingbot.data.quality import attach_archive_acceptance

ARCHIVE_DAY_SCHEMA_VERSION: Final = 1
ARCHIVE_CATALOG_SCHEMA_VERSION: Final = 1
RETENTION_PLAN_SCHEMA_VERSION: Final = 1
RETENTION_APPLY_SCHEMA_VERSION: Final = 1


class ArchiveError(RuntimeError):
    """Raised when an archive or retention safety invariant is violated."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = {} if details is None else details


@dataclass(frozen=True, slots=True)
class ArchiveDay:
    partition_date: str
    day_manifest_path: Path
    day_fingerprint: str
    audit_path: Path
    audit_sha256: str
    input_fingerprint: str
    raw_files: int
    raw_records: int
    raw_bytes: int
    quality_policy: str | None
    quality_status: str
    warning_codes: tuple[str, ...]
    warning_items: int
    warning_occurrences: int
    canonical_dataset_path: Path
    canonical_manifest_path: Path
    canonical_manifest_sha256: str
    canonical_output_fingerprint: str
    canonical_files: int
    canonical_rows: int
    canonical_bytes: int


@dataclass(frozen=True, slots=True)
class ArchiveDayResult:
    partition_date: str
    day_manifest_path: Path
    day_fingerprint: str
    audit_path: Path
    canonical_dataset_path: Path
    canonical_output_fingerprint: str
    catalog_path: Path
    catalog_fingerprint: str
    raw_files: int
    raw_records: int
    raw_bytes: int
    quality_policy: str
    quality_status: str
    warning_codes: tuple[str, ...]
    warning_items: int
    warning_occurrences: int
    canonical_files: int
    canonical_rows: int
    canonical_bytes: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_day_schema_version": ARCHIVE_DAY_SCHEMA_VERSION,
            "ok": True,
            "partition_date": self.partition_date,
            "day_manifest_path": self.day_manifest_path.as_posix(),
            "day_fingerprint": self.day_fingerprint,
            "audit_path": self.audit_path.as_posix(),
            "canonical_dataset_path": self.canonical_dataset_path.as_posix(),
            "canonical_output_fingerprint": self.canonical_output_fingerprint,
            "catalog_path": self.catalog_path.as_posix(),
            "catalog_fingerprint": self.catalog_fingerprint,
            "raw_files": self.raw_files,
            "raw_records": self.raw_records,
            "raw_bytes": self.raw_bytes,
            "quality": {
                "policy": self.quality_policy,
                "status": self.quality_status,
                "warning_codes": list(self.warning_codes),
                "warning_items": self.warning_items,
                "warning_occurrences": self.warning_occurrences,
                "training_requires_continuity_filter": (
                    self.quality_status == "gapped"
                ),
            },
            "canonical_files": self.canonical_files,
            "canonical_rows": self.canonical_rows,
            "canonical_bytes": self.canonical_bytes,
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class CatalogResult:
    path: Path
    fingerprint: str
    entries: tuple[ArchiveDay, ...]


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    path: str
    partition_date: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "partition_date": self.partition_date,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RetentionBlocker:
    code: str
    message: str
    path: str | None = None
    partition_date: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "partition_date": self.partition_date,
        }


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    raw_root: Path
    archive_root: Path
    as_of_date: str
    retention_days: int
    delete_before_date: str
    catalog_fingerprint: str
    candidates: tuple[RetentionCandidate, ...]
    blockers: tuple[RetentionBlocker, ...]
    retained_recent_files: int
    retained_recent_bytes: int
    partial_files: tuple[str, ...]
    plan_fingerprint: str

    @property
    def safe_to_apply(self) -> bool:
        return bool(self.candidates) and not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "retention_plan_schema_version": RETENTION_PLAN_SCHEMA_VERSION,
            "mode": "dry_run",
            "deletion_performed": False,
            "raw_root": self.raw_root.as_posix(),
            "archive_root": self.archive_root.as_posix(),
            "as_of_date": self.as_of_date,
            "retention_days": self.retention_days,
            "delete_before_date": self.delete_before_date,
            "catalog_fingerprint": self.catalog_fingerprint,
            "candidate_files": [item.to_dict() for item in self.candidates],
            "candidate_file_count": len(self.candidates),
            "candidate_bytes": sum(item.bytes for item in self.candidates),
            "blockers": [item.to_dict() for item in self.blockers],
            "blocker_count": len(self.blockers),
            "retained_recent_files": self.retained_recent_files,
            "retained_recent_bytes": self.retained_recent_bytes,
            "partial_files": list(self.partial_files),
            "partial_file_count": len(self.partial_files),
            "safe_to_apply": self.safe_to_apply,
            "plan_fingerprint": self.plan_fingerprint,
        }


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchiveError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"{label} is unreadable: {path}") from exc
    return _json_object(parsed, label)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArchiveError(f"{label} must be a non-empty string")
    return value


def _required_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArchiveError(f"{label} must be a non-negative integer")
    return value


def _valid_sha256(value: object, label: str) -> str:
    text = _required_string(value, label).lower()
    if len(text) != 64:
        raise ArchiveError(f"{label} must be a SHA-256 hexadecimal digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ArchiveError(f"{label} must be a SHA-256 hexadecimal digest") from exc
    return text


def _parse_date(value: date | str, label: str) -> date:
    if isinstance(value, datetime):
        raise ArchiveError(f"{label} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ArchiveError(f"{label} must be an ISO date (YYYY-MM-DD)")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ArchiveError(f"{label} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != value:
        raise ArchiveError(f"{label} must be an ISO date (YYYY-MM-DD)")
    return parsed


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    text = _required_string(value, label)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ArchiveError(f"{label} is not a safe relative path")
    return relative


def _resolve_archive_path(root: Path, value: object, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    resolved = root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ArchiveError(f"{label} escapes the archive root")
    return resolved


def _archive_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ArchiveError(f"path is outside archive root: {path}") from exc


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as target:
            target.write(rendered)
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ArchiveError(f"cannot read existing immutable file {path}: {exc}") from exc
        if existing != rendered:
            raise ArchiveError(f"immutable archive file already exists with other content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as target:
            target.write(rendered.decode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(partial, path)
        except FileExistsError as exc:
            if path.read_bytes() != rendered:
                raise ArchiveError(
                    f"immutable archive file was created concurrently: {path}"
                ) from exc
        except OSError:
            # ``os.link`` can be unavailable on some filesystems.  The archive lock
            # still makes replace safe within this application.
            os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _validate_separate_roots(raw_root: Path, archive_root: Path) -> None:
    if (
        raw_root == archive_root
        or raw_root.is_relative_to(archive_root)
        or archive_root.is_relative_to(raw_root)
    ):
        raise ArchiveError("raw and archive roots must not overlap")


@contextmanager
def _archive_lock(archive_root: Path) -> Iterator[None]:
    lock = archive_root / ".archive.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ArchiveError(
            f"another archive operation may be running; lock exists: {lock}"
        ) from exc
    try:
        yield
    finally:
        try:
            lock.rmdir()
        except OSError as exc:
            raise ArchiveError(f"could not release archive lock {lock}: {exc}") from exc


def _audit_partition_matches(path: str, partition: date) -> bool:
    parts = PurePosixPath(path).parts
    expected = (
        f"{partition.year:04d}",
        f"{partition.month:02d}",
        f"{partition.day:02d}",
    )
    return len(parts) == 6 and parts[2:5] == expected


def _assert_raw_partition_stable(
    raw_root: Path,
    partition: date,
    expected_files: dict[str, int],
) -> None:
    completed: dict[str, int] = {}
    partial: list[str] = []
    for path in raw_root.rglob("*.jsonl"):
        relative = path.relative_to(raw_root).as_posix()
        if _audit_partition_matches(relative, partition):
            try:
                completed[relative] = path.stat().st_size
            except OSError as exc:
                raise ArchiveError(f"could not stat daily raw file {path}: {exc}") from exc
    for path in raw_root.rglob("*.jsonl.partial"):
        relative = path.relative_to(raw_root).as_posix()
        if _audit_partition_matches(relative, partition):
            partial.append(relative)
    if partial:
        raise ArchiveError(
            "raw UTC partition became active while it was being archived: "
            f"{sorted(partial)[0]}"
        )
    if completed != expected_files:
        raise ArchiveError(
            "raw UTC partition changed while it was being archived; retry after "
            "late data has settled"
        )


def _canonical_stats(manifest_path: Path, partition: date) -> tuple[int, int, int]:
    manifest = _load_json(manifest_path, "canonical manifest")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ArchiveError("canonical manifest.files must be a non-empty array")
    rows = 0
    size = 0
    for index, raw_file in enumerate(raw_files):
        item = _json_object(raw_file, f"canonical manifest.files[{index}]")
        if item.get("date") != partition.isoformat():
            raise ArchiveError(
                "daily canonical dataset contains a different UTC partition"
            )
        rows += _required_nonnegative_int(
            item.get("rows"), f"canonical manifest.files[{index}].rows"
        )
        size += _required_nonnegative_int(
            item.get("bytes"), f"canonical manifest.files[{index}].bytes"
        )
    if manifest.get("output_file_count") != len(raw_files):
        raise ArchiveError("canonical output_file_count does not match files")
    return len(raw_files), rows, size


def _day_manifest_payload(
    *,
    archive_root: Path,
    partition: date,
    raw_root: Path,
    audit_path: Path,
    canonical_dataset_path: Path,
) -> dict[str, object]:
    try:
        audit = load_audit_input_manifest(audit_path)
        canonical = validate_canonical_dataset(canonical_dataset_path)
    except DatasetBuildError as exc:
        raise ArchiveError(f"daily archive validation failed: {exc}") from exc
    if any(not _audit_partition_matches(item.path, partition) for item in audit.files):
        raise ArchiveError("audit contains files outside the requested UTC partition")
    canonical_manifest = _load_json(canonical.manifest_path, "canonical manifest")
    canonical_source = _json_object(
        canonical_manifest.get("source"), "canonical manifest.source"
    )
    if canonical_source.get("audit_report_sha256") != audit.report_sha256:
        raise ArchiveError("canonical dataset copied a different audit report")
    canonical_files, canonical_rows, canonical_bytes = _canonical_stats(
        canonical.manifest_path, partition
    )
    payload: dict[str, object] = {
        "archive_day_schema_version": ARCHIVE_DAY_SCHEMA_VERSION,
        "partition_date": partition.isoformat(),
        "source": {
            "raw_root": raw_root.as_posix(),
            "audit_path": _archive_relative(audit_path, archive_root),
            "audit_sha256": audit.report_sha256,
            "audit_report_schema_version": AUDIT_REPORT_SCHEMA_VERSION,
            "input_fingerprint": audit.input_fingerprint,
            "file_count": len(audit.files),
            "records": audit.total_records,
            "bytes": audit.total_bytes,
        },
        "quality": audit.archive_quality_dict(),
        "canonical": {
            "dataset_id": canonical.dataset_id,
            "dataset_path": _archive_relative(canonical.dataset_path, archive_root),
            "manifest_path": _archive_relative(canonical.manifest_path, archive_root),
            "manifest_sha256": _sha256_file(canonical.manifest_path),
            "output_fingerprint": canonical.output_fingerprint,
            "file_count": canonical_files,
            "rows": canonical_rows,
            "bytes": canonical_bytes,
        },
    }
    payload["day_fingerprint"] = _sha256_json(payload)
    return payload


def _validate_day_manifest(
    archive_root: Path,
    manifest_path: Path,
    *,
    verify_canonical_files: bool,
) -> ArchiveDay:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(archive_root):
        raise ArchiveError("day manifest is outside archive root")
    raw = _load_json(manifest_path, "archive day manifest")
    if raw.get("archive_day_schema_version") != ARCHIVE_DAY_SCHEMA_VERSION:
        raise ArchiveError("archive day manifest uses another schema version")
    partition_text = _required_string(raw.get("partition_date"), "partition_date")
    partition = _parse_date(partition_text, "partition_date")
    expected_manifest_parent = archive_root / "days" / f"date={partition_text}"
    if manifest_path.parent != expected_manifest_parent:
        raise ArchiveError("day manifest path does not match its UTC partition")
    fingerprint = _valid_sha256(raw.get("day_fingerprint"), "day_fingerprint")
    fingerprint_payload = dict(raw)
    fingerprint_payload.pop("day_fingerprint", None)
    if _sha256_json(fingerprint_payload) != fingerprint:
        raise ArchiveError("archive day fingerprint does not match its manifest")

    source = _json_object(raw.get("source"), "archive day source")
    canonical_raw = _json_object(raw.get("canonical"), "archive day canonical")
    audit_path = _resolve_archive_path(
        archive_root, source.get("audit_path"), "source.audit_path"
    )
    audit_sha = _valid_sha256(source.get("audit_sha256"), "source.audit_sha256")
    if source.get("audit_report_schema_version") != AUDIT_REPORT_SCHEMA_VERSION:
        raise ArchiveError("daily audit schema metadata is inconsistent")
    if _sha256_file(audit_path) != audit_sha:
        raise ArchiveError("daily audit report failed SHA-256 validation")
    try:
        audit = load_audit_input_manifest(audit_path)
    except DatasetBuildError as exc:
        raise ArchiveError(f"daily audit manifest is invalid: {exc}") from exc
    audit_raw = _load_json(audit_path, "daily audit report")
    policy = _json_object(audit_raw.get("policy"), "daily audit policy")
    if policy.get("partition_date") != partition_text:
        raise ArchiveError("daily audit policy does not match archive partition")
    if any(not _audit_partition_matches(item.path, partition) for item in audit.files):
        raise ArchiveError("daily audit contains a file from another partition")
    expected_quality = audit.archive_quality_dict()
    raw_quality = raw.get("quality")
    if raw_quality is None:
        if audit.archive_acceptance_policy is not None:
            raise ArchiveError("archive day is missing quality metadata")
    elif raw_quality != expected_quality:
        raise ArchiveError("archive day quality does not match its source audit")

    dataset_path = _resolve_archive_path(
        archive_root, canonical_raw.get("dataset_path"), "canonical.dataset_path"
    )
    canonical_manifest_path = _resolve_archive_path(
        archive_root, canonical_raw.get("manifest_path"), "canonical.manifest_path"
    )
    if canonical_manifest_path != dataset_path / "manifest.json":
        raise ArchiveError("canonical manifest path does not belong to its dataset")
    expected_manifest_sha = _valid_sha256(
        canonical_raw.get("manifest_sha256"), "canonical.manifest_sha256"
    )
    if _sha256_file(canonical_manifest_path) != expected_manifest_sha:
        raise ArchiveError("canonical manifest failed SHA-256 validation")
    canonical_manifest = _load_json(canonical_manifest_path, "canonical manifest")
    canonical_source = _json_object(
        canonical_manifest.get("source"), "canonical manifest.source"
    )
    if canonical_source.get("audit_report_sha256") != audit.report_sha256:
        raise ArchiveError("canonical dataset copied a different daily audit")
    if canonical_manifest.get("dataset_id") != canonical_raw.get("dataset_id"):
        raise ArchiveError("canonical dataset ID is inconsistent")
    if canonical_manifest.get("output_fingerprint") != canonical_raw.get(
        "output_fingerprint"
    ):
        raise ArchiveError("canonical output fingerprint is inconsistent")
    if verify_canonical_files:
        try:
            canonical = validate_canonical_dataset(dataset_path)
        except DatasetBuildError as exc:
            raise ArchiveError(f"canonical archive validation failed: {exc}") from exc
        if canonical.input_fingerprint != audit.input_fingerprint:
            raise ArchiveError("canonical archive was built from another audit")
        if canonical.output_fingerprint != canonical_raw.get("output_fingerprint"):
            raise ArchiveError("canonical output fingerprint changed")
    canonical_files, canonical_rows, canonical_bytes = _canonical_stats(
        canonical_manifest_path, partition
    )

    checks: tuple[tuple[object, object, str], ...] = (
        (source.get("input_fingerprint"), audit.input_fingerprint, "input fingerprint"),
        (source.get("file_count"), len(audit.files), "raw file count"),
        (source.get("records"), audit.total_records, "raw record count"),
        (source.get("bytes"), audit.total_bytes, "raw byte count"),
        (canonical_raw.get("file_count"), canonical_files, "canonical file count"),
        (canonical_raw.get("rows"), canonical_rows, "canonical row count"),
        (canonical_raw.get("bytes"), canonical_bytes, "canonical byte count"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ArchiveError(f"archive day {label} is inconsistent")

    output_fingerprint = _valid_sha256(
        canonical_raw.get("output_fingerprint"), "canonical.output_fingerprint"
    )
    return ArchiveDay(
        partition_date=partition_text,
        day_manifest_path=manifest_path,
        day_fingerprint=fingerprint,
        audit_path=audit_path,
        audit_sha256=audit_sha,
        input_fingerprint=audit.input_fingerprint,
        raw_files=len(audit.files),
        raw_records=audit.total_records,
        raw_bytes=audit.total_bytes,
        quality_policy=audit.archive_acceptance_policy,
        quality_status=audit.archive_quality_status,
        warning_codes=audit.archive_warning_codes,
        warning_items=audit.archive_warning_items,
        warning_occurrences=audit.archive_warning_occurrences,
        canonical_dataset_path=dataset_path,
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest_sha256=expected_manifest_sha,
        canonical_output_fingerprint=output_fingerprint,
        canonical_files=canonical_files,
        canonical_rows=canonical_rows,
        canonical_bytes=canonical_bytes,
    )


def _catalog_entry(day: ArchiveDay, archive_root: Path) -> dict[str, object]:
    entry: dict[str, object] = {
        "partition_date": day.partition_date,
        "day_manifest_path": _archive_relative(day.day_manifest_path, archive_root),
        "day_manifest_sha256": _sha256_file(day.day_manifest_path),
        "day_fingerprint": day.day_fingerprint,
        "audit_path": _archive_relative(day.audit_path, archive_root),
        "audit_sha256": day.audit_sha256,
        "input_fingerprint": day.input_fingerprint,
        "raw_files": day.raw_files,
        "raw_records": day.raw_records,
        "raw_bytes": day.raw_bytes,
        "canonical_dataset_path": _archive_relative(
            day.canonical_dataset_path, archive_root
        ),
        "canonical_manifest_path": _archive_relative(
            day.canonical_manifest_path, archive_root
        ),
        "canonical_manifest_sha256": day.canonical_manifest_sha256,
        "canonical_output_fingerprint": day.canonical_output_fingerprint,
        "canonical_files": day.canonical_files,
        "canonical_rows": day.canonical_rows,
        "canonical_bytes": day.canonical_bytes,
    }
    if day.quality_policy is not None:
        entry["quality"] = {
            "policy": day.quality_policy,
            "status": day.quality_status,
            "warning_codes": list(day.warning_codes),
            "warning_items": day.warning_items,
            "warning_occurrences": day.warning_occurrences,
            "training_requires_continuity_filter": (
                day.quality_status == "gapped"
            ),
        }
    return entry


def _build_catalog_locked(archive_root: Path) -> CatalogResult:
    manifests = sorted(
        (archive_root / "days").glob("date=*/manifest.json"),
        key=lambda item: item.as_posix(),
    )
    days = tuple(
        _validate_day_manifest(
            archive_root, manifest, verify_canonical_files=False
        )
        for manifest in manifests
    )
    dates = tuple(day.partition_date for day in days)
    if len(set(dates)) != len(dates):
        raise ArchiveError("archive contains duplicate day manifests")
    if dates != tuple(sorted(dates)):
        raise ArchiveError("archive day manifests are not in chronological order")
    entries = [_catalog_entry(day, archive_root) for day in days]
    payload: dict[str, object] = {
        "archive_catalog_schema_version": ARCHIVE_CATALOG_SCHEMA_VERSION,
        "entry_count": len(entries),
        "first_partition_date": None if not dates else dates[0],
        "last_partition_date": None if not dates else dates[-1],
        "entries": entries,
    }
    fingerprint = _sha256_json(payload)
    payload["catalog_fingerprint"] = fingerprint
    path = archive_root / "catalog.json"
    _write_json_atomic(path, payload)
    return CatalogResult(path=path, fingerprint=fingerprint, entries=days)


def rebuild_archive_catalog(archive_root: str | Path) -> CatalogResult:
    """Rebuild the deterministic catalog from committed daily manifests."""

    root = Path(archive_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _archive_lock(root):
        return _build_catalog_locked(root)


def _load_catalog(archive_root: Path) -> CatalogResult:
    path = archive_root / "catalog.json"
    raw = _load_json(path, "archive catalog")
    if raw.get("archive_catalog_schema_version") != ARCHIVE_CATALOG_SCHEMA_VERSION:
        raise ArchiveError("archive catalog uses another schema version")
    expected_fingerprint = _valid_sha256(
        raw.get("catalog_fingerprint"), "catalog_fingerprint"
    )
    fingerprint_payload = dict(raw)
    fingerprint_payload.pop("catalog_fingerprint", None)
    if _sha256_json(fingerprint_payload) != expected_fingerprint:
        raise ArchiveError("archive catalog fingerprint does not match its contents")
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        raise ArchiveError("archive catalog.entries must be an array")
    if raw.get("entry_count") != len(raw_entries):
        raise ArchiveError("archive catalog entry_count is inconsistent")
    days: list[ArchiveDay] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _json_object(raw_entry, f"archive catalog.entries[{index}]")
        manifest = _resolve_archive_path(
            archive_root,
            entry.get("day_manifest_path"),
            f"archive catalog.entries[{index}].day_manifest_path",
        )
        expected_manifest_sha = _valid_sha256(
            entry.get("day_manifest_sha256"),
            f"archive catalog.entries[{index}].day_manifest_sha256",
        )
        if _sha256_file(manifest) != expected_manifest_sha:
            raise ArchiveError("catalog day manifest failed SHA-256 validation")
        day = _validate_day_manifest(
            archive_root, manifest, verify_canonical_files=False
        )
        expected_entry = _catalog_entry(day, archive_root)
        if entry != expected_entry:
            raise ArchiveError("archive catalog entry does not match its day manifest")
        days.append(day)
    dates = tuple(day.partition_date for day in days)
    if len(set(dates)) != len(dates) or dates != tuple(sorted(dates)):
        raise ArchiveError("archive catalog dates must be unique and sorted")
    expected_first = None if not dates else dates[0]
    expected_last = None if not dates else dates[-1]
    if raw.get("first_partition_date") != expected_first:
        raise ArchiveError("archive catalog first_partition_date is inconsistent")
    if raw.get("last_partition_date") != expected_last:
        raise ArchiveError("archive catalog last_partition_date is inconsistent")
    return CatalogResult(path=path, fingerprint=expected_fingerprint, entries=tuple(days))


def load_archive_catalog(
    catalog_or_root: str | Path,
    *,
    verify_canonical_files: bool = True,
) -> CatalogResult:
    """Load a catalog and optionally verify every referenced Parquet file."""

    selected = Path(catalog_or_root).expanduser().resolve()
    if selected.is_dir():
        root = selected
    else:
        if selected.name != "catalog.json":
            raise ArchiveError("archive catalog file must be named catalog.json")
        root = selected.parent
    catalog = _load_catalog(root)
    if not verify_canonical_files:
        return catalog
    verified = tuple(
        _validate_day_manifest(
            root,
            day.day_manifest_path,
            verify_canonical_files=True,
        )
        for day in catalog.entries
    )
    return CatalogResult(
        path=catalog.path,
        fingerprint=catalog.fingerprint,
        entries=verified,
    )


def archive_day(
    raw_root: str | Path,
    archive_root: str | Path,
    symbols: Sequence[str],
    kline_intervals: Sequence[str],
    *,
    partition_date: date | str,
    minimum_duration_seconds: float = 82_800,
    minimum_free_bytes: int = 0,
    scratch_dir: str | Path | None = None,
    today_utc: date | None = None,
) -> ArchiveDayResult:
    """Audit and commit one fully elapsed UTC partition to canonical Parquet."""

    selected = _parse_date(partition_date, "partition_date")
    today = datetime.now(UTC).date() if today_utc is None else today_utc
    if selected >= today:
        raise ArchiveError("only a fully elapsed UTC day can be archived")
    raw = Path(raw_root).expanduser().resolve()
    archive = Path(archive_root).expanduser().resolve()
    _validate_separate_roots(raw, archive)
    if not raw.is_dir():
        raise ArchiveError(f"raw root does not exist: {raw}")
    archive.mkdir(parents=True, exist_ok=True)

    day_manifest_path = archive / "days" / f"date={selected.isoformat()}" / "manifest.json"
    with _archive_lock(archive):
        reused = day_manifest_path.exists()
        if reused:
            day = _validate_day_manifest(
                archive, day_manifest_path, verify_canonical_files=True
            )
        else:
            report = audit_dataset(
                raw,
                symbols=symbols,
                kline_intervals=kline_intervals,
                minimum_duration_seconds=minimum_duration_seconds,
                strict=True,
                scratch_dir=scratch_dir,
                partition_date=selected,
            )
            audit_payload = report.to_dict()
            acceptance = attach_archive_acceptance(audit_payload)
            if not acceptance.ok:
                reasons = ", ".join(acceptance.reasons) or "unknown"
                raise ArchiveError(
                    f"UTC partition {selected.isoformat()} failed archive policy: {reasons}",
                    details={
                        "partition_date": selected.isoformat(),
                        "input_fingerprint": report.input_fingerprint,
                        "file_count": len(report.files),
                        "archive_acceptance": acceptance.to_dict(),
                        "errors": [item.to_dict() for item in report.errors],
                        "warnings": [item.to_dict() for item in report.warnings],
                    },
                )
            expected_raw_files = {item.path: item.bytes for item in report.files}
            _assert_raw_partition_stable(raw, selected, expected_raw_files)
            audit_path = (
                archive
                / "audits"
                / f"date={selected.isoformat()}"
                / f"audit-v{AUDIT_REPORT_SCHEMA_VERSION}-{report.input_fingerprint[:16]}.json"
            )
            _write_immutable_json(audit_path, audit_payload)
            canonical_parent = archive / "canonical" / f"date={selected.isoformat()}"
            try:
                build = build_canonical_dataset(
                    audit_report=audit_path,
                    output_root=canonical_parent,
                    source_root=raw,
                    minimum_free_bytes=minimum_free_bytes,
                )
            except DatasetBuildError as exc:
                raise ArchiveError(f"daily canonical build failed: {exc}") from exc
            _assert_raw_partition_stable(raw, selected, expected_raw_files)
            payload = _day_manifest_payload(
                archive_root=archive,
                partition=selected,
                raw_root=raw,
                audit_path=audit_path,
                canonical_dataset_path=build.dataset_path,
            )
            _write_immutable_json(day_manifest_path, payload)
            day = _validate_day_manifest(
                archive, day_manifest_path, verify_canonical_files=True
            )
        catalog = _build_catalog_locked(archive)

    return ArchiveDayResult(
        partition_date=day.partition_date,
        day_manifest_path=day.day_manifest_path,
        day_fingerprint=day.day_fingerprint,
        audit_path=day.audit_path,
        canonical_dataset_path=day.canonical_dataset_path,
        canonical_output_fingerprint=day.canonical_output_fingerprint,
        catalog_path=catalog.path,
        catalog_fingerprint=catalog.fingerprint,
        raw_files=day.raw_files,
        raw_records=day.raw_records,
        raw_bytes=day.raw_bytes,
        quality_policy=day.quality_policy or "legacy_strict_v1",
        quality_status=day.quality_status,
        warning_codes=day.warning_codes,
        warning_items=day.warning_items,
        warning_occurrences=day.warning_occurrences,
        canonical_files=day.canonical_files,
        canonical_rows=day.canonical_rows,
        canonical_bytes=day.canonical_bytes,
        reused=reused,
    )


def _raw_partition(relative_path: str) -> date | None:
    parts = PurePosixPath(relative_path).parts
    if len(parts) != 6:
        return None
    try:
        return date.fromisoformat("-".join(parts[2:5]))
    except ValueError:
        return None


def plan_raw_retention(
    raw_root: str | Path,
    archive_root: str | Path,
    *,
    retention_days: int,
    as_of_date: date | str | None = None,
) -> RetentionPlan:
    """Return a no-delete retention plan backed by raw and Parquet SHA-256 checks."""

    if (
        isinstance(retention_days, bool)
        or not isinstance(retention_days, int)
        or retention_days <= 0
    ):
        raise ArchiveError("retention_days must be a positive integer")
    as_of = (
        datetime.now(UTC).date()
        if as_of_date is None
        else _parse_date(as_of_date, "as_of_date")
    )
    if as_of > datetime.now(UTC).date():
        raise ArchiveError("as_of_date cannot be in the future")
    cutoff = as_of - timedelta(days=retention_days)
    raw = Path(raw_root).expanduser().resolve()
    archive = Path(archive_root).expanduser().resolve()
    _validate_separate_roots(raw, archive)
    if not raw.is_dir():
        raise ArchiveError(f"raw root does not exist: {raw}")
    catalog = _load_catalog(archive)
    archived_by_date = {day.partition_date: day for day in catalog.entries}

    completed = sorted(raw.rglob("*.jsonl"), key=lambda item: item.relative_to(raw).as_posix())
    partial_paths = tuple(
        item.relative_to(raw).as_posix()
        for item in sorted(
            raw.rglob("*.jsonl.partial"),
            key=lambda item: item.relative_to(raw).as_posix(),
        )
    )
    candidates: list[RetentionCandidate] = []
    blockers: list[RetentionBlocker] = []
    recent_files = 0
    recent_bytes = 0
    verified_days: dict[
        str, tuple[ArchiveDay, dict[str, AuditedInputFile]] | ArchiveError
    ] = {}

    for relative in partial_paths:
        partition = _raw_partition(relative.removesuffix(".partial"))
        if partition is None:
            blockers.append(
                RetentionBlocker(
                    code="invalid_partial_partition_path",
                    message="partial raw file has an invalid partition path",
                    path=relative,
                )
            )
        elif partition < cutoff:
            blockers.append(
                RetentionBlocker(
                    code="old_partial_file",
                    message="an old raw partition still contains an active partial file",
                    path=relative,
                    partition_date=partition.isoformat(),
                )
            )

    for path in completed:
        relative = path.relative_to(raw).as_posix()
        partition = _raw_partition(relative)
        if partition is None:
            blockers.append(
                RetentionBlocker(
                    code="invalid_partition_path",
                    message="completed raw file is not kind/symbol/YYYY/MM/DD/file.jsonl",
                    path=relative,
                )
            )
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            blockers.append(
                RetentionBlocker(
                    code="raw_stat_failed",
                    message=f"could not stat raw file: {exc}",
                    path=relative,
                    partition_date=partition.isoformat(),
                )
            )
            continue
        if partition >= cutoff:
            recent_files += 1
            recent_bytes += size
            continue

        partition_text = partition.isoformat()
        verification = verified_days.get(partition_text)
        if verification is None:
            catalog_day = archived_by_date.get(partition_text)
            if catalog_day is None:
                verification = ArchiveError("no committed archive exists for this day")
            else:
                try:
                    day = _validate_day_manifest(
                        archive,
                        catalog_day.day_manifest_path,
                        verify_canonical_files=True,
                    )
                    audit = load_audit_input_manifest(day.audit_path)
                    audit_files = {item.path: item for item in audit.files}
                    verification = (day, audit_files)
                except (ArchiveError, DatasetBuildError) as exc:
                    verification = ArchiveError(str(exc))
            verified_days[partition_text] = verification
        if isinstance(verification, ArchiveError):
            blockers.append(
                RetentionBlocker(
                    code="archive_not_verified",
                    message=str(verification),
                    path=relative,
                    partition_date=partition_text,
                )
            )
            continue
        _, audit_files = verification
        audited = audit_files.get(relative)
        if audited is None:
            blockers.append(
                RetentionBlocker(
                    code="raw_file_not_archived",
                    message="raw file is absent from the committed daily audit",
                    path=relative,
                    partition_date=partition_text,
                )
            )
            continue
        expected_size = audited.bytes
        expected_sha = audited.sha256
        if size != expected_size:
            blockers.append(
                RetentionBlocker(
                    code="raw_size_changed",
                    message="raw file size differs from the committed daily audit",
                    path=relative,
                    partition_date=partition_text,
                )
            )
            continue
        try:
            actual_sha = _sha256_file(path)
        except ArchiveError as exc:
            blockers.append(
                RetentionBlocker(
                    code="raw_hash_failed",
                    message=str(exc),
                    path=relative,
                    partition_date=partition_text,
                )
            )
            continue
        if actual_sha != expected_sha:
            blockers.append(
                RetentionBlocker(
                    code="raw_hash_changed",
                    message="raw file SHA-256 differs from the committed daily audit",
                    path=relative,
                    partition_date=partition_text,
                )
            )
            continue
        candidates.append(
            RetentionCandidate(
                path=relative,
                partition_date=partition_text,
                bytes=size,
                sha256=actual_sha,
            )
        )

    candidates.sort(key=lambda item: item.path)
    blockers.sort(
        key=lambda item: (
            "" if item.partition_date is None else item.partition_date,
            "" if item.path is None else item.path,
            item.code,
        )
    )
    fingerprint_payload = {
        "retention_plan_schema_version": RETENTION_PLAN_SCHEMA_VERSION,
        "raw_root": raw.as_posix(),
        "archive_root": archive.as_posix(),
        "as_of_date": as_of.isoformat(),
        "retention_days": retention_days,
        "delete_before_date": cutoff.isoformat(),
        "catalog_fingerprint": catalog.fingerprint,
        "candidates": [item.to_dict() for item in candidates],
        "blockers": [item.to_dict() for item in blockers],
        "retained_recent_files": recent_files,
        "retained_recent_bytes": recent_bytes,
        "partial_files": list(partial_paths),
    }
    return RetentionPlan(
        raw_root=raw,
        archive_root=archive,
        as_of_date=as_of.isoformat(),
        retention_days=retention_days,
        delete_before_date=cutoff.isoformat(),
        catalog_fingerprint=catalog.fingerprint,
        candidates=tuple(candidates),
        blockers=tuple(blockers),
        retained_recent_files=recent_files,
        retained_recent_bytes=recent_bytes,
        partial_files=partial_paths,
        plan_fingerprint=_sha256_json(fingerprint_payload),
    )


def load_retention_plan(plan_path: str | Path) -> RetentionPlan:
    """Load and fully validate a saved dry-run retention plan."""

    selected = Path(plan_path).expanduser().resolve()
    raw = _load_json(selected, "retention plan")
    if raw.get("retention_plan_schema_version") != RETENTION_PLAN_SCHEMA_VERSION:
        raise ArchiveError("retention plan uses another schema version")
    if raw.get("mode") != "dry_run":
        raise ArchiveError("retention plan mode must be dry_run")
    if raw.get("deletion_performed") is not False:
        raise ArchiveError("retention plan must not report a performed deletion")

    raw_root_text = _required_string(raw.get("raw_root"), "raw_root")
    archive_root_text = _required_string(raw.get("archive_root"), "archive_root")
    raw_root = Path(raw_root_text).expanduser().resolve()
    archive_root = Path(archive_root_text).expanduser().resolve()
    _validate_separate_roots(raw_root, archive_root)
    as_of = _parse_date(
        _required_string(raw.get("as_of_date"), "as_of_date"), "as_of_date"
    )
    if as_of > datetime.now(UTC).date():
        raise ArchiveError("retention plan as_of_date cannot be in the future")
    retention_days = _required_nonnegative_int(
        raw.get("retention_days"), "retention_days"
    )
    if retention_days == 0:
        raise ArchiveError("retention_days must be a positive integer")
    cutoff = _parse_date(
        _required_string(raw.get("delete_before_date"), "delete_before_date"),
        "delete_before_date",
    )
    if cutoff != as_of - timedelta(days=retention_days):
        raise ArchiveError("retention plan delete_before_date is inconsistent")
    catalog_fingerprint = _valid_sha256(
        raw.get("catalog_fingerprint"), "catalog_fingerprint"
    )

    raw_candidates = raw.get("candidate_files")
    if not isinstance(raw_candidates, list):
        raise ArchiveError("retention plan candidate_files must be an array")
    candidates: list[RetentionCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _json_object(
            raw_candidate, f"retention plan candidate_files[{index}]"
        )
        relative = _safe_relative_path(
            candidate.get("path"), f"candidate_files[{index}].path"
        )
        path_text = relative.as_posix()
        if not path_text.endswith(".jsonl"):
            raise ArchiveError(
                f"candidate_files[{index}].path must name a completed JSONL file"
            )
        partition = _raw_partition(path_text)
        if partition is None:
            raise ArchiveError(
                f"candidate_files[{index}].path has an invalid raw partition"
            )
        partition_text = _parse_date(
            _required_string(
                candidate.get("partition_date"),
                f"candidate_files[{index}].partition_date",
            ),
            f"candidate_files[{index}].partition_date",
        ).isoformat()
        if partition.isoformat() != partition_text:
            raise ArchiveError(
                f"candidate_files[{index}] path and partition_date differ"
            )
        if partition >= cutoff:
            raise ArchiveError(
                f"candidate_files[{index}] is not older than delete_before_date"
            )
        candidates.append(
            RetentionCandidate(
                path=path_text,
                partition_date=partition_text,
                bytes=_required_nonnegative_int(
                    candidate.get("bytes"), f"candidate_files[{index}].bytes"
                ),
                sha256=_valid_sha256(
                    candidate.get("sha256"), f"candidate_files[{index}].sha256"
                ),
            )
        )
    candidate_paths = tuple(item.path for item in candidates)
    if candidate_paths != tuple(sorted(candidate_paths)):
        raise ArchiveError("retention plan candidate_files must be sorted by path")
    if len(set(candidate_paths)) != len(candidate_paths):
        raise ArchiveError("retention plan candidate_files contain duplicate paths")
    candidate_file_count = _required_nonnegative_int(
        raw.get("candidate_file_count"), "candidate_file_count"
    )
    candidate_bytes = _required_nonnegative_int(
        raw.get("candidate_bytes"), "candidate_bytes"
    )
    if candidate_file_count != len(candidates):
        raise ArchiveError("retention plan candidate_file_count is inconsistent")
    if candidate_bytes != sum(item.bytes for item in candidates):
        raise ArchiveError("retention plan candidate_bytes is inconsistent")

    raw_blockers = raw.get("blockers")
    if not isinstance(raw_blockers, list):
        raise ArchiveError("retention plan blockers must be an array")
    blocker_count = _required_nonnegative_int(
        raw.get("blocker_count"), "blocker_count"
    )
    if raw_blockers or blocker_count != 0:
        raise ArchiveError("retention plan contains blockers")

    raw_partials = raw.get("partial_files")
    if not isinstance(raw_partials, list):
        raise ArchiveError("retention plan partial_files must be an array")
    partial_files: list[str] = []
    for index, value in enumerate(raw_partials):
        relative = _safe_relative_path(value, f"partial_files[{index}]")
        path_text = relative.as_posix()
        if not path_text.endswith(".jsonl.partial"):
            raise ArchiveError(
                f"partial_files[{index}] must name an active JSONL partial file"
            )
        if _raw_partition(path_text.removesuffix(".partial")) is None:
            raise ArchiveError(f"partial_files[{index}] has an invalid raw partition")
        partial_files.append(path_text)
    if tuple(partial_files) != tuple(sorted(partial_files)):
        raise ArchiveError("retention plan partial_files must be sorted by path")
    if len(set(partial_files)) != len(partial_files):
        raise ArchiveError("retention plan partial_files contain duplicate paths")
    partial_file_count = _required_nonnegative_int(
        raw.get("partial_file_count"), "partial_file_count"
    )
    if partial_file_count != len(partial_files):
        raise ArchiveError("retention plan partial_file_count is inconsistent")

    retained_recent_files = _required_nonnegative_int(
        raw.get("retained_recent_files"), "retained_recent_files"
    )
    retained_recent_bytes = _required_nonnegative_int(
        raw.get("retained_recent_bytes"), "retained_recent_bytes"
    )
    safe_to_apply = bool(candidates)
    if raw.get("safe_to_apply") is not safe_to_apply:
        raise ArchiveError("retention plan safe_to_apply is inconsistent")
    if not safe_to_apply:
        raise ArchiveError("retention plan has no candidate files")

    expected_fingerprint = _valid_sha256(
        raw.get("plan_fingerprint"), "plan_fingerprint"
    )
    fingerprint_payload = {
        "retention_plan_schema_version": RETENTION_PLAN_SCHEMA_VERSION,
        "raw_root": raw_root_text,
        "archive_root": archive_root_text,
        "as_of_date": as_of.isoformat(),
        "retention_days": retention_days,
        "delete_before_date": cutoff.isoformat(),
        "catalog_fingerprint": catalog_fingerprint,
        "candidates": [item.to_dict() for item in candidates],
        "blockers": [],
        "retained_recent_files": retained_recent_files,
        "retained_recent_bytes": retained_recent_bytes,
        "partial_files": partial_files,
    }
    if _sha256_json(fingerprint_payload) != expected_fingerprint:
        raise ArchiveError("retention plan fingerprint does not match its contents")
    return RetentionPlan(
        raw_root=raw_root,
        archive_root=archive_root,
        as_of_date=as_of.isoformat(),
        retention_days=retention_days,
        delete_before_date=cutoff.isoformat(),
        catalog_fingerprint=catalog_fingerprint,
        candidates=tuple(candidates),
        blockers=(),
        retained_recent_files=retained_recent_files,
        retained_recent_bytes=retained_recent_bytes,
        partial_files=tuple(partial_files),
        plan_fingerprint=expected_fingerprint,
    )


def _validate_retention_control_path(
    path: Path,
    *,
    raw_root: Path,
    archive_root: Path,
    label: str,
) -> None:
    if path in (raw_root, archive_root) or path.is_relative_to(
        raw_root
    ) or path.is_relative_to(archive_root):
        raise ArchiveError(f"{label} must be outside raw and archive roots")


def _validate_destructive_raw_root(raw_root: Path) -> None:
    filesystem_root = Path(raw_root.anchor).resolve()
    if raw_root == filesystem_root:
        raise ArchiveError("refusing retention apply against a filesystem root")
    if raw_root == Path.home().resolve():
        raise ArchiveError("refusing retention apply against the current home directory")
    if not raw_root.is_dir():
        raise ArchiveError(f"raw root does not exist: {raw_root}")


def _candidate_diff_details(
    approved: Sequence[RetentionCandidate],
    current: Sequence[RetentionCandidate],
) -> dict[str, object]:
    approved_by_path = {item.path: item for item in approved}
    current_by_path = {item.path: item for item in current}
    missing = sorted(set(approved_by_path) - set(current_by_path))
    unexpected = sorted(set(current_by_path) - set(approved_by_path))
    changed = sorted(
        path
        for path in set(approved_by_path) & set(current_by_path)
        if approved_by_path[path] != current_by_path[path]
    )
    return {
        "approved_candidate_file_count": len(approved),
        "current_candidate_file_count": len(current),
        "missing_approved_file_count": len(missing),
        "unexpected_candidate_file_count": len(unexpected),
        "changed_candidate_file_count": len(changed),
        "missing_approved_file_samples": missing[:20],
        "unexpected_candidate_file_samples": unexpected[:20],
        "changed_candidate_file_samples": changed[:20],
    }


def _verified_candidate_path(raw_root: Path, candidate: RetentionCandidate) -> Path:
    relative = _safe_relative_path(candidate.path, "retention candidate path")
    path = raw_root.joinpath(*relative.parts)
    cursor = raw_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArchiveError(
                f"retention candidate path contains a symbolic link: {candidate.path}"
            )
    try:
        resolved = path.resolve(strict=True)
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArchiveError(
            f"retention candidate is unavailable: {candidate.path}: {exc}"
        ) from exc
    if not resolved.is_relative_to(raw_root):
        raise ArchiveError(f"retention candidate escapes raw root: {candidate.path}")
    if not stat.S_ISREG(before.st_mode):
        raise ArchiveError(f"retention candidate is not a regular file: {candidate.path}")
    if before.st_size != candidate.bytes:
        raise ArchiveError(f"retention candidate size changed: {candidate.path}")
    if _sha256_file(path) != candidate.sha256:
        raise ArchiveError(f"retention candidate SHA-256 changed: {candidate.path}")
    try:
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArchiveError(
            f"retention candidate changed during verification: {candidate.path}: {exc}"
        ) from exc
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise ArchiveError(
            f"retention candidate changed during verification: {candidate.path}"
        )
    return path


def _retention_apply_payload(
    *,
    ok: bool,
    status: str,
    plan: RetentionPlan,
    plan_path: Path,
    plan_sha256: str,
    receipt_path: Path,
    current_catalog_fingerprint: str,
    started_at: str,
    completed_at: str | None,
    deleted_files: Sequence[RetentionCandidate],
    current_partition_date: str | None,
    retained_recent_files: int,
    retained_recent_bytes: int,
    partial_files: Sequence[str],
    error: str | None = None,
    failed_path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "retention_apply_schema_version": RETENTION_APPLY_SCHEMA_VERSION,
        "ok": ok,
        "mode": "apply",
        "status": status,
        "deletion_performed": bool(deleted_files),
        "started_at": started_at,
        "completed_at": completed_at,
        "raw_root": plan.raw_root.as_posix(),
        "archive_root": plan.archive_root.as_posix(),
        "source_plan_path": plan_path.as_posix(),
        "source_plan_sha256": plan_sha256,
        "receipt_path": receipt_path.as_posix(),
        "plan_fingerprint": plan.plan_fingerprint,
        "approved_catalog_fingerprint": plan.catalog_fingerprint,
        "current_catalog_fingerprint": current_catalog_fingerprint,
        "as_of_date": plan.as_of_date,
        "retention_days": plan.retention_days,
        "delete_before_date": plan.delete_before_date,
        "planned_file_count": len(plan.candidates),
        "planned_bytes": sum(item.bytes for item in plan.candidates),
        "deleted_files": [item.to_dict() for item in deleted_files],
        "deleted_file_count": len(deleted_files),
        "deleted_bytes": sum(item.bytes for item in deleted_files),
        "deleted_partition_dates": sorted(
            {item.partition_date for item in deleted_files}
        ),
        "current_partition_date": current_partition_date,
        "retained_recent_files": retained_recent_files,
        "retained_recent_bytes": retained_recent_bytes,
        "partial_files": list(partial_files),
        "partial_file_count": len(partial_files),
        "error": error,
        "failed_path": failed_path,
    }
    payload["receipt_fingerprint"] = _sha256_json(payload)
    return payload


def apply_raw_retention(
    raw_root: str | Path,
    archive_root: str | Path,
    *,
    plan_path: str | Path,
    confirmed_plan_fingerprint: str,
    receipt_path: str | Path,
) -> dict[str, object]:
    """Revalidate and apply one explicitly approved retention plan.

    The saved plan, configured roots, catalog fingerprint, canonical archives, raw
    candidate set, sizes and SHA-256 values must still match.  A durable receipt is
    written before the first unlink and after every completed UTC partition.
    """

    raw = Path(raw_root).expanduser().resolve()
    archive = Path(archive_root).expanduser().resolve()
    selected_plan = Path(plan_path).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    _validate_separate_roots(raw, archive)
    _validate_destructive_raw_root(raw)
    if not archive.is_dir():
        raise ArchiveError(f"archive root does not exist: {archive}")
    _validate_retention_control_path(
        selected_plan,
        raw_root=raw,
        archive_root=archive,
        label="retention plan",
    )
    _validate_retention_control_path(
        receipt,
        raw_root=raw,
        archive_root=archive,
        label="retention receipt",
    )
    if selected_plan == receipt:
        raise ArchiveError("retention plan and receipt paths must differ")
    if receipt.exists():
        raise ArchiveError(f"retention receipt already exists: {receipt}")

    plan = load_retention_plan(selected_plan)
    confirmation = _valid_sha256(
        confirmed_plan_fingerprint, "confirmed_plan_fingerprint"
    )
    if confirmation != plan.plan_fingerprint:
        raise ArchiveError("confirmed plan fingerprint does not match the saved plan")
    if plan.raw_root != raw:
        raise ArchiveError("saved plan raw_root does not match the configured raw root")
    if plan.archive_root != archive:
        raise ArchiveError(
            "saved plan archive_root does not match the configured archive root"
        )
    plan_sha256 = _sha256_file(selected_plan)

    with _archive_lock(archive):
        current = plan_raw_retention(
            raw,
            archive,
            retention_days=plan.retention_days,
            as_of_date=plan.as_of_date,
        )
        if current.catalog_fingerprint != plan.catalog_fingerprint:
            raise ArchiveError(
                "archive catalog changed after the approved retention plan was created",
                details={
                    "approved_catalog_fingerprint": plan.catalog_fingerprint,
                    "current_catalog_fingerprint": current.catalog_fingerprint,
                },
            )
        if current.blockers:
            raise ArchiveError(
                "current retention preflight contains blockers",
                details={
                    "blocker_count": len(current.blockers),
                    "blockers": [item.to_dict() for item in current.blockers[:100]],
                    "blockers_truncated": len(current.blockers) > 100,
                },
            )
        if current.candidates != plan.candidates:
            raise ArchiveError(
                "current retention candidates do not exactly match the approved plan",
                details=_candidate_diff_details(plan.candidates, current.candidates),
            )

        started_at = _utc_now_text()
        deleted: list[RetentionCandidate] = []
        current_partition: str | None = None
        active_candidate: RetentionCandidate | None = None
        preflight = _retention_apply_payload(
            ok=False,
            status="preflight_complete",
            plan=plan,
            plan_path=selected_plan,
            plan_sha256=plan_sha256,
            receipt_path=receipt,
            current_catalog_fingerprint=current.catalog_fingerprint,
            started_at=started_at,
            completed_at=None,
            deleted_files=deleted,
            current_partition_date=None,
            retained_recent_files=current.retained_recent_files,
            retained_recent_bytes=current.retained_recent_bytes,
            partial_files=current.partial_files,
        )
        _write_json_atomic(receipt, preflight)

        candidates_by_date: dict[str, list[RetentionCandidate]] = {}
        for candidate in plan.candidates:
            candidates_by_date.setdefault(candidate.partition_date, []).append(candidate)
        try:
            for partition_date in sorted(candidates_by_date):
                current_partition = partition_date
                for candidate in candidates_by_date[partition_date]:
                    active_candidate = candidate
                    path = _verified_candidate_path(raw, candidate)
                    try:
                        path.unlink()
                    except OSError as exc:
                        raise ArchiveError(
                            f"could not delete retention candidate {candidate.path}: {exc}"
                        ) from exc
                    deleted.append(candidate)
                active_candidate = None
                progress = _retention_apply_payload(
                    ok=False,
                    status="applying",
                    plan=plan,
                    plan_path=selected_plan,
                    plan_sha256=plan_sha256,
                    receipt_path=receipt,
                    current_catalog_fingerprint=current.catalog_fingerprint,
                    started_at=started_at,
                    completed_at=None,
                    deleted_files=deleted,
                    current_partition_date=current_partition,
                    retained_recent_files=current.retained_recent_files,
                    retained_recent_bytes=current.retained_recent_bytes,
                    partial_files=current.partial_files,
                )
                _write_json_atomic(receipt, progress)
        except (ArchiveError, OSError) as exc:
            failure = _retention_apply_payload(
                ok=False,
                status="failed",
                plan=plan,
                plan_path=selected_plan,
                plan_sha256=plan_sha256,
                receipt_path=receipt,
                current_catalog_fingerprint=current.catalog_fingerprint,
                started_at=started_at,
                completed_at=_utc_now_text(),
                deleted_files=deleted,
                current_partition_date=current_partition,
                retained_recent_files=current.retained_recent_files,
                retained_recent_bytes=current.retained_recent_bytes,
                partial_files=current.partial_files,
                error=str(exc),
                failed_path=(None if active_candidate is None else active_candidate.path),
            )
            _write_json_atomic(receipt, failure)
            raise ArchiveError(str(exc), details=failure) from exc

        result = _retention_apply_payload(
            ok=True,
            status="complete",
            plan=plan,
            plan_path=selected_plan,
            plan_sha256=plan_sha256,
            receipt_path=receipt,
            current_catalog_fingerprint=current.catalog_fingerprint,
            started_at=started_at,
            completed_at=_utc_now_text(),
            deleted_files=deleted,
            current_partition_date=None,
            retained_recent_files=current.retained_recent_files,
            retained_recent_bytes=current.retained_recent_bytes,
            partial_files=current.partial_files,
        )
        _write_json_atomic(receipt, result)
        return result
