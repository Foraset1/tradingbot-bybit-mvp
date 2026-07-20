from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from tradingbot.config import StorageConfig
from tradingbot.market.records import MarketRecord

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(slots=True)
class _Segment:
    partial_path: Path
    final_path: Path
    handle: TextIO
    opened_monotonic: float
    partition_date: str
    bytes_written: int = 0


class SegmentedJsonlWriter:
    """Append normalized records to short, recoverable JSONL segments."""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._segments: dict[tuple[str, str], _Segment] = {}
        self._last_flush = time.monotonic()
        self._config.root.mkdir(parents=True, exist_ok=True)
        self._recover_partials()

    @staticmethod
    def _safe(value: str) -> str:
        if not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"Unsafe storage path component: {value!r}")
        return value

    def _recover_partials(self) -> None:
        for partial in self._config.root.rglob("*.jsonl.partial"):
            self._truncate_incomplete_line(partial)
            recovered = partial.with_name(
                f"{partial.name.removesuffix('.jsonl.partial')}-recovered.jsonl"
            )
            counter = 1
            while recovered.exists():
                recovered = partial.with_name(
                    f"{partial.name.removesuffix('.jsonl.partial')}-recovered-{counter}.jsonl"
                )
                counter += 1
            os.replace(partial, recovered)

    @staticmethod
    def _truncate_incomplete_line(path: Path, chunk_size: int = 64 * 1024) -> None:
        """Drop only an incomplete trailing record left by an interrupted write."""
        with path.open("rb+") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return

            cursor = size
            while cursor > 0:
                start = max(0, cursor - chunk_size)
                handle.seek(start)
                chunk = handle.read(cursor - start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    handle.truncate(start + newline + 1)
                    return
                cursor = start
            handle.truncate(0)

    def _open_segment(self, record: MarketRecord, partition: datetime) -> _Segment:
        kind = self._safe(record.kind)
        symbol = self._safe(record.symbol)
        directory = (
            self._config.root
            / kind
            / symbol
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / f"{partition.day:02d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"part-{stamp}-{uuid.uuid4().hex[:8]}"
        final_path = directory / f"{stem}.jsonl"
        partial_path = directory / f"{stem}.jsonl.partial"
        handle = partial_path.open("a", encoding="utf-8", newline="\n")
        return _Segment(
            partial_path=partial_path,
            final_path=final_path,
            handle=handle,
            opened_monotonic=time.monotonic(),
            partition_date=partition.date().isoformat(),
        )

    def _finalize(self, key: tuple[str, str]) -> None:
        segment = self._segments.pop(key, None)
        if segment is None:
            return
        segment.handle.flush()
        segment.handle.close()
        os.replace(segment.partial_path, segment.final_path)

    def write(self, record: MarketRecord) -> None:
        partition = datetime.fromtimestamp(record.exchange_ts_ms / 1000, tz=UTC)
        partition_date = partition.date().isoformat()
        key = (record.kind, record.symbol)
        encoded = json.dumps(
            record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ) + "\n"
        encoded_size = len(encoded.encode("utf-8"))

        segment = self._segments.get(key)
        if segment is not None:
            age = time.monotonic() - segment.opened_monotonic
            should_rotate = (
                segment.partition_date != partition_date
                or age >= self._config.segment_seconds
                or segment.bytes_written + encoded_size > self._config.segment_max_bytes
            )
            if should_rotate:
                self._finalize(key)
                segment = None
        if segment is None:
            segment = self._open_segment(record, partition)
            self._segments[key] = segment

        segment.handle.write(encoded)
        segment.bytes_written += encoded_size

    def flush_if_due(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_flush < self._config.flush_seconds:
            return
        for segment in self._segments.values():
            segment.handle.flush()
        self._last_flush = now

    def close(self) -> None:
        for key in list(self._segments):
            self._finalize(key)
