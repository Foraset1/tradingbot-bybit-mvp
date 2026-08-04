"""Stream official Bybit trade archives into compact, immutable price history.

The public daily files can be very large.  This importer never stores the source
``.csv.gz`` and never pretends that an offline exchange timestamp is a locally
observed receive timestamp.  It streams each archive into one-second and one-minute
trade bars, records the compressed source SHA-256, and assigns a conservative
``available_at_ns`` using an explicit configured latency assumption.

This is a separate ``price_futures_v1`` profile.  It contains no order book, ticker,
queue position, funding, or open-interest history and cannot be passed to the
microstructure research builder as if those fields existed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import math
import os
import platform
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from tradingbot import __version__

HISTORY_DAY_SCHEMA_VERSION: Final = 1
HISTORY_SYMBOL_SCHEMA_VERSION: Final = 1
HISTORY_CATALOG_SCHEMA_VERSION: Final = 1
HISTORY_PROFILE: Final = "price_futures_v1"
SOURCE_NAME: Final = "bybit_public_trade_archive"
PARQUET_FORMAT_VERSION: Final = "2.6"
PARQUET_COMPRESSION: Final = "zstd"
PARQUET_COMPRESSION_LEVEL: Final = 3
BAR_BATCH_ROWS: Final = 10_000
NANOSECONDS_PER_SECOND: Final = 1_000_000_000
MILLISECONDS_PER_DAY: Final = 86_400_000
EXPECTED_MINUTES_PER_DAY: Final = 1_440
USER_AGENT: Final = f"tradingbot-bybit/{__version__} history-import"

LOGGER = logging.getLogger(__name__)


class HistoryImportError(RuntimeError):
    """Raised when official history cannot be imported without ambiguity."""


class _ArchiveResponse(Protocol):
    headers: Any

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


_BAR_METADATA: Final = {
    b"tradingbot.history_day_schema_version": str(HISTORY_DAY_SCHEMA_VERSION).encode("ascii"),
    b"tradingbot.dataset_profile": HISTORY_PROFILE.encode("ascii"),
    b"tradingbot.causal_key": b"available_at_ns",
}

TRADE_BAR_SCHEMA: Final = pa.schema(
    (
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval_seconds", pa.int32(), nullable=False),
        pa.field("start_ms", pa.int64(), nullable=False),
        pa.field("end_ms", pa.int64(), nullable=False),
        pa.field("available_at_ns", pa.int64(), nullable=False),
        pa.field("first_event_ns", pa.int64(), nullable=False),
        pa.field("last_event_ns", pa.int64(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("turnover", pa.float64(), nullable=False),
        pa.field("buy_volume", pa.float64(), nullable=False),
        pa.field("sell_volume", pa.float64(), nullable=False),
        pa.field("buy_turnover", pa.float64(), nullable=False),
        pa.field("sell_turnover", pa.float64(), nullable=False),
        pa.field("trade_count", pa.int64(), nullable=False),
    ),
    metadata=_BAR_METADATA,
)


@dataclass(frozen=True, slots=True)
class HistoryFile:
    path: str
    kind: str
    symbol: str
    date: str
    rows: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "symbol": self.symbol,
            "date": self.date,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class HistorySource:
    symbol: str
    partition_date: str
    url: str
    compressed_bytes: int
    compressed_sha256: str
    etag: str | None
    last_modified: str | None
    content_length: int | None
    csv_header: tuple[str, ...]
    source_rows: int
    first_event_ns: int
    last_event_ns: int
    adjacent_duplicate_trade_ids: int

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "partition_date": self.partition_date,
            "url": self.url,
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "content_length": self.content_length,
            "csv_header": list(self.csv_header),
            "source_rows": self.source_rows,
            "first_event_ns": self.first_event_ns,
            "last_event_ns": self.last_event_ns,
            "source_retained": False,
            "global_trade_id_deduplication": False,
            "adjacent_duplicate_trade_ids": self.adjacent_duplicate_trade_ids,
        }


@dataclass(frozen=True, slots=True)
class _SymbolArtifact:
    symbol: str
    source: HistorySource
    files: tuple[HistoryFile, ...]
    rows_by_kind: dict[str, int]
    missing_minutes: int
    seconds_with_trades: int
    symbol_manifest_sha256: str
    reused: bool


@dataclass(frozen=True, slots=True)
class HistoryDayResult:
    partition_date: str
    dataset_path: Path
    manifest_path: Path
    parameters_fingerprint: str
    output_fingerprint: str
    symbols: tuple[str, ...]
    source_rows: int
    output_files: int
    output_rows: int
    output_bytes: int
    output_rows_by_kind: dict[str, int]
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
            "dataset_profile": HISTORY_PROFILE,
            "partition_date": self.partition_date,
            "dataset_path": self.dataset_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "parameters_fingerprint": self.parameters_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "symbols": list(self.symbols),
            "source_rows": self.source_rows,
            "output_files": self.output_files,
            "output_rows": self.output_rows,
            "output_bytes": self.output_bytes,
            "output_rows_by_kind": dict(sorted(self.output_rows_by_kind.items())),
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class HistoryRangeResult:
    start_date: str
    end_date: str
    history_root: Path
    catalog_path: Path
    catalog_fingerprint: str
    days: int
    imported_days: int
    reused_days: int
    source_rows: int
    output_rows: int
    output_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
            "history_catalog_schema_version": HISTORY_CATALOG_SCHEMA_VERSION,
            "dataset_profile": HISTORY_PROFILE,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "history_root": self.history_root.as_posix(),
            "catalog_path": self.catalog_path.as_posix(),
            "catalog_fingerprint": self.catalog_fingerprint,
            "days": self.days,
            "imported_days": self.imported_days,
            "reused_days": self.reused_days,
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "output_bytes": self.output_bytes,
        }


@dataclass(slots=True)
class _TradeBar:
    symbol: str
    interval_seconds: int
    start_ms: int
    first_event_ns: int
    last_event_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    buy_turnover: Decimal
    sell_turnover: Decimal
    trade_count: int

    @classmethod
    def first(
        cls,
        *,
        symbol: str,
        interval_seconds: int,
        start_ms: int,
        event_ns: int,
        price: Decimal,
        size: Decimal,
        side: str,
    ) -> _TradeBar:
        turnover = price * size
        is_buy = side == "Buy"
        return cls(
            symbol=symbol,
            interval_seconds=interval_seconds,
            start_ms=start_ms,
            first_event_ns=event_ns,
            last_event_ns=event_ns,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            turnover=turnover,
            buy_volume=size if is_buy else Decimal(0),
            sell_volume=Decimal(0) if is_buy else size,
            buy_turnover=turnover if is_buy else Decimal(0),
            sell_turnover=Decimal(0) if is_buy else turnover,
            trade_count=1,
        )

    def observe(self, *, event_ns: int, price: Decimal, size: Decimal, side: str) -> None:
        turnover = price * size
        self.last_event_ns = event_ns
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.turnover += turnover
        if side == "Buy":
            self.buy_volume += size
            self.buy_turnover += turnover
        else:
            self.sell_volume += size
            self.sell_turnover += turnover
        self.trade_count += 1

    def to_row(self, assumed_latency_ms: int) -> dict[str, object]:
        end_ms = self.start_ms + self.interval_seconds * 1_000 - 1
        available_at_ns = (end_ms + 1 + assumed_latency_ms) * 1_000_000
        return {
            "schema_version": HISTORY_DAY_SCHEMA_VERSION,
            "source": SOURCE_NAME,
            "symbol": self.symbol,
            "interval_seconds": self.interval_seconds,
            "start_ms": self.start_ms,
            "end_ms": end_ms,
            "available_at_ns": available_at_ns,
            "first_event_ns": self.first_event_ns,
            "last_event_ns": self.last_event_ns,
            "open": _finite_float(self.open, "bar.open"),
            "high": _finite_float(self.high, "bar.high"),
            "low": _finite_float(self.low, "bar.low"),
            "close": _finite_float(self.close, "bar.close"),
            "volume": _finite_float(self.volume, "bar.volume"),
            "turnover": _finite_float(self.turnover, "bar.turnover"),
            "buy_volume": _finite_float(self.buy_volume, "bar.buy_volume"),
            "sell_volume": _finite_float(self.sell_volume, "bar.sell_volume"),
            "buy_turnover": _finite_float(self.buy_turnover, "bar.buy_turnover"),
            "sell_turnover": _finite_float(self.sell_turnover, "bar.sell_turnover"),
            "trade_count": self.trade_count,
        }


class _HashingReader(io.RawIOBase):
    def __init__(self, source: _ArchiveResponse) -> None:
        super().__init__()
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        if not isinstance(chunk, bytes):
            raise OSError("archive response returned non-byte data")
        self.digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


def _finite_float(value: Decimal, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise HistoryImportError(f"{label} cannot be represented as a finite float")
    return converted


class _BarWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(
            path,
            TRADE_BAR_SCHEMA,
            version=PARQUET_FORMAT_VERSION,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="1.0",
            write_page_index=True,
            write_page_checksum=True,
        )
        self.rows = 0
        self.buffer: list[dict[str, object]] = []
        self.closed = False

    def add(self, bar: _TradeBar, assumed_latency_ms: int) -> None:
        self.buffer.append(bar.to_row(assumed_latency_ms))
        self.rows += 1
        if len(self.buffer) >= BAR_BATCH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=TRADE_BAR_SCHEMA)
        self.writer.write_table(table, row_group_size=len(self.buffer))
        self.buffer.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self.writer.close()
        self.closed = True

    def abort(self) -> None:
        self.buffer.clear()
        if not self.closed:
            with suppress(Exception):
                self.writer.close()
            self.closed = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise HistoryImportError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as target:
            target.write(rendered)
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, path)
    finally:
        with suppress(FileNotFoundError):
            partial.unlink()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HistoryImportError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryImportError(f"{label} is unreadable: {path}") from exc
    return _json_object(parsed, label)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HistoryImportError(f"{label} must be a non-empty string")
    return value


def _required_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HistoryImportError(f"{label} must be an integer >= {minimum}")
    return value


def _required_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise HistoryImportError(f"{label} must be a boolean")
    return value


def _valid_sha256(value: object, label: str) -> str:
    text = _required_string(value, label).lower()
    if len(text) != 64:
        raise HistoryImportError(f"{label} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise HistoryImportError(f"{label} must be a SHA-256 digest") from exc
    return text


def _parse_partition_date(value: str, *, require_complete: bool = True) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HistoryImportError(f"invalid UTC date: {value!r}") from exc
    if parsed.isoformat() != value:
        raise HistoryImportError(f"UTC date must use YYYY-MM-DD: {value!r}")
    if require_complete and parsed >= datetime.now(UTC).date():
        raise HistoryImportError(f"history date must be a completed UTC day, got {value}")
    return parsed


def _validate_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    result = tuple(symbols)
    if not result:
        raise HistoryImportError("at least one symbol is required")
    if len(result) != len(set(result)):
        raise HistoryImportError("history symbols contain duplicates")
    for symbol in result:
        if symbol != symbol.upper() or not symbol.endswith("USDT") or not symbol.isalnum():
            raise HistoryImportError(f"unsupported history symbol: {symbol!r}")
    return result


def _validate_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "public.bybit.com"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/trading"
    ):
        raise HistoryImportError("public_base_url must be https://public.bybit.com/trading")
    return normalized


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise HistoryImportError(f"unsafe {label}: {value!r}")
    return candidate


def _resolve_relative(root: Path, value: str, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise HistoryImportError(f"{label} escapes history root: {value!r}")
    return candidate


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def _files_fingerprint(files: Sequence[HistoryFile]) -> str:
    return _sha256_json([item.to_dict() for item in sorted(files, key=lambda x: x.path)])


def _parameters_payload(
    *,
    partition_date: str,
    symbols: Sequence[str],
    public_base_url: str,
    assumed_latency_ms: int,
    maximum_missing_minutes: int,
) -> dict[str, object]:
    return {
        "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
        "dataset_profile": HISTORY_PROFILE,
        "partition_date": partition_date,
        "symbols": list(symbols),
        "public_base_url": public_base_url,
        "assumed_latency_ms": assumed_latency_ms,
        "maximum_missing_minutes": maximum_missing_minutes,
        "output_intervals_seconds": [1, 60],
        "retain_source_archive": False,
        "retain_individual_trades": False,
    }


def _archive_url(base_url: str, symbol: str, partition_date: str) -> str:
    return f"{base_url}/{symbol}/{symbol}{partition_date}.csv.gz"


def _open_url(url: str, timeout_seconds: int) -> _ArchiveResponse:
    request = Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": USER_AGENT},
        method="GET",
    )
    return cast(_ArchiveResponse, urlopen(request, timeout=timeout_seconds))


def _header(response: _ArchiveResponse, name: str) -> str | None:
    value: object = response.headers.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HistoryImportError(f"HTTP header {name} is not text")
    return value


def _content_length(response: _ArchiveResponse) -> int | None:
    raw = _header(response, "Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise HistoryImportError("HTTP Content-Length is not an integer") from exc
    if value <= 0:
        raise HistoryImportError("HTTP Content-Length must be positive")
    return value


def _decimal(value: object, label: str, *, positive: bool) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise HistoryImportError(f"{label} is missing")
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise HistoryImportError(f"{label} is not numeric: {value!r}") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        requirement = "positive and finite" if positive else "finite"
        raise HistoryImportError(f"{label} must be {requirement}")
    return parsed


def _event_ns(value: object, label: str) -> int:
    seconds = _decimal(value, label, positive=True)
    scaled = seconds * NANOSECONDS_PER_SECOND
    integral = scaled.to_integral_value(rounding=ROUND_FLOOR)
    if scaled != integral:
        raise HistoryImportError(f"{label} has precision finer than one nanosecond")
    return int(integral)


def _field_map(fieldnames: Sequence[str] | None) -> tuple[tuple[str, ...], dict[str, str]]:
    if not fieldnames:
        raise HistoryImportError("archive CSV has no header")
    header = tuple(item.strip() for item in fieldnames)
    if any(not item for item in header):
        raise HistoryImportError("archive CSV contains a blank header")
    normalized: dict[str, str] = {}
    for item in header:
        key = item.lower()
        if key in normalized:
            raise HistoryImportError(f"archive CSV repeats header {item!r}")
        normalized[key] = item
    aliases = {
        "timestamp": ("timestamp",),
        "symbol": ("symbol",),
        "side": ("side",),
        "size": ("size",),
        "price": ("price",),
        "trade_id": ("trdmatchid", "trade_id", "tradeid"),
    }
    result: dict[str, str] = {}
    for logical, candidates in aliases.items():
        match = next((normalized[item] for item in candidates if item in normalized), None)
        if match is None:
            raise HistoryImportError(f"archive CSV is missing required column {logical}")
        result[logical] = match
    return header, result


def _row_value(row: dict[str, str | None], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HistoryImportError(f"{label} is missing")
    return value.strip()


def _bar_start_ms(event_ns: int, interval_seconds: int) -> int:
    interval_ns = interval_seconds * NANOSECONDS_PER_SECOND
    return (event_ns // interval_ns) * interval_seconds * 1_000


def _check_disk(path: Path, minimum_free_bytes: int, additional_bytes: int = 0) -> None:
    free = shutil.disk_usage(path).free
    required = minimum_free_bytes + additional_bytes
    if free < required:
        raise HistoryImportError(f"insufficient disk space: {free} bytes free, {required} required")


def _history_file(
    *, path: Path, relative_path: str, kind: str, symbol: str, partition_date: str, rows: int
) -> HistoryFile:
    return HistoryFile(
        path=relative_path,
        kind=kind,
        symbol=symbol,
        date=partition_date,
        rows=rows,
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _process_archive(
    *,
    response: _ArchiveResponse,
    url: str,
    symbol: str,
    partition_date: str,
    temp_symbol_dir: Path,
    assumed_latency_ms: int,
    maximum_missing_minutes: int,
    minimum_free_bytes: int,
) -> _SymbolArtifact:
    final_url = response.geturl()
    parsed_final = urlparse(final_url)
    if (
        parsed_final.scheme != "https"
        or parsed_final.hostname != "public.bybit.com"
        or final_url != url
    ):
        raise HistoryImportError(f"archive redirected away from expected URL: {final_url}")
    content_length = _content_length(response)
    _check_disk(
        temp_symbol_dir.parent,
        minimum_free_bytes,
        0 if content_length is None else content_length,
    )
    one_second_path = (
        temp_symbol_dir / f"date={partition_date}" / "kind=trade_bar_1s" / "part-00000.parquet"
    )
    one_minute_path = (
        temp_symbol_dir / f"date={partition_date}" / "kind=trade_bar_1m" / "part-00000.parquet"
    )
    second_writer = _BarWriter(one_second_path)
    minute_writer = _BarWriter(one_minute_path)
    hashing_reader = _HashingReader(response)
    source_rows = 0
    adjacent_duplicates = 0
    first_event_ns: int | None = None
    previous_event_ns: int | None = None
    previous_trade_id: str | None = None
    current_second: _TradeBar | None = None
    current_minute: _TradeBar | None = None
    csv_header: tuple[str, ...] = ()
    selected_fields: dict[str, str] = {}
    parsed_date = _parse_partition_date(partition_date)
    day_start_ns = (
        int(datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC).timestamp())
        * NANOSECONDS_PER_SECOND
    )
    day_end_ns = day_start_ns + 86_400 * NANOSECONDS_PER_SECOND
    try:
        with (
            gzip.GzipFile(fileobj=hashing_reader, mode="rb") as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8-sig", errors="strict", newline="") as text,
        ):
            reader = csv.DictReader(text)
            csv_header, selected_fields = _field_map(reader.fieldnames)
            for row in reader:
                source_rows += 1
                line_label = f"{symbol}/{partition_date}:{reader.line_num}"
                if None in row:
                    raise HistoryImportError(f"{line_label} has more values than CSV headers")
                event_ns = _event_ns(
                    row.get(selected_fields["timestamp"]),
                    f"{line_label}.timestamp",
                )
                if not day_start_ns <= event_ns < day_end_ns:
                    raise HistoryImportError(f"{line_label} timestamp is outside requested UTC day")
                if previous_event_ns is not None and event_ns < previous_event_ns:
                    raise HistoryImportError(f"{line_label} timestamp moved backwards")
                row_symbol = _row_value(row, selected_fields["symbol"], f"{line_label}.symbol")
                if row_symbol != symbol:
                    raise HistoryImportError(
                        f"{line_label} contains symbol {row_symbol!r}, expected {symbol}"
                    )
                side = _row_value(row, selected_fields["side"], f"{line_label}.side")
                if side not in {"Buy", "Sell"}:
                    raise HistoryImportError(f"{line_label}.side must be Buy or Sell")
                price = _decimal(
                    row.get(selected_fields["price"]),
                    f"{line_label}.price",
                    positive=True,
                )
                size = _decimal(
                    row.get(selected_fields["size"]),
                    f"{line_label}.size",
                    positive=True,
                )
                trade_id = _row_value(
                    row,
                    selected_fields["trade_id"],
                    f"{line_label}.trade_id",
                )
                if trade_id == previous_trade_id:
                    adjacent_duplicates += 1
                    raise HistoryImportError(f"{line_label} repeats adjacent trade ID {trade_id!r}")

                second_start = _bar_start_ms(event_ns, 1)
                if current_second is None or current_second.start_ms != second_start:
                    if current_second is not None:
                        second_writer.add(current_second, assumed_latency_ms)
                    current_second = _TradeBar.first(
                        symbol=symbol,
                        interval_seconds=1,
                        start_ms=second_start,
                        event_ns=event_ns,
                        price=price,
                        size=size,
                        side=side,
                    )
                else:
                    current_second.observe(event_ns=event_ns, price=price, size=size, side=side)

                minute_start = _bar_start_ms(event_ns, 60)
                if current_minute is None or current_minute.start_ms != minute_start:
                    if current_minute is not None:
                        minute_writer.add(current_minute, assumed_latency_ms)
                    current_minute = _TradeBar.first(
                        symbol=symbol,
                        interval_seconds=60,
                        start_ms=minute_start,
                        event_ns=event_ns,
                        price=price,
                        size=size,
                        side=side,
                    )
                else:
                    current_minute.observe(event_ns=event_ns, price=price, size=size, side=side)

                if first_event_ns is None:
                    first_event_ns = event_ns
                previous_event_ns = event_ns
                previous_trade_id = trade_id
                if source_rows % 250_000 == 0:
                    _check_disk(temp_symbol_dir, minimum_free_bytes)
                    LOGGER.info(
                        "History progress %s %s: %,d source trades",
                        partition_date,
                        symbol,
                        source_rows,
                    )

        if source_rows == 0 or first_event_ns is None or previous_event_ns is None:
            raise HistoryImportError(f"archive contains no trades: {url}")
        assert current_second is not None and current_minute is not None
        second_writer.add(current_second, assumed_latency_ms)
        minute_writer.add(current_minute, assumed_latency_ms)
        second_writer.close()
        minute_writer.close()
    except Exception:
        second_writer.abort()
        minute_writer.abort()
        raise

    if content_length is not None and hashing_reader.bytes_read != content_length:
        raise HistoryImportError(
            "compressed archive length mismatch: "
            f"read {hashing_reader.bytes_read}, expected {content_length}"
        )
    missing_minutes = EXPECTED_MINUTES_PER_DAY - minute_writer.rows
    if missing_minutes < 0:
        raise HistoryImportError(
            f"derived more than {EXPECTED_MINUTES_PER_DAY} minute bars for {symbol}"
        )
    if missing_minutes > maximum_missing_minutes:
        raise HistoryImportError(
            f"{symbol}/{partition_date} is missing {missing_minutes} trade minutes; "
            f"configured maximum is {maximum_missing_minutes}"
        )

    prefix = f"symbols/symbol={symbol}"
    second_relative = f"{prefix}/date={partition_date}/kind=trade_bar_1s/part-00000.parquet"
    minute_relative = f"{prefix}/date={partition_date}/kind=trade_bar_1m/part-00000.parquet"
    files = (
        _history_file(
            path=one_second_path,
            relative_path=second_relative,
            kind="trade_bar_1s",
            symbol=symbol,
            partition_date=partition_date,
            rows=second_writer.rows,
        ),
        _history_file(
            path=one_minute_path,
            relative_path=minute_relative,
            kind="trade_bar_1m",
            symbol=symbol,
            partition_date=partition_date,
            rows=minute_writer.rows,
        ),
    )
    source = HistorySource(
        symbol=symbol,
        partition_date=partition_date,
        url=url,
        compressed_bytes=hashing_reader.bytes_read,
        compressed_sha256=hashing_reader.sha256,
        etag=_header(response, "ETag"),
        last_modified=_header(response, "Last-Modified"),
        content_length=content_length,
        csv_header=csv_header,
        source_rows=source_rows,
        first_event_ns=first_event_ns,
        last_event_ns=previous_event_ns,
        adjacent_duplicate_trade_ids=adjacent_duplicates,
    )
    return _SymbolArtifact(
        symbol=symbol,
        source=source,
        files=files,
        rows_by_kind={
            "trade_bar_1m": minute_writer.rows,
            "trade_bar_1s": second_writer.rows,
        },
        missing_minutes=missing_minutes,
        seconds_with_trades=second_writer.rows,
        symbol_manifest_sha256="",
        reused=False,
    )


def _symbol_manifest_payload(
    artifact: _SymbolArtifact, parameters_fingerprint: str
) -> dict[str, object]:
    return {
        "history_symbol_schema_version": HISTORY_SYMBOL_SCHEMA_VERSION,
        "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
        "dataset_profile": HISTORY_PROFILE,
        "symbol": artifact.symbol,
        "partition_date": artifact.source.partition_date,
        "parameters_fingerprint": parameters_fingerprint,
        "source": artifact.source.to_dict(),
        "quality": {
            "source_timestamp_order": "nondecreasing",
            "adjacent_trade_ids_unique": True,
            "global_trade_id_deduplication": False,
            "missing_minutes": artifact.missing_minutes,
            "seconds_with_trades": artifact.seconds_with_trades,
            "synthetic_bars": 0,
        },
        "output_rows_by_kind": dict(sorted(artifact.rows_by_kind.items())),
        "output_fingerprint": _files_fingerprint(artifact.files),
        "files": [item.to_dict() for item in artifact.files],
    }


def _build_symbol(
    *,
    stage_path: Path,
    symbol: str,
    partition_date: str,
    public_base_url: str,
    parameters_fingerprint: str,
    assumed_latency_ms: int,
    request_timeout_seconds: int,
    download_attempts: int,
    maximum_missing_minutes: int,
    minimum_free_bytes: int,
) -> _SymbolArtifact:
    final_symbol_dir = stage_path / "symbols" / f"symbol={symbol}"
    if final_symbol_dir.exists():
        return _validate_symbol_artifact(
            stage_path=stage_path,
            symbol_dir=final_symbol_dir,
            expected_symbol=symbol,
            expected_date=partition_date,
            expected_parameters_fingerprint=parameters_fingerprint,
            expected_url=_archive_url(public_base_url, symbol, partition_date),
            reused=True,
        )

    url = _archive_url(public_base_url, symbol, partition_date)
    last_error: Exception | None = None
    for attempt in range(1, download_attempts + 1):
        temp_symbol_dir = stage_path / f".symbol={symbol}.tmp-{uuid.uuid4().hex}"
        temp_symbol_dir.mkdir()
        response: _ArchiveResponse | None = None
        try:
            LOGGER.info(
                "Importing official Bybit history %s %s (attempt %d/%d)",
                partition_date,
                symbol,
                attempt,
                download_attempts,
            )
            response = _open_url(url, request_timeout_seconds)
            artifact = _process_archive(
                response=response,
                url=url,
                symbol=symbol,
                partition_date=partition_date,
                temp_symbol_dir=temp_symbol_dir,
                assumed_latency_ms=assumed_latency_ms,
                maximum_missing_minutes=maximum_missing_minutes,
                minimum_free_bytes=minimum_free_bytes,
            )
            manifest_path = temp_symbol_dir / "symbol-manifest.json"
            _write_json_atomic(
                manifest_path,
                _symbol_manifest_payload(artifact, parameters_fingerprint),
            )
            final_symbol_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_symbol_dir, final_symbol_dir)
            return _validate_symbol_artifact(
                stage_path=stage_path,
                symbol_dir=final_symbol_dir,
                expected_symbol=symbol,
                expected_date=partition_date,
                expected_parameters_fingerprint=parameters_fingerprint,
                expected_url=url,
                reused=False,
            )
        except HistoryImportError as exc:
            last_error = exc
            LOGGER.warning(
                "History import failed for %s %s on attempt %d/%d: %s",
                partition_date,
                symbol,
                attempt,
                download_attempts,
                exc,
            )
            break
        except (OSError, EOFError) as exc:
            last_error = exc
            LOGGER.warning(
                "Transient history download failure for %s %s on attempt %d/%d: %s",
                partition_date,
                symbol,
                attempt,
                download_attempts,
                exc,
            )
            if attempt < download_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if response is not None:
                with suppress(Exception):
                    response.close()
            if temp_symbol_dir.is_dir() and temp_symbol_dir.parent == stage_path:
                shutil.rmtree(temp_symbol_dir, ignore_errors=True)
    assert last_error is not None
    raise HistoryImportError(
        f"failed to import {symbol}/{partition_date} after {download_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _history_file_from_dict(value: object, label: str) -> HistoryFile:
    item = _json_object(value, label)
    return HistoryFile(
        path=_required_string(item.get("path"), f"{label}.path"),
        kind=_required_string(item.get("kind"), f"{label}.kind"),
        symbol=_required_string(item.get("symbol"), f"{label}.symbol"),
        date=_required_string(item.get("date"), f"{label}.date"),
        rows=_required_int(item.get("rows"), f"{label}.rows", minimum=1),
        bytes=_required_int(item.get("bytes"), f"{label}.bytes", minimum=1),
        sha256=_valid_sha256(item.get("sha256"), f"{label}.sha256"),
    )


def _history_source_from_dict(value: object, label: str) -> HistorySource:
    source = _json_object(value, label)
    raw_header = source.get("csv_header")
    if not isinstance(raw_header, list) or not all(
        isinstance(item, str) and item for item in raw_header
    ):
        raise HistoryImportError(f"{label}.csv_header must be a list of strings")
    etag = source.get("etag")
    last_modified = source.get("last_modified")
    content_length = source.get("content_length")
    if etag is not None and not isinstance(etag, str):
        raise HistoryImportError(f"{label}.etag must be text or null")
    if last_modified is not None and not isinstance(last_modified, str):
        raise HistoryImportError(f"{label}.last_modified must be text or null")
    if content_length is not None:
        content_length = _required_int(content_length, f"{label}.content_length", minimum=1)
    if _required_bool(source.get("source_retained"), f"{label}.source_retained"):
        raise HistoryImportError(f"{label}.source_retained must be false")
    if _required_bool(
        source.get("global_trade_id_deduplication"),
        f"{label}.global_trade_id_deduplication",
    ):
        raise HistoryImportError(f"{label}.global_trade_id_deduplication must be false")
    return HistorySource(
        symbol=_required_string(source.get("symbol"), f"{label}.symbol"),
        partition_date=_required_string(source.get("partition_date"), f"{label}.partition_date"),
        url=_required_string(source.get("url"), f"{label}.url"),
        compressed_bytes=_required_int(
            source.get("compressed_bytes"), f"{label}.compressed_bytes", minimum=1
        ),
        compressed_sha256=_valid_sha256(
            source.get("compressed_sha256"), f"{label}.compressed_sha256"
        ),
        etag=etag,
        last_modified=last_modified,
        content_length=content_length,
        csv_header=tuple(cast(list[str], raw_header)),
        source_rows=_required_int(source.get("source_rows"), f"{label}.source_rows", minimum=1),
        first_event_ns=_required_int(
            source.get("first_event_ns"), f"{label}.first_event_ns", minimum=1
        ),
        last_event_ns=_required_int(
            source.get("last_event_ns"), f"{label}.last_event_ns", minimum=1
        ),
        adjacent_duplicate_trade_ids=_required_int(
            source.get("adjacent_duplicate_trade_ids"),
            f"{label}.adjacent_duplicate_trade_ids",
        ),
    )


def _validate_parquet_file(root: Path, item: HistoryFile) -> None:
    path = _resolve_relative(root, item.path, "history file path")
    if not path.is_file():
        raise HistoryImportError(f"history file is missing: {path}")
    stat = path.stat()
    if stat.st_size != item.bytes:
        raise HistoryImportError(f"history file size changed: {path}")
    if _sha256_file(path) != item.sha256:
        raise HistoryImportError(f"history file SHA-256 changed: {path}")
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise HistoryImportError(f"history Parquet is unreadable: {path}") from exc
    if parquet_file.metadata.num_rows != item.rows:
        raise HistoryImportError(f"history Parquet row count changed: {path}")
    if not parquet_file.schema_arrow.equals(TRADE_BAR_SCHEMA, check_metadata=True):
        raise HistoryImportError(f"history Parquet schema changed: {path}")


def _validate_symbol_artifact(
    *,
    stage_path: Path,
    symbol_dir: Path,
    expected_symbol: str,
    expected_date: str,
    expected_parameters_fingerprint: str,
    expected_url: str,
    reused: bool,
) -> _SymbolArtifact:
    manifest_path = symbol_dir / "symbol-manifest.json"
    manifest = _load_json(manifest_path, "history symbol manifest")
    if manifest.get("history_symbol_schema_version") != HISTORY_SYMBOL_SCHEMA_VERSION:
        raise HistoryImportError(f"unsupported history symbol manifest: {manifest_path}")
    if manifest.get("history_day_schema_version") != HISTORY_DAY_SCHEMA_VERSION:
        raise HistoryImportError(f"history day schema mismatch: {manifest_path}")
    if manifest.get("dataset_profile") != HISTORY_PROFILE:
        raise HistoryImportError(f"history profile mismatch: {manifest_path}")
    if manifest.get("symbol") != expected_symbol or manifest.get("partition_date") != expected_date:
        raise HistoryImportError(f"history symbol/date mismatch: {manifest_path}")
    if manifest.get("parameters_fingerprint") != expected_parameters_fingerprint:
        raise HistoryImportError(f"history parameter mismatch: {manifest_path}")
    source = _history_source_from_dict(manifest.get("source"), "source")
    if (
        source.symbol != expected_symbol
        or source.partition_date != expected_date
        or source.url != expected_url
    ):
        raise HistoryImportError(f"history source mismatch: {manifest_path}")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        raise HistoryImportError(f"history symbol must contain exactly two files: {manifest_path}")
    files = tuple(
        _history_file_from_dict(item, f"files[{index}]") for index, item in enumerate(raw_files)
    )
    expected_kinds = {"trade_bar_1s", "trade_bar_1m"}
    if {item.kind for item in files} != expected_kinds:
        raise HistoryImportError(f"history symbol has unexpected kinds: {manifest_path}")
    prefix = f"symbols/symbol={expected_symbol}/date={expected_date}/"
    for item in files:
        if item.symbol != expected_symbol or item.date != expected_date:
            raise HistoryImportError(f"history descriptor partition mismatch: {item.path}")
        if not item.path.startswith(prefix):
            raise HistoryImportError(f"history descriptor path mismatch: {item.path}")
        _validate_parquet_file(stage_path, item)
    if manifest.get("output_fingerprint") != _files_fingerprint(files):
        raise HistoryImportError(f"history symbol fingerprint mismatch: {manifest_path}")
    raw_rows = _json_object(manifest.get("output_rows_by_kind"), "output_rows_by_kind")
    rows_by_kind = {
        key: _required_int(value, f"output_rows_by_kind.{key}", minimum=1)
        for key, value in raw_rows.items()
    }
    actual_rows = {item.kind: item.rows for item in files}
    if rows_by_kind != actual_rows:
        raise HistoryImportError(f"history row totals mismatch: {manifest_path}")
    quality = _json_object(manifest.get("quality"), "quality")
    if quality.get("source_timestamp_order") != "nondecreasing":
        raise HistoryImportError(f"history timestamp quality mismatch: {manifest_path}")
    if quality.get("adjacent_trade_ids_unique") is not True:
        raise HistoryImportError(f"history duplicate quality mismatch: {manifest_path}")
    if quality.get("global_trade_id_deduplication") is not False:
        raise HistoryImportError(f"history deduplication claim mismatch: {manifest_path}")
    if _required_int(quality.get("synthetic_bars"), "quality.synthetic_bars") != 0:
        raise HistoryImportError(f"history must not contain synthetic bars: {manifest_path}")
    missing_minutes = _required_int(quality.get("missing_minutes"), "quality.missing_minutes")
    seconds_with_trades = _required_int(
        quality.get("seconds_with_trades"), "quality.seconds_with_trades", minimum=1
    )
    return _SymbolArtifact(
        symbol=expected_symbol,
        source=source,
        files=files,
        rows_by_kind=rows_by_kind,
        missing_minutes=missing_minutes,
        seconds_with_trades=seconds_with_trades,
        symbol_manifest_sha256=_sha256_file(manifest_path),
        reused=reused,
    )


def _day_manifest_payload(
    *,
    partition_date: str,
    symbols: Sequence[str],
    parameters: dict[str, object],
    parameters_fingerprint: str,
    artifacts: Sequence[_SymbolArtifact],
) -> dict[str, object]:
    files = tuple(
        sorted(
            (item for artifact in artifacts for item in artifact.files),
            key=lambda item: item.path,
        )
    )
    rows_by_kind: Counter[str] = Counter()
    for item in files:
        rows_by_kind[item.kind] += item.rows
    return {
        "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
        "dataset_profile": HISTORY_PROFILE,
        "dataset_id": f"bybit-history-v{HISTORY_DAY_SCHEMA_VERSION}-{partition_date}",
        "partition_date": partition_date,
        "symbols": list(symbols),
        "parameters": parameters,
        "parameters_fingerprint": parameters_fingerprint,
        "builder": {
            "package_version": __version__,
            "pyarrow_version": pa.__version__,
            "parquet_format_version": PARQUET_FORMAT_VERSION,
            "compression": PARQUET_COMPRESSION,
            "compression_level": PARQUET_COMPRESSION_LEVEL,
            "bar_batch_rows": BAR_BATCH_ROWS,
        },
        "causality": {
            "timestamp_basis": "bar_end_plus_assumed_latency",
            "assumed_latency_ms": parameters["assumed_latency_ms"],
            "offline_receive_timestamp_available": False,
            "availability_field": "available_at_ns",
        },
        "coverage": {
            "bars_are_trade_derived": True,
            "synthetic_bars": 0,
            "source_archives_retained": False,
            "individual_trades_retained": False,
            "orderbook_available": False,
            "ticker_available": False,
            "queue_position_available": False,
            "funding_available": False,
            "open_interest_available": False,
        },
        "known_limitations": [
            "Global trade-ID deduplication is not performed during streaming aggregation.",
            "One-second bars cannot determine event order when both barriers occur "
            "within one second.",
            "Maker queue position cannot be reconstructed without historical L2/L3 data.",
        ],
        "schemas": {"trade_bar": _schema_manifest(TRADE_BAR_SCHEMA)},
        "sources": [artifact.source.to_dict() for artifact in artifacts],
        "symbol_manifests": [
            {
                "symbol": artifact.symbol,
                "path": f"symbols/symbol={artifact.symbol}/symbol-manifest.json",
                "sha256": artifact.symbol_manifest_sha256,
            }
            for artifact in artifacts
        ],
        "source_rows": sum(artifact.source.source_rows for artifact in artifacts),
        "output_rows_by_kind": dict(sorted(rows_by_kind.items())),
        "output_file_count": len(files),
        "output_rows": sum(item.rows for item in files),
        "output_bytes": sum(item.bytes for item in files),
        "output_fingerprint": _files_fingerprint(files),
        "files": [item.to_dict() for item in files],
    }


def _result_from_manifest(
    dataset_path: Path, manifest: dict[str, Any], *, reused: bool
) -> HistoryDayResult:
    raw_symbols = manifest.get("symbols")
    if not isinstance(raw_symbols, list) or not all(
        isinstance(item, str) and item for item in raw_symbols
    ):
        raise HistoryImportError("history manifest symbols are invalid")
    raw_rows = _json_object(manifest.get("output_rows_by_kind"), "output_rows_by_kind")
    rows_by_kind = {
        key: _required_int(value, f"output_rows_by_kind.{key}", minimum=1)
        for key, value in raw_rows.items()
    }
    return HistoryDayResult(
        partition_date=_required_string(manifest.get("partition_date"), "partition_date"),
        dataset_path=dataset_path,
        manifest_path=dataset_path / "manifest.json",
        parameters_fingerprint=_valid_sha256(
            manifest.get("parameters_fingerprint"), "parameters_fingerprint"
        ),
        output_fingerprint=_valid_sha256(manifest.get("output_fingerprint"), "output_fingerprint"),
        symbols=tuple(cast(list[str], raw_symbols)),
        source_rows=_required_int(manifest.get("source_rows"), "source_rows", minimum=1),
        output_files=_required_int(
            manifest.get("output_file_count"), "output_file_count", minimum=1
        ),
        output_rows=_required_int(manifest.get("output_rows"), "output_rows", minimum=1),
        output_bytes=_required_int(manifest.get("output_bytes"), "output_bytes", minimum=1),
        output_rows_by_kind=rows_by_kind,
        reused=reused,
    )


def validate_history_day(
    dataset_path: str | Path,
    *,
    expected_parameters_fingerprint: str | None = None,
    expected_symbols: Sequence[str] | None = None,
) -> HistoryDayResult:
    """Deeply validate an existing imported day, including every Parquet SHA."""

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_dir():
        raise HistoryImportError(f"history day does not exist: {path}")
    manifest = _load_json(path / "manifest.json", "history day manifest")
    if manifest.get("history_day_schema_version") != HISTORY_DAY_SCHEMA_VERSION:
        raise HistoryImportError(f"unsupported history day schema: {path}")
    if manifest.get("dataset_profile") != HISTORY_PROFILE:
        raise HistoryImportError(f"history profile mismatch: {path}")
    partition_date = _required_string(manifest.get("partition_date"), "partition_date")
    _parse_partition_date(partition_date)
    if path.name != f"day={partition_date}":
        raise HistoryImportError(f"history day directory does not match manifest: {path}")
    parameters = _json_object(manifest.get("parameters"), "parameters")
    parameters_fingerprint = _valid_sha256(
        manifest.get("parameters_fingerprint"), "parameters_fingerprint"
    )
    if _sha256_json(parameters) != parameters_fingerprint:
        raise HistoryImportError(f"history parameters fingerprint mismatch: {path}")
    if (
        expected_parameters_fingerprint is not None
        and parameters_fingerprint != expected_parameters_fingerprint
    ):
        raise HistoryImportError(f"history day uses different import parameters: {path}")
    raw_symbols = manifest.get("symbols")
    if not isinstance(raw_symbols, list) or not all(
        isinstance(item, str) and item for item in raw_symbols
    ):
        raise HistoryImportError(f"history day symbols are invalid: {path}")
    symbols = tuple(cast(list[str], raw_symbols))
    _validate_symbols(symbols)
    if expected_symbols is not None and symbols != tuple(expected_symbols):
        raise HistoryImportError(f"history day symbol set differs: {path}")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise HistoryImportError(f"history day has no file descriptors: {path}")
    files = tuple(
        _history_file_from_dict(item, f"files[{index}]") for index, item in enumerate(raw_files)
    )
    if len(files) != len(symbols) * 2:
        raise HistoryImportError(f"history day file count is inconsistent: {path}")
    for item in files:
        if item.symbol not in symbols or item.date != partition_date:
            raise HistoryImportError(f"history file partition mismatch: {item.path}")
        _validate_parquet_file(path, item)
    if manifest.get("output_fingerprint") != _files_fingerprint(files):
        raise HistoryImportError(f"history day output fingerprint mismatch: {path}")
    if manifest.get("output_file_count") != len(files):
        raise HistoryImportError(f"history day output file count mismatch: {path}")
    if manifest.get("output_rows") != sum(item.rows for item in files):
        raise HistoryImportError(f"history day output row count mismatch: {path}")
    if manifest.get("output_bytes") != sum(item.bytes for item in files):
        raise HistoryImportError(f"history day output byte count mismatch: {path}")
    symbol_manifests = manifest.get("symbol_manifests")
    if not isinstance(symbol_manifests, list) or len(symbol_manifests) != len(symbols):
        raise HistoryImportError(f"history symbol manifests are inconsistent: {path}")
    for index, raw_item in enumerate(symbol_manifests):
        symbol_manifest = _json_object(raw_item, f"symbol_manifests[{index}]")
        relative = _required_string(symbol_manifest.get("path"), f"symbol_manifests[{index}].path")
        manifest_path = _resolve_relative(path, relative, "symbol manifest path")
        if _sha256_file(manifest_path) != _valid_sha256(
            symbol_manifest.get("sha256"), f"symbol_manifests[{index}].sha256"
        ):
            raise HistoryImportError(f"history symbol manifest SHA changed: {manifest_path}")
    return _result_from_manifest(path, manifest, reused=True)


@contextmanager
def _history_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".history-import.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise HistoryImportError(
            f"history import lock exists: {lock_path}; verify no import is running "
            "before removing a stale lock"
        ) from exc
    try:
        payload = {
            "pid": os.getpid(),
            "host": platform.node(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _prepare_stage(
    *,
    history_root: Path,
    partition_date: str,
    parameters: dict[str, object],
    parameters_fingerprint: str,
) -> Path:
    stage = history_root / (f".day={partition_date}.{parameters_fingerprint[:12]}.staging")
    parameters_path = stage / "build-parameters.json"
    expected: dict[str, object] = {
        "parameters": parameters,
        "parameters_fingerprint": parameters_fingerprint,
    }
    if stage.exists():
        if not stage.is_dir():
            raise HistoryImportError(f"history staging path is not a directory: {stage}")
        existing = _load_json(parameters_path, "history staging parameters")
        if existing != expected:
            raise HistoryImportError(f"history staging parameters changed: {stage}")
        for child in stage.glob(".symbol=*.tmp-*"):
            if child.is_dir() and child.parent == stage:
                shutil.rmtree(child, ignore_errors=True)
        return stage
    stage.mkdir()
    _write_json_atomic(parameters_path, expected)
    return stage


def _import_day_unlocked(
    *,
    history_root: Path,
    partition_date: str,
    symbols: tuple[str, ...],
    public_base_url: str,
    assumed_latency_ms: int,
    request_timeout_seconds: int,
    download_attempts: int,
    maximum_missing_minutes: int,
    minimum_free_bytes: int,
) -> HistoryDayResult:
    parameters = _parameters_payload(
        partition_date=partition_date,
        symbols=symbols,
        public_base_url=public_base_url,
        assumed_latency_ms=assumed_latency_ms,
        maximum_missing_minutes=maximum_missing_minutes,
    )
    parameters_fingerprint = _sha256_json(parameters)
    final_path = history_root / f"day={partition_date}"
    if final_path.exists():
        return validate_history_day(
            final_path,
            expected_parameters_fingerprint=parameters_fingerprint,
            expected_symbols=symbols,
        )
    _check_disk(history_root, minimum_free_bytes)
    stage = _prepare_stage(
        history_root=history_root,
        partition_date=partition_date,
        parameters=parameters,
        parameters_fingerprint=parameters_fingerprint,
    )
    artifacts = tuple(
        _build_symbol(
            stage_path=stage,
            symbol=symbol,
            partition_date=partition_date,
            public_base_url=public_base_url,
            parameters_fingerprint=parameters_fingerprint,
            assumed_latency_ms=assumed_latency_ms,
            request_timeout_seconds=request_timeout_seconds,
            download_attempts=download_attempts,
            maximum_missing_minutes=maximum_missing_minutes,
            minimum_free_bytes=minimum_free_bytes,
        )
        for symbol in symbols
    )
    manifest = _day_manifest_payload(
        partition_date=partition_date,
        symbols=symbols,
        parameters=parameters,
        parameters_fingerprint=parameters_fingerprint,
        artifacts=artifacts,
    )
    _write_json_atomic(stage / "manifest.json", manifest)
    if final_path.exists():
        raise HistoryImportError(f"history day appeared concurrently: {final_path}")
    os.replace(stage, final_path)
    result = validate_history_day(
        final_path,
        expected_parameters_fingerprint=parameters_fingerprint,
        expected_symbols=symbols,
    )
    return HistoryDayResult(
        partition_date=result.partition_date,
        dataset_path=result.dataset_path,
        manifest_path=result.manifest_path,
        parameters_fingerprint=result.parameters_fingerprint,
        output_fingerprint=result.output_fingerprint,
        symbols=result.symbols,
        source_rows=result.source_rows,
        output_files=result.output_files,
        output_rows=result.output_rows,
        output_bytes=result.output_bytes,
        output_rows_by_kind=result.output_rows_by_kind,
        reused=False,
    )


def _catalog_payload(history_root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for day_path in sorted(history_root.glob("day=????-??-??")):
        if not day_path.is_dir():
            continue
        manifest_path = day_path / "manifest.json"
        manifest = _load_json(manifest_path, "history catalog day manifest")
        if manifest.get("history_day_schema_version") != HISTORY_DAY_SCHEMA_VERSION:
            raise HistoryImportError(f"unsupported history day in catalog: {day_path}")
        if manifest.get("dataset_profile") != HISTORY_PROFILE:
            raise HistoryImportError(f"mixed history profile in catalog: {day_path}")
        partition_date = _required_string(manifest.get("partition_date"), "catalog partition_date")
        if day_path.name != f"day={partition_date}":
            raise HistoryImportError(f"catalog day path mismatch: {day_path}")
        entries.append(
            {
                "partition_date": partition_date,
                "path": day_path.relative_to(history_root).as_posix(),
                "manifest_path": manifest_path.relative_to(history_root).as_posix(),
                "manifest_sha256": _sha256_file(manifest_path),
                "parameters_fingerprint": _valid_sha256(
                    manifest.get("parameters_fingerprint"),
                    "catalog parameters_fingerprint",
                ),
                "output_fingerprint": _valid_sha256(
                    manifest.get("output_fingerprint"),
                    "catalog output_fingerprint",
                ),
                "symbols": manifest.get("symbols"),
                "source_rows": _required_int(
                    manifest.get("source_rows"), "catalog source_rows", minimum=1
                ),
                "output_rows": _required_int(
                    manifest.get("output_rows"), "catalog output_rows", minimum=1
                ),
                "output_bytes": _required_int(
                    manifest.get("output_bytes"), "catalog output_bytes", minimum=1
                ),
            }
        )
    fingerprint = _sha256_json(entries)
    return {
        "history_catalog_schema_version": HISTORY_CATALOG_SCHEMA_VERSION,
        "history_day_schema_version": HISTORY_DAY_SCHEMA_VERSION,
        "dataset_profile": HISTORY_PROFILE,
        "entry_count": len(entries),
        "catalog_fingerprint": fingerprint,
        "entries": entries,
    }


def _write_catalog(history_root: Path) -> tuple[Path, str]:
    payload = _catalog_payload(history_root)
    catalog_path = history_root / "catalog.json"
    _write_json_atomic(catalog_path, payload)
    fingerprint = _valid_sha256(payload.get("catalog_fingerprint"), "catalog_fingerprint")
    return catalog_path, fingerprint


def import_bybit_history_day(
    *,
    history_root: str | Path,
    partition_date: str,
    symbols: Sequence[str],
    public_base_url: str = "https://public.bybit.com/trading",
    assumed_latency_ms: int = 1_000,
    request_timeout_seconds: int = 60,
    download_attempts: int = 3,
    maximum_missing_minutes: int = 0,
    minimum_free_bytes: int = 0,
) -> HistoryDayResult:
    """Import or deeply validate one completed UTC day for all selected symbols."""

    root = Path(history_root).expanduser().resolve()
    parsed_date = _parse_partition_date(partition_date)
    validated_symbols = _validate_symbols(symbols)
    base_url = _validate_base_url(public_base_url)
    _validate_options(
        assumed_latency_ms=assumed_latency_ms,
        request_timeout_seconds=request_timeout_seconds,
        download_attempts=download_attempts,
        maximum_missing_minutes=maximum_missing_minutes,
        minimum_free_bytes=minimum_free_bytes,
    )
    root.mkdir(parents=True, exist_ok=True)
    with _history_lock(root):
        result = _import_day_unlocked(
            history_root=root,
            partition_date=parsed_date.isoformat(),
            symbols=validated_symbols,
            public_base_url=base_url,
            assumed_latency_ms=assumed_latency_ms,
            request_timeout_seconds=request_timeout_seconds,
            download_attempts=download_attempts,
            maximum_missing_minutes=maximum_missing_minutes,
            minimum_free_bytes=minimum_free_bytes,
        )
        _write_catalog(root)
        return result


def _validate_options(
    *,
    assumed_latency_ms: int,
    request_timeout_seconds: int,
    download_attempts: int,
    maximum_missing_minutes: int,
    minimum_free_bytes: int,
) -> None:
    if not 0 <= assumed_latency_ms <= 60_000:
        raise HistoryImportError("assumed_latency_ms must be within [0, 60000]")
    if not 1 <= request_timeout_seconds <= 300:
        raise HistoryImportError("request_timeout_seconds must be within [1, 300]")
    if not 1 <= download_attempts <= 10:
        raise HistoryImportError("download_attempts must be within [1, 10]")
    if not 0 <= maximum_missing_minutes < EXPECTED_MINUTES_PER_DAY:
        raise HistoryImportError("maximum_missing_minutes must be within [0, 1440)")
    if minimum_free_bytes < 0:
        raise HistoryImportError("minimum_free_bytes must be non-negative")


def import_bybit_history_range(
    *,
    history_root: str | Path,
    start_date: str,
    end_date: str,
    symbols: Sequence[str],
    public_base_url: str = "https://public.bybit.com/trading",
    assumed_latency_ms: int = 1_000,
    request_timeout_seconds: int = 60,
    download_attempts: int = 3,
    maximum_missing_minutes: int = 0,
    minimum_free_bytes: int = 0,
) -> HistoryRangeResult:
    """Import an inclusive date range, committing and cataloging each day atomically."""

    root = Path(history_root).expanduser().resolve()
    first = _parse_partition_date(start_date)
    last = _parse_partition_date(end_date)
    if first > last:
        raise HistoryImportError("start_date cannot be after end_date")
    validated_symbols = _validate_symbols(symbols)
    base_url = _validate_base_url(public_base_url)
    _validate_options(
        assumed_latency_ms=assumed_latency_ms,
        request_timeout_seconds=request_timeout_seconds,
        download_attempts=download_attempts,
        maximum_missing_minutes=maximum_missing_minutes,
        minimum_free_bytes=minimum_free_bytes,
    )
    root.mkdir(parents=True, exist_ok=True)
    results: list[HistoryDayResult] = []
    catalog_path = root / "catalog.json"
    catalog_fingerprint = ""
    with _history_lock(root):
        current = first
        while current <= last:
            result = _import_day_unlocked(
                history_root=root,
                partition_date=current.isoformat(),
                symbols=validated_symbols,
                public_base_url=base_url,
                assumed_latency_ms=assumed_latency_ms,
                request_timeout_seconds=request_timeout_seconds,
                download_attempts=download_attempts,
                maximum_missing_minutes=maximum_missing_minutes,
                minimum_free_bytes=minimum_free_bytes,
            )
            results.append(result)
            catalog_path, catalog_fingerprint = _write_catalog(root)
            current += timedelta(days=1)
    return HistoryRangeResult(
        start_date=first.isoformat(),
        end_date=last.isoformat(),
        history_root=root,
        catalog_path=catalog_path,
        catalog_fingerprint=catalog_fingerprint,
        days=len(results),
        imported_days=sum(not item.reused for item in results),
        reused_days=sum(item.reused for item in results),
        source_rows=sum(item.source_rows for item in results),
        output_rows=sum(item.output_rows for item in results),
        output_bytes=sum(item.output_bytes for item in results),
    )
