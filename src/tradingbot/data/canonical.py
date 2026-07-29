"""Build a deterministic, typed Parquet layer from an audited raw snapshot.

The builder deliberately accepts an audit report rather than scanning arbitrary
directories.  The report is the immutable input manifest: every listed JSONL
segment is hashed again while it is parsed, and files not present in the report
are ignored.

Raw events remain untouched.  Closed kline revisions are canonicalized with a
last-write-wins rule on ``received_at_ns`` for
``(symbol, interval, start_ms)``.  Different payloads with the same causal
timestamp are rejected because their order cannot be reconstructed safely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeGuard

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from tradingbot import __version__
from tradingbot.data.audit import AUDIT_REPORT_SCHEMA_VERSION

DATASET_SCHEMA_VERSION: Final = 1
PARQUET_FORMAT_VERSION: Final = "2.6"
PARQUET_COMPRESSION: Final = "zstd"
PARQUET_COMPRESSION_LEVEL: Final = 3
DEFAULT_BATCH_ROWS: Final = 10_000
ORDERBOOK_BATCH_ROWS: Final = 1_000

LOGGER = logging.getLogger(__name__)


class DatasetBuildError(RuntimeError):
    """Raised when audited input cannot be converted without ambiguity."""


@dataclass(frozen=True, slots=True)
class AuditedInputFile:
    path: str
    bytes: int
    lines: int
    records: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AuditInputManifest:
    report_path: Path
    report_sha256: str
    dataset_root: str
    input_fingerprint: str
    files: tuple[AuditedInputFile, ...]
    total_bytes: int
    total_records: int
    expected_symbols: tuple[str, ...]
    kline_intervals: tuple[str, ...]
    kline_revisions: int
    duplicate_klines: int


@dataclass(frozen=True, slots=True)
class DatasetFile:
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
class DatasetBuildResult:
    dataset_id: str
    dataset_path: Path
    manifest_path: Path
    input_fingerprint: str
    output_fingerprint: str
    source_files: int
    source_records: int
    output_files: int
    output_rows: int
    output_rows_by_kind: dict[str, int]
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "source_files": self.source_files,
            "source_records": self.source_records,
            "output_files": self.output_files,
            "output_rows": self.output_rows,
            "output_rows_by_kind": dict(sorted(self.output_rows_by_kind.items())),
            "reused": self.reused,
        }


_SCHEMA_METADATA: Final = {
    b"tradingbot.dataset_schema_version": str(DATASET_SCHEMA_VERSION).encode("ascii"),
    b"tradingbot.causal_key": b"received_at_ns",
}

_COMMON_FIELDS: Final = (
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("session_id", pa.string()),
    pa.field("exchange_ts_ms", pa.int64(), nullable=False),
    pa.field("received_at_ns", pa.int64(), nullable=False),
    pa.field("source_path", pa.string(), nullable=False),
    pa.field("source_line", pa.int32(), nullable=False),
)

ORDERBOOK_SCHEMA: Final = pa.schema(
    (
        *_COMMON_FIELDS,
        pa.field("matching_engine_ts_ms", pa.int64(), nullable=False),
        pa.field("update_id", pa.int64(), nullable=False),
        pa.field("sequence", pa.int64(), nullable=False),
        pa.field("bid_prices", pa.list_(pa.float64()), nullable=False),
        pa.field("bid_sizes", pa.list_(pa.float64()), nullable=False),
        pa.field("ask_prices", pa.list_(pa.float64()), nullable=False),
        pa.field("ask_sizes", pa.list_(pa.float64()), nullable=False),
    ),
    metadata=_SCHEMA_METADATA,
)

TRADES_SCHEMA: Final = pa.schema(
    (
        *_COMMON_FIELDS,
        pa.field("event_ts_ms", pa.int64(), nullable=False),
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("tick_direction", pa.string()),
        pa.field("sequence", pa.int64()),
        pa.field("is_block_trade", pa.bool_()),
        pa.field("is_rpi_trade", pa.bool_()),
    ),
    metadata=_SCHEMA_METADATA,
)

TICKER_SCHEMA: Final = pa.schema(
    (
        *_COMMON_FIELDS,
        pa.field("last_price", pa.float64()),
        pa.field("index_price", pa.float64()),
        pa.field("mark_price", pa.float64()),
        pa.field("bid_price", pa.float64()),
        pa.field("bid_size", pa.float64()),
        pa.field("ask_price", pa.float64()),
        pa.field("ask_size", pa.float64()),
        pa.field("open_interest", pa.float64()),
        pa.field("open_interest_value", pa.float64()),
        pa.field("funding_rate", pa.float64()),
        pa.field("next_funding_time_ms", pa.int64()),
        pa.field("funding_interval_hours", pa.int32()),
        pa.field("volume_24h", pa.float64()),
        pa.field("turnover_24h", pa.float64()),
        pa.field("price_24h_fraction", pa.float64()),
        pa.field("high_price_24h", pa.float64()),
        pa.field("low_price_24h", pa.float64()),
        pa.field("previous_price_1h", pa.float64()),
        pa.field("previous_price_24h", pa.float64()),
        pa.field("tick_direction", pa.string()),
    ),
    metadata=_SCHEMA_METADATA,
)

KLINE_SCHEMA: Final = pa.schema(
    (
        *_COMMON_FIELDS,
        pa.field("interval", pa.string(), nullable=False),
        pa.field("start_ms", pa.int64(), nullable=False),
        pa.field("end_ms", pa.int64(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("turnover", pa.float64(), nullable=False),
        pa.field("payload_sha256", pa.string(), nullable=False),
    ),
    metadata=_SCHEMA_METADATA,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DatasetBuildError("audit file SHA-256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DatasetBuildError(
            "audit file SHA-256 must contain 64 hexadecimal characters"
        ) from exc
    return value.lower()


def _plain_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_nonnegative_int(value: object, field: str) -> int:
    if not _plain_int(value) or value < 0:
        raise DatasetBuildError(f"{field} must be a non-negative integer")
    return value


def _required_positive_int(value: object, field: str) -> int:
    if not _plain_int(value) or value <= 0:
        raise DatasetBuildError(f"{field} must be a positive integer")
    return value


def _payload_int(
    value: object,
    field: str,
    *,
    required: bool = True,
    nonnegative: bool = False,
) -> int | None:
    if value is None or value == "":
        if required:
            raise DatasetBuildError(f"{field} is required")
        return None
    if isinstance(value, bool):
        raise DatasetBuildError(f"{field} must be an integer")
    try:
        if _plain_int(value):
            result = value
        elif isinstance(value, str):
            result = int(value)
        else:
            raise TypeError
    except (TypeError, ValueError) as exc:
        raise DatasetBuildError(f"{field} must be an integer") from exc
    if nonnegative and result < 0:
        raise DatasetBuildError(f"{field} must be non-negative")
    if not nonnegative and result <= 0:
        raise DatasetBuildError(f"{field} must be positive")
    return result


def _payload_float(
    value: object,
    field: str,
    *,
    required: bool = True,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if value is None or value == "":
        if required:
            raise DatasetBuildError(f"{field} is required")
        return None
    if isinstance(value, bool):
        raise DatasetBuildError(f"{field} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DatasetBuildError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise DatasetBuildError(f"{field} must be finite")
    if positive and result <= 0:
        raise DatasetBuildError(f"{field} must be positive")
    if nonnegative and result < 0:
        raise DatasetBuildError(f"{field} must be non-negative")
    return result


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetBuildError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DatasetBuildError(f"{field} must be a string when present")
    return value


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise DatasetBuildError(f"{field} must be a boolean when present")
    return value


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DatasetBuildError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetBuildError(f"unsafe {label}: {value!r}")
    return path


def _audited_files_fingerprint(files: Sequence[AuditedInputFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_files_fingerprint(files: Sequence[DatasetFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetBuildError(f"{label} must be a JSON object")
    return dict(value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetBuildError(f"{label} must be a non-empty array")
    result = tuple(_required_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise DatasetBuildError(f"{label} contains duplicates")
    return result


def load_audit_input_manifest(path: str | Path) -> AuditInputManifest:
    """Load and validate the strict audit report used as the source manifest."""

    report_path = Path(path).expanduser().resolve()
    try:
        report_bytes = report_path.read_bytes()
        parsed: object = json.loads(report_bytes)
    except FileNotFoundError as exc:
        raise DatasetBuildError(f"audit report does not exist: {report_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(f"cannot read audit report {report_path}: {exc}") from exc
    report = _json_object(parsed, "audit report")

    if report.get("audit_report_schema_version") != AUDIT_REPORT_SCHEMA_VERSION:
        raise DatasetBuildError(
            f"audit report schema must be {AUDIT_REPORT_SCHEMA_VERSION}"
        )
    readiness = _json_object(report.get("readiness"), "readiness")
    if readiness.get("ok") is not True or readiness.get("strict") is not True:
        raise DatasetBuildError("dataset build requires a successful strict audit")
    if readiness.get("reasons") != []:
        raise DatasetBuildError("successful audit must not contain readiness reasons")
    if report.get("errors") != [] or report.get("warnings") != []:
        raise DatasetBuildError("dataset build requires an audit with no errors or warnings")
    if (
        report.get("partial_file_count") != 0
        or report.get("partial_files") != []
        or report.get("missing_expected_streams") != []
        or report.get("short_streams") != []
    ):
        raise DatasetBuildError("audit report contains incomplete input streams")

    raw_files = report.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DatasetBuildError("audit report must contain a non-empty file manifest")
    files: list[AuditedInputFile] = []
    paths: set[str] = set()
    for index, raw_item in enumerate(raw_files):
        item = _json_object(raw_item, f"files[{index}]")
        relative = _safe_relative_path(item.get("path"), f"files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in paths:
            raise DatasetBuildError(f"duplicate source path in audit report: {relative_text}")
        paths.add(relative_text)
        files.append(
            AuditedInputFile(
                path=relative_text,
                bytes=_required_nonnegative_int(
                    item.get("bytes"), f"files[{index}].bytes"
                ),
                lines=_required_nonnegative_int(
                    item.get("lines"), f"files[{index}].lines"
                ),
                records=_required_nonnegative_int(
                    item.get("records"), f"files[{index}].records"
                ),
                sha256=_valid_sha256(item.get("sha256")),
            )
        )
    files.sort(key=lambda item: item.path)
    if report.get("file_count") != len(files):
        raise DatasetBuildError("audit file_count does not match files manifest")

    input_fingerprint = _valid_sha256(report.get("input_fingerprint"))
    if _audited_files_fingerprint(files) != input_fingerprint:
        raise DatasetBuildError("audit input_fingerprint does not match its file manifest")

    totals = _json_object(report.get("totals"), "totals")
    total_bytes = _required_nonnegative_int(totals.get("bytes"), "totals.bytes")
    total_records = _required_nonnegative_int(totals.get("records"), "totals.records")
    if total_bytes != sum(item.bytes for item in files):
        raise DatasetBuildError("audit totals.bytes does not match file manifest")
    if total_records != sum(item.records for item in files):
        raise DatasetBuildError("audit totals.records does not match file manifest")

    policy = _json_object(report.get("policy"), "policy")
    symbols = _string_tuple(policy.get("expected_symbols"), "policy.expected_symbols")
    intervals = _string_tuple(policy.get("kline_intervals"), "policy.kline_intervals")
    streams = _json_object(report.get("streams"), "streams")
    kline_revisions = 0
    duplicate_klines = 0
    for name, raw_stream in streams.items():
        stream = _json_object(raw_stream, f"streams.{name}")
        kline_revisions += _required_nonnegative_int(
            stream.get("kline_revisions"), f"streams.{name}.kline_revisions"
        )
        duplicate_klines += _required_nonnegative_int(
            stream.get("duplicate_klines"), f"streams.{name}.duplicate_klines"
        )

    return AuditInputManifest(
        report_path=report_path,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        dataset_root=_required_string(report.get("dataset_root"), "dataset_root"),
        input_fingerprint=input_fingerprint,
        files=tuple(files),
        total_bytes=total_bytes,
        total_records=total_records,
        expected_symbols=symbols,
        kline_intervals=intervals,
        kline_revisions=kline_revisions,
        duplicate_klines=duplicate_klines,
    )


def _schema_for_kind(kind: str) -> pa.Schema:
    if kind == "orderbook":
        return ORDERBOOK_SCHEMA
    if kind == "trades":
        return TRADES_SCHEMA
    if kind == "ticker":
        return TICKER_SCHEMA
    if kind.startswith("kline_"):
        return KLINE_SCHEMA
    raise DatasetBuildError(f"unsupported market record kind: {kind!r}")


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def _common_row(
    raw: dict[str, Any],
    source_path: str,
    source_line: int,
) -> dict[str, Any]:
    schema_version = _required_positive_int(raw.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise DatasetBuildError(f"unsupported raw schema_version: {schema_version}")
    session_value = raw.get("session_id")
    session_id = _optional_string(session_value, "session_id")
    return {
        "schema_version": schema_version,
        "source": _required_string(raw.get("source"), "source"),
        "session_id": session_id,
        "kind": _required_string(raw.get("kind"), "kind"),
        "symbol": _required_string(raw.get("symbol"), "symbol"),
        "exchange_ts_ms": _required_positive_int(
            raw.get("exchange_ts_ms"), "exchange_ts_ms"
        ),
        "received_at_ns": _required_positive_int(
            raw.get("received_at_ns"), "received_at_ns"
        ),
        "source_path": source_path,
        "source_line": source_line,
    }


def _levels(payload: dict[str, Any], field: str) -> tuple[list[float], list[float]]:
    raw_levels = payload.get(field)
    if not isinstance(raw_levels, list) or not raw_levels:
        raise DatasetBuildError(f"orderbook {field} must be a non-empty array")
    prices: list[float] = []
    sizes: list[float] = []
    for index, raw_level in enumerate(raw_levels):
        if not isinstance(raw_level, list) or len(raw_level) < 2:
            raise DatasetBuildError(f"orderbook {field}[{index}] is invalid")
        price = _payload_float(
            raw_level[0], f"orderbook {field}[{index}].price", positive=True
        )
        size = _payload_float(
            raw_level[1], f"orderbook {field}[{index}].size", nonnegative=True
        )
        assert price is not None and size is not None
        prices.append(price)
        sizes.append(size)
    return prices, sizes


def _orderbook_row(
    raw: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    payload = _json_object(raw.get("payload"), "orderbook payload")
    bid_prices, bid_sizes = _levels(payload, "bids")
    ask_prices, ask_sizes = _levels(payload, "asks")
    result = dict(common)
    result.update(
        {
            "matching_engine_ts_ms": _payload_int(
                payload.get("matching_engine_ts_ms"),
                "matching_engine_ts_ms",
                nonnegative=True,
            ),
            "update_id": _payload_int(
                payload.get("update_id"), "update_id", nonnegative=True
            ),
            "sequence": _payload_int(
                payload.get("sequence"), "sequence", nonnegative=True
            ),
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
        }
    )
    return result


_TICKER_FLOAT_FIELDS: Final = {
    "last_price": "lastPrice",
    "index_price": "indexPrice",
    "mark_price": "markPrice",
    "bid_price": "bid1Price",
    "bid_size": "bid1Size",
    "ask_price": "ask1Price",
    "ask_size": "ask1Size",
    "open_interest": "openInterest",
    "open_interest_value": "openInterestValue",
    "funding_rate": "fundingRate",
    "volume_24h": "volume24h",
    "turnover_24h": "turnover24h",
    "price_24h_fraction": "price24hPcnt",
    "high_price_24h": "highPrice24h",
    "low_price_24h": "lowPrice24h",
    "previous_price_1h": "prevPrice1h",
    "previous_price_24h": "prevPrice24h",
}


def _ticker_row(raw: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    payload = _json_object(raw.get("payload"), "ticker payload")
    result = dict(common)
    for target, source in _TICKER_FLOAT_FIELDS.items():
        result[target] = _payload_float(
            payload.get(source), f"ticker.{source}", required=False
        )
    result.update(
        {
            "next_funding_time_ms": _payload_int(
                payload.get("nextFundingTime"),
                "ticker.nextFundingTime",
                required=False,
                nonnegative=True,
            ),
            "funding_interval_hours": _payload_int(
                payload.get("fundingIntervalHour"),
                "ticker.fundingIntervalHour",
                required=False,
            ),
            "tick_direction": _optional_string(
                payload.get("tickDirection"), "ticker.tickDirection"
            ),
        }
    )
    return result


def _trade_rows(
    raw: dict[str, Any], common: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    payload = raw.get("payload")
    if not isinstance(payload, list) or not payload:
        raise DatasetBuildError("trades payload must be a non-empty array")
    for index, raw_trade in enumerate(payload):
        trade = _json_object(raw_trade, f"trades payload[{index}]")
        side = _required_string(trade.get("S"), f"trade[{index}].S")
        if side not in {"Buy", "Sell"}:
            raise DatasetBuildError(f"trade[{index}].S must be Buy or Sell")
        price = _payload_float(trade.get("p"), f"trade[{index}].p", positive=True)
        size = _payload_float(trade.get("v"), f"trade[{index}].v", positive=True)
        assert price is not None and size is not None
        result = dict(common)
        result.update(
            {
                "event_ts_ms": _payload_int(trade.get("T"), f"trade[{index}].T"),
                "trade_id": _required_string(
                    trade.get("i"), f"trade[{index}].i"
                ),
                "side": side,
                "price": price,
                "size": size,
                "tick_direction": _optional_string(
                    trade.get("L"), f"trade[{index}].L"
                ),
                "sequence": _payload_int(
                    trade.get("seq"),
                    f"trade[{index}].seq",
                    required=False,
                    nonnegative=True,
                ),
                "is_block_trade": _optional_bool(
                    trade.get("BT"), f"trade[{index}].BT"
                ),
                "is_rpi_trade": _optional_bool(
                    trade.get("RPI"), f"trade[{index}].RPI"
                ),
            }
        )
        yield result


def _kline_row(raw: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    payload = _json_object(raw.get("payload"), "kline payload")
    interval = _required_string(payload.get("interval"), "kline.interval")
    expected_kind = f"kline_{interval}"
    if common["kind"] != expected_kind:
        raise DatasetBuildError(
            f"kline interval {interval!r} does not match kind {common['kind']!r}"
        )
    if payload.get("confirm") is not True:
        raise DatasetBuildError("canonical dataset accepts only confirmed klines")
    values: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        value = _payload_float(payload.get(field), f"kline.{field}", positive=True)
        assert value is not None
        values[field] = value
    for field in ("volume", "turnover"):
        value = _payload_float(payload.get(field), f"kline.{field}", nonnegative=True)
        assert value is not None
        values[field] = value
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    result = dict(common)
    result.update(
        {
            "interval": interval,
            "start_ms": _payload_int(
                payload.get("start"), "kline.start", nonnegative=True
            ),
            "end_ms": _payload_int(payload.get("end"), "kline.end"),
            **values,
            "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        }
    )
    return result


class _CanonicalKlines:
    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        self.database = sqlite3.connect(database_path)
        self.database.execute("PRAGMA journal_mode=OFF")
        self.database.execute("PRAGMA synchronous=OFF")
        self.database.execute("PRAGMA temp_store=MEMORY")
        self.database.execute("PRAGMA cache_size=-65536")
        self.database.execute(
            "CREATE TABLE klines ("
            "kind TEXT NOT NULL, symbol TEXT NOT NULL, start_ms INTEGER NOT NULL, "
            "received_at_ns INTEGER NOT NULL, payload_sha256 TEXT NOT NULL, "
            "row_json TEXT NOT NULL, "
            "PRIMARY KEY(kind, symbol, start_ms)"
            ") WITHOUT ROWID"
        )
        self.closed = False

    def observe(self, row: dict[str, Any]) -> None:
        kind = _required_string(row.get("kind"), "canonical kline kind")
        symbol = _required_string(row.get("symbol"), "canonical kline symbol")
        start_ms = _required_nonnegative_int(row.get("start_ms"), "canonical start_ms")
        received_at_ns = _required_positive_int(
            row.get("received_at_ns"), "canonical received_at_ns"
        )
        payload_sha256 = _valid_sha256(row.get("payload_sha256"))
        row_json = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        existing = self.database.execute(
            "SELECT received_at_ns, payload_sha256 "
            "FROM klines WHERE kind = ? AND symbol = ? AND start_ms = ?",
            (kind, symbol, start_ms),
        ).fetchone()
        if existing is None:
            self.database.execute(
                "INSERT INTO klines("
                "kind, symbol, start_ms, received_at_ns, payload_sha256, row_json"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (kind, symbol, start_ms, received_at_ns, payload_sha256, row_json),
            )
            return
        previous_received_at, previous_payload_sha = existing
        if (
            received_at_ns == previous_received_at
            and payload_sha256 != previous_payload_sha
        ):
            raise DatasetBuildError(
                "ambiguous kline revision: different payloads share received_at_ns "
                f"for {kind}/{symbol}/{start_ms}"
            )
        if received_at_ns > previous_received_at:
            self.database.execute(
                "UPDATE klines SET "
                "received_at_ns = ?, payload_sha256 = ?, row_json = ? "
                "WHERE kind = ? AND symbol = ? AND start_ms = ?",
                (
                    received_at_ns,
                    payload_sha256,
                    row_json,
                    kind,
                    symbol,
                    start_ms,
                ),
            )

    def rows(self) -> Iterator[dict[str, Any]]:
        self.database.commit()
        cursor = self.database.execute(
            "SELECT row_json FROM klines ORDER BY kind, symbol, start_ms"
        )
        for (row_json,) in cursor:
            parsed: object = json.loads(row_json)
            yield _json_object(parsed, "canonical kline row")

    def count(self) -> int:
        result = self.database.execute("SELECT COUNT(*) FROM klines").fetchone()
        assert result is not None
        return int(result[0])

    def close(self) -> None:
        if self.closed:
            return
        self.database.close()
        self.closed = True


def _utc_date(exchange_ts_ms: int) -> str:
    return datetime.fromtimestamp(exchange_ts_ms / 1_000, tz=UTC).date().isoformat()


class _ParquetSink:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.current_key: tuple[str, str, str] | None = None
        self.current_schema: pa.Schema | None = None
        self.current_path: Path | None = None
        self.current_writer: pq.ParquetWriter | None = None
        self.current_rows = 0
        self.buffer: list[dict[str, Any]] = []
        self.part_numbers: Counter[tuple[str, str, str]] = Counter()
        self.files: list[DatasetFile] = []
        self.rows_by_kind: Counter[str] = Counter()

    @staticmethod
    def _batch_limit(kind: str) -> int:
        return ORDERBOOK_BATCH_ROWS if kind == "orderbook" else DEFAULT_BATCH_ROWS

    def add(self, row: dict[str, Any]) -> None:
        kind = _required_string(row.get("kind"), "output kind")
        symbol = _required_string(row.get("symbol"), "output symbol")
        exchange_ts_ms = _required_positive_int(
            row.get("exchange_ts_ms"), "output exchange_ts_ms"
        )
        date = _utc_date(exchange_ts_ms)
        key = (kind, symbol, date)
        if self.current_key != key:
            self._close_partition()
            self._open_partition(key, _schema_for_kind(kind))
        self.buffer.append(row)
        self.current_rows += 1
        self.rows_by_kind[kind] += 1
        if len(self.buffer) >= self._batch_limit(kind):
            self._flush()

    def _open_partition(
        self, key: tuple[str, str, str], schema: pa.Schema
    ) -> None:
        kind, symbol, date = key
        part = self.part_numbers[key]
        self.part_numbers[key] += 1
        relative = (
            Path("market")
            / f"kind={kind}"
            / f"symbol={symbol}"
            / f"date={date}"
            / f"part-{part:05d}.parquet"
        )
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.current_key = key
        self.current_schema = schema
        self.current_path = path
        self.current_rows = 0
        self.current_writer = pq.ParquetWriter(
            path,
            schema,
            version=PARQUET_FORMAT_VERSION,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="1.0",
            write_page_index=True,
            write_page_checksum=True,
        )

    def _flush(self) -> None:
        if not self.buffer:
            return
        assert self.current_schema is not None
        assert self.current_writer is not None
        table = pa.Table.from_pylist(self.buffer, schema=self.current_schema)
        self.current_writer.write_table(table, row_group_size=len(self.buffer))
        self.buffer.clear()

    def _close_partition(self) -> None:
        if self.current_writer is None:
            return
        self._flush()
        writer = self.current_writer
        path = self.current_path
        key = self.current_key
        rows = self.current_rows
        writer.close()
        assert path is not None and key is not None
        kind, symbol, date = key
        relative = path.relative_to(self.root).as_posix()
        self.files.append(
            DatasetFile(
                path=relative,
                kind=kind,
                symbol=symbol,
                date=date,
                rows=rows,
                bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
        self.current_key = None
        self.current_schema = None
        self.current_path = None
        self.current_writer = None
        self.current_rows = 0

    def close(self) -> None:
        self._close_partition()

    def abort(self) -> None:
        self.buffer.clear()
        if self.current_writer is not None:
            with suppress(Exception):
                self.current_writer.close()
        self.current_writer = None


def _resolve_input_file(root: Path, relative_path: str) -> Path:
    relative = _safe_relative_path(relative_path, "source path")
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root):
        raise DatasetBuildError(f"source path escapes dataset root: {relative_path}")
    return candidate


def _process_source_file(
    root: Path,
    item: AuditedInputFile,
    sink: _ParquetSink,
    klines: _CanonicalKlines,
    source_records_by_kind: Counter[str],
) -> None:
    source_path = _resolve_input_file(root, item.path)
    path_parts = PurePosixPath(item.path).parts
    if len(path_parts) != 6:
        raise DatasetBuildError(
            f"source path must be kind/symbol/YYYY/MM/DD/file.jsonl: {item.path}"
        )
    expected_kind, expected_symbol = path_parts[:2]
    if not source_path.is_file():
        raise DatasetBuildError(f"audited source file is missing: {source_path}")
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    record_count = 0
    with source_path.open("rb") as source:
        for line_count, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            try:
                parsed: object = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(
                    f"invalid JSON in {item.path}:{line_count}: {exc}"
                ) from exc
            raw = _json_object(parsed, f"{item.path}:{line_count}")
            common = _common_row(raw, item.path, line_count)
            kind = _required_string(common.get("kind"), "kind")
            symbol = _required_string(common.get("symbol"), "symbol")
            if kind != expected_kind or symbol != expected_symbol:
                raise DatasetBuildError(
                    f"record wrapper does not match partition path at "
                    f"{item.path}:{line_count}"
                )
            if common["source"] != "bybit":
                raise DatasetBuildError(
                    f"unsupported source at {item.path}:{line_count}: "
                    f"{common['source']!r}"
                )
            source_records_by_kind[kind] += 1
            record_count += 1
            if kind == "orderbook":
                sink.add(_orderbook_row(raw, common))
            elif kind == "ticker":
                sink.add(_ticker_row(raw, common))
            elif kind == "trades":
                for row in _trade_rows(raw, common):
                    sink.add(row)
            elif kind.startswith("kline_"):
                klines.observe(_kline_row(raw, common))
            else:
                raise DatasetBuildError(
                    f"unsupported kind {kind!r} in {item.path}:{line_count}"
                )
    actual_sha256 = digest.hexdigest()
    if byte_count != item.bytes:
        raise DatasetBuildError(
            f"source byte count changed for {item.path}: "
            f"expected {item.bytes}, found {byte_count}"
        )
    if line_count != item.lines:
        raise DatasetBuildError(
            f"source line count changed for {item.path}: "
            f"expected {item.lines}, found {line_count}"
        )
    if record_count != item.records:
        raise DatasetBuildError(
            f"source record count changed for {item.path}: "
            f"expected {item.records}, found {record_count}"
        )
    if actual_sha256 != item.sha256:
        raise DatasetBuildError(
            f"source SHA-256 changed for {item.path}: "
            f"expected {item.sha256}, found {actual_sha256}"
        )


def _safe_output_root(source_root: Path, output_root: Path) -> None:
    if (
        output_root == source_root
        or output_root.is_relative_to(source_root)
        or source_root.is_relative_to(output_root)
    ):
        raise DatasetBuildError("source and output roots must not overlap")


def _manifest_result(
    dataset_path: Path,
    manifest: dict[str, Any],
    *,
    reused: bool,
) -> DatasetBuildResult:
    rows_by_kind_raw = _json_object(
        manifest.get("output_rows_by_kind"), "output_rows_by_kind"
    )
    rows_by_kind = {
        key: _required_nonnegative_int(value, f"output_rows_by_kind.{key}")
        for key, value in rows_by_kind_raw.items()
    }
    source = _json_object(manifest.get("source"), "source")
    return DatasetBuildResult(
        dataset_id=_required_string(manifest.get("dataset_id"), "dataset_id"),
        dataset_path=dataset_path,
        manifest_path=dataset_path / "manifest.json",
        input_fingerprint=_valid_sha256(source.get("input_fingerprint")),
        output_fingerprint=_valid_sha256(manifest.get("output_fingerprint")),
        source_files=_required_nonnegative_int(
            source.get("file_count"), "source.file_count"
        ),
        source_records=_required_nonnegative_int(
            source.get("records"), "source.records"
        ),
        output_files=_required_nonnegative_int(
            manifest.get("output_file_count"), "output_file_count"
        ),
        output_rows=sum(rows_by_kind.values()),
        output_rows_by_kind=rows_by_kind,
        reused=reused,
    )


def _validate_existing_dataset(
    dataset_path: Path,
    audit: AuditInputManifest,
    dataset_id: str,
) -> DatasetBuildResult:
    manifest_path = dataset_path / "manifest.json"
    try:
        parsed: object = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(
            f"existing dataset manifest is unreadable: {manifest_path}"
        ) from exc
    manifest = _json_object(parsed, "dataset manifest")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetBuildError("existing dataset uses a different schema version")
    if manifest.get("dataset_id") != dataset_id:
        raise DatasetBuildError("existing dataset_id does not match its directory")
    source = _json_object(manifest.get("source"), "source")
    if source.get("input_fingerprint") != audit.input_fingerprint:
        raise DatasetBuildError("existing dataset was built from different input")

    audit_copy = dataset_path / "source-audit.json"
    stored_audit_sha256 = _valid_sha256(source.get("audit_report_sha256"))
    if not audit_copy.is_file() or _sha256_file(audit_copy) != stored_audit_sha256:
        raise DatasetBuildError("existing source-audit.json failed SHA-256 validation")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise DatasetBuildError("existing dataset manifest.files must be an array")
    files: list[DatasetFile] = []
    for index, raw_file in enumerate(raw_files):
        item = _json_object(raw_file, f"manifest.files[{index}]")
        relative = _safe_relative_path(
            item.get("path"), f"manifest.files[{index}].path"
        )
        actual = dataset_path.joinpath(*relative.parts).resolve()
        if not actual.is_relative_to(dataset_path) or not actual.is_file():
            raise DatasetBuildError(f"existing dataset file is missing: {relative}")
        expected_bytes = _required_nonnegative_int(
            item.get("bytes"), f"manifest.files[{index}].bytes"
        )
        expected_sha = _valid_sha256(item.get("sha256"))
        if actual.stat().st_size != expected_bytes or _sha256_file(actual) != expected_sha:
            raise DatasetBuildError(f"existing dataset file is corrupted: {relative}")
        parquet_rows = pq.ParquetFile(actual).metadata.num_rows
        expected_rows = _required_nonnegative_int(
            item.get("rows"), f"manifest.files[{index}].rows"
        )
        if parquet_rows != expected_rows:
            raise DatasetBuildError(
                f"existing dataset row count is inconsistent: {relative}"
            )
        files.append(
            DatasetFile(
                path=relative.as_posix(),
                kind=_required_string(
                    item.get("kind"), f"manifest.files[{index}].kind"
                ),
                symbol=_required_string(
                    item.get("symbol"), f"manifest.files[{index}].symbol"
                ),
                date=_required_string(
                    item.get("date"), f"manifest.files[{index}].date"
                ),
                rows=expected_rows,
                bytes=expected_bytes,
                sha256=expected_sha,
            )
        )
    if len(files) != manifest.get("output_file_count"):
        raise DatasetBuildError("existing output_file_count is inconsistent")
    if _dataset_files_fingerprint(files) != manifest.get("output_fingerprint"):
        raise DatasetBuildError("existing output_fingerprint is inconsistent")
    rows_by_kind_raw = _json_object(
        manifest.get("output_rows_by_kind"), "output_rows_by_kind"
    )
    rows_by_kind = {
        key: _required_nonnegative_int(value, f"output_rows_by_kind.{key}")
        for key, value in rows_by_kind_raw.items()
    }
    descriptor_rows: Counter[str] = Counter()
    for dataset_file in files:
        descriptor_rows[dataset_file.kind] += dataset_file.rows
    if dict(sorted(descriptor_rows.items())) != dict(sorted(rows_by_kind.items())):
        raise DatasetBuildError("existing output row totals are inconsistent")
    return _manifest_result(dataset_path, manifest, reused=True)


def validate_canonical_dataset(dataset_path: str | Path) -> DatasetBuildResult:
    """Validate every file in an existing canonical dataset.

    Research stages use this public gate instead of trusting a directory name or
    scanning arbitrary Parquet files.  It verifies the copied audit report,
    every Parquet SHA-256 and row count, and both dataset fingerprints.
    """

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_dir():
        raise DatasetBuildError(f"canonical dataset does not exist: {path}")
    audit_path = path / "source-audit.json"
    if not audit_path.is_file():
        raise DatasetBuildError(
            f"canonical dataset is missing source-audit.json: {path}"
        )
    audit = load_audit_input_manifest(audit_path)
    return _validate_existing_dataset(path, audit, path.name)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(".json.partial")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as target:
        target.write(rendered)
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, path)


def build_canonical_dataset(
    audit_report: str | Path,
    output_root: str | Path,
    *,
    source_root: str | Path | None = None,
    minimum_free_bytes: int = 0,
) -> DatasetBuildResult:
    """Build or validate the canonical dataset identified by its input fingerprint."""

    audit = load_audit_input_manifest(audit_report)
    root = Path(audit.dataset_root if source_root is None else source_root)
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetBuildError(f"source dataset root does not exist: {root}")
    destination_root = Path(output_root).expanduser().resolve()
    _safe_output_root(root, destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    dataset_id = (
        f"canonical-v{DATASET_SCHEMA_VERSION}-{audit.input_fingerprint[:16]}"
    )
    final_path = destination_root / dataset_id
    if final_path.exists():
        return _validate_existing_dataset(final_path, audit, dataset_id)

    if minimum_free_bytes < 0:
        raise DatasetBuildError("minimum_free_bytes must be non-negative")
    free_bytes = shutil.disk_usage(destination_root).free
    required_free_bytes = minimum_free_bytes + audit.total_bytes
    if free_bytes < required_free_bytes:
        raise DatasetBuildError(
            "insufficient disk space for canonical build: "
            f"{free_bytes} bytes free, {required_free_bytes} required"
        )

    staging_path = destination_root / f".{dataset_id}.tmp-{uuid.uuid4().hex}"
    staging_path.mkdir()
    sink = _ParquetSink(staging_path)
    klines = _CanonicalKlines(staging_path / ".canonical-klines.sqlite")
    source_records_by_kind: Counter[str] = Counter()
    processed_bytes = 0
    started = time.monotonic()
    last_progress = started
    try:
        audit_copy = staging_path / "source-audit.json"
        shutil.copyfile(audit.report_path, audit_copy)
        if _sha256_file(audit_copy) != audit.report_sha256:
            raise DatasetBuildError("audit report changed while the build was starting")
        LOGGER.info(
            "Building %s from %d audited files (%.2f GiB)",
            dataset_id,
            len(audit.files),
            audit.total_bytes / (1024**3),
        )
        for index, item in enumerate(audit.files, start=1):
            _process_source_file(root, item, sink, klines, source_records_by_kind)
            processed_bytes += item.bytes
            now = time.monotonic()
            if now - last_progress >= 10 or index == len(audit.files):
                elapsed = max(now - started, 0.001)
                percent = (
                    100.0
                    if audit.total_bytes == 0
                    else min(100.0, processed_bytes / audit.total_bytes * 100)
                )
                LOGGER.info(
                    "Dataset progress: %d/%d files, %.1f%%, %.1f MiB/s",
                    index,
                    len(audit.files),
                    percent,
                    processed_bytes / (1024**2) / elapsed,
                )
                last_progress = now

        canonical_kline_rows = klines.count()
        for row in klines.rows():
            sink.add(row)
        sink.close()
        klines.close()
        (staging_path / ".canonical-klines.sqlite").unlink()

        files = tuple(sorted(sink.files, key=lambda item: item.path))
        output_fingerprint = _dataset_files_fingerprint(files)
        source_kline_rows = sum(
            count
            for kind, count in source_records_by_kind.items()
            if kind.startswith("kline_")
        )
        manifest: dict[str, object] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "builder": {
                "package_version": __version__,
                "pyarrow_version": pa.__version__,
                "parquet_format_version": PARQUET_FORMAT_VERSION,
                "compression": PARQUET_COMPRESSION,
                "compression_level": PARQUET_COMPRESSION_LEVEL,
                "default_batch_rows": DEFAULT_BATCH_ROWS,
                "orderbook_batch_rows": ORDERBOOK_BATCH_ROWS,
            },
            "source": {
                "audit_report_schema_version": AUDIT_REPORT_SCHEMA_VERSION,
                "audit_report_copy": "source-audit.json",
                "audit_report_sha256": audit.report_sha256,
                "audit_dataset_root": audit.dataset_root,
                "resolved_dataset_root": root.as_posix(),
                "input_fingerprint": audit.input_fingerprint,
                "file_count": len(audit.files),
                "bytes": audit.total_bytes,
                "records": audit.total_records,
                "expected_symbols": list(audit.expected_symbols),
                "kline_intervals": list(audit.kline_intervals),
            },
            "canonicalization": {
                "kline_key": ["symbol", "interval", "start_ms"],
                "selection": "maximum received_at_ns",
                "equal_timestamp_different_payload": "error",
                "reported_kline_revisions": audit.kline_revisions,
                "reported_exact_redeliveries": audit.duplicate_klines,
                "source_kline_rows": source_kline_rows,
                "canonical_kline_rows": canonical_kline_rows,
                "collapsed_kline_rows": source_kline_rows - canonical_kline_rows,
            },
            "partitioning": ["kind", "symbol", "exchange_utc_date"],
            "schemas": {
                "orderbook": _schema_manifest(ORDERBOOK_SCHEMA),
                "ticker": _schema_manifest(TICKER_SCHEMA),
                "trades": _schema_manifest(TRADES_SCHEMA),
                "kline": _schema_manifest(KLINE_SCHEMA),
            },
            "source_records_by_kind": dict(sorted(source_records_by_kind.items())),
            "output_rows_by_kind": dict(sorted(sink.rows_by_kind.items())),
            "output_file_count": len(files),
            "output_fingerprint": output_fingerprint,
            "files": [item.to_dict() for item in files],
        }
        _write_manifest(staging_path / "manifest.json", manifest)
        os.replace(staging_path, final_path)
        LOGGER.info(
            "Canonical dataset ready at %s (%d Parquet files)",
            final_path,
            len(files),
        )
        return _manifest_result(final_path, manifest, reused=False)
    except Exception:
        sink.abort()
        klines.close()
        if staging_path.is_dir() and staging_path.parent == destination_root:
            shutil.rmtree(staging_path, ignore_errors=True)
        raise
