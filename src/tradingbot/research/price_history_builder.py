"""Build a causal price-only research dataset from official Bybit trade bars.

The source profile deliberately contains no order book, ticker, funding, open
interest, or individual-trade ordering.  This builder keeps that limitation
explicit: it emits a separate research profile and resolves barriers only to
one-second precision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from tradingbot import __version__
from tradingbot.data.bybit_history import (
    HISTORY_CATALOG_SCHEMA_VERSION,
    HISTORY_DAY_SCHEMA_VERSION,
    HISTORY_PROFILE,
    HistoryImportError,
    validate_history_day,
)
from tradingbot.research.contracts import (
    KLINE_RETURN_WINDOWS_MINUTES,
    KLINE_VOLATILITY_WINDOWS_MINUTES,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_FORMAT_VERSION,
    PRICE_FEATURE_SCHEMA,
    PRICE_LABEL_SCHEMA,
    PRICE_RESEARCH_PROFILE,
    RESEARCH_SCHEMA_VERSION,
    TRADE_WINDOWS_SECONDS,
    PriceResearchParameters,
    ResearchBuildError,
    ResearchBuildResult,
    ResearchFile,
)

LOGGER = logging.getLogger(__name__)

NS_PER_MILLISECOND: Final = 1_000_000
NS_PER_SECOND: Final = 1_000_000_000
MS_PER_MINUTE: Final = 60_000
NS_PER_MINUTE: Final = 60 * NS_PER_SECOND
NS_PER_DAY: Final = 24 * 60 * NS_PER_MINUTE
BTC_SYMBOL: Final = "BTCUSDT"
MINUTE_COLUMNS: Final = (
    "interval_seconds",
    "start_ms",
    "end_ms",
    "available_at_ns",
    "high",
    "low",
    "close",
    "volume",
)
SECOND_COLUMNS: Final = (
    "interval_seconds",
    "start_ms",
    "end_ms",
    "available_at_ns",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "buy_volume",
    "sell_volume",
    "trade_count",
)


@dataclass(frozen=True, slots=True)
class _HistoryFile:
    path: Path
    kind: str
    symbol: str
    partition_date: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _HistoryDay:
    partition_date: str
    path: Path
    manifest_sha256: str
    parameters_fingerprint: str
    output_fingerprint: str
    output_rows: int
    output_bytes: int
    files: dict[tuple[str, str], _HistoryFile]


@dataclass(frozen=True, slots=True)
class _HistorySelection:
    history_root: Path
    catalog_path: Path
    symbols: tuple[str, ...]
    days: tuple[_HistoryDay, ...]
    start_date: str
    end_date: str
    dataset_id: str
    output_fingerprint: str
    manifest_payload: dict[str, object]
    manifest_bytes: bytes
    manifest_sha256: str
    total_bytes: int


@dataclass(frozen=True, slots=True)
class _BarrierResult:
    outcome: str
    hit_index: int | None
    timeout_price: float | None
    outcome_return_bps: float | None
    future_trade_count: int
    resolution: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ResearchBuildError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(rendered.encode("utf-8"))


def _render_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchBuildError(f"{label} is unreadable: {path}") from exc
    if not isinstance(parsed, dict):
        raise ResearchBuildError(f"{label} must be a JSON object")
    return cast(dict[str, Any], parsed)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResearchBuildError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchBuildError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchBuildError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label).lower()
    if len(text) != 64:
        raise ResearchBuildError(f"{label} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ResearchBuildError(f"{label} must be a SHA-256 digest") from exc
    return text


def _partition_date(value: object, label: str) -> date:
    text = _string(value, label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ResearchBuildError(f"{label} must use YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise ResearchBuildError(f"{label} must use canonical YYYY-MM-DD")
    return parsed


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_string(value, label))
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ResearchBuildError(f"{label} is not a safe relative path")
    return path


def _resolve_relative(root: Path, value: object, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    resolved = root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ResearchBuildError(f"{label} escapes its root")
    return resolved


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _catalog_entries_fingerprint(entries: object) -> str:
    if not isinstance(entries, list):
        raise ResearchBuildError("history catalog.entries must be an array")
    return _sha256_json(entries)


def _history_file(
    day_path: Path,
    raw: object,
    *,
    index: int,
    expected_date: str,
    symbols: tuple[str, ...],
) -> _HistoryFile:
    item = _object(raw, f"history day files[{index}]")
    kind = _string(item.get("kind"), f"history day files[{index}].kind")
    if kind not in {"trade_bar_1s", "trade_bar_1m"}:
        raise ResearchBuildError(f"unsupported history file kind: {kind}")
    symbol = _string(item.get("symbol"), f"history day files[{index}].symbol")
    if symbol not in symbols:
        raise ResearchBuildError(f"history file contains an unknown symbol: {symbol}")
    partition = _string(item.get("date"), f"history day files[{index}].date")
    if partition != expected_date:
        raise ResearchBuildError("history file date differs from its day")
    path = _resolve_relative(day_path, item.get("path"), f"history files[{index}].path")
    return _HistoryFile(
        path=path,
        kind=kind,
        symbol=symbol,
        partition_date=partition,
        rows=_nonnegative_int(item.get("rows"), f"history files[{index}].rows"),
        bytes=_nonnegative_int(item.get("bytes"), f"history files[{index}].bytes"),
        sha256=_sha256(item.get("sha256"), f"history files[{index}].sha256"),
    )


def _load_history_selection(
    catalog_path: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> _HistorySelection:
    catalog = Path(catalog_path).expanduser().resolve()
    history_root = catalog.parent
    if catalog.name != "catalog.json" or not catalog.is_file():
        raise ResearchBuildError(f"history catalog does not exist: {catalog}")
    raw_catalog = _load_json(catalog, "history catalog")
    if raw_catalog.get("history_catalog_schema_version") != HISTORY_CATALOG_SCHEMA_VERSION:
        raise ResearchBuildError("unsupported history catalog schema")
    if raw_catalog.get("history_day_schema_version") != HISTORY_DAY_SCHEMA_VERSION:
        raise ResearchBuildError("history catalog day schema is inconsistent")
    if raw_catalog.get("dataset_profile") != HISTORY_PROFILE:
        raise ResearchBuildError("history catalog is not price_futures_v1")
    raw_entries = raw_catalog.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ResearchBuildError("history catalog has no entries")
    if raw_catalog.get("entry_count") != len(raw_entries):
        raise ResearchBuildError("history catalog entry_count is inconsistent")
    if raw_catalog.get("catalog_fingerprint") != _catalog_entries_fingerprint(raw_entries):
        raise ResearchBuildError("history catalog fingerprint is inconsistent")

    first = _partition_date(start_date, "start_date")
    last = _partition_date(end_date, "end_date")
    if first > last:
        raise ResearchBuildError("start_date cannot be after end_date")
    by_date: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry = _object(raw_entry, f"history catalog.entries[{index}]")
        partition = _partition_date(
            entry.get("partition_date"), f"history catalog.entries[{index}].partition_date"
        ).isoformat()
        if partition in by_date:
            raise ResearchBuildError(f"history catalog repeats {partition}")
        by_date[partition] = entry

    selected_dates: list[str] = []
    current = first
    while current <= last:
        text = current.isoformat()
        if text not in by_date:
            raise ResearchBuildError(f"history catalog is missing selected day {text}")
        selected_dates.append(text)
        current += timedelta(days=1)

    days: list[_HistoryDay] = []
    expected_symbols: tuple[str, ...] | None = None
    expected_parameters: str | None = None
    source_entries: list[dict[str, object]] = []
    for partition in selected_dates:
        entry = by_date[partition]
        day_path = _resolve_relative(history_root, entry.get("path"), "catalog day path")
        manifest_path = _resolve_relative(
            history_root, entry.get("manifest_path"), "catalog day manifest path"
        )
        if manifest_path != day_path / "manifest.json":
            raise ResearchBuildError(f"catalog manifest path is inconsistent for {partition}")
        try:
            validated = validate_history_day(
                day_path,
                expected_parameters_fingerprint=expected_parameters,
                expected_symbols=expected_symbols,
            )
        except HistoryImportError as exc:
            raise ResearchBuildError(f"history day {partition} failed validation: {exc}") from exc
        if validated.partition_date != partition:
            raise ResearchBuildError(f"history day result differs for {partition}")
        if expected_symbols is None:
            expected_symbols = validated.symbols
            expected_parameters = validated.parameters_fingerprint
            if BTC_SYMBOL not in expected_symbols:
                raise ResearchBuildError("price research requires BTCUSDT context")
        manifest_sha = _sha256_file(manifest_path)
        if manifest_sha != _sha256(entry.get("manifest_sha256"), "entry.manifest_sha256"):
            raise ResearchBuildError(f"catalog manifest SHA differs for {partition}")
        if validated.output_fingerprint != _sha256(
            entry.get("output_fingerprint"), "entry.output_fingerprint"
        ):
            raise ResearchBuildError(f"catalog output fingerprint differs for {partition}")
        if validated.parameters_fingerprint != _sha256(
            entry.get("parameters_fingerprint"), "entry.parameters_fingerprint"
        ):
            raise ResearchBuildError(f"catalog parameter fingerprint differs for {partition}")
        raw_entry_symbols = entry.get("symbols")
        if raw_entry_symbols != list(validated.symbols):
            raise ResearchBuildError(f"catalog symbol list differs for {partition}")
        manifest = _load_json(manifest_path, "history day manifest")
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise ResearchBuildError(f"history day {partition} has no files")
        file_map: dict[tuple[str, str], _HistoryFile] = {}
        assert expected_symbols is not None
        for index, raw_file in enumerate(raw_files):
            item = _history_file(
                day_path,
                raw_file,
                index=index,
                expected_date=partition,
                symbols=expected_symbols,
            )
            key = (item.symbol, item.kind)
            if key in file_map:
                raise ResearchBuildError(f"history day repeats {item.symbol}/{item.kind}")
            file_map[key] = item
        expected_keys = {
            (symbol, kind)
            for symbol in expected_symbols
            for kind in ("trade_bar_1s", "trade_bar_1m")
        }
        if set(file_map) != expected_keys:
            raise ResearchBuildError(f"history day {partition} has incomplete symbol files")
        output_rows = _nonnegative_int(entry.get("output_rows"), "entry.output_rows")
        output_bytes = _nonnegative_int(entry.get("output_bytes"), "entry.output_bytes")
        if output_rows != validated.output_rows or output_bytes != validated.output_bytes:
            raise ResearchBuildError(f"catalog row/byte totals differ for {partition}")
        day = _HistoryDay(
            partition_date=partition,
            path=day_path,
            manifest_sha256=manifest_sha,
            parameters_fingerprint=validated.parameters_fingerprint,
            output_fingerprint=validated.output_fingerprint,
            output_rows=output_rows,
            output_bytes=output_bytes,
            files=file_map,
        )
        days.append(day)
        source_entries.append(
            {
                "partition_date": partition,
                "manifest_sha256": manifest_sha,
                "parameters_fingerprint": validated.parameters_fingerprint,
                "output_fingerprint": validated.output_fingerprint,
                "output_rows": output_rows,
                "output_bytes": output_bytes,
            }
        )

    assert expected_symbols is not None
    selection_fingerprint = _sha256_json(source_entries)
    dataset_id = f"history-selection-v1-{selection_fingerprint[:16]}"
    total_bytes = sum(day.output_bytes for day in days)
    source_manifest: dict[str, object] = {
        "dataset_schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_profile": HISTORY_PROFILE,
        "output_fingerprint": selection_fingerprint,
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "days": len(days),
        "symbols": list(expected_symbols),
        "bytes": total_bytes,
        "source_capabilities": {
            "trade_bar_1s": True,
            "trade_bar_1m": True,
            "individual_trades": False,
            "orderbook": False,
            "ticker": False,
            "funding": False,
            "open_interest": False,
        },
        "entries": source_entries,
    }
    manifest_bytes = _render_json(source_manifest)
    return _HistorySelection(
        history_root=history_root,
        catalog_path=catalog,
        symbols=expected_symbols,
        days=tuple(days),
        start_date=first.isoformat(),
        end_date=last.isoformat(),
        dataset_id=dataset_id,
        output_fingerprint=selection_fingerprint,
        manifest_payload=source_manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        total_bytes=total_bytes,
    )


def _int64(table: pa.Table, name: str) -> np.ndarray[Any, np.dtype[np.int64]]:
    values = pc.cast(table.column(name).combine_chunks(), pa.int64())
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def _float64(table: pa.Table, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = pc.cast(table.column(name).combine_chunks(), pa.float64())
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)


def _read_bars(files: Sequence[_HistoryFile], columns: Sequence[str]) -> pa.Table:
    tables = [pq.ParquetFile(item.path).read(columns=list(columns)) for item in files]
    if not tables:
        return pa.table({name: [] for name in columns})
    return pa.concat_tables(tables)


class _MinuteSeries:
    def __init__(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ResearchBuildError("minute trade-bar table is empty")
        intervals = _int64(table, "interval_seconds")
        starts = _int64(table, "start_ms")
        available = _int64(table, "available_at_ns")
        high = _float64(table, "high")
        low = _float64(table, "low")
        close = _float64(table, "close")
        volume = _float64(table, "volume")
        if np.any(intervals != 60):
            raise ResearchBuildError("minute history contains another interval")
        if (
            np.any(~np.isfinite(high))
            or np.any(~np.isfinite(low))
            or np.any(~np.isfinite(close))
            or np.any(close <= 0)
            or np.any(~np.isfinite(volume))
            or np.any(volume < 0)
        ):
            raise ResearchBuildError("minute history contains invalid numeric values")
        order = np.argsort(starts, kind="stable")
        self.starts_ms = starts[order]
        self.available_at_ns = available[order]
        self.high = high[order]
        self.low = low[order]
        self.close = close[order]
        self.volume = volume[order]
        if np.any(self.starts_ms[1:] <= self.starts_ms[:-1]):
            raise ResearchBuildError("minute history contains duplicate or unordered bars")

    def features_at(
        self,
        decision_at_ns: int,
        *,
        history_minutes: int,
        volatility_lookback_minutes: int,
    ) -> tuple[dict[str, object] | None, float | None, str | None]:
        decision_at_ms = decision_at_ns // NS_PER_MILLISECOND
        target_start = decision_at_ms // MS_PER_MINUTE * MS_PER_MINUTE - MS_PER_MINUTE
        position = int(np.searchsorted(self.starts_ms, target_start, side="left"))
        if position >= len(self.starts_ms) or int(self.starts_ms[position]) != target_start:
            return None, None, "missing_latest_minute_bar"
        first = position - history_minutes
        if first < 0:
            return None, None, "missing_minute_history"
        starts = self.starts_ms[first : position + 1]
        expected = target_start - np.arange(history_minutes, -1, -1) * MS_PER_MINUTE
        if not np.array_equal(starts, expected):
            return None, None, "missing_minute_history"
        available = self.available_at_ns[first : position + 1]
        if np.any(available > decision_at_ns):
            return None, None, "minute_bar_not_yet_available"
        closes = self.close[first : position + 1]
        highs = self.high[first : position + 1]
        lows = self.low[first : position + 1]
        volumes = self.volume[first : position + 1]
        log_returns = np.diff(np.log(closes))
        result: dict[str, object] = {
            "latest_minute_bar_available_at_ns": int(available[-1]),
            "minute_bar_age_ms": (
                decision_at_ns - int(available[-1])
            )
            / NS_PER_MILLISECOND,
            "close_price": float(closes[-1]),
        }
        for window in KLINE_RETURN_WINDOWS_MINUTES:
            result[f"return_{window}m_fraction"] = float(
                closes[-1] / closes[-1 - window] - 1
            )
        for window in KLINE_VOLATILITY_WINDOWS_MINUTES:
            window_returns = log_returns[-window:]
            result[f"realized_volatility_{window}m_fraction"] = float(
                math.sqrt(float(np.dot(window_returns, window_returns)))
            )
        true_ranges = np.maximum.reduce(
            (
                highs[-14:] - lows[-14:],
                np.abs(highs[-14:] - closes[-15:-1]),
                np.abs(lows[-14:] - closes[-15:-1]),
            )
        )
        result["atr_14_bps"] = float(np.mean(true_ranges) / closes[-1] * 10_000)
        result["range_1m_bps"] = float((highs[-1] - lows[-1]) / closes[-1] * 10_000)
        mean_volume_60 = float(np.mean(volumes[-60:]))
        result["volume_ratio_5m_to_60m"] = (
            0.0 if mean_volume_60 == 0 else float(np.mean(volumes[-5:])) / mean_volume_60
        )
        label_returns = log_returns[-volatility_lookback_minutes:]
        label_volatility = float(math.sqrt(float(np.dot(label_returns, label_returns))))
        return result, label_volatility, None


class _SecondSeries:
    def __init__(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ResearchBuildError("one-second trade-bar table is empty")
        intervals = _int64(table, "interval_seconds")
        starts = _int64(table, "start_ms")
        available = _int64(table, "available_at_ns")
        high = _float64(table, "high")
        low = _float64(table, "low")
        close = _float64(table, "close")
        volume = _float64(table, "volume")
        turnover = _float64(table, "turnover")
        buy_volume = _float64(table, "buy_volume")
        sell_volume = _float64(table, "sell_volume")
        trade_count = _int64(table, "trade_count")
        if np.any(intervals != 1):
            raise ResearchBuildError("one-second history contains another interval")
        if (
            np.any(~np.isfinite(high))
            or np.any(~np.isfinite(low))
            or np.any(~np.isfinite(close))
            or np.any(close <= 0)
            or np.any(~np.isfinite(volume))
            or np.any(volume <= 0)
            or np.any(~np.isfinite(turnover))
            or np.any(turnover <= 0)
            or np.any(trade_count <= 0)
        ):
            raise ResearchBuildError("one-second history contains invalid numeric values")
        order = np.lexsort((starts, available))
        self.starts_ms = starts[order]
        self.available_at_ns = available[order]
        self.high = high[order]
        self.low = low[order]
        self.close = close[order]
        self.volume = volume[order]
        self.turnover = turnover[order]
        self.trade_count = trade_count[order]
        signed = buy_volume[order] - sell_volume[order]
        if np.any(self.available_at_ns[1:] <= self.available_at_ns[:-1]):
            raise ResearchBuildError("one-second history has duplicate availability keys")
        self.prefix_volume = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.volume, dtype=np.float64))
        )
        self.prefix_turnover = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.turnover, dtype=np.float64))
        )
        self.prefix_signed = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(signed, dtype=np.float64))
        )
        self.prefix_count = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(self.trade_count, dtype=np.int64))
        )

    def features_at(
        self, decision_at_ns: int, *, maximum_trade_age_ms: int
    ) -> tuple[dict[str, object] | None, str | None]:
        end = int(np.searchsorted(self.available_at_ns, decision_at_ns, side="right"))
        if end == 0:
            return None, "no_available_trade_bar"
        latest_available = int(self.available_at_ns[end - 1])
        age_ms = (decision_at_ns - latest_available) / NS_PER_MILLISECOND
        if age_ms > maximum_trade_age_ms:
            return None, "stale_trade_bar"
        result: dict[str, object] = {
            "latest_second_bar_available_at_ns": latest_available,
            "trade_age_ms": age_ms,
            "reference_price": float(self.close[end - 1]),
        }
        for seconds in TRADE_WINDOWS_SECONDS:
            start = int(
                np.searchsorted(
                    self.available_at_ns,
                    decision_at_ns - seconds * NS_PER_SECOND,
                    side="left",
                )
            )
            count = int(self.prefix_count[end] - self.prefix_count[start])
            volume = float(self.prefix_volume[end] - self.prefix_volume[start])
            turnover = float(self.prefix_turnover[end] - self.prefix_turnover[start])
            signed = float(self.prefix_signed[end] - self.prefix_signed[start])
            suffix = f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"
            result[f"trade_count_{suffix}"] = count
            result[f"trade_base_volume_{suffix}"] = volume
            result[f"trade_notional_{suffix}"] = turnover
            result[f"trade_imbalance_{suffix}"] = 0.0 if volume == 0 else signed / volume
            result[f"trade_return_{suffix}_fraction"] = (
                0.0 if end - start < 2 else float(self.close[end - 1] / self.close[start] - 1)
            )
        return result, None

    def barrier_outcome(
        self,
        *,
        decision_at_ns: int,
        label_end_ns: int,
        side: str,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        stop_distance_bps: float,
        take_profit_distance_bps: float,
    ) -> _BarrierResult:
        start = int(np.searchsorted(self.available_at_ns, decision_at_ns, side="right"))
        end = int(np.searchsorted(self.available_at_ns, label_end_ns, side="right"))
        future_trade_count = int(self.prefix_count[end] - self.prefix_count[start])
        if side == "LONG":
            tp_hits = self.high[start:end] >= take_profit_price
            sl_hits = self.low[start:end] <= stop_price
        elif side == "SHORT":
            tp_hits = self.low[start:end] <= take_profit_price
            sl_hits = self.high[start:end] >= stop_price
        else:
            raise ResearchBuildError(f"unsupported label side: {side}")
        candidates = np.flatnonzero(tp_hits | sl_hits)
        if len(candidates):
            offset = int(candidates[0])
            hit_index = start + offset
            if bool(tp_hits[offset]) and bool(sl_hits[offset]):
                return _BarrierResult(
                    outcome="AMBIGUOUS",
                    hit_index=hit_index,
                    timeout_price=None,
                    outcome_return_bps=None,
                    future_trade_count=future_trade_count,
                    resolution="same_one_second_bar_crossed_both_barriers",
                )
            if bool(tp_hits[offset]):
                return _BarrierResult(
                    outcome="TP_FIRST",
                    hit_index=hit_index,
                    timeout_price=None,
                    outcome_return_bps=take_profit_distance_bps,
                    future_trade_count=future_trade_count,
                    resolution="one_second_bar_crossed_take_profit",
                )
            return _BarrierResult(
                outcome="SL_FIRST",
                hit_index=hit_index,
                timeout_price=None,
                outcome_return_bps=-stop_distance_bps,
                future_trade_count=future_trade_count,
                resolution="one_second_bar_crossed_stop",
            )
        timeout_price = None if end == start else float(self.close[end - 1])
        if timeout_price is None:
            outcome_return = None
            resolution = "no_public_trade_bar_in_complete_horizon"
        else:
            direction = 1.0 if side == "LONG" else -1.0
            outcome_return = direction * (timeout_price / entry_price - 1) * 10_000
            resolution = "complete_horizon_no_barrier"
        return _BarrierResult(
            outcome="TIMEOUT",
            hit_index=None,
            timeout_price=timeout_price,
            outcome_return_bps=outcome_return,
            future_trade_count=future_trade_count,
            resolution=resolution,
        )


def _utc_day_start_ns(partition_date: str) -> int:
    instant = datetime.combine(date.fromisoformat(partition_date), datetime.min.time(), tzinfo=UTC)
    return int(instant.timestamp()) * NS_PER_SECOND


def _utc_date_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / NS_PER_SECOND, tz=UTC).date().isoformat()


def _time_features(decision_at_ns: int) -> dict[str, float]:
    instant = datetime.fromtimestamp(decision_at_ns / NS_PER_SECOND, tz=UTC)
    hour = instant.hour + instant.minute / 60 + instant.second / 3_600
    weekday = instant.weekday() + hour / 24
    hour_angle = 2 * math.pi * hour / 24
    weekday_angle = 2 * math.pi * weekday / 7
    return {
        "utc_hour_sin": math.sin(hour_angle),
        "utc_hour_cos": math.cos(hour_angle),
        "utc_weekday_sin": math.sin(weekday_angle),
        "utc_weekday_cos": math.cos(weekday_angle),
    }


def _decision_id(source_dataset_id: str, symbol: str, decision_at_ns: int) -> str:
    raw = f"{source_dataset_id}|{symbol}|{decision_at_ns}".encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchBuildError(f"feature {key} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResearchBuildError(f"feature {key} is not finite")
    return result


def _label_rows(
    *,
    source_dataset_id: str,
    feature: dict[str, object],
    label_volatility: float,
    seconds: _SecondSeries,
    parameters: PriceResearchParameters,
    coverage_end_ns: int,
    quality: Counter[str],
) -> list[dict[str, object]]:
    decision_at_ns = int(cast(int, feature["decision_at_ns"]))
    entry_price = _number(feature, "reference_price")
    per_minute_volatility = label_volatility / math.sqrt(
        parameters.volatility_lookback_minutes
    )
    rows: list[dict[str, object]] = []
    for horizon_minutes in parameters.label_horizons_minutes:
        label_end_ns = decision_at_ns + horizon_minutes * NS_PER_MINUTE
        if label_end_ns > coverage_end_ns:
            quality[f"labels_skipped_incomplete_{horizon_minutes}m"] += 2
            continue
        horizon_volatility_bps = (
            per_minute_volatility
            * math.sqrt(horizon_minutes)
            * parameters.stop_volatility_multiple
            * 10_000
        )
        stop_distance_bps = min(
            parameters.maximum_stop_bps,
            max(parameters.minimum_stop_bps, horizon_volatility_bps),
        )
        take_profit_distance_bps = stop_distance_bps * parameters.take_profit_multiple
        for side in ("LONG", "SHORT"):
            direction = 1.0 if side == "LONG" else -1.0
            stop_price = entry_price * (1 - direction * stop_distance_bps / 10_000)
            take_profit_price = entry_price * (
                1 + direction * take_profit_distance_bps / 10_000
            )
            outcome = seconds.barrier_outcome(
                decision_at_ns=decision_at_ns,
                label_end_ns=label_end_ns,
                side=side,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                stop_distance_bps=stop_distance_bps,
                take_profit_distance_bps=take_profit_distance_bps,
            )
            hit_at_ns = (
                None
                if outcome.hit_index is None
                else int(seconds.available_at_ns[outcome.hit_index])
            )
            rows.append(
                {
                    "research_schema_version": RESEARCH_SCHEMA_VERSION,
                    "decision_id": feature["decision_id"],
                    "source_dataset_id": source_dataset_id,
                    "symbol": feature["symbol"],
                    "decision_at_ns": decision_at_ns,
                    "decision_utc_date": feature["decision_utc_date"],
                    "side": side,
                    "horizon_minutes": horizon_minutes,
                    "label_end_ns": label_end_ns,
                    "entry_reference_price": entry_price,
                    "stop_distance_bps": stop_distance_bps,
                    "take_profit_distance_bps": take_profit_distance_bps,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "outcome": outcome.outcome,
                    "hit_at_ns": hit_at_ns,
                    "hit_event_ts_ms": None,
                    "hit_sequence": None,
                    "hit_trade_price": None,
                    "time_to_hit_ms": (
                        None
                        if hit_at_ns is None
                        else (hit_at_ns - decision_at_ns) / NS_PER_MILLISECOND
                    ),
                    "timeout_price": outcome.timeout_price,
                    "outcome_return_bps": outcome.outcome_return_bps,
                    "future_trade_count": outcome.future_trade_count,
                    "resolution": outcome.resolution,
                }
            )
            quality[
                f"label_{horizon_minutes}m_{side.lower()}_{outcome.outcome.lower()}"
            ] += 1
    return rows


def _load_symbol_series(
    days_by_date: dict[str, _HistoryDay],
    selected_dates: tuple[str, ...],
    *,
    day_index: int,
    symbol: str,
) -> tuple[_MinuteSeries, _SecondSeries]:
    nearby = [day_index]
    if day_index > 0:
        nearby.insert(0, day_index - 1)
    if day_index + 1 < len(selected_dates):
        nearby.append(day_index + 1)
    minute_indices = [index for index in nearby if index <= day_index]
    minute_files = [
        days_by_date[selected_dates[index]].files[(symbol, "trade_bar_1m")]
        for index in minute_indices
    ]
    second_files = [
        days_by_date[selected_dates[index]].files[(symbol, "trade_bar_1s")]
        for index in nearby
    ]
    return (
        _MinuteSeries(_read_bars(minute_files, MINUTE_COLUMNS)),
        _SecondSeries(_read_bars(second_files, SECOND_COLUMNS)),
    )


def _build_symbol_day(
    *,
    source_dataset_id: str,
    symbol: str,
    partition_date: str,
    minute: _MinuteSeries,
    seconds: _SecondSeries,
    parameters: PriceResearchParameters,
    coverage_end_ns: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    day_start_ns = _utc_day_start_ns(partition_date)
    interval_ns = parameters.decision_interval_seconds * NS_PER_SECOND
    first_decision = day_start_ns + parameters.decision_offset_seconds * NS_PER_SECOND
    last_decision = day_start_ns + NS_PER_DAY - interval_ns + (
        parameters.decision_offset_seconds * NS_PER_SECOND
    )
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    quality: Counter[str] = Counter()
    decision_at_ns = first_decision
    while decision_at_ns <= last_decision:
        quality["candidate_decisions"] += 1
        minute_features, label_volatility, reason = minute.features_at(
            decision_at_ns,
            history_minutes=parameters.kline_history_minutes,
            volatility_lookback_minutes=parameters.volatility_lookback_minutes,
        )
        if minute_features is None or label_volatility is None:
            quality[f"skipped_{reason}"] += 1
            decision_at_ns += interval_ns
            continue
        second_features, reason = seconds.features_at(
            decision_at_ns, maximum_trade_age_ms=parameters.maximum_trade_age_ms
        )
        if second_features is None:
            quality[f"skipped_{reason}"] += 1
            decision_at_ns += interval_ns
            continue
        feature: dict[str, object] = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "decision_id": _decision_id(source_dataset_id, symbol, decision_at_ns),
            "source_dataset_id": source_dataset_id,
            "symbol": symbol,
            "decision_at_ns": decision_at_ns,
            "decision_at_ms": decision_at_ns // NS_PER_MILLISECOND,
            "decision_utc_date": _utc_date_from_ns(decision_at_ns),
        }
        feature.update(minute_features)
        feature.update(second_features)
        feature.update(_time_features(decision_at_ns))
        features.append(feature)
        labels.extend(
            _label_rows(
                source_dataset_id=source_dataset_id,
                feature=feature,
                label_volatility=label_volatility,
                seconds=seconds,
                parameters=parameters,
                coverage_end_ns=coverage_end_ns,
                quality=quality,
            )
        )
        quality["features_emitted"] += 1
        decision_at_ns += interval_ns
    return features, labels, quality


def _add_btc_context(
    rows: list[dict[str, object]],
    *,
    symbol: str,
    btc_by_decision: dict[int, dict[str, object]],
) -> None:
    for row in rows:
        decision_at_ns = int(cast(int, row["decision_at_ns"]))
        btc = btc_by_decision.get(decision_at_ns)
        if btc is None:
            for key in (
                "btc_return_5m_fraction",
                "btc_return_15m_fraction",
                "btc_return_60m_fraction",
                "btc_realized_volatility_15m_fraction",
                "btc_trade_imbalance_60s",
                "relative_return_5m_fraction",
                "relative_return_15m_fraction",
                "relative_return_60m_fraction",
            ):
                row[key] = None
            continue
        row["btc_return_5m_fraction"] = btc["return_5m_fraction"]
        row["btc_return_15m_fraction"] = btc["return_15m_fraction"]
        row["btc_return_60m_fraction"] = btc["return_60m_fraction"]
        row["btc_realized_volatility_15m_fraction"] = btc[
            "realized_volatility_15m_fraction"
        ]
        row["btc_trade_imbalance_60s"] = btc["trade_imbalance_1m"]
        for window in (5, 15, 60):
            row[f"relative_return_{window}m_fraction"] = (
                0.0
                if symbol == BTC_SYMBOL
                else _number(row, f"return_{window}m_fraction")
                - _number(btc, f"return_{window}m_fraction")
            )


def _write_partition(
    root: Path,
    *,
    table_name: str,
    symbol: str,
    partition_date: str,
    rows: list[dict[str, object]],
) -> ResearchFile | None:
    if not rows:
        return None
    schema = PRICE_FEATURE_SCHEMA if table_name == "features" else PRICE_LABEL_SCHEMA
    if table_name == "features":
        rows.sort(key=lambda row: int(cast(int, row["decision_at_ns"])))
    else:
        rows.sort(
            key=lambda row: (
                int(cast(int, row["decision_at_ns"])),
                int(cast(int, row["horizon_minutes"])),
                cast(str, row["side"]),
            )
        )
    relative = (
        Path(f"table={table_name}")
        / f"symbol={symbol}"
        / f"date={partition_date}"
        / "part-00000.parquet"
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        version=PARQUET_FORMAT_VERSION,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        use_dictionary=True,
        write_statistics=True,
        write_page_checksum=True,
    )
    return ResearchFile(
        path=relative.as_posix(),
        table=table_name,
        symbol=symbol,
        date=partition_date,
        rows=len(rows),
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _files_fingerprint(files: Sequence[ResearchFile]) -> str:
    return _sha256_json(
        [item.to_dict() for item in sorted(files, key=lambda item: item.path)]
    )


def _write_json_atomic(path: Path, value: object) -> None:
    partial = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with partial.open("xb") as target:
            target.write(_render_json(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _result(path: Path, manifest: dict[str, Any], *, reused: bool) -> ResearchBuildResult:
    source = _object(manifest.get("source"), "research manifest.source")
    parameters = _object(manifest.get("parameters"), "research manifest.parameters")
    rows = _object(manifest.get("output_rows"), "research manifest.output_rows")
    return ResearchBuildResult(
        research_dataset_id=_string(manifest.get("research_dataset_id"), "research_dataset_id"),
        dataset_path=path,
        manifest_path=path / "manifest.json",
        source_dataset_id=_string(source.get("dataset_id"), "source.dataset_id"),
        source_output_fingerprint=_sha256(
            source.get("output_fingerprint"), "source.output_fingerprint"
        ),
        parameter_fingerprint=_sha256(
            parameters.get("fingerprint"), "parameters.fingerprint"
        ),
        input_fingerprint=_sha256(manifest.get("input_fingerprint"), "input_fingerprint"),
        output_fingerprint=_sha256(
            manifest.get("output_fingerprint"), "output_fingerprint"
        ),
        feature_rows=_nonnegative_int(rows.get("features"), "output_rows.features"),
        label_rows=_nonnegative_int(rows.get("labels"), "output_rows.labels"),
        output_files=_nonnegative_int(
            manifest.get("output_file_count"), "output_file_count"
        ),
        reused=reused,
    )


def _validate_existing(
    path: Path,
    *,
    research_dataset_id: str,
    input_fingerprint: str,
    source_manifest_sha256: str,
) -> ResearchBuildResult:
    try:
        from tradingbot.research.evaluation_contracts import EvaluationError
        from tradingbot.research.evaluation_dataset import validate_research_dataset

        validated = validate_research_dataset(path)
    except (EvaluationError, OSError, pa.ArrowInvalid) as exc:
        raise ResearchBuildError(f"existing price research dataset is corrupted: {exc}") from exc
    if validated.research_dataset_id != research_dataset_id:
        raise ResearchBuildError("existing price research dataset ID is inconsistent")
    if validated.input_fingerprint != input_fingerprint:
        raise ResearchBuildError("existing price research dataset uses another input")
    if validated.research_profile != PRICE_RESEARCH_PROFILE:
        raise ResearchBuildError("existing research dataset uses another profile")
    manifest = validated.manifest
    source = _object(manifest.get("source"), "research manifest.source")
    if source.get("manifest_sha256") != source_manifest_sha256:
        raise ResearchBuildError("existing source selection manifest differs")
    return _result(path, manifest, reused=True)


def _safe_roots(source_root: Path, output_root: Path) -> None:
    if (
        source_root == output_root
        or source_root.is_relative_to(output_root)
        or output_root.is_relative_to(source_root)
    ):
        raise ResearchBuildError("history and research output roots must not overlap")


def build_price_research_dataset(
    history_catalog: str | Path,
    output_root: str | Path,
    *,
    start_date: str,
    end_date: str,
    parameters: PriceResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ResearchBuildResult:
    """Build deterministic causal features and one-second market labels."""

    selected_parameters = PriceResearchParameters() if parameters is None else parameters
    selected_parameters.validate()
    if minimum_free_bytes < 0:
        raise ResearchBuildError("minimum_free_bytes must be non-negative")
    selection = _load_history_selection(
        history_catalog, start_date=start_date, end_date=end_date
    )
    destination_root = Path(output_root).expanduser().resolve()
    _safe_roots(selection.history_root, destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    required_free = minimum_free_bytes + selection.total_bytes
    if shutil.disk_usage(destination_root).free < required_free:
        raise ResearchBuildError(
            f"insufficient disk space for price research build; {required_free} bytes required"
        )
    parameter_payload = selected_parameters.to_dict()
    parameter_fingerprint = _sha256_json(parameter_payload)
    input_payload = {
        "research_schema_version": RESEARCH_SCHEMA_VERSION,
        "research_profile": PRICE_RESEARCH_PROFILE,
        "package_version": __version__,
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "source_dataset_id": selection.dataset_id,
        "source_manifest_sha256": selection.manifest_sha256,
        "source_output_fingerprint": selection.output_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
    }
    input_fingerprint = _sha256_json(input_payload)
    research_dataset_id = f"research-price-v{RESEARCH_SCHEMA_VERSION}-{input_fingerprint[:16]}"
    final_path = destination_root / research_dataset_id
    if final_path.exists():
        return _validate_existing(
            final_path,
            research_dataset_id=research_dataset_id,
            input_fingerprint=input_fingerprint,
            source_manifest_sha256=selection.manifest_sha256,
        )

    staging = destination_root / f".{research_dataset_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "source-manifest.json").write_bytes(selection.manifest_bytes)
        selected_dates = tuple(day.partition_date for day in selection.days)
        days_by_date = {day.partition_date: day for day in selection.days}
        coverage_end_ns = _utc_day_start_ns(selection.end_date) + NS_PER_DAY
        files: list[ResearchFile] = []
        total_rows: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        horizons: Counter[str] = Counter()
        quality_by_symbol: dict[str, Counter[str]] = {
            symbol: Counter() for symbol in selection.symbols
        }
        build_symbols = (BTC_SYMBOL,) + tuple(
            symbol for symbol in selection.symbols if symbol != BTC_SYMBOL
        )
        for day_index, partition in enumerate(selected_dates):
            LOGGER.info("Building price research partition %s", partition)
            btc_features: list[dict[str, object]] | None = None
            btc_by_decision: dict[int, dict[str, object]] = {}
            for symbol in build_symbols:
                minute, seconds = _load_symbol_series(
                    days_by_date,
                    selected_dates,
                    day_index=day_index,
                    symbol=symbol,
                )
                features, labels, quality = _build_symbol_day(
                    source_dataset_id=selection.dataset_id,
                    symbol=symbol,
                    partition_date=partition,
                    minute=minute,
                    seconds=seconds,
                    parameters=selected_parameters,
                    coverage_end_ns=coverage_end_ns,
                )
                if symbol == BTC_SYMBOL:
                    btc_features = features
                    btc_by_decision = {
                        int(cast(int, row["decision_at_ns"])): row for row in features
                    }
                elif btc_features is None:
                    raise ResearchBuildError("BTCUSDT must precede altcoins in history symbols")
                _add_btc_context(
                    features,
                    symbol=symbol,
                    btc_by_decision=btc_by_decision,
                )
                for table_name, rows in (("features", features), ("labels", labels)):
                    output = _write_partition(
                        staging,
                        table_name=table_name,
                        symbol=symbol,
                        partition_date=partition,
                        rows=rows,
                    )
                    if output is not None:
                        files.append(output)
                        total_rows[table_name] += output.rows
                for row in labels:
                    outcomes[cast(str, row["outcome"])] += 1
                    horizons[f"{int(cast(int, row['horizon_minutes']))}m"] += 1
                quality_by_symbol[symbol].update(quality)
                LOGGER.info(
                    "%s %s ready: %d features, %d labels",
                    partition,
                    symbol,
                    len(features),
                    len(labels),
                )
        if total_rows["features"] <= 0 or total_rows["labels"] <= 0:
            raise ResearchBuildError("price research build produced no modelable rows")
        output_fingerprint = _files_fingerprint(files)
        manifest: dict[str, object] = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "research_profile": PRICE_RESEARCH_PROFILE,
            "research_dataset_id": research_dataset_id,
            "input_fingerprint": input_fingerprint,
            "builder": {
                "package_version": __version__,
                "pyarrow_version": pa.__version__,
                "numpy_version": np.__version__,
                "parquet_format_version": PARQUET_FORMAT_VERSION,
                "compression": PARQUET_COMPRESSION,
                "compression_level": PARQUET_COMPRESSION_LEVEL,
            },
            "source": {
                "dataset_id": selection.dataset_id,
                "dataset_profile": HISTORY_PROFILE,
                "dataset_path": selection.history_root.as_posix(),
                "manifest_copy": "source-manifest.json",
                "manifest_sha256": selection.manifest_sha256,
                "output_fingerprint": selection.output_fingerprint,
                "symbols": list(selection.symbols),
                "start_date": selection.start_date,
                "end_date": selection.end_date,
                "days": len(selection.days),
                "bytes": selection.total_bytes,
            },
            "parameters": {**parameter_payload, "fingerprint": parameter_fingerprint},
            "causality": {
                "feature_rule": "available_at_ns <= decision_at_ns",
                "decision_grid": "UTC epoch aligned",
                "label_rule": (
                    "decision_at_ns < trade_bar_1s.available_at_ns <= label_end_ns"
                ),
                "barrier_resolution": "one_second_bars",
                "same_bar_double_cross": "AMBIGUOUS",
                "execution_labels_included": False,
                "maker_fill_claimed": False,
            },
            "source_capabilities": selection.manifest_payload["source_capabilities"],
            "unavailable_features": [
                "orderbook",
                "spread",
                "ticker",
                "mark_index_basis",
                "funding",
                "open_interest",
                "maker_queue_position",
                "partial_fill",
            ],
            "schemas": {
                "features": _schema_manifest(PRICE_FEATURE_SCHEMA),
                "labels": _schema_manifest(PRICE_LABEL_SCHEMA),
            },
            "quality_by_symbol": {
                symbol: dict(sorted(values.items()))
                for symbol, values in quality_by_symbol.items()
            },
            "label_outcomes": dict(sorted(outcomes.items())),
            "labels_by_horizon": dict(sorted(horizons.items())),
            "output_rows": {
                "features": total_rows["features"],
                "labels": total_rows["labels"],
            },
            "output_file_count": len(files),
            "output_fingerprint": output_fingerprint,
            "files": [
                item.to_dict() for item in sorted(files, key=lambda item: item.path)
            ],
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        os.replace(staging, final_path)
        LOGGER.info(
            "Price research dataset ready at %s (%d features, %d labels)",
            final_path,
            total_rows["features"],
            total_rows["labels"],
        )
        return _result(final_path, manifest, reused=False)
    except Exception:
        if staging.is_dir() and staging.parent == destination_root:
            shutil.rmtree(staging, ignore_errors=True)
        raise
