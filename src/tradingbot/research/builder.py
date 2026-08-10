"""Build causal features and market labels from a canonical Parquet dataset.

Feature rows are evaluated on a fixed UTC grid.  Every source record used by a
feature satisfies ``received_at_ns <= decision_at_ns``.  Labels live in a
separate table and deliberately look forward through public trades; they are
market-movement labels, not claims that a maker order would have filled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tradingbot import __version__
from tradingbot.data.canonical import (
    DatasetBuildError,
    validate_canonical_dataset,
)
from tradingbot.research.contracts import (
    BOOK_DEPTH_LEVELS,
    FEATURE_SCHEMA,
    KLINE_RETURN_WINDOWS_MINUTES,
    KLINE_VOLATILITY_WINDOWS_MINUTES,
    LABEL_SCHEMA,
    MICROSTRUCTURE_RESEARCH_PROFILE,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_FORMAT_VERSION,
    RESEARCH_SCHEMA_VERSION,
    TRADE_WINDOWS_SECONDS,
    ResearchBuildError,
    ResearchBuildResult,
    ResearchFile,
    ResearchParameters,
)

LOGGER = logging.getLogger(__name__)

NS_PER_SECOND: Final = 1_000_000_000
NS_PER_MILLISECOND: Final = 1_000_000
MS_PER_MINUTE: Final = 60_000
MISSING_SEQUENCE: Final = np.iinfo(np.int64).max


@dataclass(frozen=True, slots=True)
class _CanonicalFile:
    path: str
    kind: str
    symbol: str
    date: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _CanonicalSource:
    dataset_id: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    output_fingerprint: str
    symbols: tuple[str, ...]
    files: tuple[_CanonicalFile, ...]
    total_bytes: int

    def paths(self, kind: str, symbol: str) -> tuple[Path, ...]:
        selected = tuple(
            self.root.joinpath(*PurePosixPath(item.path).parts)
            for item in self.files
            if item.kind == kind and item.symbol == symbol
        )
        if not selected:
            raise ResearchBuildError(
                f"canonical dataset has no {kind} files for {symbol}"
            )
        return selected


@dataclass(frozen=True, slots=True)
class _BarrierResult:
    outcome: str
    hit_index: int | None
    timeout_price: float | None
    outcome_return_bps: float | None
    resolution: str
    future_trade_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ResearchBuildError(f"{label} must be a SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ResearchBuildError(
            f"{label} must be a SHA-256 hexadecimal digest"
        ) from exc
    return value.lower()


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchBuildError(f"{label} must be a non-empty string")
    return value


def _required_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchBuildError(f"{label} must be a non-negative integer")
    return value


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResearchBuildError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    text = _required_string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ResearchBuildError(f"{label} is not a safe relative path")
    return path


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchBuildError(f"{label} is unreadable: {path}") from exc
    return _json_object(parsed, label)


def _load_canonical_source(dataset_path: str | Path) -> _CanonicalSource:
    try:
        validated = validate_canonical_dataset(dataset_path)
    except DatasetBuildError as exc:
        raise ResearchBuildError(f"canonical dataset validation failed: {exc}") from exc

    root = validated.dataset_path
    manifest_path = validated.manifest_path
    manifest = _load_json(manifest_path, "canonical manifest")
    raw_source = _json_object(manifest.get("source"), "canonical manifest.source")
    raw_symbols = raw_source.get("expected_symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ResearchBuildError(
            "canonical manifest.source.expected_symbols must be a non-empty array"
        )
    symbols = tuple(
        _required_string(value, "canonical source symbol") for value in raw_symbols
    )
    if len(set(symbols)) != len(symbols):
        raise ResearchBuildError("canonical source symbols contain duplicates")
    if "BTCUSDT" not in symbols:
        raise ResearchBuildError("research features require BTCUSDT as the market anchor")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ResearchBuildError("canonical manifest.files must be a non-empty array")
    files: list[_CanonicalFile] = []
    total_bytes = 0
    for index, raw in enumerate(raw_files):
        item = _json_object(raw, f"canonical manifest.files[{index}]")
        relative = _safe_relative_path(
            item.get("path"), f"canonical manifest.files[{index}].path"
        )
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root):
            raise ResearchBuildError(f"canonical file escapes its dataset: {relative}")
        size = _required_nonnegative_int(
            item.get("bytes"), f"canonical manifest.files[{index}].bytes"
        )
        total_bytes += size
        files.append(
            _CanonicalFile(
                path=relative.as_posix(),
                kind=_required_string(
                    item.get("kind"), f"canonical manifest.files[{index}].kind"
                ),
                symbol=_required_string(
                    item.get("symbol"), f"canonical manifest.files[{index}].symbol"
                ),
                date=_required_string(
                    item.get("date"), f"canonical manifest.files[{index}].date"
                ),
                rows=_required_nonnegative_int(
                    item.get("rows"), f"canonical manifest.files[{index}].rows"
                ),
                bytes=size,
                sha256=_valid_sha256(
                    item.get("sha256"), f"canonical manifest.files[{index}].sha256"
                ),
            )
        )

    required_kinds = {"orderbook", "ticker", "trades", "kline_1"}
    for symbol in symbols:
        actual = {item.kind for item in files if item.symbol == symbol}
        missing = required_kinds - actual
        if missing:
            raise ResearchBuildError(
                f"canonical dataset is missing required kinds for {symbol}: "
                f"{', '.join(sorted(missing))}"
            )

    return _CanonicalSource(
        dataset_id=validated.dataset_id,
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        output_fingerprint=validated.output_fingerprint,
        symbols=tuple(sorted(symbols)),
        files=tuple(sorted(files, key=lambda item: item.path)),
        total_bytes=total_bytes,
    )


def _load_archive_catalog_source(catalog_path: str | Path) -> _CanonicalSource:
    from tradingbot.data.archive import ArchiveError, load_archive_catalog

    try:
        catalog = load_archive_catalog(
            catalog_path,
            verify_canonical_files=False,
        )
    except ArchiveError as exc:
        raise ResearchBuildError(f"archive catalog validation failed: {exc}") from exc
    if not catalog.entries:
        raise ResearchBuildError("archive catalog contains no committed days")
    parsed_dates = tuple(date.fromisoformat(item.partition_date) for item in catalog.entries)
    for previous, current in zip(parsed_dates, parsed_dates[1:], strict=False):
        if current != previous + timedelta(days=1):
            raise ResearchBuildError(
                "research archive catalog must contain consecutive UTC days"
            )

    archive_root = catalog.path.parent.resolve()
    daily_sources = tuple(
        _load_canonical_source(day.canonical_dataset_path) for day in catalog.entries
    )
    symbols = daily_sources[0].symbols
    combined_files: list[_CanonicalFile] = []
    for day, source in zip(catalog.entries, daily_sources, strict=True):
        if source.symbols != symbols:
            raise ResearchBuildError(
                f"archive day {day.partition_date} uses another symbol universe"
            )
        try:
            dataset_relative = source.root.relative_to(archive_root)
        except ValueError as exc:
            raise ResearchBuildError(
                f"archive day {day.partition_date} escapes the archive root"
            ) from exc
        for item in source.files:
            combined = PurePosixPath(dataset_relative.as_posix()) / item.path
            combined_files.append(
                _CanonicalFile(
                    path=combined.as_posix(),
                    kind=item.kind,
                    symbol=item.symbol,
                    date=item.date,
                    rows=item.rows,
                    bytes=item.bytes,
                    sha256=item.sha256,
                )
            )
    if len({item.path for item in combined_files}) != len(combined_files):
        raise ResearchBuildError("archive catalog resolves to duplicate Parquet paths")
    return _CanonicalSource(
        dataset_id=f"archive-catalog-v1-{catalog.fingerprint[:16]}",
        root=archive_root,
        manifest_path=catalog.path,
        manifest_sha256=_sha256_file(catalog.path),
        output_fingerprint=catalog.fingerprint,
        symbols=symbols,
        files=tuple(sorted(combined_files, key=lambda item: item.path)),
        total_bytes=sum(item.bytes for item in combined_files),
    )


def _read_parquet_files(paths: tuple[Path, ...], columns: list[str]) -> pa.Table:
    tables = [pq.ParquetFile(path).read(columns=columns) for path in paths]
    if not tables:
        raise ResearchBuildError("no canonical Parquet files were selected")
    try:
        return pa.concat_tables(tables)
    except pa.ArrowInvalid as exc:
        raise ResearchBuildError("canonical Parquet schemas are inconsistent") from exc


def _int64_array(
    table: pa.Table,
    name: str,
    *,
    null_value: int | None = None,
) -> NDArray[np.int64]:
    column = table.column(name).combine_chunks()
    if column.null_count:
        if null_value is None:
            raise ResearchBuildError(f"{name} unexpectedly contains null values")
        column = pc.fill_null(column, pa.scalar(null_value, type=pa.int64()))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _float64_array(
    table: pa.Table,
    name: str,
    *,
    null_value: float = math.nan,
) -> NDArray[np.float64]:
    column = table.column(name).combine_chunks()
    if column.null_count:
        column = pc.fill_null(column, pa.scalar(null_value, type=pa.float64()))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _scalar_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(cast(float | int | str, value))
    return number if math.isfinite(number) else None


class _OrderBookSeries:
    def __init__(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ResearchBuildError("orderbook table is empty")
        received = _int64_array(table, "received_at_ns")
        order = np.argsort(received, kind="stable")
        self.table = table
        self.received_at_ns = received[order]
        self.original_indices = order

    @property
    def first_received_at_ns(self) -> int:
        return int(self.received_at_ns[0])

    @property
    def last_received_at_ns(self) -> int:
        return int(self.received_at_ns[-1])

    def features_at(
        self, decision_at_ns: int, maximum_age_ms: int
    ) -> tuple[dict[str, object] | None, str | None]:
        position = int(
            np.searchsorted(self.received_at_ns, decision_at_ns, side="right")
        ) - 1
        if position < 0:
            return None, "missing_orderbook"
        received_at_ns = int(self.received_at_ns[position])
        age_ms = (decision_at_ns - received_at_ns) / NS_PER_MILLISECOND
        if age_ms > maximum_age_ms:
            return None, "stale_orderbook"
        index = int(self.original_indices[position])
        bid_prices = cast(list[float], self.table["bid_prices"][index].as_py())
        bid_sizes = cast(list[float], self.table["bid_sizes"][index].as_py())
        ask_prices = cast(list[float], self.table["ask_prices"][index].as_py())
        ask_sizes = cast(list[float], self.table["ask_sizes"][index].as_py())
        if (
            len(bid_prices) < max(BOOK_DEPTH_LEVELS)
            or len(ask_prices) < max(BOOK_DEPTH_LEVELS)
            or len(bid_prices) != len(bid_sizes)
            or len(ask_prices) != len(ask_sizes)
        ):
            return None, "short_orderbook"
        best_bid = float(bid_prices[0])
        best_ask = float(ask_prices[0])
        best_bid_size = float(bid_sizes[0])
        best_ask_size = float(ask_sizes[0])
        if (
            best_bid <= 0
            or best_ask <= best_bid
            or best_bid_size <= 0
            or best_ask_size <= 0
        ):
            return None, "invalid_orderbook"
        mid = (best_bid + best_ask) / 2
        top_size = best_bid_size + best_ask_size
        microprice = (
            best_ask * best_bid_size + best_bid * best_ask_size
        ) / top_size
        result: dict[str, object] = {
            "book_received_at_ns": received_at_ns,
            "book_age_ms": age_ms,
            "reference_mid_price": mid,
            "best_bid_price": best_bid,
            "best_ask_price": best_ask,
            "best_bid_size": best_bid_size,
            "best_ask_size": best_ask_size,
            "spread_bps": (best_ask - best_bid) / mid * 10_000,
            "microprice": microprice,
            "microprice_offset_bps": (microprice / mid - 1) * 10_000,
        }
        for level in BOOK_DEPTH_LEVELS:
            bid_depth = math.fsum(float(value) for value in bid_sizes[:level])
            ask_depth = math.fsum(float(value) for value in ask_sizes[:level])
            total_depth = bid_depth + ask_depth
            if total_depth <= 0:
                return None, "invalid_orderbook"
            bid_notional = math.fsum(
                float(price) * float(size)
                for price, size in zip(
                    bid_prices[:level], bid_sizes[:level], strict=True
                )
            )
            ask_notional = math.fsum(
                float(price) * float(size)
                for price, size in zip(
                    ask_prices[:level], ask_sizes[:level], strict=True
                )
            )
            result[f"bid_depth_{level}"] = bid_depth
            result[f"ask_depth_{level}"] = ask_depth
            result[f"book_imbalance_{level}"] = (
                bid_depth - ask_depth
            ) / total_depth
            result[f"depth_notional_{level}"] = bid_notional + ask_notional
        return result, None


class _TickerSeries:
    def __init__(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ResearchBuildError("ticker table is empty")
        received = _int64_array(table, "received_at_ns")
        order = np.argsort(received, kind="stable")
        self.received_at_ns = received[order]
        self.open_interest = _float64_array(table, "open_interest")[order]
        self.mark_price = _float64_array(table, "mark_price")[order]
        self.index_price = _float64_array(table, "index_price")[order]
        self.funding_rate = _float64_array(table, "funding_rate")[order]
        self.next_funding_time_ms = _int64_array(
            table, "next_funding_time_ms", null_value=-1
        )[order]

    @property
    def first_received_at_ns(self) -> int:
        return int(self.received_at_ns[0])

    @property
    def last_received_at_ns(self) -> int:
        return int(self.received_at_ns[-1])

    def _position(self, at_ns: int) -> int:
        return int(np.searchsorted(self.received_at_ns, at_ns, side="right")) - 1

    def _oi_change(self, current_position: int, at_ns: int, seconds: int) -> float | None:
        previous_position = self._position(at_ns - seconds * NS_PER_SECOND)
        if previous_position < 0:
            return None
        current = float(self.open_interest[current_position])
        previous = float(self.open_interest[previous_position])
        if not math.isfinite(current) or not math.isfinite(previous) or previous <= 0:
            return None
        return current / previous - 1

    def features_at(
        self, decision_at_ns: int, maximum_age_ms: int
    ) -> tuple[dict[str, object] | None, str | None]:
        position = self._position(decision_at_ns)
        if position < 0:
            return None, "missing_ticker"
        received_at_ns = int(self.received_at_ns[position])
        age_ms = (decision_at_ns - received_at_ns) / NS_PER_MILLISECOND
        if age_ms > maximum_age_ms:
            return None, "stale_ticker"
        mark_price = float(self.mark_price[position])
        index_price = float(self.index_price[position])
        open_interest = float(self.open_interest[position])
        funding_rate = float(self.funding_rate[position])
        next_funding_time_ms = int(self.next_funding_time_ms[position])
        mark = mark_price if math.isfinite(mark_price) and mark_price > 0 else None
        index = index_price if math.isfinite(index_price) and index_price > 0 else None
        basis = None if mark is None or index is None else (mark / index - 1) * 10_000
        funding = funding_rate if math.isfinite(funding_rate) else None
        oi = open_interest if math.isfinite(open_interest) and open_interest >= 0 else None
        minutes_to_funding = (
            None
            if next_funding_time_ms < 0
            else (
                next_funding_time_ms - decision_at_ns / NS_PER_MILLISECOND
            )
            / MS_PER_MINUTE
        )
        return {
            "ticker_received_at_ns": received_at_ns,
            "ticker_age_ms": age_ms,
            "mark_price": mark,
            "index_price": index,
            "mark_index_basis_bps": basis,
            "open_interest": oi,
            "open_interest_change_5m_fraction": self._oi_change(
                position, decision_at_ns, 300
            ),
            "open_interest_change_15m_fraction": self._oi_change(
                position, decision_at_ns, 900
            ),
            "funding_rate": funding,
            "minutes_to_funding": minutes_to_funding,
        }, None


class _KlineSeries:
    def __init__(self, table: pa.Table) -> None:
        rows = cast(list[dict[str, object]], table.to_pylist())
        if not rows:
            raise ResearchBuildError("kline_1 table is empty")
        self.rows_by_start: dict[int, dict[str, object]] = {}
        for row in rows:
            if row.get("interval") != "1":
                raise ResearchBuildError("kline_1 partition contains another interval")
            start_ms = int(cast(int, row["start_ms"]))
            if start_ms in self.rows_by_start:
                raise ResearchBuildError(
                    f"canonical kline table contains duplicate start_ms {start_ms}"
                )
            self.rows_by_start[start_ms] = row

    def features_at(
        self, decision_at_ns: int, history_minutes: int
    ) -> tuple[dict[str, object] | None, str | None]:
        decision_at_ms = decision_at_ns // NS_PER_MILLISECOND
        target_start_ms = (
            decision_at_ms // MS_PER_MINUTE * MS_PER_MINUTE - MS_PER_MINUTE
        )
        starts = [
            target_start_ms - offset * MS_PER_MINUTE
            for offset in range(history_minutes, -1, -1)
        ]
        rows: list[dict[str, object]] = []
        for start_ms in starts:
            row = self.rows_by_start.get(start_ms)
            if row is None:
                return None, "missing_kline_history"
            received_at_ns = int(cast(int, row["received_at_ns"]))
            if received_at_ns > decision_at_ns:
                return None, "kline_not_yet_received"
            if int(cast(int, row["end_ms"])) >= decision_at_ms:
                return None, "unclosed_kline"
            rows.append(row)

        closes = np.asarray(
            [float(cast(float | int | str, row["close"])) for row in rows],
            dtype=np.float64,
        )
        highs = np.asarray(
            [float(cast(float | int | str, row["high"])) for row in rows],
            dtype=np.float64,
        )
        lows = np.asarray(
            [float(cast(float | int | str, row["low"])) for row in rows],
            dtype=np.float64,
        )
        volumes = np.asarray(
            [float(cast(float | int | str, row["volume"])) for row in rows],
            dtype=np.float64,
        )
        if (
            np.any(~np.isfinite(closes))
            or np.any(closes <= 0)
            or np.any(~np.isfinite(highs))
            or np.any(~np.isfinite(lows))
            or np.any(~np.isfinite(volumes))
            or np.any(volumes < 0)
        ):
            return None, "invalid_kline"
        log_returns = np.diff(np.log(closes))
        result: dict[str, object] = {
            "latest_kline_received_at_ns": max(
                int(cast(int, row["received_at_ns"])) for row in rows
            ),
            "close_price": float(closes[-1]),
        }
        for window in KLINE_RETURN_WINDOWS_MINUTES:
            result[f"return_{window}m_fraction"] = (
                float(closes[-1] / closes[-1 - window] - 1)
            )
        for window in KLINE_VOLATILITY_WINDOWS_MINUTES:
            result[f"realized_volatility_{window}m_fraction"] = float(
                math.sqrt(float(np.dot(log_returns[-window:], log_returns[-window:])))
            )

        true_ranges: list[float] = []
        for index in range(len(rows) - 14, len(rows)):
            previous_close = float(closes[index - 1])
            true_ranges.append(
                max(
                    float(highs[index] - lows[index]),
                    abs(float(highs[index] - previous_close)),
                    abs(float(lows[index] - previous_close)),
                )
            )
        result["atr_14_bps"] = (
            math.fsum(true_ranges) / len(true_ranges) / float(closes[-1]) * 10_000
        )
        result["range_1m_bps"] = (
            float(highs[-1] - lows[-1]) / float(closes[-1]) * 10_000
        )
        mean_volume_60 = float(np.mean(volumes[-60:]))
        result["volume_ratio_5m_to_60m"] = (
            0.0
            if mean_volume_60 == 0
            else float(np.mean(volumes[-5:])) / mean_volume_60
        )
        return result, None


class _TradeSeries:
    def __init__(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ResearchBuildError("trades table is empty")
        received = _int64_array(table, "received_at_ns")
        event_ts_ms = _int64_array(table, "event_ts_ms")
        sequence = _int64_array(
            table, "sequence", null_value=int(MISSING_SEQUENCE)
        )
        price = _float64_array(table, "price")
        size = _float64_array(table, "size")
        side = table.column("side").combine_chunks()
        is_buy = np.asarray(
            pc.equal(side, pa.scalar("Buy")).to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        )
        is_sell = np.asarray(
            pc.equal(side, pa.scalar("Sell")).to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        )
        if np.any(~(is_buy | is_sell)):
            raise ResearchBuildError("trades contain a side other than Buy or Sell")
        if (
            np.any(~np.isfinite(price))
            or np.any(price <= 0)
            or np.any(~np.isfinite(size))
            or np.any(size <= 0)
        ):
            raise ResearchBuildError("trades contain invalid price or size")

        order = np.lexsort((sequence, event_ts_ms, received))
        self.received_at_ns = received[order]
        self.event_ts_ms = event_ts_ms[order]
        self.sequence = sequence[order]
        self.price = price[order]
        self.size = size[order]
        self.is_buy = is_buy[order]

        notional = self.price * self.size
        signed_size = np.where(self.is_buy, self.size, -self.size)
        self.prefix_size = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.size, dtype=np.float64))
        )
        self.prefix_notional = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(notional, dtype=np.float64))
        )
        self.prefix_signed_size = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(signed_size, dtype=np.float64),
            )
        )

        seconds = self.received_at_ns // NS_PER_SECOND
        starts = np.flatnonzero(
            np.concatenate(
                (np.ones(1, dtype=np.bool_), seconds[1:] != seconds[:-1])
            )
        ).astype(np.int64)
        ends = np.concatenate(
            (starts[1:], np.asarray([len(seconds)], dtype=np.int64))
        )
        self.bucket_seconds = seconds[starts]
        self.bucket_starts = starts
        self.bucket_ends = ends
        self.bucket_high = np.maximum.reduceat(self.price, starts)
        self.bucket_low = np.minimum.reduceat(self.price, starts)

    @property
    def first_received_at_ns(self) -> int:
        return int(self.received_at_ns[0])

    @property
    def last_received_at_ns(self) -> int:
        return int(self.received_at_ns[-1])

    def features_at(self, decision_at_ns: int) -> dict[str, object]:
        end = int(np.searchsorted(self.received_at_ns, decision_at_ns, side="right"))
        latest_received = None if end == 0 else int(self.received_at_ns[end - 1])
        result: dict[str, object] = {
            "latest_trade_received_at_ns": latest_received,
            "trade_age_ms": (
                None
                if latest_received is None
                else (decision_at_ns - latest_received) / NS_PER_MILLISECOND
            ),
        }
        for seconds in TRADE_WINDOWS_SECONDS:
            start = int(
                np.searchsorted(
                    self.received_at_ns,
                    decision_at_ns - seconds * NS_PER_SECOND,
                    side="left",
                )
            )
            count = end - start
            base_volume = float(self.prefix_size[end] - self.prefix_size[start])
            notional = float(
                self.prefix_notional[end] - self.prefix_notional[start]
            )
            signed_volume = float(
                self.prefix_signed_size[end] - self.prefix_signed_size[start]
            )
            suffix = f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"
            result[f"trade_count_{suffix}"] = count
            result[f"trade_base_volume_{suffix}"] = base_volume
            result[f"trade_notional_{suffix}"] = notional
            result[f"trade_imbalance_{suffix}"] = (
                0.0 if base_volume == 0 else signed_volume / base_volume
            )
            result[f"trade_return_{suffix}_fraction"] = (
                0.0
                if count < 2
                else float(self.price[end - 1] / self.price[start] - 1)
            )
        return result

    def _ordering_key(self, index: int) -> tuple[int, int, int]:
        return (
            int(self.received_at_ns[index]),
            int(self.event_ts_ms[index]),
            int(self.sequence[index]),
        )

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
    ) -> _BarrierResult | None:
        if self.last_received_at_ns < label_end_ns:
            return None
        start = int(
            np.searchsorted(self.received_at_ns, decision_at_ns, side="right")
        )
        end = int(
            np.searchsorted(self.received_at_ns, label_end_ns, side="right")
        )
        future_count = end - start
        if future_count <= 0:
            return _BarrierResult(
                outcome="TIMEOUT",
                hit_index=None,
                timeout_price=None,
                outcome_return_bps=None,
                resolution="no_public_trade_in_complete_horizon",
                future_trade_count=0,
            )

        first_second = decision_at_ns // NS_PER_SECOND
        last_second = label_end_ns // NS_PER_SECOND
        bucket_start = int(
            np.searchsorted(self.bucket_seconds, first_second, side="left")
        )
        bucket_end = int(
            np.searchsorted(self.bucket_seconds, last_second, side="right")
        )
        highs = self.bucket_high[bucket_start:bucket_end]
        lows = self.bucket_low[bucket_start:bucket_end]
        if side == "LONG":
            candidate_mask = (highs >= take_profit_price) | (lows <= stop_price)
        elif side == "SHORT":
            candidate_mask = (lows <= take_profit_price) | (highs >= stop_price)
        else:
            raise ResearchBuildError(f"unsupported label side: {side}")

        candidate_offsets = np.flatnonzero(candidate_mask)
        for offset in candidate_offsets:
            bucket = bucket_start + int(offset)
            raw_start = max(start, int(self.bucket_starts[bucket]))
            raw_end = min(end, int(self.bucket_ends[bucket]))
            if raw_start >= raw_end:
                continue
            prices = self.price[raw_start:raw_end]
            if side == "LONG":
                tp_hits = np.flatnonzero(prices >= take_profit_price)
                sl_hits = np.flatnonzero(prices <= stop_price)
            else:
                tp_hits = np.flatnonzero(prices <= take_profit_price)
                sl_hits = np.flatnonzero(prices >= stop_price)
            tp_index = (
                None if len(tp_hits) == 0 else raw_start + int(tp_hits[0])
            )
            sl_index = (
                None if len(sl_hits) == 0 else raw_start + int(sl_hits[0])
            )
            if tp_index is None and sl_index is None:
                continue
            if tp_index is not None and sl_index is not None:
                tp_key = self._ordering_key(tp_index)
                sl_key = self._ordering_key(sl_index)
                if tp_key == sl_key:
                    return _BarrierResult(
                        outcome="AMBIGUOUS",
                        hit_index=None,
                        timeout_price=None,
                        outcome_return_bps=None,
                        resolution="equal_received_event_sequence_key",
                        future_trade_count=future_count,
                    )
                take_profit_first = tp_key < sl_key
            else:
                take_profit_first = tp_index is not None
            if take_profit_first:
                assert tp_index is not None
                return _BarrierResult(
                    outcome="TP_FIRST",
                    hit_index=tp_index,
                    timeout_price=None,
                    outcome_return_bps=take_profit_distance_bps,
                    resolution="public_trade_received_event_sequence",
                    future_trade_count=future_count,
                )
            assert sl_index is not None
            return _BarrierResult(
                outcome="SL_FIRST",
                hit_index=sl_index,
                timeout_price=None,
                outcome_return_bps=-stop_distance_bps,
                resolution="public_trade_received_event_sequence",
                future_trade_count=future_count,
            )

        timeout_price = float(self.price[end - 1])
        direction = 1.0 if side == "LONG" else -1.0
        timeout_return_bps = direction * (timeout_price / entry_price - 1) * 10_000
        return _BarrierResult(
            outcome="TIMEOUT",
            hit_index=None,
            timeout_price=timeout_price,
            outcome_return_bps=timeout_return_bps,
            resolution="complete_horizon_no_barrier",
            future_trade_count=future_count,
        )


def _load_symbol_series(
    source: _CanonicalSource, symbol: str
) -> tuple[_OrderBookSeries, _TickerSeries, _KlineSeries, _TradeSeries]:
    orderbook = _OrderBookSeries(
        _read_parquet_files(
            source.paths("orderbook", symbol),
            ["received_at_ns", "bid_prices", "bid_sizes", "ask_prices", "ask_sizes"],
        )
    )
    ticker = _TickerSeries(
        _read_parquet_files(
            source.paths("ticker", symbol),
            [
                "received_at_ns",
                "mark_price",
                "index_price",
                "open_interest",
                "funding_rate",
                "next_funding_time_ms",
            ],
        )
    )
    klines = _KlineSeries(
        _read_parquet_files(
            source.paths("kline_1", symbol),
            [
                "received_at_ns",
                "interval",
                "start_ms",
                "end_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
    )
    trades = _TradeSeries(
        _read_parquet_files(
            source.paths("trades", symbol),
            [
                "received_at_ns",
                "event_ts_ms",
                "sequence",
                "side",
                "price",
                "size",
            ],
        )
    )
    return orderbook, ticker, klines, trades


def _first_grid_at_or_after(
    value_ns: int, interval_ns: int, offset_ns: int
) -> int:
    return ((value_ns - offset_ns + interval_ns - 1) // interval_ns) * interval_ns + offset_ns


def _last_grid_at_or_before(
    value_ns: int, interval_ns: int, offset_ns: int
) -> int:
    return (value_ns - offset_ns) // interval_ns * interval_ns + offset_ns


def _decision_id(source_dataset_id: str, symbol: str, decision_at_ns: int) -> str:
    value = f"{source_dataset_id}|{symbol}|{decision_at_ns}".encode("ascii")
    return hashlib.sha256(value).hexdigest()[:32]


def _time_features(decision_at_ns: int) -> dict[str, float]:
    instant = datetime.fromtimestamp(decision_at_ns / NS_PER_SECOND, tz=UTC)
    hour = (
        instant.hour
        + instant.minute / 60
        + instant.second / 3_600
        + instant.microsecond / 3_600_000_000
    )
    weekday = instant.weekday() + hour / 24
    hour_angle = 2 * math.pi * hour / 24
    weekday_angle = 2 * math.pi * weekday / 7
    return {
        "utc_hour_sin": math.sin(hour_angle),
        "utc_hour_cos": math.cos(hour_angle),
        "utc_weekday_sin": math.sin(weekday_angle),
        "utc_weekday_cos": math.cos(weekday_angle),
    }


def _utc_date_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / NS_PER_SECOND, tz=UTC).date().isoformat()


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchBuildError(f"feature {key} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ResearchBuildError(f"feature {key} is not finite")
    return number


def _label_rows(
    *,
    source_dataset_id: str,
    symbol: str,
    feature: dict[str, object],
    trades: _TradeSeries,
    parameters: ResearchParameters,
    quality: Counter[str],
) -> list[dict[str, object]]:
    decision_at_ns = int(cast(int, feature["decision_at_ns"]))
    decision_utc_date = cast(str, feature["decision_utc_date"])
    entry_price = _number(feature, "reference_mid_price")
    realised_60 = _number(feature, "realized_volatility_60m_fraction")
    per_minute_volatility = realised_60 / math.sqrt(60)
    labels: list[dict[str, object]] = []
    for horizon_minutes in parameters.label_horizons_minutes:
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
        take_profit_distance_bps = (
            stop_distance_bps * parameters.take_profit_multiple
        )
        label_end_ns = (
            decision_at_ns + horizon_minutes * 60 * NS_PER_SECOND
        )
        for side in ("LONG", "SHORT"):
            if side == "LONG":
                stop_price = entry_price * (1 - stop_distance_bps / 10_000)
                take_profit_price = entry_price * (
                    1 + take_profit_distance_bps / 10_000
                )
            else:
                stop_price = entry_price * (1 + stop_distance_bps / 10_000)
                take_profit_price = entry_price * (
                    1 - take_profit_distance_bps / 10_000
                )
            outcome = trades.barrier_outcome(
                decision_at_ns=decision_at_ns,
                label_end_ns=label_end_ns,
                side=side,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                stop_distance_bps=stop_distance_bps,
                take_profit_distance_bps=take_profit_distance_bps,
            )
            if outcome is None:
                quality[f"labels_skipped_incomplete_{horizon_minutes}m"] += 1
                continue
            hit_index = outcome.hit_index
            hit_sequence: int | None = None
            if hit_index is not None:
                raw_sequence = int(trades.sequence[hit_index])
                if raw_sequence != int(MISSING_SEQUENCE):
                    hit_sequence = raw_sequence
            row: dict[str, object] = {
                "research_schema_version": RESEARCH_SCHEMA_VERSION,
                "decision_id": feature["decision_id"],
                "source_dataset_id": source_dataset_id,
                "symbol": symbol,
                "decision_at_ns": decision_at_ns,
                "decision_utc_date": decision_utc_date,
                "side": side,
                "horizon_minutes": horizon_minutes,
                "label_end_ns": label_end_ns,
                "entry_reference_price": entry_price,
                "stop_distance_bps": stop_distance_bps,
                "take_profit_distance_bps": take_profit_distance_bps,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "outcome": outcome.outcome,
                "hit_at_ns": (
                    None
                    if hit_index is None
                    else int(trades.received_at_ns[hit_index])
                ),
                "hit_event_ts_ms": (
                    None if hit_index is None else int(trades.event_ts_ms[hit_index])
                ),
                "hit_sequence": hit_sequence,
                "hit_trade_price": (
                    None if hit_index is None else float(trades.price[hit_index])
                ),
                "time_to_hit_ms": (
                    None
                    if hit_index is None
                    else (
                        int(trades.received_at_ns[hit_index]) - decision_at_ns
                    )
                    / NS_PER_MILLISECOND
                ),
                "timeout_price": outcome.timeout_price,
                "outcome_return_bps": outcome.outcome_return_bps,
                "future_trade_count": outcome.future_trade_count,
                "resolution": outcome.resolution,
            }
            labels.append(row)
            quality[
                f"label_{horizon_minutes}m_{side.lower()}_{outcome.outcome.lower()}"
            ] += 1
    return labels


def _build_symbol(
    source: _CanonicalSource,
    symbol: str,
    parameters: ResearchParameters,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    orderbook, ticker, klines, trades = _load_symbol_series(source, symbol)
    interval_ns = parameters.decision_interval_seconds * NS_PER_SECOND
    offset_ns = parameters.decision_offset_seconds * NS_PER_SECOND
    first_available = max(
        orderbook.first_received_at_ns,
        ticker.first_received_at_ns,
        trades.first_received_at_ns,
    )
    last_available = min(
        orderbook.last_received_at_ns,
        ticker.last_received_at_ns,
        trades.last_received_at_ns,
    )
    first_decision = _first_grid_at_or_after(
        first_available, interval_ns, offset_ns
    )
    last_decision = _last_grid_at_or_before(last_available, interval_ns, offset_ns)
    if first_decision > last_decision:
        raise ResearchBuildError(f"{symbol} has no complete decision interval")

    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    quality: Counter[str] = Counter()
    decision_at_ns = first_decision
    while decision_at_ns <= last_decision:
        quality["candidate_decisions"] += 1
        book_features, reason = orderbook.features_at(
            decision_at_ns, parameters.max_orderbook_age_ms
        )
        if book_features is None:
            quality[f"skipped_{reason}"] += 1
            decision_at_ns += interval_ns
            continue
        ticker_features, reason = ticker.features_at(
            decision_at_ns, parameters.max_ticker_age_ms
        )
        if ticker_features is None:
            quality[f"skipped_{reason}"] += 1
            decision_at_ns += interval_ns
            continue
        kline_features, reason = klines.features_at(
            decision_at_ns, parameters.kline_history_minutes
        )
        if kline_features is None:
            quality[f"skipped_{reason}"] += 1
            decision_at_ns += interval_ns
            continue

        decision_at_ms = decision_at_ns // NS_PER_MILLISECOND
        feature: dict[str, object] = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "decision_id": _decision_id(
                source.dataset_id, symbol, decision_at_ns
            ),
            "source_dataset_id": source.dataset_id,
            "symbol": symbol,
            "decision_at_ns": decision_at_ns,
            "decision_at_ms": decision_at_ms,
            "decision_utc_date": _utc_date_from_ns(decision_at_ns),
        }
        feature.update(book_features)
        feature.update(ticker_features)
        feature.update(kline_features)
        feature.update(trades.features_at(decision_at_ns))
        feature.update(_time_features(decision_at_ns))
        features.append(feature)
        labels.extend(
            _label_rows(
                source_dataset_id=source.dataset_id,
                symbol=symbol,
                feature=feature,
                trades=trades,
                parameters=parameters,
                quality=quality,
            )
        )
        quality["features_emitted"] += 1
        decision_at_ns += interval_ns
    return features, labels, quality


def _add_btc_context(
    features_by_symbol: dict[str, list[dict[str, object]]],
) -> None:
    btc_rows = {
        int(cast(int, row["decision_at_ns"])): row
        for row in features_by_symbol["BTCUSDT"]
    }
    for symbol, rows in features_by_symbol.items():
        for row in rows:
            decision_at_ns = int(cast(int, row["decision_at_ns"]))
            btc = btc_rows.get(decision_at_ns)
            if btc is None:
                for key in (
                    "btc_return_5m_fraction",
                    "btc_return_15m_fraction",
                    "btc_return_60m_fraction",
                    "btc_realized_volatility_15m_fraction",
                    "btc_trade_imbalance_60s",
                    "btc_spread_bps",
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
            row["btc_spread_bps"] = btc["spread_bps"]
            row["relative_return_5m_fraction"] = _number(
                row, "return_5m_fraction"
            ) - _number(btc, "return_5m_fraction")
            row["relative_return_15m_fraction"] = _number(
                row, "return_15m_fraction"
            ) - _number(btc, "return_15m_fraction")
            row["relative_return_60m_fraction"] = _number(
                row, "return_60m_fraction"
            ) - _number(btc, "return_60m_fraction")
            if symbol == "BTCUSDT":
                row["relative_return_5m_fraction"] = 0.0
                row["relative_return_15m_fraction"] = 0.0
                row["relative_return_60m_fraction"] = 0.0


def _research_files_fingerprint(files: list[ResearchFile]) -> str:
    return _sha256_json([item.to_dict() for item in sorted(files, key=lambda x: x.path)])


def _write_parquet_outputs(
    root: Path,
    features_by_symbol: dict[str, list[dict[str, object]]],
    labels_by_symbol: dict[str, list[dict[str, object]]],
) -> list[ResearchFile]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for symbol, rows in features_by_symbol.items():
        for row in rows:
            groups[("features", symbol, cast(str, row["decision_utc_date"]))].append(
                row
            )
    for symbol, rows in labels_by_symbol.items():
        for row in rows:
            groups[("labels", symbol, cast(str, row["decision_utc_date"]))].append(
                row
            )

    files: list[ResearchFile] = []
    for (table_name, symbol, partition_date), rows in sorted(groups.items()):
        schema = FEATURE_SCHEMA if table_name == "features" else LABEL_SCHEMA
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
        arrow_table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            arrow_table,
            path,
            version=PARQUET_FORMAT_VERSION,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="1.0",
            write_page_index=True,
            write_page_checksum=True,
            row_group_size=min(10_000, len(rows)),
        )
        files.append(
            ResearchFile(
                path=relative.as_posix(),
                table=table_name,
                symbol=symbol,
                date=partition_date,
                rows=len(rows),
                bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    return files


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(".json.partial")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as target:
        target.write(rendered)
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, path)


def _manifest_result(
    dataset_path: Path, manifest: dict[str, Any], *, reused: bool
) -> ResearchBuildResult:
    source = _json_object(manifest.get("source"), "research manifest.source")
    parameters = _json_object(
        manifest.get("parameters"), "research manifest.parameters"
    )
    rows = _json_object(manifest.get("output_rows"), "research manifest.output_rows")
    return ResearchBuildResult(
        research_dataset_id=_required_string(
            manifest.get("research_dataset_id"), "research_dataset_id"
        ),
        dataset_path=dataset_path,
        manifest_path=dataset_path / "manifest.json",
        source_dataset_id=_required_string(
            source.get("dataset_id"), "source.dataset_id"
        ),
        source_output_fingerprint=_valid_sha256(
            source.get("output_fingerprint"), "source.output_fingerprint"
        ),
        parameter_fingerprint=_valid_sha256(
            parameters.get("fingerprint"), "parameters.fingerprint"
        ),
        input_fingerprint=_valid_sha256(
            manifest.get("input_fingerprint"), "input_fingerprint"
        ),
        output_fingerprint=_valid_sha256(
            manifest.get("output_fingerprint"), "output_fingerprint"
        ),
        feature_rows=_required_nonnegative_int(rows.get("features"), "rows.features"),
        label_rows=_required_nonnegative_int(rows.get("labels"), "rows.labels"),
        output_files=_required_nonnegative_int(
            manifest.get("output_file_count"), "output_file_count"
        ),
        reused=reused,
    )


def _validate_existing_research_dataset(
    dataset_path: Path,
    *,
    research_dataset_id: str,
    input_fingerprint: str,
    source: _CanonicalSource,
) -> ResearchBuildResult:
    manifest = _load_json(dataset_path / "manifest.json", "research manifest")
    if manifest.get("research_schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ResearchBuildError("existing research dataset uses another schema version")
    if manifest.get("research_dataset_id") != research_dataset_id:
        raise ResearchBuildError(
            "existing research dataset ID does not match its directory"
        )
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise ResearchBuildError("existing research dataset was built from another input")
    source_manifest_copy = dataset_path / "source-manifest.json"
    raw_source = _json_object(manifest.get("source"), "research manifest.source")
    expected_source_sha = _valid_sha256(
        raw_source.get("manifest_sha256"), "source.manifest_sha256"
    )
    if (
        not source_manifest_copy.is_file()
        or _sha256_file(source_manifest_copy) != expected_source_sha
        or expected_source_sha != source.manifest_sha256
    ):
        raise ResearchBuildError(
            "existing research source-manifest.json failed validation"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ResearchBuildError("existing research manifest.files must be an array")
    files: list[ResearchFile] = []
    rows: Counter[str] = Counter()
    for index, raw in enumerate(raw_files):
        item = _json_object(raw, f"research manifest.files[{index}]")
        relative = _safe_relative_path(
            item.get("path"), f"research manifest.files[{index}].path"
        )
        actual = dataset_path.joinpath(*relative.parts).resolve()
        if not actual.is_relative_to(dataset_path) or not actual.is_file():
            raise ResearchBuildError(f"research output file is missing: {relative}")
        expected_bytes = _required_nonnegative_int(
            item.get("bytes"), f"research manifest.files[{index}].bytes"
        )
        expected_sha = _valid_sha256(
            item.get("sha256"), f"research manifest.files[{index}].sha256"
        )
        expected_rows = _required_nonnegative_int(
            item.get("rows"), f"research manifest.files[{index}].rows"
        )
        if (
            actual.stat().st_size != expected_bytes
            or _sha256_file(actual) != expected_sha
            or pq.ParquetFile(actual).metadata.num_rows != expected_rows
        ):
            raise ResearchBuildError(f"research output file is corrupted: {relative}")
        table_name = _required_string(
            item.get("table"), f"research manifest.files[{index}].table"
        )
        rows[table_name] += expected_rows
        files.append(
            ResearchFile(
                path=relative.as_posix(),
                table=table_name,
                symbol=_required_string(
                    item.get("symbol"), f"research manifest.files[{index}].symbol"
                ),
                date=_required_string(
                    item.get("date"), f"research manifest.files[{index}].date"
                ),
                rows=expected_rows,
                bytes=expected_bytes,
                sha256=expected_sha,
            )
        )
    if len(files) != manifest.get("output_file_count"):
        raise ResearchBuildError("existing output_file_count is inconsistent")
    if _research_files_fingerprint(files) != manifest.get("output_fingerprint"):
        raise ResearchBuildError("existing output_fingerprint is inconsistent")
    expected_rows_by_table = _json_object(
        manifest.get("output_rows"), "research manifest.output_rows"
    )
    if dict(sorted(rows.items())) != {
        key: _required_nonnegative_int(value, f"output_rows.{key}")
        for key, value in sorted(expected_rows_by_table.items())
    }:
        raise ResearchBuildError("existing output row totals are inconsistent")
    return _manifest_result(dataset_path, manifest, reused=True)


def _safe_output_root(source_root: Path, output_root: Path) -> None:
    if (
        source_root == output_root
        or source_root.is_relative_to(output_root)
        or output_root.is_relative_to(source_root)
    ):
        raise ResearchBuildError("canonical and research output roots must not overlap")


def _build_research_dataset(
    source: _CanonicalSource,
    output_root: str | Path,
    *,
    parameters: ResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ResearchBuildResult:
    selected_parameters = ResearchParameters() if parameters is None else parameters
    selected_parameters.validate()
    destination_root = Path(output_root).expanduser().resolve()
    _safe_output_root(source.root, destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    if minimum_free_bytes < 0:
        raise ResearchBuildError("minimum_free_bytes must be non-negative")
    required_free_bytes = minimum_free_bytes + source.total_bytes
    free_bytes = shutil.disk_usage(destination_root).free
    if free_bytes < required_free_bytes:
        raise ResearchBuildError(
            "insufficient disk space for research build: "
            f"{free_bytes} bytes free, {required_free_bytes} required"
        )

    parameter_payload = selected_parameters.to_dict()
    parameter_fingerprint = _sha256_json(parameter_payload)
    input_payload = {
        "research_schema_version": RESEARCH_SCHEMA_VERSION,
        "package_version": __version__,
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "source_dataset_id": source.dataset_id,
        "source_manifest_sha256": source.manifest_sha256,
        "source_output_fingerprint": source.output_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
    }
    input_fingerprint = _sha256_json(input_payload)
    research_dataset_id = (
        f"research-v{RESEARCH_SCHEMA_VERSION}-{input_fingerprint[:16]}"
    )
    final_path = destination_root / research_dataset_id
    if final_path.exists():
        return _validate_existing_research_dataset(
            final_path,
            research_dataset_id=research_dataset_id,
            input_fingerprint=input_fingerprint,
            source=source,
        )

    staging_path = (
        destination_root / f".{research_dataset_id}.tmp-{uuid.uuid4().hex}"
    )
    staging_path.mkdir()
    try:
        source_manifest_copy = staging_path / "source-manifest.json"
        shutil.copyfile(source.manifest_path, source_manifest_copy)
        if _sha256_file(source_manifest_copy) != source.manifest_sha256:
            raise ResearchBuildError(
                "canonical manifest changed while research build was starting"
            )

        features_by_symbol: dict[str, list[dict[str, object]]] = {}
        labels_by_symbol: dict[str, list[dict[str, object]]] = {}
        quality_by_symbol: dict[str, dict[str, int]] = {}
        for symbol in source.symbols:
            LOGGER.info("Building causal features and labels for %s", symbol)
            features, labels, quality = _build_symbol(
                source, symbol, selected_parameters
            )
            features_by_symbol[symbol] = features
            labels_by_symbol[symbol] = labels
            quality_by_symbol[symbol] = dict(sorted(quality.items()))
            LOGGER.info(
                "%s ready: %d feature rows, %d label rows",
                symbol,
                len(features),
                len(labels),
            )
        _add_btc_context(features_by_symbol)
        feature_rows = sum(len(rows) for rows in features_by_symbol.values())
        label_rows = sum(len(rows) for rows in labels_by_symbol.values())
        if feature_rows == 0:
            raise ResearchBuildError("research build produced no eligible features")
        if label_rows == 0:
            raise ResearchBuildError("research build produced no complete labels")

        files = _write_parquet_outputs(
            staging_path, features_by_symbol, labels_by_symbol
        )
        output_fingerprint = _research_files_fingerprint(files)
        outcome_counts: Counter[str] = Counter()
        horizon_counts: Counter[str] = Counter()
        for rows in labels_by_symbol.values():
            for row in rows:
                outcome_counts[cast(str, row["outcome"])] += 1
                horizon_counts[f"{int(cast(int, row['horizon_minutes']))}m"] += 1
        manifest: dict[str, object] = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "research_profile": MICROSTRUCTURE_RESEARCH_PROFILE,
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
                "dataset_id": source.dataset_id,
                "dataset_path": source.root.as_posix(),
                "manifest_copy": "source-manifest.json",
                "manifest_sha256": source.manifest_sha256,
                "output_fingerprint": source.output_fingerprint,
                "symbols": list(source.symbols),
                "bytes": source.total_bytes,
            },
            "parameters": {
                **parameter_payload,
                "fingerprint": parameter_fingerprint,
            },
            "causality": {
                "feature_rule": "received_at_ns <= decision_at_ns",
                "decision_grid": "UTC epoch aligned",
                "label_rule": "decision_at_ns < trade.received_at_ns <= label_end_ns",
                "trade_order": [
                    "received_at_ns",
                    "event_ts_ms",
                    "sequence",
                ],
                "equal_order_key": "AMBIGUOUS",
                "execution_labels_included": False,
                "maker_fill_claimed": False,
            },
            "schemas": {
                "features": _schema_manifest(FEATURE_SCHEMA),
                "labels": _schema_manifest(LABEL_SCHEMA),
            },
            "quality_by_symbol": quality_by_symbol,
            "label_outcomes": dict(sorted(outcome_counts.items())),
            "labels_by_horizon": dict(sorted(horizon_counts.items())),
            "output_rows": {
                "features": feature_rows,
                "labels": label_rows,
            },
            "output_file_count": len(files),
            "output_fingerprint": output_fingerprint,
            "files": [item.to_dict() for item in sorted(files, key=lambda x: x.path)],
        }
        _write_json_atomic(staging_path / "manifest.json", manifest)
        os.replace(staging_path, final_path)
        LOGGER.info(
            "Research dataset ready at %s (%d features, %d labels)",
            final_path,
            feature_rows,
            label_rows,
        )
        return _manifest_result(final_path, manifest, reused=False)
    except Exception:
        if staging_path.is_dir() and staging_path.parent == destination_root:
            shutil.rmtree(staging_path, ignore_errors=True)
        raise


def build_research_dataset(
    canonical_dataset: str | Path,
    output_root: str | Path,
    *,
    parameters: ResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ResearchBuildResult:
    """Build causal features and labels from one canonical dataset."""

    return _build_research_dataset(
        _load_canonical_source(canonical_dataset),
        output_root,
        parameters=parameters,
        minimum_free_bytes=minimum_free_bytes,
    )


def build_research_dataset_from_catalog(
    archive_catalog: str | Path,
    output_root: str | Path,
    *,
    parameters: ResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ResearchBuildResult:
    """Build causal features and labels from consecutive daily archives."""

    return _build_research_dataset(
        _load_archive_catalog_source(archive_catalog),
        output_root,
        parameters=parameters,
        minimum_free_bytes=minimum_free_bytes,
    )
