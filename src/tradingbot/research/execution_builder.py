"""Build immutable execution-aware maker labels from live microstructure data.

Features use only records received by the decision timestamp.  Execution labels
look forward through the first observable activation book and public trades.
They are a conservative queue proxy, not a claim about a real exchange order.
"""

from __future__ import annotations

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
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from tradingbot import __version__
from tradingbot.research.builder import (
    MISSING_SEQUENCE,
    NS_PER_MILLISECOND,
    NS_PER_SECOND,
    _add_btc_context,
    _CanonicalSource,
    _decision_id,
    _first_grid_at_or_after,
    _KlineSeries,
    _last_grid_at_or_before,
    _load_archive_catalog_source,
    _load_canonical_source,
    _load_symbol_series,
    _number,
    _OrderBookSeries,
    _required_nonnegative_int,
    _required_string,
    _research_files_fingerprint,
    _safe_output_root,
    _safe_relative_path,
    _schema_manifest,
    _sha256_file,
    _sha256_json,
    _time_features,
    _TradeSeries,
    _utc_date_from_ns,
    _valid_sha256,
    _write_json_atomic,
)
from tradingbot.research.contracts import (
    EXECUTION_FEATURE_SCHEMA,
    EXECUTION_LABEL_SCHEMA,
    EXECUTION_RESEARCH_PROFILE,
    EXECUTION_RESEARCH_SCHEMA_VERSION,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_FORMAT_VERSION,
    RESEARCH_SCHEMA_VERSION,
    ExecutionResearchBuildResult,
    ExecutionResearchParameters,
    ResearchBuildError,
    ResearchFile,
)

LOGGER = logging.getLogger(__name__)

_PRICE_RELATIVE_TOLERANCE: Final = 1e-10


@dataclass(frozen=True, slots=True)
class _ActivationSnapshot:
    received_at_ns: int
    best_bid_price: float
    best_ask_price: float
    post_only_valid: bool
    queue_ahead_size: float | None


@dataclass(frozen=True, slots=True)
class _MakerFillResult:
    status: str
    resolution: str
    entry_window_trade_count: int
    contra_trade_count: int
    contra_volume_at_entry_price: float
    filled_size: float
    first_fill_index: int | None
    full_fill_index: int | None


@dataclass(frozen=True, slots=True)
class _ContinuityGuard:
    books: _OrderBookSeries
    klines: _KlineSeries
    maximum_gap_ms: int

    def covers(self, start_ns: int, end_ns: int) -> bool:
        return self.books.has_continuous_coverage(
            start_ns,
            end_ns,
            self.maximum_gap_ms,
        ) and bool(self.klines.has_complete_coverage(start_ns, end_ns))


def _price_tolerance(price: float) -> float:
    return max(abs(price) * _PRICE_RELATIVE_TOLERANCE, 1e-12)


def _matching_level_size(
    prices: list[float], sizes: list[float], target: float
) -> float | None:
    tolerance = _price_tolerance(target)
    for price, size in zip(prices, sizes, strict=True):
        if abs(float(price) - target) <= tolerance:
            value = float(size)
            return value if math.isfinite(value) and value >= 0 else None
    return None


def _activation_snapshot(
    books: _OrderBookSeries,
    *,
    submitted_at_ns: int,
    activation_max_delay_ms: int,
    side: str,
    entry_price: float,
) -> tuple[_ActivationSnapshot | None, str | None]:
    position = int(
        np.searchsorted(books.received_at_ns, submitted_at_ns, side="left")
    )
    if position >= len(books.received_at_ns):
        return None, "missing_activation_orderbook"
    received_at_ns = int(books.received_at_ns[position])
    if (
        received_at_ns - submitted_at_ns
        > activation_max_delay_ms * NS_PER_MILLISECOND
    ):
        return None, "stale_activation_orderbook"

    index = int(books.original_indices[position])
    bid_prices = cast(list[float], books.table["bid_prices"][index].as_py())
    bid_sizes = cast(list[float], books.table["bid_sizes"][index].as_py())
    ask_prices = cast(list[float], books.table["ask_prices"][index].as_py())
    ask_sizes = cast(list[float], books.table["ask_sizes"][index].as_py())
    if (
        not bid_prices
        or not ask_prices
        or len(bid_prices) != len(bid_sizes)
        or len(ask_prices) != len(ask_sizes)
    ):
        return None, "invalid_activation_orderbook"
    best_bid = float(bid_prices[0])
    best_ask = float(ask_prices[0])
    if not (
        math.isfinite(best_bid)
        and math.isfinite(best_ask)
        and best_bid > 0
        and best_ask > best_bid
    ):
        return None, "invalid_activation_orderbook"

    tolerance = _price_tolerance(entry_price)
    if side == "LONG":
        post_only_valid = entry_price < best_ask - tolerance
        if not post_only_valid:
            queue_ahead = None
        elif entry_price > best_bid + tolerance:
            queue_ahead = 0.0
        else:
            queue_ahead = _matching_level_size(
                bid_prices, bid_sizes, entry_price
            )
    elif side == "SHORT":
        post_only_valid = entry_price > best_bid + tolerance
        if not post_only_valid:
            queue_ahead = None
        elif entry_price < best_ask - tolerance:
            queue_ahead = 0.0
        else:
            queue_ahead = _matching_level_size(
                ask_prices, ask_sizes, entry_price
            )
    else:
        raise ResearchBuildError(f"unsupported execution side: {side}")

    if post_only_valid and queue_ahead is None:
        return None, "entry_level_missing_from_activation_book"
    return (
        _ActivationSnapshot(
            received_at_ns=received_at_ns,
            best_bid_price=best_bid,
            best_ask_price=best_ask,
            post_only_valid=post_only_valid,
            queue_ahead_size=queue_ahead,
        ),
        None,
    )


def _maker_fills(
    trades: _TradeSeries,
    *,
    activation_at_ns: int,
    entry_window_end_ns: int,
    side: str,
    entry_price: float,
    queue_ahead_size: float,
    order_sizes: tuple[float, ...],
    queue_ahead_multiplier: float,
) -> tuple[_MakerFillResult, ...] | None:
    if trades.last_received_at_ns < entry_window_end_ns:
        return None
    start = int(
        np.searchsorted(trades.received_at_ns, activation_at_ns, side="right")
    )
    end = int(
        np.searchsorted(
            trades.received_at_ns, entry_window_end_ns, side="left"
        )
    )
    required_queue = queue_ahead_size * queue_ahead_multiplier
    exact_volume = 0.0
    contra_count = 0
    first_fill_indices: list[int | None] = [None] * len(order_sizes)
    filled_sizes = [0.0] * len(order_sizes)
    results: list[_MakerFillResult | None] = [None] * len(order_sizes)
    tolerance = _price_tolerance(entry_price)

    for index in range(start, end):
        if not bool(trades.is_visible_execution_trade[index]):
            continue
        price = float(trades.price[index])
        is_buy = bool(trades.is_buy[index])
        if side == "LONG":
            is_contra = not is_buy and price <= entry_price + tolerance
            traded_through = price < entry_price - tolerance
        elif side == "SHORT":
            is_contra = is_buy and price >= entry_price - tolerance
            traded_through = price > entry_price + tolerance
        else:
            raise ResearchBuildError(f"unsupported execution side: {side}")
        if not is_contra:
            continue
        contra_count += 1
        if traded_through:
            for scenario, order_size in enumerate(order_sizes):
                if results[scenario] is not None:
                    continue
                if first_fill_indices[scenario] is None:
                    first_fill_indices[scenario] = index
                results[scenario] = _MakerFillResult(
                    status="FULL_FILL",
                    resolution="public_trade_through_entry_price",
                    entry_window_trade_count=end - start,
                    contra_trade_count=contra_count,
                    contra_volume_at_entry_price=exact_volume,
                    filled_size=order_size,
                    first_fill_index=first_fill_indices[scenario],
                    full_fill_index=index,
                )
            break

        exact_volume += float(trades.size[index])
        executable = max(0.0, exact_volume - required_queue)
        for scenario, order_size in enumerate(order_sizes):
            if results[scenario] is not None:
                continue
            filled_size = min(order_size, executable)
            if filled_size > 0 and first_fill_indices[scenario] is None:
                first_fill_indices[scenario] = index
            filled_sizes[scenario] = filled_size
            if filled_size + max(order_size * 1e-12, 1e-15) >= order_size:
                results[scenario] = _MakerFillResult(
                    status="FULL_FILL",
                    resolution="visible_queue_depleted_by_public_trades",
                    entry_window_trade_count=end - start,
                    contra_trade_count=contra_count,
                    contra_volume_at_entry_price=exact_volume,
                    filled_size=order_size,
                    first_fill_index=first_fill_indices[scenario],
                    full_fill_index=index,
                )
        if all(result is not None for result in results):
            break

    for scenario in range(len(order_sizes)):
        if results[scenario] is not None:
            continue
        filled_size = filled_sizes[scenario]
        results[scenario] = _MakerFillResult(
            status="PARTIAL_FILL" if filled_size > 0 else "NO_FILL",
            resolution=(
                "entry_ttl_expired_after_partial_fill"
                if filled_size > 0
                else "entry_ttl_expired_without_fill"
            ),
            entry_window_trade_count=end - start,
            contra_trade_count=contra_count,
            contra_volume_at_entry_price=exact_volume,
            filled_size=filled_size,
            first_fill_index=first_fill_indices[scenario],
            full_fill_index=None,
        )
    return tuple(cast(_MakerFillResult, result) for result in results)


def _maker_fill(
    trades: _TradeSeries,
    *,
    activation_at_ns: int,
    entry_window_end_ns: int,
    side: str,
    entry_price: float,
    queue_ahead_size: float,
    order_size: float,
    queue_ahead_multiplier: float,
) -> _MakerFillResult | None:
    """Single-size compatibility wrapper used by focused unit tests."""

    results = _maker_fills(
        trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=entry_window_end_ns,
        side=side,
        entry_price=entry_price,
        queue_ahead_size=queue_ahead_size,
        order_sizes=(order_size,),
        queue_ahead_multiplier=queue_ahead_multiplier,
    )
    return None if results is None else results[0]


def _barrier_prices(
    feature: dict[str, object],
    parameters: ExecutionResearchParameters,
    *,
    side: str,
    horizon_minutes: int,
    entry_price: float,
) -> tuple[float, float, float, float]:
    realised_60 = _number(feature, "realized_volatility_60m_fraction")
    per_minute_volatility = realised_60 / math.sqrt(60)
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
    return (
        stop_distance_bps,
        take_profit_distance_bps,
        stop_price,
        take_profit_price,
    )


def _sequence_or_none(trades: _TradeSeries, index: int | None) -> int | None:
    if index is None:
        return None
    value = int(trades.sequence[index])
    return None if value == int(MISSING_SEQUENCE) else value


def _execution_label_rows(
    *,
    source_dataset_id: str,
    symbol: str,
    feature: dict[str, object],
    books: _OrderBookSeries,
    trades: _TradeSeries,
    parameters: ExecutionResearchParameters,
    quality: Counter[str],
    continuity: _ContinuityGuard | None = None,
) -> list[dict[str, object]]:
    decision_at_ns = int(cast(int, feature["decision_at_ns"]))
    decision_utc_date = cast(str, feature["decision_utc_date"])
    submitted_at_ns = (
        decision_at_ns
        + parameters.submission_latency_ms * NS_PER_MILLISECOND
    )
    entry_window_end_ns = (
        decision_at_ns + parameters.entry_ttl_seconds * NS_PER_SECOND
    )
    labels: list[dict[str, object]] = []

    for side in ("LONG", "SHORT"):
        entry_price = _number(
            feature,
            "best_bid_price" if side == "LONG" else "best_ask_price",
        )
        activation, reason = _activation_snapshot(
            books,
            submitted_at_ns=submitted_at_ns,
            activation_max_delay_ms=parameters.activation_max_delay_ms,
            side=side,
            entry_price=entry_price,
        )
        if activation is None:
            quality[f"skipped_{reason}"] += 1
            continue

        order_sizes = tuple(
            order_notional / entry_price
            for order_notional in parameters.order_notionals_usdt
        )
        if activation.post_only_valid:
            if continuity is not None and not continuity.covers(
                activation.received_at_ns,
                entry_window_end_ns,
            ):
                quality["entry_scenarios_skipped_discontinuous_window"] += len(
                    order_sizes
                )
                continue
            if activation.queue_ahead_size is None:
                raise ResearchBuildError(
                    "valid PostOnly activation unexpectedly has no queue size"
                )
            fills = _maker_fills(
                trades,
                activation_at_ns=activation.received_at_ns,
                entry_window_end_ns=entry_window_end_ns,
                side=side,
                entry_price=entry_price,
                queue_ahead_size=activation.queue_ahead_size,
                order_sizes=order_sizes,
                queue_ahead_multiplier=parameters.queue_ahead_multiplier,
            )
            if fills is None:
                quality["entry_scenarios_skipped_incomplete_trade_window"] += len(
                    order_sizes
                )
                continue
        else:
            fills = tuple(
                _MakerFillResult(
                    status="NO_FILL",
                    resolution="post_only_would_cross_at_observed_activation",
                    entry_window_trade_count=0,
                    contra_trade_count=0,
                    contra_volume_at_entry_price=0.0,
                    filled_size=0.0,
                    first_fill_index=None,
                    full_fill_index=None,
                )
                for _ in order_sizes
            )

        for order_notional, order_size, fill in zip(
            parameters.order_notionals_usdt,
            order_sizes,
            fills,
            strict=True,
        ):
            queue_required_size: float | None = (
                None
                if activation.queue_ahead_size is None
                else activation.queue_ahead_size
                * parameters.queue_ahead_multiplier
                + order_size
            )

            quality[f"entry_scenarios_{fill.status.lower()}"] += 1
            first_fill_at_ns = (
                None
                if fill.first_fill_index is None
                else int(trades.received_at_ns[fill.first_fill_index])
            )
            full_fill_at_ns = (
                None
                if fill.full_fill_index is None
                else int(trades.received_at_ns[fill.full_fill_index])
            )
            for horizon_minutes in parameters.position_horizons_minutes:
                (
                    stop_distance_bps,
                    take_profit_distance_bps,
                    stop_price,
                    take_profit_price,
                ) = _barrier_prices(
                    feature,
                    parameters,
                    side=side,
                    horizon_minutes=horizon_minutes,
                    entry_price=entry_price,
                )
                row: dict[str, object] = {
                    "execution_research_schema_version": (
                        EXECUTION_RESEARCH_SCHEMA_VERSION
                    ),
                    "decision_id": feature["decision_id"],
                    "source_dataset_id": source_dataset_id,
                    "symbol": symbol,
                    "decision_at_ns": decision_at_ns,
                    "decision_utc_date": decision_utc_date,
                    "side": side,
                    "horizon_minutes": horizon_minutes,
                    "order_notional_usdt": order_notional,
                    "submitted_at_ns": submitted_at_ns,
                    "activation_at_ns": activation.received_at_ns,
                    "activation_delay_ms": (
                        activation.received_at_ns - submitted_at_ns
                    )
                    / NS_PER_MILLISECOND,
                    "entry_window_end_ns": entry_window_end_ns,
                    "entry_limit_price": entry_price,
                    "order_size_base": order_size,
                    "activation_best_bid_price": activation.best_bid_price,
                    "activation_best_ask_price": activation.best_ask_price,
                    "post_only_valid": activation.post_only_valid,
                    "queue_ahead_size_base": activation.queue_ahead_size,
                    "queue_required_size_base": queue_required_size,
                    "entry_window_trade_count": fill.entry_window_trade_count,
                    "contra_trade_count": fill.contra_trade_count,
                    "contra_volume_at_entry_price_base": (
                        fill.contra_volume_at_entry_price
                    ),
                    "fill_status": fill.status,
                    "fill_fraction": fill.filled_size / order_size,
                    "filled_size_base": fill.filled_size,
                    "first_fill_at_ns": first_fill_at_ns,
                    "full_fill_at_ns": full_fill_at_ns,
                    "full_fill_event_ts_ms": (
                        None
                        if fill.full_fill_index is None
                        else int(trades.event_ts_ms[fill.full_fill_index])
                    ),
                    "full_fill_sequence": _sequence_or_none(
                        trades, fill.full_fill_index
                    ),
                    "full_fill_trade_price": (
                        None
                        if fill.full_fill_index is None
                        else float(trades.price[fill.full_fill_index])
                    ),
                    "time_to_full_fill_ms": (
                        None
                        if full_fill_at_ns is None
                        else (full_fill_at_ns - decision_at_ns)
                        / NS_PER_MILLISECOND
                    ),
                    "stop_distance_bps": stop_distance_bps,
                    "take_profit_distance_bps": take_profit_distance_bps,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "position_end_ns": None,
                    "outcome": fill.status,
                    "hit_at_ns": None,
                    "hit_event_ts_ms": None,
                    "hit_sequence": None,
                    "hit_trade_price": None,
                    "time_from_fill_to_hit_ms": None,
                    "timeout_price": None,
                    "outcome_return_bps": None,
                    "future_trade_count": 0,
                    "resolution": fill.resolution,
                }
                if fill.status != "FULL_FILL":
                    labels.append(row)
                    quality[f"label_{horizon_minutes}m_{fill.status.lower()}"] += 1
                    continue

                if fill.full_fill_index is None or full_fill_at_ns is None:
                    raise ResearchBuildError(
                        "FULL_FILL execution label is missing its ordering key"
                    )
                position_end_ns = (
                    full_fill_at_ns
                    + horizon_minutes * 60 * NS_PER_SECOND
                )
                if continuity is not None and not continuity.covers(
                    full_fill_at_ns,
                    position_end_ns,
                ):
                    quality[
                        f"labels_skipped_discontinuous_position_{horizon_minutes}m"
                    ] += 1
                    continue
                if trades.last_received_at_ns < position_end_ns:
                    quality[
                        f"labels_skipped_incomplete_position_{horizon_minutes}m"
                    ] += 1
                    continue
                full_fill_trade_price = float(
                    trades.price[fill.full_fill_index]
                )
                fill_trade_crossed_barrier = (
                    side == "LONG" and full_fill_trade_price <= stop_price
                ) or (
                    side == "SHORT" and full_fill_trade_price >= stop_price
                )
                if fill_trade_crossed_barrier:
                    row.update(
                        {
                            "position_end_ns": position_end_ns,
                            "outcome": "AMBIGUOUS",
                            "resolution": "fill_trade_also_crossed_stop_barrier",
                        }
                    )
                    labels.append(row)
                    quality[f"label_{horizon_minutes}m_ambiguous"] += 1
                    continue

                outcome = trades.barrier_outcome(
                    decision_at_ns=full_fill_at_ns,
                    label_end_ns=position_end_ns,
                    side=side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    stop_distance_bps=stop_distance_bps,
                    take_profit_distance_bps=take_profit_distance_bps,
                    start_after_index=fill.full_fill_index,
                    execution_eligible_only=True,
                )
                if outcome is None:
                    quality[
                        f"labels_skipped_incomplete_position_{horizon_minutes}m"
                    ] += 1
                    continue
                hit_index = outcome.hit_index
                hit_at_ns = (
                    None
                    if hit_index is None
                    else int(trades.received_at_ns[hit_index])
                )
                row.update(
                    {
                        "position_end_ns": position_end_ns,
                        "outcome": outcome.outcome,
                        "hit_at_ns": hit_at_ns,
                        "hit_event_ts_ms": (
                            None
                            if hit_index is None
                            else int(trades.event_ts_ms[hit_index])
                        ),
                        "hit_sequence": _sequence_or_none(trades, hit_index),
                        "hit_trade_price": (
                            None
                            if hit_index is None
                            else float(trades.price[hit_index])
                        ),
                        "time_from_fill_to_hit_ms": (
                            None
                            if hit_at_ns is None
                            else (hit_at_ns - full_fill_at_ns)
                            / NS_PER_MILLISECOND
                        ),
                        "timeout_price": outcome.timeout_price,
                        "outcome_return_bps": outcome.outcome_return_bps,
                        "future_trade_count": outcome.future_trade_count,
                        "resolution": outcome.resolution,
                    }
                )
                labels.append(row)
                quality[
                    f"label_{horizon_minutes}m_{outcome.outcome.lower()}"
                ] += 1
    quality["execution_labels_emitted"] += len(labels)
    return labels


def _build_execution_symbol(
    source: _CanonicalSource,
    symbol: str,
    parameters: ExecutionResearchParameters,
    *,
    decision_start_ns: int | None = None,
    decision_end_ns: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    books, ticker, klines, trades = _load_symbol_series(source, symbol)
    continuity = _ContinuityGuard(
        books=books,
        klines=klines,
        maximum_gap_ms=parameters.maximum_continuity_gap_ms,
    )
    interval_ns = parameters.decision_interval_seconds * NS_PER_SECOND
    offset_ns = parameters.decision_offset_seconds * NS_PER_SECOND
    first_available = max(
        books.first_received_at_ns,
        ticker.first_received_at_ns,
        trades.first_received_at_ns,
    )
    last_available = min(
        books.last_received_at_ns,
        ticker.last_received_at_ns,
        trades.last_received_at_ns,
    )
    first_decision = _first_grid_at_or_after(
        first_available, interval_ns, offset_ns
    )
    last_decision = _last_grid_at_or_before(
        last_available, interval_ns, offset_ns
    )
    if decision_start_ns is not None:
        first_decision = max(
            first_decision,
            _first_grid_at_or_after(
                decision_start_ns, interval_ns, offset_ns
            ),
        )
    if decision_end_ns is not None:
        if decision_start_ns is not None and decision_end_ns <= decision_start_ns:
            raise ResearchBuildError(
                "execution decision range must have a positive duration"
            )
        last_decision = min(
            last_decision,
            _last_grid_at_or_before(
                decision_end_ns - 1, interval_ns, offset_ns
            ),
        )
    if first_decision > last_decision:
        raise ResearchBuildError(f"{symbol} has no complete decision interval")

    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    quality: Counter[str] = Counter()
    decision_at_ns = first_decision
    while decision_at_ns <= last_decision:
        quality["candidate_decisions"] += 1
        feature_window_start_ns = decision_at_ns - (
            parameters.kline_history_minutes + 1
        ) * 60 * NS_PER_SECOND
        if not continuity.covers(feature_window_start_ns, decision_at_ns):
            quality["skipped_discontinuous_feature_window"] += 1
            decision_at_ns += interval_ns
            continue
        book_features, reason = books.features_at(
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

        feature: dict[str, object] = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "decision_id": _decision_id(
                source.dataset_id, symbol, decision_at_ns
            ),
            "source_dataset_id": source.dataset_id,
            "symbol": symbol,
            "decision_at_ns": decision_at_ns,
            "decision_at_ms": decision_at_ns // NS_PER_MILLISECOND,
            "decision_utc_date": _utc_date_from_ns(decision_at_ns),
        }
        feature.update(book_features)
        feature.update(ticker_features)
        feature.update(kline_features)
        feature.update(trades.features_at(decision_at_ns))
        feature.update(_time_features(decision_at_ns))
        features.append(feature)
        labels.extend(
            _execution_label_rows(
                source_dataset_id=source.dataset_id,
                symbol=symbol,
                feature=feature,
                books=books,
                trades=trades,
                parameters=parameters,
                quality=quality,
                continuity=continuity,
            )
        )
        quality["features_emitted"] += 1
        decision_at_ns += interval_ns

    return features, labels, quality


def _source_partition_dates(source: _CanonicalSource) -> tuple[date, ...]:
    try:
        partition_dates = tuple(
            sorted({date.fromisoformat(item.date) for item in source.files})
        )
    except ValueError as exc:
        raise ResearchBuildError(
            "canonical source contains a non-ISO partition date"
        ) from exc
    if not partition_dates:
        raise ResearchBuildError("canonical source contains no partition dates")
    return partition_dates


def _source_window(
    source: _CanonicalSource,
    target_date: date,
) -> _CanonicalSource:
    """Keep one UTC output day plus adjacent context without changing provenance."""

    selected_dates = {
        (target_date - timedelta(days=1)).isoformat(),
        target_date.isoformat(),
        (target_date + timedelta(days=1)).isoformat(),
    }
    files = tuple(item for item in source.files if item.date in selected_dates)
    if not files:
        raise ResearchBuildError(
            f"canonical source has no files for execution date {target_date}"
        )
    for symbol in source.symbols:
        actual = {item.kind for item in files if item.symbol == symbol}
        missing = {"orderbook", "ticker", "trades", "kline_1"} - actual
        if missing:
            raise ResearchBuildError(
                f"execution window for {target_date} is missing "
                f"{', '.join(sorted(missing))} for {symbol}"
            )
    return _CanonicalSource(
        dataset_id=source.dataset_id,
        root=source.root,
        manifest_path=source.manifest_path,
        manifest_sha256=source.manifest_sha256,
        output_fingerprint=source.output_fingerprint,
        symbols=source.symbols,
        files=files,
        total_bytes=sum(item.bytes for item in files),
        gapped_dates=tuple(
            value for value in source.gapped_dates if value in selected_dates
        ),
    )


def _utc_day_bounds_ns(partition_date: date) -> tuple[int, int]:
    start = datetime(
        partition_date.year,
        partition_date.month,
        partition_date.day,
        tzinfo=UTC,
    )
    start_ns = int(start.timestamp()) * NS_PER_SECOND
    return start_ns, start_ns + 24 * 60 * 60 * NS_PER_SECOND


def _write_execution_outputs(
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
            groups[
                (
                    "execution_labels",
                    symbol,
                    cast(str, row["decision_utc_date"]),
                )
            ].append(row)

    files: list[ResearchFile] = []
    for (table_name, symbol, partition_date), rows in sorted(groups.items()):
        schema = (
            EXECUTION_FEATURE_SCHEMA
            if table_name == "features"
            else EXECUTION_LABEL_SCHEMA
        )
        if table_name == "features":
            rows.sort(key=lambda row: int(cast(int, row["decision_at_ns"])))
        else:
            rows.sort(
                key=lambda row: (
                    int(cast(int, row["decision_at_ns"])),
                    int(cast(int, row["horizon_minutes"])),
                    cast(str, row["side"]),
                    float(cast(float, row["order_notional_usdt"])),
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


def _manifest_result(
    dataset_path: Path, manifest: dict[str, Any], *, reused: bool
) -> ExecutionResearchBuildResult:
    source = cast(dict[str, object], manifest["source"])
    parameters = cast(dict[str, object], manifest["parameters"])
    rows = cast(dict[str, object], manifest["output_rows"])
    return ExecutionResearchBuildResult(
        execution_dataset_id=_required_string(
            manifest.get("execution_dataset_id"), "execution_dataset_id"
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
        feature_rows=_required_nonnegative_int(
            rows.get("features"), "output_rows.features"
        ),
        execution_label_rows=_required_nonnegative_int(
            rows.get("execution_labels"), "output_rows.execution_labels"
        ),
        output_files=_required_nonnegative_int(
            manifest.get("output_file_count"), "output_file_count"
        ),
        reused=reused,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchBuildError(
            f"execution research manifest is unreadable: {path}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ResearchBuildError("execution research manifest must be a JSON object")
    return cast(dict[str, Any], parsed)


def _validate_existing_execution_dataset(
    dataset_path: Path,
    *,
    execution_dataset_id: str,
    input_fingerprint: str,
    source: _CanonicalSource,
) -> ExecutionResearchBuildResult:
    manifest = _load_manifest(dataset_path / "manifest.json")
    if (
        manifest.get("execution_research_schema_version")
        != EXECUTION_RESEARCH_SCHEMA_VERSION
    ):
        raise ResearchBuildError(
            "existing execution dataset uses another schema version"
        )
    if manifest.get("research_profile") != EXECUTION_RESEARCH_PROFILE:
        raise ResearchBuildError("existing execution dataset uses another profile")
    if manifest.get("execution_dataset_id") != execution_dataset_id:
        raise ResearchBuildError(
            "existing execution dataset ID does not match its directory"
        )
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise ResearchBuildError(
            "existing execution dataset was built from another input"
        )

    source_manifest_copy = dataset_path / "source-manifest.json"
    raw_source = manifest.get("source")
    if not isinstance(raw_source, dict):
        raise ResearchBuildError("execution manifest.source must be an object")
    expected_source_sha = _valid_sha256(
        raw_source.get("manifest_sha256"), "source.manifest_sha256"
    )
    if (
        not source_manifest_copy.is_file()
        or _sha256_file(source_manifest_copy) != expected_source_sha
        or expected_source_sha != source.manifest_sha256
    ):
        raise ResearchBuildError(
            "existing execution source-manifest.json failed validation"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ResearchBuildError("execution manifest.files must be an array")
    files: list[ResearchFile] = []
    rows: Counter[str] = Counter()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise ResearchBuildError(f"execution manifest.files[{index}] is invalid")
        relative = _safe_relative_path(
            raw.get("path"), f"execution manifest.files[{index}].path"
        )
        actual = dataset_path.joinpath(*PurePosixPath(relative).parts).resolve()
        if not actual.is_relative_to(dataset_path) or not actual.is_file():
            raise ResearchBuildError(f"execution output file is missing: {relative}")
        expected_bytes = _required_nonnegative_int(
            raw.get("bytes"), f"execution manifest.files[{index}].bytes"
        )
        expected_rows = _required_nonnegative_int(
            raw.get("rows"), f"execution manifest.files[{index}].rows"
        )
        expected_sha = _valid_sha256(
            raw.get("sha256"), f"execution manifest.files[{index}].sha256"
        )
        if (
            actual.stat().st_size != expected_bytes
            or _sha256_file(actual) != expected_sha
            or pq.ParquetFile(actual).metadata.num_rows != expected_rows
        ):
            raise ResearchBuildError(f"execution output file is corrupted: {relative}")
        table_name = _required_string(
            raw.get("table"), f"execution manifest.files[{index}].table"
        )
        rows[table_name] += expected_rows
        files.append(
            ResearchFile(
                path=relative.as_posix(),
                table=table_name,
                symbol=_required_string(
                    raw.get("symbol"),
                    f"execution manifest.files[{index}].symbol",
                ),
                date=_required_string(
                    raw.get("date"), f"execution manifest.files[{index}].date"
                ),
                rows=expected_rows,
                bytes=expected_bytes,
                sha256=expected_sha,
            )
        )
    if len(files) != manifest.get("output_file_count"):
        raise ResearchBuildError("existing execution output_file_count is inconsistent")
    if _research_files_fingerprint(files) != manifest.get("output_fingerprint"):
        raise ResearchBuildError("existing execution output_fingerprint is inconsistent")
    expected_rows_by_table = manifest.get("output_rows")
    if not isinstance(expected_rows_by_table, dict):
        raise ResearchBuildError("execution manifest.output_rows must be an object")
    normalized_expected_rows = {
        str(key): _required_nonnegative_int(value, f"output_rows.{key}")
        for key, value in expected_rows_by_table.items()
    }
    if dict(sorted(rows.items())) != dict(sorted(normalized_expected_rows.items())):
        raise ResearchBuildError("existing execution output row totals are inconsistent")
    return _manifest_result(dataset_path, manifest, reused=True)


def _build_execution_research(
    source: _CanonicalSource,
    output_root: str | Path,
    *,
    parameters: ExecutionResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ExecutionResearchBuildResult:
    selected = (
        ExecutionResearchParameters() if parameters is None else parameters
    )
    selected.validate()
    destination_root = Path(output_root).expanduser().resolve()
    _safe_output_root(source.root, destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    if minimum_free_bytes < 0:
        raise ResearchBuildError("minimum_free_bytes must be non-negative")
    required_free_bytes = minimum_free_bytes + source.total_bytes
    free_bytes = shutil.disk_usage(destination_root).free
    if free_bytes < required_free_bytes:
        raise ResearchBuildError(
            "insufficient disk space for execution research build: "
            f"{free_bytes} bytes free, {required_free_bytes} required"
        )

    parameter_payload = selected.to_dict()
    parameter_fingerprint = _sha256_json(parameter_payload)
    input_payload = {
        "execution_research_schema_version": EXECUTION_RESEARCH_SCHEMA_VERSION,
        "research_profile": EXECUTION_RESEARCH_PROFILE,
        "package_version": __version__,
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "source_dataset_id": source.dataset_id,
        "source_manifest_sha256": source.manifest_sha256,
        "source_output_fingerprint": source.output_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
    }
    input_fingerprint = _sha256_json(input_payload)
    execution_dataset_id = (
        f"execution-research-v{EXECUTION_RESEARCH_SCHEMA_VERSION}-"
        f"{input_fingerprint[:16]}"
    )
    final_path = destination_root / execution_dataset_id
    if final_path.exists():
        return _validate_existing_execution_dataset(
            final_path,
            execution_dataset_id=execution_dataset_id,
            input_fingerprint=input_fingerprint,
            source=source,
        )

    staging_path = (
        destination_root / f".{execution_dataset_id}.tmp-{uuid.uuid4().hex}"
    )
    staging_path.mkdir()
    try:
        source_manifest_copy = staging_path / "source-manifest.json"
        shutil.copyfile(source.manifest_path, source_manifest_copy)
        if _sha256_file(source_manifest_copy) != source.manifest_sha256:
            raise ResearchBuildError(
                "source manifest changed while execution build was starting"
            )

        partition_dates = _source_partition_dates(source)
        quality_counters = {
            symbol: Counter[str]() for symbol in source.symbols
        }
        files: list[ResearchFile] = []
        outcomes: Counter[str] = Counter()
        fill_statuses: Counter[str] = Counter()
        horizons: Counter[str] = Counter()
        notionals: Counter[str] = Counter()
        feature_rows = 0
        execution_label_rows = 0
        for partition_index, partition_date in enumerate(
            partition_dates, start=1
        ):
            LOGGER.info(
                "Building execution partition %s (%d/%d)",
                partition_date,
                partition_index,
                len(partition_dates),
            )
            window_source = _source_window(source, partition_date)
            decision_start_ns, decision_end_ns = _utc_day_bounds_ns(
                partition_date
            )
            features_by_symbol: dict[str, list[dict[str, object]]] = {}
            labels_by_symbol: dict[str, list[dict[str, object]]] = {}
            for symbol in source.symbols:
                LOGGER.info(
                    "Building execution-aware features and labels for %s/%s",
                    partition_date,
                    symbol,
                )
                features, labels, quality = _build_execution_symbol(
                    window_source,
                    symbol,
                    selected,
                    decision_start_ns=decision_start_ns,
                    decision_end_ns=decision_end_ns,
                )
                features_by_symbol[symbol] = features
                labels_by_symbol[symbol] = labels
                quality_counters[symbol].update(quality)
                LOGGER.info(
                    "%s/%s execution data ready: %d feature rows, %d labels",
                    partition_date,
                    symbol,
                    len(features),
                    len(labels),
                )

            _add_btc_context(features_by_symbol)
            feature_rows += sum(
                len(rows) for rows in features_by_symbol.values()
            )
            execution_label_rows += sum(
                len(rows) for rows in labels_by_symbol.values()
            )
            partition_files = _write_execution_outputs(
                staging_path, features_by_symbol, labels_by_symbol
            )
            files.extend(partition_files)
            for rows in labels_by_symbol.values():
                for row in rows:
                    outcomes[cast(str, row["outcome"])] += 1
                    fill_statuses[cast(str, row["fill_status"])] += 1
                    horizons[
                        f"{int(cast(int, row['horizon_minutes']))}m"
                    ] += 1
                    notionals[
                        format(
                            float(cast(float, row["order_notional_usdt"])),
                            "g",
                        )
                    ] += 1

        if feature_rows == 0:
            raise ResearchBuildError(
                "execution research build produced no eligible features"
            )
        if execution_label_rows == 0:
            raise ResearchBuildError(
                "execution research build produced no complete labels"
            )

        if len({item.path for item in files}) != len(files):
            raise ResearchBuildError(
                "execution partition build produced duplicate output paths"
            )
        output_fingerprint = _research_files_fingerprint(files)
        quality_by_symbol = {
            symbol: dict(sorted(counter.items()))
            for symbol, counter in quality_counters.items()
        }

        manifest: dict[str, object] = {
            "execution_research_schema_version": (
                EXECUTION_RESEARCH_SCHEMA_VERSION
            ),
            "research_profile": EXECUTION_RESEARCH_PROFILE,
            "execution_dataset_id": execution_dataset_id,
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
                "manifest_path": source.manifest_path.as_posix(),
                "manifest_copy": "source-manifest.json",
                "manifest_sha256": source.manifest_sha256,
                "output_fingerprint": source.output_fingerprint,
                "symbols": list(source.symbols),
                "bytes": source.total_bytes,
                "gapped_partition_dates": list(source.gapped_dates),
                "partition_dates": [
                    value.isoformat() for value in partition_dates
                ],
            },
            "processing": {
                "mode": "utc_day_with_adjacent_context",
                "output_partition_count": len(partition_dates),
                "maximum_source_partitions_loaded_per_symbol": 3,
            },
            "parameters": {
                **parameter_payload,
                "fingerprint": parameter_fingerprint,
            },
            "causality": {
                "feature_rule": "received_at_ns <= decision_at_ns",
                "entry_limit": "decision best bid for LONG; best ask for SHORT",
                "submission_rule": "decision_at_ns + submission_latency_ms",
                "activation_rule": (
                    "first orderbook received at or after submitted_at_ns within "
                    "activation_max_delay_ms"
                ),
                "post_only_rule": (
                    "LONG limit < activation ask; SHORT limit > activation bid"
                ),
                "queue_rule": (
                    "activation price-level size times queue_ahead_multiplier; "
                    "cancellations do not advance the queue"
                ),
                "fill_rule": (
                    "opposite-side non-block, non-RPI public trades deplete the "
                    "exact-price queue; a trade through the limit implies full fill"
                ),
                "position_rule": (
                    "TP/SL horizon starts only after the full-fill ordering key; "
                    "block and RPI prints cannot resolve a barrier"
                ),
                "continuity_rule": (
                    "feature, maker-entry, and post-fill windows must remain in one "
                    "observed WebSocket session, stay within the configured maximum "
                    "orderbook gap, and contain every intersecting one-minute kline"
                ),
                "trade_order": [
                    "received_at_ns",
                    "event_ts_ms",
                    "sequence",
                ],
                "partial_fill_class": "PARTIAL_FILL",
                "no_fill_class": "NO_FILL",
            },
            "scope": {
                "real_exchange_orders_observed": False,
                "maker_fill_is_proxy": True,
                "queue_cancellations_modeled": False,
                "hidden_liquidity_modeled": False,
                "block_and_rpi_trades_excluded_from_queue": True,
                "partial_fills_retained": True,
                "gapped_source_supported_with_window_filter": True,
                "eligible_for_fill_model_training": True,
                "eligible_for_profitability_conclusion": False,
            },
            "schemas": {
                "features": _schema_manifest(EXECUTION_FEATURE_SCHEMA),
                "execution_labels": _schema_manifest(EXECUTION_LABEL_SCHEMA),
            },
            "quality_by_symbol": quality_by_symbol,
            "execution_outcomes": dict(sorted(outcomes.items())),
            "fill_statuses": dict(sorted(fill_statuses.items())),
            "labels_by_horizon": dict(sorted(horizons.items())),
            "labels_by_order_notional_usdt": dict(sorted(notionals.items())),
            "output_rows": {
                "features": feature_rows,
                "execution_labels": execution_label_rows,
            },
            "output_file_count": len(files),
            "output_fingerprint": output_fingerprint,
            "files": [
                item.to_dict() for item in sorted(files, key=lambda item: item.path)
            ],
        }
        _write_json_atomic(staging_path / "manifest.json", manifest)
        os.replace(staging_path, final_path)
        LOGGER.info(
            "Execution research dataset ready at %s (%d features, %d labels)",
            final_path,
            feature_rows,
            execution_label_rows,
        )
        return _manifest_result(final_path, manifest, reused=False)
    except Exception:
        if staging_path.is_dir() and staging_path.parent == destination_root:
            shutil.rmtree(staging_path, ignore_errors=True)
        raise


def build_execution_research_dataset(
    canonical_dataset: str | Path,
    output_root: str | Path,
    *,
    parameters: ExecutionResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ExecutionResearchBuildResult:
    """Build execution-aware labels from one canonical dataset."""

    return _build_execution_research(
        _load_canonical_source(canonical_dataset, allow_gapped=True),
        output_root,
        parameters=parameters,
        minimum_free_bytes=minimum_free_bytes,
    )


def build_execution_research_dataset_from_catalog(
    archive_catalog: str | Path,
    output_root: str | Path,
    *,
    parameters: ExecutionResearchParameters | None = None,
    minimum_free_bytes: int = 0,
) -> ExecutionResearchBuildResult:
    """Build execution-aware labels from consecutive immutable archive days."""

    return _build_execution_research(
        _load_archive_catalog_source(archive_catalog, allow_gapped=True),
        output_root,
        parameters=parameters,
        minimum_free_bytes=minimum_free_bytes,
    )
