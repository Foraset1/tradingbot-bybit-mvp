from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa  # type: ignore[import-untyped]

RESEARCH_SCHEMA_VERSION: Final = 1
MICROSTRUCTURE_RESEARCH_PROFILE: Final = "microstructure_research_v1"
PRICE_RESEARCH_PROFILE: Final = "price_futures_research_v1"
EXECUTION_RESEARCH_SCHEMA_VERSION: Final = 1
EXECUTION_RESEARCH_PROFILE: Final = "execution_microstructure_v1"
PARQUET_FORMAT_VERSION: Final = "2.6"
PARQUET_COMPRESSION: Final = "zstd"
PARQUET_COMPRESSION_LEVEL: Final = 3

BOOK_DEPTH_LEVELS: Final = (1, 5, 10, 25, 50)
KLINE_RETURN_WINDOWS_MINUTES: Final = (1, 3, 5, 15, 60)
KLINE_VOLATILITY_WINDOWS_MINUTES: Final = (5, 15, 60)
TRADE_WINDOWS_SECONDS: Final = (5, 30, 60, 300, 900)
DEFAULT_LABEL_HORIZONS_MINUTES: Final = (5, 15, 30, 60)
DEFAULT_EXECUTION_HORIZONS_MINUTES: Final = (15, 30)
DEFAULT_EXECUTION_ORDER_NOTIONALS_USDT: Final = (50.0, 100.0, 250.0, 500.0)


class ResearchBuildError(RuntimeError):
    """Raised when a causal research dataset cannot be built safely."""


@dataclass(frozen=True, slots=True)
class ResearchParameters:
    """Versioned parameters for the standard MVP research dataset."""

    decision_interval_seconds: int = 60
    decision_offset_seconds: int = 5
    kline_history_minutes: int = 60
    max_orderbook_age_ms: int = 2_500
    max_ticker_age_ms: int = 2_500
    label_horizons_minutes: tuple[int, ...] = DEFAULT_LABEL_HORIZONS_MINUTES
    volatility_lookback_minutes: int = 60
    stop_volatility_multiple: float = 1.0
    take_profit_multiple: float = 1.5
    minimum_stop_bps: float = 10.0
    maximum_stop_bps: float = 250.0

    def validate(self) -> None:
        if self.decision_interval_seconds <= 0:
            raise ResearchBuildError("decision_interval_seconds must be positive")
        if not 0 <= self.decision_offset_seconds < self.decision_interval_seconds:
            raise ResearchBuildError(
                "decision_offset_seconds must be within the decision interval"
            )
        if self.kline_history_minutes < max(KLINE_RETURN_WINDOWS_MINUTES):
            raise ResearchBuildError(
                "kline_history_minutes is too short for the feature contract"
            )
        if self.max_orderbook_age_ms <= 0 or self.max_ticker_age_ms <= 0:
            raise ResearchBuildError("staleness limits must be positive")
        if (
            not self.label_horizons_minutes
            or tuple(sorted(set(self.label_horizons_minutes)))
            != self.label_horizons_minutes
            or any(value <= 0 for value in self.label_horizons_minutes)
        ):
            raise ResearchBuildError(
                "label_horizons_minutes must be unique, positive, and sorted"
            )
        if self.volatility_lookback_minutes < 2:
            raise ResearchBuildError("volatility_lookback_minutes must be at least 2")
        if self.stop_volatility_multiple <= 0 or self.take_profit_multiple <= 0:
            raise ResearchBuildError("barrier volatility multiples must be positive")
        if (
            self.minimum_stop_bps <= 0
            or self.maximum_stop_bps < self.minimum_stop_bps
        ):
            raise ResearchBuildError("stop barrier bounds are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_interval_seconds": self.decision_interval_seconds,
            "decision_offset_seconds": self.decision_offset_seconds,
            "kline_history_minutes": self.kline_history_minutes,
            "max_orderbook_age_ms": self.max_orderbook_age_ms,
            "max_ticker_age_ms": self.max_ticker_age_ms,
            "label_horizons_minutes": list(self.label_horizons_minutes),
            "volatility_lookback_minutes": self.volatility_lookback_minutes,
            "stop_volatility_multiple": self.stop_volatility_multiple,
            "take_profit_multiple": self.take_profit_multiple,
            "minimum_stop_bps": self.minimum_stop_bps,
            "maximum_stop_bps": self.maximum_stop_bps,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResearchParameters:
    """Pre-registered assumptions for conservative maker execution labels."""

    decision_interval_seconds: int = 60
    decision_offset_seconds: int = 5
    kline_history_minutes: int = 60
    max_orderbook_age_ms: int = 2_500
    max_ticker_age_ms: int = 2_500
    position_horizons_minutes: tuple[int, ...] = DEFAULT_EXECUTION_HORIZONS_MINUTES
    volatility_lookback_minutes: int = 60
    stop_volatility_multiple: float = 1.0
    take_profit_multiple: float = 1.5
    minimum_stop_bps: float = 10.0
    maximum_stop_bps: float = 250.0
    order_notionals_usdt: tuple[float, ...] = (
        DEFAULT_EXECUTION_ORDER_NOTIONALS_USDT
    )
    submission_latency_ms: int = 250
    activation_max_delay_ms: int = 2_500
    entry_ttl_seconds: int = 30
    queue_ahead_multiplier: float = 1.0

    def feature_parameters(self) -> ResearchParameters:
        return ResearchParameters(
            decision_interval_seconds=self.decision_interval_seconds,
            decision_offset_seconds=self.decision_offset_seconds,
            kline_history_minutes=self.kline_history_minutes,
            max_orderbook_age_ms=self.max_orderbook_age_ms,
            max_ticker_age_ms=self.max_ticker_age_ms,
            label_horizons_minutes=self.position_horizons_minutes,
            volatility_lookback_minutes=self.volatility_lookback_minutes,
            stop_volatility_multiple=self.stop_volatility_multiple,
            take_profit_multiple=self.take_profit_multiple,
            minimum_stop_bps=self.minimum_stop_bps,
            maximum_stop_bps=self.maximum_stop_bps,
        )

    def validate(self) -> None:
        self.feature_parameters().validate()
        if self.volatility_lookback_minutes != 60:
            raise ResearchBuildError(
                "execution volatility_lookback_minutes must remain 60 in schema v1"
            )
        if any(value > 60 for value in self.position_horizons_minutes):
            raise ResearchBuildError(
                "position_horizons_minutes cannot exceed the one-hour MVP limit"
            )
        if (
            not self.order_notionals_usdt
            or tuple(sorted(set(self.order_notionals_usdt)))
            != self.order_notionals_usdt
            or any(
                not math.isfinite(value) or value <= 0
                for value in self.order_notionals_usdt
            )
        ):
            raise ResearchBuildError(
                "order_notionals_usdt must be unique, positive, finite, and sorted"
            )
        if self.submission_latency_ms < 0:
            raise ResearchBuildError("submission_latency_ms must be non-negative")
        if self.activation_max_delay_ms <= 0:
            raise ResearchBuildError("activation_max_delay_ms must be positive")
        if self.entry_ttl_seconds <= 0:
            raise ResearchBuildError("entry_ttl_seconds must be positive")
        if (
            self.submission_latency_ms + self.activation_max_delay_ms
            >= self.entry_ttl_seconds * 1_000
        ):
            raise ResearchBuildError(
                "entry TTL must extend beyond submission plus activation delay"
            )
        if not math.isfinite(self.queue_ahead_multiplier) or (
            self.queue_ahead_multiplier < 1.0
        ):
            raise ResearchBuildError(
                "queue_ahead_multiplier must be finite and at least 1.0"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_interval_seconds": self.decision_interval_seconds,
            "decision_offset_seconds": self.decision_offset_seconds,
            "kline_history_minutes": self.kline_history_minutes,
            "max_orderbook_age_ms": self.max_orderbook_age_ms,
            "max_ticker_age_ms": self.max_ticker_age_ms,
            "position_horizons_minutes": list(self.position_horizons_minutes),
            "volatility_lookback_minutes": self.volatility_lookback_minutes,
            "stop_volatility_multiple": self.stop_volatility_multiple,
            "take_profit_multiple": self.take_profit_multiple,
            "minimum_stop_bps": self.minimum_stop_bps,
            "maximum_stop_bps": self.maximum_stop_bps,
            "order_notionals_usdt": list(self.order_notionals_usdt),
            "submission_latency_ms": self.submission_latency_ms,
            "activation_max_delay_ms": self.activation_max_delay_ms,
            "entry_ttl_seconds": self.entry_ttl_seconds,
            "queue_ahead_multiplier": self.queue_ahead_multiplier,
        }


@dataclass(frozen=True, slots=True)
class PriceResearchParameters:
    """Versioned parameters for research built from official trade-bar history."""

    decision_interval_seconds: int = 60
    decision_offset_seconds: int = 5
    kline_history_minutes: int = 60
    maximum_trade_age_ms: int = 10_000
    label_horizons_minutes: tuple[int, ...] = DEFAULT_LABEL_HORIZONS_MINUTES
    volatility_lookback_minutes: int = 60
    stop_volatility_multiple: float = 1.0
    take_profit_multiple: float = 1.5
    minimum_stop_bps: float = 10.0
    maximum_stop_bps: float = 250.0

    def validate(self) -> None:
        if self.decision_interval_seconds <= 0:
            raise ResearchBuildError("decision_interval_seconds must be positive")
        if not 0 <= self.decision_offset_seconds < self.decision_interval_seconds:
            raise ResearchBuildError(
                "decision_offset_seconds must be within the decision interval"
            )
        if self.kline_history_minutes < max(KLINE_RETURN_WINDOWS_MINUTES):
            raise ResearchBuildError(
                "kline_history_minutes is too short for the price feature contract"
            )
        if self.maximum_trade_age_ms <= 0:
            raise ResearchBuildError("maximum_trade_age_ms must be positive")
        if (
            not self.label_horizons_minutes
            or tuple(sorted(set(self.label_horizons_minutes)))
            != self.label_horizons_minutes
            or any(value <= 0 for value in self.label_horizons_minutes)
        ):
            raise ResearchBuildError(
                "label_horizons_minutes must be unique, positive, and sorted"
            )
        if not 2 <= self.volatility_lookback_minutes <= self.kline_history_minutes:
            raise ResearchBuildError(
                "volatility_lookback_minutes must be within the minute history"
            )
        if self.stop_volatility_multiple <= 0 or self.take_profit_multiple <= 0:
            raise ResearchBuildError("barrier volatility multiples must be positive")
        if self.minimum_stop_bps <= 0 or self.maximum_stop_bps < self.minimum_stop_bps:
            raise ResearchBuildError("stop barrier bounds are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_interval_seconds": self.decision_interval_seconds,
            "decision_offset_seconds": self.decision_offset_seconds,
            "kline_history_minutes": self.kline_history_minutes,
            "maximum_trade_age_ms": self.maximum_trade_age_ms,
            "label_horizons_minutes": list(self.label_horizons_minutes),
            "volatility_lookback_minutes": self.volatility_lookback_minutes,
            "stop_volatility_multiple": self.stop_volatility_multiple,
            "take_profit_multiple": self.take_profit_multiple,
            "minimum_stop_bps": self.minimum_stop_bps,
            "maximum_stop_bps": self.maximum_stop_bps,
        }


@dataclass(frozen=True, slots=True)
class ResearchFile:
    path: str
    table: str
    symbol: str
    date: str
    rows: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "table": self.table,
            "symbol": self.symbol,
            "date": self.date,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResearchBuildResult:
    research_dataset_id: str
    dataset_path: Path
    manifest_path: Path
    source_dataset_id: str
    source_output_fingerprint: str
    parameter_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    feature_rows: int
    label_rows: int
    output_files: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "research_dataset_id": self.research_dataset_id,
            "dataset_path": self.dataset_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "source_dataset_id": self.source_dataset_id,
            "source_output_fingerprint": self.source_output_fingerprint,
            "parameter_fingerprint": self.parameter_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "feature_rows": self.feature_rows,
            "label_rows": self.label_rows,
            "output_files": self.output_files,
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResearchBuildResult:
    execution_dataset_id: str
    dataset_path: Path
    manifest_path: Path
    source_dataset_id: str
    source_output_fingerprint: str
    parameter_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    feature_rows: int
    execution_label_rows: int
    output_files: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_research_schema_version": EXECUTION_RESEARCH_SCHEMA_VERSION,
            "research_profile": EXECUTION_RESEARCH_PROFILE,
            "execution_dataset_id": self.execution_dataset_id,
            "dataset_path": self.dataset_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "source_dataset_id": self.source_dataset_id,
            "source_output_fingerprint": self.source_output_fingerprint,
            "parameter_fingerprint": self.parameter_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "feature_rows": self.feature_rows,
            "execution_label_rows": self.execution_label_rows,
            "output_files": self.output_files,
            "reused": self.reused,
        }


_RESEARCH_METADATA: Final = {
    b"tradingbot.research_schema_version": str(RESEARCH_SCHEMA_VERSION).encode(
        "ascii"
    ),
    b"tradingbot.feature_cutoff": b"received_at_ns <= decision_at_ns",
    b"tradingbot.execution_labels": b"not_included",
}


def _required(name: str, data_type: pa.DataType) -> pa.Field:
    return pa.field(name, data_type, nullable=False)


_FEATURE_FIELDS: list[pa.Field] = [
    _required("research_schema_version", pa.int32()),
    _required("decision_id", pa.string()),
    _required("source_dataset_id", pa.string()),
    _required("symbol", pa.string()),
    _required("decision_at_ns", pa.int64()),
    _required("decision_at_ms", pa.int64()),
    _required("decision_utc_date", pa.string()),
    _required("book_received_at_ns", pa.int64()),
    _required("ticker_received_at_ns", pa.int64()),
    _required("latest_kline_received_at_ns", pa.int64()),
    pa.field("latest_trade_received_at_ns", pa.int64()),
    _required("book_age_ms", pa.float64()),
    _required("ticker_age_ms", pa.float64()),
    pa.field("trade_age_ms", pa.float64()),
    _required("reference_mid_price", pa.float64()),
    _required("best_bid_price", pa.float64()),
    _required("best_ask_price", pa.float64()),
    _required("best_bid_size", pa.float64()),
    _required("best_ask_size", pa.float64()),
    _required("spread_bps", pa.float64()),
    _required("microprice", pa.float64()),
    _required("microprice_offset_bps", pa.float64()),
]

for _level in BOOK_DEPTH_LEVELS:
    _FEATURE_FIELDS.extend(
        (
            _required(f"bid_depth_{_level}", pa.float64()),
            _required(f"ask_depth_{_level}", pa.float64()),
            _required(f"book_imbalance_{_level}", pa.float64()),
            _required(f"depth_notional_{_level}", pa.float64()),
        )
    )

_FEATURE_FIELDS.extend(
    _required(f"return_{window}m_fraction", pa.float64())
    for window in KLINE_RETURN_WINDOWS_MINUTES
)
_FEATURE_FIELDS.extend(
    _required(f"realized_volatility_{window}m_fraction", pa.float64())
    for window in KLINE_VOLATILITY_WINDOWS_MINUTES
)
_FEATURE_FIELDS.extend(
    (
        _required("close_price", pa.float64()),
        _required("atr_14_bps", pa.float64()),
        _required("range_1m_bps", pa.float64()),
        _required("volume_ratio_5m_to_60m", pa.float64()),
        pa.field("mark_price", pa.float64()),
        pa.field("index_price", pa.float64()),
        pa.field("mark_index_basis_bps", pa.float64()),
        pa.field("open_interest", pa.float64()),
        pa.field("open_interest_change_5m_fraction", pa.float64()),
        pa.field("open_interest_change_15m_fraction", pa.float64()),
        pa.field("funding_rate", pa.float64()),
        pa.field("minutes_to_funding", pa.float64()),
    )
)

for _seconds in TRADE_WINDOWS_SECONDS:
    _suffix = f"{_seconds}s" if _seconds < 60 else f"{_seconds // 60}m"
    _FEATURE_FIELDS.extend(
        (
            _required(f"trade_count_{_suffix}", pa.int64()),
            _required(f"trade_base_volume_{_suffix}", pa.float64()),
            _required(f"trade_notional_{_suffix}", pa.float64()),
            _required(f"trade_imbalance_{_suffix}", pa.float64()),
            _required(f"trade_return_{_suffix}_fraction", pa.float64()),
        )
    )

_FEATURE_FIELDS.extend(
    (
        _required("utc_hour_sin", pa.float64()),
        _required("utc_hour_cos", pa.float64()),
        _required("utc_weekday_sin", pa.float64()),
        _required("utc_weekday_cos", pa.float64()),
        pa.field("btc_return_5m_fraction", pa.float64()),
        pa.field("btc_return_15m_fraction", pa.float64()),
        pa.field("btc_return_60m_fraction", pa.float64()),
        pa.field("btc_realized_volatility_15m_fraction", pa.float64()),
        pa.field("btc_trade_imbalance_60s", pa.float64()),
        pa.field("btc_spread_bps", pa.float64()),
        pa.field("relative_return_5m_fraction", pa.float64()),
        pa.field("relative_return_15m_fraction", pa.float64()),
        pa.field("relative_return_60m_fraction", pa.float64()),
    )
)

FEATURE_SCHEMA: Final = pa.schema(_FEATURE_FIELDS, metadata=_RESEARCH_METADATA)

LABEL_SCHEMA: Final = pa.schema(
    (
        _required("research_schema_version", pa.int32()),
        _required("decision_id", pa.string()),
        _required("source_dataset_id", pa.string()),
        _required("symbol", pa.string()),
        _required("decision_at_ns", pa.int64()),
        _required("decision_utc_date", pa.string()),
        _required("side", pa.string()),
        _required("horizon_minutes", pa.int32()),
        _required("label_end_ns", pa.int64()),
        _required("entry_reference_price", pa.float64()),
        _required("stop_distance_bps", pa.float64()),
        _required("take_profit_distance_bps", pa.float64()),
        _required("stop_price", pa.float64()),
        _required("take_profit_price", pa.float64()),
        _required("outcome", pa.string()),
        pa.field("hit_at_ns", pa.int64()),
        pa.field("hit_event_ts_ms", pa.int64()),
        pa.field("hit_sequence", pa.int64()),
        pa.field("hit_trade_price", pa.float64()),
        pa.field("time_to_hit_ms", pa.float64()),
        pa.field("timeout_price", pa.float64()),
        pa.field("outcome_return_bps", pa.float64()),
        _required("future_trade_count", pa.int64()),
        _required("resolution", pa.string()),
    ),
    metadata=_RESEARCH_METADATA,
)


_PRICE_RESEARCH_METADATA: Final = {
    b"tradingbot.research_schema_version": str(RESEARCH_SCHEMA_VERSION).encode(
        "ascii"
    ),
    b"tradingbot.research_profile": PRICE_RESEARCH_PROFILE.encode("ascii"),
    b"tradingbot.feature_cutoff": b"available_at_ns <= decision_at_ns",
    b"tradingbot.execution_labels": b"not_included",
}

_PRICE_FEATURE_FIELDS: list[pa.Field] = [
    _required("research_schema_version", pa.int32()),
    _required("decision_id", pa.string()),
    _required("source_dataset_id", pa.string()),
    _required("symbol", pa.string()),
    _required("decision_at_ns", pa.int64()),
    _required("decision_at_ms", pa.int64()),
    _required("decision_utc_date", pa.string()),
    _required("latest_minute_bar_available_at_ns", pa.int64()),
    pa.field("latest_second_bar_available_at_ns", pa.int64()),
    _required("minute_bar_age_ms", pa.float64()),
    pa.field("trade_age_ms", pa.float64()),
    _required("reference_price", pa.float64()),
    _required("close_price", pa.float64()),
]

_PRICE_FEATURE_FIELDS.extend(
    _required(f"return_{window}m_fraction", pa.float64())
    for window in KLINE_RETURN_WINDOWS_MINUTES
)
_PRICE_FEATURE_FIELDS.extend(
    _required(f"realized_volatility_{window}m_fraction", pa.float64())
    for window in KLINE_VOLATILITY_WINDOWS_MINUTES
)
_PRICE_FEATURE_FIELDS.extend(
    (
        _required("atr_14_bps", pa.float64()),
        _required("range_1m_bps", pa.float64()),
        _required("volume_ratio_5m_to_60m", pa.float64()),
    )
)

for _seconds in TRADE_WINDOWS_SECONDS:
    _suffix = f"{_seconds}s" if _seconds < 60 else f"{_seconds // 60}m"
    _PRICE_FEATURE_FIELDS.extend(
        (
            _required(f"trade_count_{_suffix}", pa.int64()),
            _required(f"trade_base_volume_{_suffix}", pa.float64()),
            _required(f"trade_notional_{_suffix}", pa.float64()),
            _required(f"trade_imbalance_{_suffix}", pa.float64()),
            _required(f"trade_return_{_suffix}_fraction", pa.float64()),
        )
    )

_PRICE_FEATURE_FIELDS.extend(
    (
        _required("utc_hour_sin", pa.float64()),
        _required("utc_hour_cos", pa.float64()),
        _required("utc_weekday_sin", pa.float64()),
        _required("utc_weekday_cos", pa.float64()),
        pa.field("btc_return_5m_fraction", pa.float64()),
        pa.field("btc_return_15m_fraction", pa.float64()),
        pa.field("btc_return_60m_fraction", pa.float64()),
        pa.field("btc_realized_volatility_15m_fraction", pa.float64()),
        pa.field("btc_trade_imbalance_60s", pa.float64()),
        pa.field("relative_return_5m_fraction", pa.float64()),
        pa.field("relative_return_15m_fraction", pa.float64()),
        pa.field("relative_return_60m_fraction", pa.float64()),
    )
)

PRICE_FEATURE_SCHEMA: Final = pa.schema(
    _PRICE_FEATURE_FIELDS, metadata=_PRICE_RESEARCH_METADATA
)

PRICE_LABEL_SCHEMA: Final = pa.schema(
    tuple(LABEL_SCHEMA), metadata=_PRICE_RESEARCH_METADATA
)


_EXECUTION_RESEARCH_METADATA: Final = {
    b"tradingbot.execution_research_schema_version": str(
        EXECUTION_RESEARCH_SCHEMA_VERSION
    ).encode("ascii"),
    b"tradingbot.research_profile": EXECUTION_RESEARCH_PROFILE.encode("ascii"),
    b"tradingbot.feature_cutoff": b"received_at_ns <= decision_at_ns",
    b"tradingbot.execution_label_rule": (
        b"future orderbook activation and public trades after decision_at_ns"
    ),
}

EXECUTION_FEATURE_SCHEMA: Final = pa.schema(
    tuple(FEATURE_SCHEMA), metadata=_EXECUTION_RESEARCH_METADATA
)

EXECUTION_LABEL_SCHEMA: Final = pa.schema(
    (
        _required("execution_research_schema_version", pa.int32()),
        _required("decision_id", pa.string()),
        _required("source_dataset_id", pa.string()),
        _required("symbol", pa.string()),
        _required("decision_at_ns", pa.int64()),
        _required("decision_utc_date", pa.string()),
        _required("side", pa.string()),
        _required("horizon_minutes", pa.int32()),
        _required("order_notional_usdt", pa.float64()),
        _required("submitted_at_ns", pa.int64()),
        _required("activation_at_ns", pa.int64()),
        _required("activation_delay_ms", pa.float64()),
        _required("entry_window_end_ns", pa.int64()),
        _required("entry_limit_price", pa.float64()),
        _required("order_size_base", pa.float64()),
        _required("activation_best_bid_price", pa.float64()),
        _required("activation_best_ask_price", pa.float64()),
        _required("post_only_valid", pa.bool_()),
        pa.field("queue_ahead_size_base", pa.float64()),
        pa.field("queue_required_size_base", pa.float64()),
        _required("entry_window_trade_count", pa.int64()),
        _required("contra_trade_count", pa.int64()),
        _required("contra_volume_at_entry_price_base", pa.float64()),
        _required("fill_status", pa.string()),
        _required("fill_fraction", pa.float64()),
        _required("filled_size_base", pa.float64()),
        pa.field("first_fill_at_ns", pa.int64()),
        pa.field("full_fill_at_ns", pa.int64()),
        pa.field("full_fill_event_ts_ms", pa.int64()),
        pa.field("full_fill_sequence", pa.int64()),
        pa.field("full_fill_trade_price", pa.float64()),
        pa.field("time_to_full_fill_ms", pa.float64()),
        _required("stop_distance_bps", pa.float64()),
        _required("take_profit_distance_bps", pa.float64()),
        _required("stop_price", pa.float64()),
        _required("take_profit_price", pa.float64()),
        pa.field("position_end_ns", pa.int64()),
        _required("outcome", pa.string()),
        pa.field("hit_at_ns", pa.int64()),
        pa.field("hit_event_ts_ms", pa.int64()),
        pa.field("hit_sequence", pa.int64()),
        pa.field("hit_trade_price", pa.float64()),
        pa.field("time_from_fill_to_hit_ms", pa.float64()),
        pa.field("timeout_price", pa.float64()),
        pa.field("outcome_return_bps", pa.float64()),
        _required("future_trade_count", pa.int64()),
        _required("resolution", pa.string()),
    ),
    metadata=_EXECUTION_RESEARCH_METADATA,
)
