from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa  # type: ignore[import-untyped]

RESEARCH_SCHEMA_VERSION: Final = 1
PARQUET_FORMAT_VERSION: Final = "2.6"
PARQUET_COMPRESSION: Final = "zstd"
PARQUET_COMPRESSION_LEVEL: Final = 3

BOOK_DEPTH_LEVELS: Final = (1, 5, 10, 25, 50)
KLINE_RETURN_WINDOWS_MINUTES: Final = (1, 3, 5, 15, 60)
KLINE_VOLATILITY_WINDOWS_MINUTES: Final = (5, 15, 60)
TRADE_WINDOWS_SECONDS: Final = (5, 30, 60, 300, 900)
DEFAULT_LABEL_HORIZONS_MINUTES: Final = (5, 15, 30, 60)


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
