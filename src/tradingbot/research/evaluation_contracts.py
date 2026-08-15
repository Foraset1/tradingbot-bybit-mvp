from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

EVALUATION_SCHEMA_VERSION: Final = 2
NS_PER_MINUTE: Final = 60 * 1_000_000_000
NS_PER_DAY: Final = 24 * 60 * NS_PER_MINUTE

OUTCOME_NAMES: Final = ("SL_FIRST", "TIMEOUT", "TP_FIRST")
OUTCOME_TO_INDEX: Final = {name: index for index, name in enumerate(OUTCOME_NAMES)}

FEATURE_PROFILES: Final = ("full", "no_calendar")
CALENDAR_FEATURE_NAMES: Final = frozenset(
    {
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_weekday_sin",
        "utc_weekday_cos",
    }
)

DIRECT_FEATURE_COLUMNS: Final = (
    "book_age_ms",
    "ticker_age_ms",
    "trade_age_ms",
    "spread_bps",
    "microprice_offset_bps",
    "book_imbalance_1",
    "book_imbalance_5",
    "book_imbalance_10",
    "book_imbalance_25",
    "book_imbalance_50",
    "return_1m_fraction",
    "return_3m_fraction",
    "return_5m_fraction",
    "return_15m_fraction",
    "return_60m_fraction",
    "realized_volatility_5m_fraction",
    "realized_volatility_15m_fraction",
    "realized_volatility_60m_fraction",
    "atr_14_bps",
    "range_1m_bps",
    "volume_ratio_5m_to_60m",
    "mark_index_basis_bps",
    "open_interest_change_5m_fraction",
    "open_interest_change_15m_fraction",
    "funding_rate",
    "minutes_to_funding",
    "trade_imbalance_5s",
    "trade_return_5s_fraction",
    "trade_imbalance_30s",
    "trade_return_30s_fraction",
    "trade_imbalance_1m",
    "trade_return_1m_fraction",
    "trade_imbalance_5m",
    "trade_return_5m_fraction",
    "trade_imbalance_15m",
    "trade_return_15m_fraction",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_weekday_sin",
    "utc_weekday_cos",
    "btc_return_5m_fraction",
    "btc_return_15m_fraction",
    "btc_return_60m_fraction",
    "btc_realized_volatility_15m_fraction",
    "btc_trade_imbalance_60s",
    "btc_spread_bps",
    "relative_return_5m_fraction",
    "relative_return_15m_fraction",
    "relative_return_60m_fraction",
)

LOG1P_FEATURE_COLUMNS: Final = (
    "best_bid_size",
    "best_ask_size",
    "bid_depth_1",
    "ask_depth_1",
    "depth_notional_1",
    "bid_depth_5",
    "ask_depth_5",
    "depth_notional_5",
    "bid_depth_10",
    "ask_depth_10",
    "depth_notional_10",
    "bid_depth_25",
    "ask_depth_25",
    "depth_notional_25",
    "bid_depth_50",
    "ask_depth_50",
    "depth_notional_50",
    "open_interest",
    "trade_count_5s",
    "trade_base_volume_5s",
    "trade_notional_5s",
    "trade_count_30s",
    "trade_base_volume_30s",
    "trade_notional_30s",
    "trade_count_1m",
    "trade_base_volume_1m",
    "trade_notional_1m",
    "trade_count_5m",
    "trade_base_volume_5m",
    "trade_notional_5m",
    "trade_count_15m",
    "trade_base_volume_15m",
    "trade_notional_15m",
)

PRICE_DIRECT_FEATURE_COLUMNS: Final = (
    "minute_bar_age_ms",
    "trade_age_ms",
    "return_1m_fraction",
    "return_3m_fraction",
    "return_5m_fraction",
    "return_15m_fraction",
    "return_60m_fraction",
    "realized_volatility_5m_fraction",
    "realized_volatility_15m_fraction",
    "realized_volatility_60m_fraction",
    "atr_14_bps",
    "range_1m_bps",
    "volume_ratio_5m_to_60m",
    "trade_imbalance_5s",
    "trade_return_5s_fraction",
    "trade_imbalance_30s",
    "trade_return_30s_fraction",
    "trade_imbalance_1m",
    "trade_return_1m_fraction",
    "trade_imbalance_5m",
    "trade_return_5m_fraction",
    "trade_imbalance_15m",
    "trade_return_15m_fraction",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_weekday_sin",
    "utc_weekday_cos",
    "btc_return_5m_fraction",
    "btc_return_15m_fraction",
    "btc_return_60m_fraction",
    "btc_realized_volatility_15m_fraction",
    "btc_trade_imbalance_60s",
    "relative_return_5m_fraction",
    "relative_return_15m_fraction",
    "relative_return_60m_fraction",
)

PRICE_LOG1P_FEATURE_COLUMNS: Final = tuple(
    name
    for suffix in ("5s", "30s", "1m", "5m", "15m")
    for name in (
        f"trade_count_{suffix}",
        f"trade_base_volume_{suffix}",
        f"trade_notional_{suffix}",
    )
)


class EvaluationError(RuntimeError):
    """Raised when an offline evaluation would violate its integrity contract."""


@dataclass(frozen=True, slots=True)
class EvaluationParameters:
    horizon_minutes: int
    embargo_minutes: int
    minimum_train_days: int
    test_days: int
    maximum_folds: int
    acceptance_minimum_days: int
    minimum_train_rows: int
    minimum_test_rows: int
    calibration_days: int
    minimum_calibration_rows: int
    minimum_symbol_coverage_fraction: float
    maker_fee_bps: float
    taker_fee_bps: float
    entry_adverse_selection_bps: float
    stop_slippage_bps: float
    timeout_slippage_bps: float
    minimum_expected_net_bps: float
    lightgbm_estimators: int
    lightgbm_learning_rate: float
    lightgbm_num_leaves: int
    lightgbm_min_child_samples: int
    logistic_max_training_rows: int
    training_threads: int
    random_seed: int
    max_notional_fraction: float
    max_planned_risk_fraction: float
    rolling_24h_loss_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "embargo_minutes": self.embargo_minutes,
            "minimum_train_days": self.minimum_train_days,
            "test_days": self.test_days,
            "maximum_folds": self.maximum_folds,
            "acceptance_minimum_days": self.acceptance_minimum_days,
            "minimum_train_rows": self.minimum_train_rows,
            "minimum_test_rows": self.minimum_test_rows,
            "calibration_days": self.calibration_days,
            "minimum_calibration_rows": self.minimum_calibration_rows,
            "minimum_symbol_coverage_fraction": (
                self.minimum_symbol_coverage_fraction
            ),
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "entry_adverse_selection_bps": self.entry_adverse_selection_bps,
            "stop_slippage_bps": self.stop_slippage_bps,
            "timeout_slippage_bps": self.timeout_slippage_bps,
            "minimum_expected_net_bps": self.minimum_expected_net_bps,
            "lightgbm_estimators": self.lightgbm_estimators,
            "lightgbm_learning_rate": self.lightgbm_learning_rate,
            "lightgbm_num_leaves": self.lightgbm_num_leaves,
            "lightgbm_min_child_samples": self.lightgbm_min_child_samples,
            "logistic_max_training_rows": self.logistic_max_training_rows,
            "training_threads": self.training_threads,
            "random_seed": self.random_seed,
            "max_notional_fraction": self.max_notional_fraction,
            "max_planned_risk_fraction": self.max_planned_risk_fraction,
            "rolling_24h_loss_fraction": self.rolling_24h_loss_fraction,
        }


@dataclass(frozen=True, slots=True)
class ResearchDataset:
    root: Path
    research_dataset_id: str
    research_profile: str
    source_dataset_id: str
    input_fingerprint: str
    output_fingerprint: str
    symbols: tuple[str, ...]
    feature_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    feature_rows: int
    label_rows: int
    manifest: dict[str, Any]


@dataclass(slots=True)
class PreparedData:
    x: NDArray[np.float32]
    y: NDArray[np.int64]
    feature_names: tuple[str, ...]
    decision_ids: NDArray[np.bytes_]
    decision_at_ns: NDArray[np.int64]
    label_end_ns: NDArray[np.int64]
    hit_at_ns: NDArray[np.int64]
    symbol_codes: NDArray[np.int16]
    symbols: tuple[str, ...]
    side_codes: NDArray[np.int8]
    outcome_return_bps: NDArray[np.float64]
    stop_distance_bps: NDArray[np.float64]
    take_profit_distance_bps: NDArray[np.float64]
    funding_rate: NDArray[np.float64]
    minutes_to_funding: NDArray[np.float64]
    excluded_ambiguous_rows: int
    excluded_unpriced_rows: int

    @property
    def rows(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True, slots=True)
class TemporalFold:
    fold: int
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_start_ns: int
    train_end_ns: int
    test_start_ns: int
    test_end_ns: int
    mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "mode": self.mode,
            "train_rows": len(self.train_indices),
            "test_rows": len(self.test_indices),
            "train_start_ns": self.train_start_ns,
            "train_end_ns": self.train_end_ns,
            "test_start_ns": self.test_start_ns,
            "test_end_ns": self.test_end_ns,
        }


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    fit_indices: NDArray[np.int64]
    calibration_indices: NDArray[np.int64]
    calibration_start_ns: int
    calibration_end_ns: int
    fit_purge_cutoff_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_rows": len(self.fit_indices),
            "calibration_rows": len(self.calibration_indices),
            "calibration_start_ns": self.calibration_start_ns,
            "calibration_end_ns": self.calibration_end_ns,
            "fit_purge_cutoff_ns": self.fit_purge_cutoff_ns,
        }


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    model_name: str
    fold: int
    row_indices: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    expected_net_bps: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    experiment_id: str
    experiment_path: Path
    manifest_path: Path
    report_path: Path
    data_mode: str
    data_span_days: float
    folds: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "experiment_path": self.experiment_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "report_path": self.report_path.as_posix(),
            "data_mode": self.data_mode,
            "data_span_days": self.data_span_days,
            "folds": self.folds,
            "reused": self.reused,
        }
