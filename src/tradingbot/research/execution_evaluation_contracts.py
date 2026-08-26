from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

EXECUTION_EVALUATION_SCHEMA_VERSION: Final = 1

FILL_NAMES: Final = ("NO_FILL", "PARTIAL_FILL", "FULL_FILL")
FILL_TO_INDEX: Final = {name: index for index, name in enumerate(FILL_NAMES)}

EXECUTION_OUTCOME_NAMES: Final = ("SL_FIRST", "TIMEOUT", "TP_FIRST")
EXECUTION_OUTCOME_TO_INDEX: Final = {
    name: index for index, name in enumerate(EXECUTION_OUTCOME_NAMES)
}


@dataclass(frozen=True, slots=True)
class ExecutionResearchDataset:
    root: Path
    execution_dataset_id: str
    source_dataset_id: str
    input_fingerprint: str
    output_fingerprint: str
    symbols: tuple[str, ...]
    partition_dates: tuple[str, ...]
    feature_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    feature_rows: int
    label_rows: int
    manifest: dict[str, Any]


@dataclass(slots=True)
class ExecutionPreparedData:
    x: NDArray[np.float32]
    feature_names: tuple[str, ...]
    decision_ids: NDArray[np.bytes_]
    decision_at_ns: NDArray[np.int64]
    label_end_ns: NDArray[np.int64]
    entry_window_end_ns: NDArray[np.int64]
    position_end_ns: NDArray[np.int64]
    hit_at_ns: NDArray[np.int64]
    full_fill_at_ns: NDArray[np.int64]
    symbol_codes: NDArray[np.int16]
    symbols: tuple[str, ...]
    side_codes: NDArray[np.int8]
    fill_y: NDArray[np.int64]
    outcome_y: NDArray[np.int64]
    fill_fraction: NDArray[np.float64]
    outcome_return_bps: NDArray[np.float64]
    stop_distance_bps: NDArray[np.float64]
    take_profit_distance_bps: NDArray[np.float64]
    funding_rate: NDArray[np.float64]
    minutes_to_funding: NDArray[np.float64]
    activation_delay_ms: NDArray[np.float64]
    time_to_full_fill_ms: NDArray[np.float64]
    post_only_valid: NDArray[np.bool_]
    horizon_minutes: int
    order_notional_usdt: float
    excluded_ambiguous_full_fills: int
    excluded_unpriced_full_fills: int

    @property
    def rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def full_fill_mask(self) -> NDArray[np.bool_]:
        return np.asarray(
            self.fill_y == FILL_TO_INDEX["FULL_FILL"], dtype=np.bool_
        )


@dataclass(frozen=True, slots=True)
class ExecutionPredictionBatch:
    model_name: str
    fold: int
    row_indices: NDArray[np.int64]
    fill_probabilities: NDArray[np.float64]
    outcome_probabilities: NDArray[np.float64]
    expected_net_bps: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExecutionEvaluationResult:
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
            "execution_evaluation_schema_version": (
                EXECUTION_EVALUATION_SCHEMA_VERSION
            ),
            "experiment_id": self.experiment_id,
            "experiment_path": self.experiment_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "report_path": self.report_path.as_posix(),
            "data_mode": self.data_mode,
            "data_span_days": self.data_span_days,
            "folds": self.folds,
            "reused": self.reused,
        }
