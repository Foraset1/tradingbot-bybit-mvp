from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tradingbot.research.evaluation_contracts import (
    NS_PER_DAY,
    NS_PER_MINUTE,
    CalibrationSplit,
    EvaluationError,
    EvaluationParameters,
    TemporalFold,
)
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_OUTCOME_NAMES,
    FILL_NAMES,
    ExecutionPreparedData,
)


def _require_classes(
    data: ExecutionPreparedData,
    indices: NDArray[np.int64],
    *,
    label: str,
) -> None:
    if len(np.unique(data.fill_y[indices])) != len(FILL_NAMES):
        raise EvaluationError(f"{label} does not contain every fill class")
    full = indices[data.outcome_y[indices] >= 0]
    if len(np.unique(data.outcome_y[full])) != len(EXECUTION_OUTCOME_NAMES):
        raise EvaluationError(f"{label} does not contain every execution outcome")


def build_execution_calibration_split(
    data: ExecutionPreparedData,
    fold: TemporalFold,
    parameters: EvaluationParameters,
) -> CalibrationSplit:
    """Reserve one strictly older, purged calibration window for both models."""

    outer_train = fold.train_indices
    unique_train_decisions = np.unique(data.decision_at_ns[outer_train])
    if len(unique_train_decisions) < 10:
        raise EvaluationError(f"fold {fold.fold} has too few decisions for calibration")
    outer_purge_cutoff_ns = (
        fold.test_start_ns - parameters.embargo_minutes * NS_PER_MINUTE
    )
    calibration_start_ns = (
        outer_purge_cutoff_ns - parameters.calibration_days * NS_PER_DAY
    )
    if calibration_start_ns <= int(unique_train_decisions[0]):
        if fold.mode != "technical_smoke":
            raise EvaluationError(
                f"fold {fold.fold} cannot reserve {parameters.calibration_days} "
                "calibration days"
            )
        offset = min(
            len(unique_train_decisions) - 1,
            max(1, int(len(unique_train_decisions) * 0.70)),
        )
        calibration_start_ns = int(unique_train_decisions[offset])

    fit_purge_cutoff_ns = (
        calibration_start_ns - parameters.embargo_minutes * NS_PER_MINUTE
    )
    fit_indices = outer_train[
        data.label_end_ns[outer_train] <= fit_purge_cutoff_ns
    ].astype(np.int64, copy=False)
    calibration_indices = outer_train[
        data.decision_at_ns[outer_train] >= calibration_start_ns
    ].astype(np.int64, copy=False)
    minimum_fit_rows = (
        parameters.minimum_train_rows
        if fold.mode == "walk_forward"
        else max(len(FILL_NAMES) * 20, parameters.minimum_train_rows // 4)
    )
    if len(fit_indices) < minimum_fit_rows:
        raise EvaluationError(
            f"fold {fold.fold} has only {len(fit_indices)} calibration-fit rows; "
            f"{minimum_fit_rows} required"
        )
    if len(calibration_indices) < parameters.minimum_calibration_rows:
        raise EvaluationError(
            f"fold {fold.fold} has only {len(calibration_indices)} calibration rows; "
            f"{parameters.minimum_calibration_rows} required"
        )
    _require_classes(data, fit_indices, label=f"fold {fold.fold} calibration-fit")
    _require_classes(
        data,
        calibration_indices,
        label=f"fold {fold.fold} calibration",
    )
    if np.intersect1d(fit_indices, calibration_indices).size:
        raise AssertionError("execution fit and calibration rows overlap")
    if int(np.max(data.label_end_ns[fit_indices])) > fit_purge_cutoff_ns:
        raise AssertionError("execution fit labels cross the inner purge boundary")
    if int(np.max(data.label_end_ns[calibration_indices])) > outer_purge_cutoff_ns:
        raise AssertionError(
            "execution calibration labels cross the outer purge boundary"
        )
    return CalibrationSplit(
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        calibration_start_ns=calibration_start_ns,
        calibration_end_ns=int(
            np.max(data.decision_at_ns[calibration_indices])
        ),
        fit_purge_cutoff_ns=fit_purge_cutoff_ns,
    )


def _fold_for_window(
    data: ExecutionPreparedData,
    parameters: EvaluationParameters,
    *,
    fold_number: int,
    test_start_ns: int,
    test_end_ns: int,
    mode: str,
) -> TemporalFold:
    purge_cutoff_ns = test_start_ns - parameters.embargo_minutes * NS_PER_MINUTE
    train_indices = np.flatnonzero(data.label_end_ns <= purge_cutoff_ns).astype(
        np.int64
    )
    test_indices = np.flatnonzero(
        (data.decision_at_ns >= test_start_ns)
        & (data.decision_at_ns < test_end_ns)
    ).astype(np.int64)
    if len(train_indices) < parameters.minimum_train_rows:
        raise EvaluationError(
            f"fold {fold_number} has only {len(train_indices)} train rows; "
            f"{parameters.minimum_train_rows} required"
        )
    if len(test_indices) < parameters.minimum_test_rows:
        raise EvaluationError(
            f"fold {fold_number} has only {len(test_indices)} test rows; "
            f"{parameters.minimum_test_rows} required"
        )
    _require_classes(data, train_indices, label=f"fold {fold_number} train")
    if int(np.max(data.label_end_ns[train_indices])) > purge_cutoff_ns:
        raise AssertionError("purged execution labels cross the fold boundary")
    if int(np.min(data.decision_at_ns[test_indices])) < test_start_ns:
        raise AssertionError("execution test rows start before their fold")
    return TemporalFold(
        fold=fold_number,
        train_indices=train_indices,
        test_indices=test_indices,
        train_start_ns=int(np.min(data.decision_at_ns[train_indices])),
        train_end_ns=int(np.max(data.decision_at_ns[train_indices])),
        test_start_ns=test_start_ns,
        test_end_ns=test_end_ns,
        mode=mode,
    )


def build_execution_temporal_folds(
    data: ExecutionPreparedData,
    parameters: EvaluationParameters,
) -> tuple[TemporalFold, ...]:
    """Create expanding UTC folds without splitting a decision timestamp."""

    if data.rows == 0:
        raise EvaluationError("cannot split an empty execution dataset")
    first_ns = int(np.min(data.decision_at_ns))
    end_ns = int(np.max(data.decision_at_ns)) + NS_PER_MINUTE
    span_ns = end_ns - first_ns
    normal_requirement_ns = (
        parameters.minimum_train_days + parameters.test_days
    ) * NS_PER_DAY
    candidate_windows: list[tuple[int, int]] = []
    if span_ns >= normal_requirement_ns:
        test_start_ns = first_ns + parameters.minimum_train_days * NS_PER_DAY
        test_width_ns = parameters.test_days * NS_PER_DAY
        while test_start_ns + test_width_ns <= end_ns:
            candidate_windows.append((test_start_ns, test_start_ns + test_width_ns))
            test_start_ns += test_width_ns
        candidate_windows = candidate_windows[-parameters.maximum_folds :]

    if candidate_windows:
        folds = tuple(
            _fold_for_window(
                data,
                parameters,
                fold_number=index,
                test_start_ns=start,
                test_end_ns=end,
                mode="walk_forward",
            )
            for index, (start, end) in enumerate(candidate_windows, start=1)
        )
    else:
        unique_decisions = np.unique(data.decision_at_ns)
        if len(unique_decisions) < 10:
            raise EvaluationError("at least ten execution decisions are required")
        test_offset = min(
            len(unique_decisions) - 1,
            int(len(unique_decisions) * 0.70),
        )
        folds = (
            _fold_for_window(
                data,
                parameters,
                fold_number=1,
                test_start_ns=int(unique_decisions[test_offset]),
                test_end_ns=end_ns,
                mode="technical_smoke",
            ),
        )

    seen_test_rows: set[int] = set()
    for fold in folds:
        overlap = seen_test_rows.intersection(
            int(value) for value in fold.test_indices
        )
        if overlap:
            raise AssertionError("execution temporal test folds overlap")
        seen_test_rows.update(int(value) for value in fold.test_indices)
    return folds
