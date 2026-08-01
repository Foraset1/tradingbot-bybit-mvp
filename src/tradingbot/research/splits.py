from __future__ import annotations

import numpy as np

from tradingbot.research.evaluation_contracts import (
    NS_PER_DAY,
    NS_PER_MINUTE,
    OUTCOME_NAMES,
    EvaluationError,
    EvaluationParameters,
    PreparedData,
    TemporalFold,
)


def _fold_for_window(
    data: PreparedData,
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
    classes = np.unique(data.y[train_indices])
    if len(classes) != len(OUTCOME_NAMES):
        raise EvaluationError(
            f"fold {fold_number} train data does not contain all market outcomes"
        )
    if int(np.max(data.label_end_ns[train_indices])) > purge_cutoff_ns:
        raise AssertionError("purged train labels cross the fold boundary")
    if int(np.min(data.decision_at_ns[test_indices])) < test_start_ns:
        raise AssertionError("test rows start before their fold")
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


def build_temporal_folds(
    data: PreparedData, parameters: EvaluationParameters
) -> tuple[TemporalFold, ...]:
    """Create expanding UTC folds, falling back to an explicit smoke-only split."""

    if data.rows == 0:
        raise EvaluationError("cannot split an empty evaluation dataset")
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
            raise EvaluationError("at least ten decision timestamps are required")
        test_offset = min(len(unique_decisions) - 1, int(len(unique_decisions) * 0.70))
        test_start_ns = int(unique_decisions[test_offset])
        folds = (
            _fold_for_window(
                data,
                parameters,
                fold_number=1,
                test_start_ns=test_start_ns,
                test_end_ns=end_ns,
                mode="technical_smoke",
            ),
        )

    seen_test_rows: set[int] = set()
    for fold in folds:
        overlap = seen_test_rows.intersection(int(value) for value in fold.test_indices)
        if overlap:
            raise AssertionError("temporal test folds overlap")
        seen_test_rows.update(int(value) for value in fold.test_indices)
    return folds
