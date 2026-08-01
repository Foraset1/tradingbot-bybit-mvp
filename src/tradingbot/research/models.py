from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from tradingbot.research.evaluation_contracts import (
    OUTCOME_NAMES,
    EvaluationError,
    EvaluationParameters,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    name: str
    probabilities: NDArray[np.float64]
    model_text: str | None
    feature_importance: NDArray[np.float64] | None


def _aligned_probabilities(
    probabilities: NDArray[np.float64], classes: NDArray[np.int64]
) -> NDArray[np.float64]:
    aligned = np.zeros((probabilities.shape[0], len(OUTCOME_NAMES)), dtype=np.float64)
    for source_column, outcome_index in enumerate(classes):
        if not 0 <= int(outcome_index) < len(OUTCOME_NAMES):
            raise EvaluationError(f"model returned an unknown class: {outcome_index}")
        aligned[:, int(outcome_index)] = probabilities[:, source_column]
    row_sums = np.sum(aligned, axis=1)
    if np.any(~np.isfinite(aligned)) or np.any(aligned < 0) or np.any(row_sums <= 0):
        raise EvaluationError("model returned invalid class probabilities")
    return aligned / row_sums[:, None]


def _prior_prediction(
    y_train: NDArray[np.int64], test_rows: int
) -> ModelPrediction:
    counts = np.bincount(y_train, minlength=len(OUTCOME_NAMES)).astype(np.float64)
    probabilities = counts / np.sum(counts)
    return ModelPrediction(
        name="class_prior",
        probabilities=np.repeat(probabilities[None, :], test_rows, axis=0),
        model_text=None,
        feature_importance=None,
    )


def _logistic_prediction(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    *,
    seed: int,
) -> ModelPrediction:
    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    max_iter=300,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    raw = np.asarray(pipeline.predict_proba(x_test), dtype=np.float64)
    classifier = cast(Any, pipeline.named_steps["classifier"])
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    return ModelPrediction(
        name="logistic",
        probabilities=_aligned_probabilities(raw, classes),
        model_text=None,
        feature_importance=None,
    )


def _lightgbm_prediction(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    parameters: EvaluationParameters,
) -> ModelPrediction:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(OUTCOME_NAMES),
        n_estimators=parameters.lightgbm_estimators,
        learning_rate=parameters.lightgbm_learning_rate,
        num_leaves=parameters.lightgbm_num_leaves,
        min_child_samples=parameters.lightgbm_min_child_samples,
        subsample=1.0,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=parameters.random_seed,
        n_jobs=parameters.training_threads,
        deterministic=True,
        force_col_wise=True,
        data_random_seed=parameters.random_seed,
        feature_fraction_seed=parameters.random_seed,
        bagging_seed=parameters.random_seed,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    raw = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    booster = model.booster_
    return ModelPrediction(
        name="lightgbm",
        probabilities=_aligned_probabilities(raw, classes),
        model_text=booster.model_to_string(),
        feature_importance=np.asarray(
            booster.feature_importance(importance_type="gain"), dtype=np.float64
        ),
    )


def fit_fold_models(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    parameters: EvaluationParameters,
) -> tuple[ModelPrediction, ...]:
    if len(np.unique(y_train)) != len(OUTCOME_NAMES):
        raise EvaluationError("training rows must contain all supported outcomes")
    LOGGER.info(
        "Fitting baselines and LightGBM on %d rows; predicting %d rows",
        len(y_train),
        len(x_test),
    )
    return (
        _prior_prediction(y_train, len(x_test)),
        _logistic_prediction(
            x_train,
            y_train,
            x_test,
            seed=parameters.random_seed,
        ),
        _lightgbm_prediction(x_train, y_train, x_test, parameters),
    )


def classification_metrics(
    y_true: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> dict[str, object]:
    if len(y_true) != len(probabilities) or probabilities.shape[1] != len(
        OUTCOME_NAMES
    ):
        raise EvaluationError("classification metric inputs have incompatible shapes")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    clipped /= np.sum(clipped, axis=1, keepdims=True)
    selected = clipped[np.arange(len(y_true)), y_true]
    log_loss = float(-np.mean(np.log(selected)))
    one_hot = np.eye(len(OUTCOME_NAMES), dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1)))
    predicted = np.argmax(clipped, axis=1)
    accuracy = float(np.mean(predicted == y_true))
    confidence = np.max(clipped, axis=1)
    correctness = (predicted == y_true).astype(np.float64)
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence <= upper) if upper >= 1.0 else (confidence < upper)
        )
        if np.any(mask):
            calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correctness[mask]))
            )
    class_counts = np.bincount(y_true, minlength=len(OUTCOME_NAMES))
    return {
        "rows": len(y_true),
        "log_loss": log_loss,
        "multiclass_brier": brier,
        "accuracy": accuracy,
        "expected_calibration_error_10_bins": calibration_error,
        "outcomes": {
            name: int(class_counts[index]) for index, name in enumerate(OUTCOME_NAMES)
        },
    }


def timeout_return_estimate(
    *,
    y_train: NDArray[np.int64],
    returns_train: NDArray[np.float64],
    symbols_train: NDArray[np.int16],
    sides_train: NDArray[np.int8],
    symbols_test: NDArray[np.int16],
    sides_test: NDArray[np.int8],
) -> NDArray[np.float64]:
    timeout_index = OUTCOME_NAMES.index("TIMEOUT")
    timeout_mask = y_train == timeout_index
    if not np.any(timeout_mask):
        raise EvaluationError("training fold has no TIMEOUT return observations")
    global_mean = float(np.mean(returns_train[timeout_mask]))
    side_means: dict[int, float] = {}
    group_means: dict[tuple[int, int], float] = {}
    for side in (-1, 1):
        mask = timeout_mask & (sides_train == side)
        if np.any(mask):
            side_means[side] = float(np.mean(returns_train[mask]))
    for symbol in np.unique(symbols_train):
        for side in (-1, 1):
            mask = timeout_mask & (symbols_train == symbol) & (sides_train == side)
            if np.count_nonzero(mask) >= 20:
                group_means[(int(symbol), side)] = float(np.mean(returns_train[mask]))
    result = np.empty(len(symbols_test), dtype=np.float64)
    for index, (symbol, side) in enumerate(zip(symbols_test, sides_test, strict=True)):
        result[index] = group_means.get(
            (int(symbol), int(side)), side_means.get(int(side), global_mean)
        )
    return result
