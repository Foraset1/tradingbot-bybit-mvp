from __future__ import annotations

import ctypes
import gc
import logging
import sys
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from tradingbot.research.evaluation_contracts import (
    OUTCOME_NAMES,
    EvaluationError,
    EvaluationParameters,
)

LOGGER = logging.getLogger(__name__)

TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
PRIOR_WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75)


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    name: str
    calibration_probabilities: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    model_text: str | None
    feature_importance: NDArray[np.float64] | None
    training_rows_available: int
    training_rows_used: int


def _resident_memory_mib() -> float | None:
    """Return current Linux RSS for operational telemetry when available."""

    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def release_unused_process_memory() -> None:
    """Return released native allocations to Linux between large model fits."""

    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        trim = cast(Any, ctypes.CDLL(None).malloc_trim)
        trim(0)
    except (AttributeError, OSError):
        return


def _time_uniform_training_sample(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    *,
    maximum_rows: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Bound the baseline fit deterministically while retaining the full time span."""

    if maximum_rows <= 0:
        raise EvaluationError("logistic maximum training rows must be positive")
    if len(x_train) != len(y_train):
        raise EvaluationError("training feature and target rows do not match")
    if maximum_rows < len(OUTCOME_NAMES):
        raise EvaluationError(
            "logistic maximum training rows must cover every outcome"
        )
    if len(y_train) <= maximum_rows:
        return x_train, y_train
    indices = np.linspace(
        0,
        len(y_train) - 1,
        num=maximum_rows,
        dtype=np.int64,
    )
    sampled_outcomes = set(int(value) for value in y_train[indices])
    for outcome_index in range(len(OUTCOME_NAMES)):
        if outcome_index in sampled_outcomes:
            continue
        matching = np.flatnonzero(y_train == outcome_index)
        if not len(matching):
            raise EvaluationError("logistic training rows do not contain every outcome")
        indices[outcome_index] = int(matching[len(matching) // 2])
    indices.sort()
    return x_train[indices], y_train[indices]


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    temperature: float
    prior_weight: float
    class_prior: NDArray[np.float64]
    calibration_log_loss_before: float
    calibration_log_loss_after: float

    def transform(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _calibrated_probabilities(
            probabilities,
            temperature=self.temperature,
            prior_weight=self.prior_weight,
            class_prior=self.class_prior,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "temperature_plus_prior_shrinkage_grid",
            "temperature": self.temperature,
            "prior_weight": self.prior_weight,
            "class_prior": {
                name: float(self.class_prior[index])
                for index, name in enumerate(OUTCOME_NAMES)
            },
            "calibration_log_loss_before": self.calibration_log_loss_before,
            "calibration_log_loss_after": self.calibration_log_loss_after,
        }


def _aligned_probabilities(
    probabilities: NDArray[np.float64], classes: NDArray[np.int64]
) -> NDArray[np.float64]:
    aligned = np.zeros((probabilities.shape[0], len(OUTCOME_NAMES)), dtype=np.float64)
    for source_column, outcome_index in enumerate(classes):
        if not 0 <= int(outcome_index) < len(OUTCOME_NAMES):
            raise EvaluationError(f"model returned an unknown class: {outcome_index}")
        aligned[:, int(outcome_index)] = probabilities[:, source_column]
    return _normalized_probabilities(aligned)


def _normalized_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(OUTCOME_NAMES):
        raise EvaluationError("model returned an invalid probability shape")
    row_sums = np.sum(probabilities, axis=1)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(row_sums <= 0)
    ):
        raise EvaluationError("model returned invalid class probabilities")
    return probabilities / row_sums[:, None]


def _multiclass_log_loss(
    y_true: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    clipped = np.clip(_normalized_probabilities(probabilities), 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def _calibrated_probabilities(
    probabilities: NDArray[np.float64],
    *,
    temperature: float,
    prior_weight: float,
    class_prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    if temperature <= 0 or not 0 <= prior_weight < 1:
        raise EvaluationError("probability calibrator parameters are invalid")
    normalized = np.clip(_normalized_probabilities(probabilities), 1e-12, 1.0)
    logits = np.log(normalized) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= np.sum(scaled, axis=1, keepdims=True)
    blended = (1.0 - prior_weight) * scaled + prior_weight * class_prior[None, :]
    return _normalized_probabilities(blended)


def fit_probability_calibrator(
    probabilities: NDArray[np.float64], y_true: NDArray[np.int64]
) -> ProbabilityCalibrator:
    """Fit a deterministic calibrator on a dedicated historical window only."""

    if len(probabilities) != len(y_true) or len(y_true) == 0:
        raise EvaluationError("probability calibration inputs are incompatible")
    if len(np.unique(y_true)) != len(OUTCOME_NAMES):
        raise EvaluationError("probability calibration requires every outcome")
    counts = np.bincount(y_true, minlength=len(OUTCOME_NAMES)).astype(np.float64)
    # One pseudo-count avoids a zero prior without materially changing a real fold.
    class_prior = (counts + 1.0) / (np.sum(counts) + len(OUTCOME_NAMES))
    before = _multiclass_log_loss(y_true, probabilities)
    best_loss = before
    best_temperature = 1.0
    best_prior_weight = 0.0
    for temperature in TEMPERATURE_GRID:
        for prior_weight in PRIOR_WEIGHT_GRID:
            candidate = _calibrated_probabilities(
                probabilities,
                temperature=temperature,
                prior_weight=prior_weight,
                class_prior=class_prior,
            )
            loss = _multiclass_log_loss(y_true, candidate)
            if loss < best_loss - 1e-15:
                best_loss = loss
                best_temperature = temperature
                best_prior_weight = prior_weight
    return ProbabilityCalibrator(
        temperature=best_temperature,
        prior_weight=best_prior_weight,
        class_prior=class_prior,
        calibration_log_loss_before=before,
        calibration_log_loss_after=best_loss,
    )


def _prior_prediction(
    y_train: NDArray[np.int64], calibration_rows: int, test_rows: int
) -> ModelPrediction:
    counts = np.bincount(y_train, minlength=len(OUTCOME_NAMES)).astype(np.float64)
    probabilities = counts / np.sum(counts)
    return ModelPrediction(
        name="class_prior",
        calibration_probabilities=np.repeat(
            probabilities[None, :], calibration_rows, axis=0
        ),
        probabilities=np.repeat(probabilities[None, :], test_rows, axis=0),
        model_text=None,
        feature_importance=None,
        training_rows_available=len(y_train),
        training_rows_used=len(y_train),
    )


def _logistic_prediction(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_calibration: NDArray[np.float32],
    x_test: NDArray[np.float32],
    *,
    seed: int,
    maximum_training_rows: int,
    training_threads: int,
) -> ModelPrediction:
    available_rows = len(y_train)
    x_fit, y_fit = _time_uniform_training_sample(
        x_train,
        y_train,
        maximum_rows=maximum_training_rows,
    )
    LOGGER.info(
        "Fitting logistic baseline on %d/%d time-uniform rows (RSS %.1f MiB)",
        len(y_fit),
        available_rows,
        _resident_memory_mib() or -1.0,
    )
    pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
    with threadpool_limits(limits=training_threads):
        pipeline.fit(x_fit, y_fit)
    classifier = cast(Any, pipeline.named_steps["classifier"])
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    calibration_raw = np.asarray(
        pipeline.predict_proba(x_calibration), dtype=np.float64
    )
    test_raw = np.asarray(pipeline.predict_proba(x_test), dtype=np.float64)
    LOGGER.info(
        "Logistic baseline ready (RSS %.1f MiB)",
        _resident_memory_mib() or -1.0,
    )
    return ModelPrediction(
        name="logistic",
        calibration_probabilities=_aligned_probabilities(calibration_raw, classes),
        probabilities=_aligned_probabilities(test_raw, classes),
        model_text=None,
        feature_importance=None,
        training_rows_available=available_rows,
        training_rows_used=len(y_fit),
    )


def _lightgbm_prediction(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_calibration: NDArray[np.float32],
    x_test: NDArray[np.float32],
    parameters: EvaluationParameters,
) -> ModelPrediction:
    LOGGER.info(
        "Fitting LightGBM on %d rows with %d threads (RSS %.1f MiB)",
        len(y_train),
        parameters.training_threads,
        _resident_memory_mib() or -1.0,
    )
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
    with threadpool_limits(limits=parameters.training_threads):
        model.fit(x_train, y_train)
    classes = np.asarray(model.classes_, dtype=np.int64)
    calibration_raw = np.asarray(
        model.predict_proba(x_calibration), dtype=np.float64
    )
    test_raw = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    LOGGER.info("LightGBM ready (RSS %.1f MiB)", _resident_memory_mib() or -1.0)
    booster = model.booster_
    return ModelPrediction(
        name="lightgbm",
        calibration_probabilities=_aligned_probabilities(calibration_raw, classes),
        probabilities=_aligned_probabilities(test_raw, classes),
        model_text=booster.model_to_string(),
        feature_importance=np.asarray(
            booster.feature_importance(importance_type="gain"), dtype=np.float64
        ),
        training_rows_available=len(y_train),
        training_rows_used=len(y_train),
    )


def fit_fold_models(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_calibration: NDArray[np.float32],
    x_test: NDArray[np.float32],
    parameters: EvaluationParameters,
) -> tuple[ModelPrediction, ...]:
    if len(np.unique(y_train)) != len(OUTCOME_NAMES):
        raise EvaluationError("training rows must contain all supported outcomes")
    if len(x_calibration) == 0 or len(x_test) == 0:
        raise EvaluationError("calibration and test matrices must be non-empty")
    LOGGER.info(
        "Fitting baselines and LightGBM on %d rows; calibrating on %d; predicting %d",
        len(y_train),
        len(x_calibration),
        len(x_test),
    )
    prior = _prior_prediction(y_train, len(x_calibration), len(x_test))
    logistic = _logistic_prediction(
        x_train,
        y_train,
        x_calibration,
        x_test,
        seed=parameters.random_seed,
        maximum_training_rows=parameters.logistic_max_training_rows,
        training_threads=parameters.training_threads,
    )
    release_unused_process_memory()
    lightgbm_prediction = _lightgbm_prediction(
        x_train,
        y_train,
        x_calibration,
        x_test,
        parameters,
    )
    release_unused_process_memory()
    return prior, logistic, lightgbm_prediction


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
        "mean_probabilities": {
            name: float(np.mean(clipped[:, index]))
            for index, name in enumerate(OUTCOME_NAMES)
        },
        "observed_fractions": {
            name: float(class_counts[index] / len(y_true))
            for index, name in enumerate(OUTCOME_NAMES)
        },
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
