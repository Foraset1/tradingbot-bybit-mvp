from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

import lightgbm
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tradingbot import __version__
from tradingbot.config import AppConfig
from tradingbot.research.evaluation_contracts import (
    NS_PER_DAY,
    EvaluationError,
)
from tradingbot.research.evaluator import evaluation_parameters
from tradingbot.research.execution_backtest import (
    combine_execution_prediction_batches,
    expected_execution_net_bps,
    partial_fraction_estimate,
    run_execution_one_position_backtest,
    timeout_return_estimate,
)
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_EVALUATION_SCHEMA_VERSION,
    EXECUTION_OUTCOME_NAMES,
    FILL_NAMES,
    ExecutionEvaluationResult,
    ExecutionPredictionBatch,
    ExecutionPreparedData,
)
from tradingbot.research.execution_evaluation_dataset import (
    prepare_execution_evaluation_data,
    validate_execution_research_dataset,
)
from tradingbot.research.execution_splits import (
    build_execution_calibration_split,
    build_execution_temporal_folds,
)
from tradingbot.research.models import (
    ModelPrediction,
    classification_metrics,
    fit_fold_models,
    fit_probability_calibrator,
    release_unused_process_memory,
)

LOGGER = logging.getLogger(__name__)

EXECUTION_ATTEMPT_SCHEMA = pa.schema(
    [
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("fold", pa.int16(), nullable=False),
        pa.field("decision_at_ns", pa.int64(), nullable=False),
        pa.field("exit_at_ns", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("horizon_minutes", pa.int32(), nullable=False),
        pa.field("order_notional_usdt", pa.float64(), nullable=False),
        pa.field("fill_status", pa.string(), nullable=False),
        pa.field("fill_fraction", pa.float64(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("probability_no_fill", pa.float64(), nullable=False),
        pa.field("probability_partial_fill", pa.float64(), nullable=False),
        pa.field("probability_full_fill", pa.float64(), nullable=False),
        pa.field("probability_sl_first", pa.float64(), nullable=False),
        pa.field("probability_timeout", pa.float64(), nullable=False),
        pa.field("probability_tp_first", pa.float64(), nullable=False),
        pa.field("expected_net_bps", pa.float64(), nullable=False),
        pa.field("candidate_count", pa.int32(), nullable=False),
        pa.field("eligible_candidate_count", pa.int32(), nullable=False),
        pa.field("expected_margin_to_second_bps", pa.float64(), nullable=True),
        pa.field("gross_return_bps", pa.float64(), nullable=False),
        pa.field("fee_bps", pa.float64(), nullable=False),
        pa.field("slippage_bps", pa.float64(), nullable=False),
        pa.field("funding_cost_bps", pa.float64(), nullable=False),
        pa.field("net_return_bps", pa.float64(), nullable=False),
        pa.field("notional_fraction", pa.float64(), nullable=False),
        pa.field("realized_equity_return_fraction", pa.float64(), nullable=False),
        pa.field("equity_before", pa.float64(), nullable=False),
        pa.field("equity_after", pa.float64(), nullable=False),
    ],
    metadata={
        b"tradingbot.execution_evaluation_schema_version": str(
            EXECUTION_EVALUATION_SCHEMA_VERSION
        ).encode("ascii"),
        b"tradingbot.execution_assumption": (
            b"public_visible_queue_proxy_with_partial_unwind"
        ),
    },
)


def _matrix_view(
    data: ExecutionPreparedData,
    rows: NDArray[np.int64],
) -> NDArray[np.float32]:
    if len(rows) == 0:
        return data.x[:0]
    start = int(rows[0])
    stop = int(rows[-1]) + 1
    contiguous = (stop - start == len(rows)) and (
        len(rows) == 1 or bool(np.all(rows[1:] > rows[:-1]))
    )
    return data.x[start:stop] if contiguous else data.x[rows]


def _release_preparation_memory(data: ExecutionPreparedData) -> None:
    arrow_before = int(pa.total_allocated_bytes())
    gc.collect()
    pa.default_memory_pool().release_unused()
    gc.collect()
    release_unused_process_memory()
    arrow_after = int(pa.total_allocated_bytes())
    LOGGER.info(
        "Prepared %d execution rows: matrix %.1f MiB, IDs %.1f MiB; "
        "Arrow pool %.1f -> %.1f MiB",
        data.rows,
        data.x.nbytes / 1024**2,
        data.decision_ids.nbytes / 1024**2,
        arrow_before / 1024**2,
        arrow_after / 1024**2,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    rendered = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as target:
        target.write(rendered)
        target.flush()
        os.fsync(target.fileno())


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise EvaluationError(f"{label} is not a safe relative path")
    return path


def _environment_payload() -> dict[str, object]:
    return {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "scikit_learn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
    }


def _output_descriptor(
    root: Path,
    path: Path,
    *,
    rows: int | None = None,
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def _existing_result(
    path: Path,
    *,
    experiment_id: str,
    input_fingerprint: str,
) -> ExecutionEvaluationResult:
    try:
        manifest_raw: object = json.loads((path / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            "existing execution evaluation manifest is unreadable"
        ) from exc
    if not isinstance(manifest_raw, dict):
        raise EvaluationError(
            "existing execution evaluation manifest must be an object"
        )
    manifest = cast(dict[str, Any], manifest_raw)
    if (
        manifest.get("execution_evaluation_schema_version")
        != EXECUTION_EVALUATION_SCHEMA_VERSION
    ):
        raise EvaluationError(
            "existing execution evaluation uses another schema version"
        )
    if manifest.get("experiment_id") != experiment_id or path.name != experiment_id:
        raise EvaluationError("existing execution evaluation ID is inconsistent")
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise EvaluationError(
            "existing execution evaluation was built from another input"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EvaluationError(
            "existing execution evaluation manifest has no output files"
        )

    descriptors: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise EvaluationError(f"existing files[{index}] must be an object")
        item = cast(dict[str, object], raw)
        relative = _safe_relative_path(item.get("path"), f"files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise EvaluationError(f"duplicate existing output: {relative_text}")
        seen.add(relative_text)
        actual = path.joinpath(*relative.parts).resolve()
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            not actual.is_file()
            or not actual.is_relative_to(path)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or actual.stat().st_size != size
            or _sha256_file(actual) != digest
        ):
            raise EvaluationError(
                f"existing execution evaluation output is corrupted: {relative}"
            )
        descriptor: dict[str, object] = {
            "path": relative_text,
            "bytes": size,
            "sha256": digest,
        }
        rows = item.get("rows")
        if rows is not None:
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise EvaluationError(f"files[{index}].rows is invalid")
            if (
                actual.suffix == ".parquet"
                and pq.ParquetFile(actual).metadata.num_rows != rows
            ):
                raise EvaluationError(
                    f"existing Parquet row count is wrong: {relative}"
                )
            descriptor["rows"] = rows
        descriptors.append(descriptor)
    if _sha256_json(
        sorted(descriptors, key=lambda item: cast(str, item["path"]))
    ) != manifest.get("output_fingerprint"):
        raise EvaluationError(
            "existing execution evaluation output fingerprint is inconsistent"
        )

    report_path = path / "report.json"
    try:
        report_raw: object = json.loads(report_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            "existing execution evaluation report is unreadable"
        ) from exc
    if not isinstance(report_raw, dict):
        raise EvaluationError("existing execution evaluation report must be an object")
    report = cast(dict[str, Any], report_raw)
    if report.get("input_fingerprint") != input_fingerprint:
        raise EvaluationError("existing execution report input is inconsistent")
    gate = report.get("data_gate")
    if not isinstance(gate, dict):
        raise EvaluationError("existing execution evaluation has no data gate")
    mode = gate.get("mode")
    span_days = gate.get("data_span_days")
    fold_count = report.get("fold_count")
    if not isinstance(mode, str):
        raise EvaluationError("existing execution data gate mode is invalid")
    if isinstance(span_days, bool) or not isinstance(span_days, (int, float)):
        raise EvaluationError("existing execution data span is invalid")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise EvaluationError("existing execution fold count is invalid")
    return ExecutionEvaluationResult(
        experiment_id=experiment_id,
        experiment_path=path,
        manifest_path=path / "manifest.json",
        report_path=report_path,
        data_mode=mode,
        data_span_days=float(span_days),
        folds=fold_count,
        reused=True,
    )


def _model_map(outputs: tuple[ModelPrediction, ...]) -> dict[str, ModelPrediction]:
    result = {output.name: output for output in outputs}
    if set(result) != {"class_prior", "logistic", "lightgbm"}:
        raise EvaluationError("execution evaluator received an unknown model family")
    return result


def _conditional_classification_metrics(
    y_true: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    if len(y_true) == 0:
        return {"rows": 0, "status": "unavailable_no_full_fills"}
    return classification_metrics(
        y_true,
        probabilities,
        class_names=EXECUTION_OUTCOME_NAMES,
    )


def _importance_report(
    values: Iterable[NDArray[np.float64]],
    feature_names: tuple[str, ...],
) -> list[dict[str, object]]:
    arrays = tuple(values)
    if not arrays:
        return []
    mean_importance = np.mean(np.stack(arrays), axis=0)
    total = float(np.sum(mean_importance))
    normalized = mean_importance if total <= 0 else mean_importance / total
    ranked = np.argsort(-normalized)[:25]
    return [
        {
            "feature": feature_names[int(index)],
            "gain_fraction": float(normalized[int(index)]),
        }
        for index in ranked
    ]


def _attempt_group_summary(
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    fill_counts = Counter(cast(str, item["fill_status"]) for item in attempts)
    outcome_counts = Counter(cast(str, item["outcome"]) for item in attempts)
    count = len(attempts)
    full_attempts = [item for item in attempts if item["fill_status"] == "FULL_FILL"]
    equity_changes = [
        float(cast(float, item["realized_equity_return_fraction"]))
        for item in attempts
    ]
    return {
        "order_attempts": count,
        "fill_statuses": dict(sorted(fill_counts.items())),
        "outcomes": dict(sorted(outcome_counts.items())),
        "any_fill_rate": (
            None
            if count == 0
            else (fill_counts["PARTIAL_FILL"] + fill_counts["FULL_FILL"]) / count
        ),
        "full_fill_rate": None if count == 0 else fill_counts["FULL_FILL"] / count,
        "full_fill_win_rate": (
            None
            if not full_attempts
            else sum(float(cast(float, item["net_return_bps"])) > 0 for item in full_attempts)
            / len(full_attempts)
        ),
        "mean_expected_net_bps": (
            None
            if count == 0
            else float(
                np.mean(
                    [float(cast(float, item["expected_net_bps"])) for item in attempts]
                )
            )
        ),
        "mean_realized_equity_return_fraction": (
            None if count == 0 else float(np.mean(equity_changes))
        ),
        "sum_realized_equity_return_fraction": (
            None if count == 0 else float(np.sum(equity_changes))
        ),
        "mean_full_fill_net_return_bps": (
            None
            if not full_attempts
            else float(
                np.mean(
                    [
                        float(cast(float, item["net_return_bps"]))
                        for item in full_attempts
                    ]
                )
            )
        ),
        "total_fee_bps_on_filled_notional": float(
            np.sum([float(cast(float, item["fee_bps"])) for item in attempts])
        ),
        "total_slippage_bps_on_filled_notional": float(
            np.sum([float(cast(float, item["slippage_bps"])) for item in attempts])
        ),
        "total_funding_cost_bps_on_filled_notional": float(
            np.sum(
                [float(cast(float, item["funding_cost_bps"])) for item in attempts]
            )
        ),
    }


def _group_attempts(
    attempts: list[dict[str, object]],
    key: str,
    *,
    expected_labels: Iterable[str] = (),
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in attempts:
        value = item[key]
        label = f"{value:.12g}" if isinstance(value, float) else str(value)
        grouped[label].append(item)
    for label in expected_labels:
        grouped.setdefault(label, [])
    return {
        label: _attempt_group_summary(items)
        for label, items in sorted(grouped.items())
    }


def _attempt_diagnostics(
    attempts: list[dict[str, object]],
    *,
    symbols: tuple[str, ...],
    folds: Iterable[int],
    horizon_minutes: int,
    order_notional_usdt: float,
) -> dict[str, object]:
    return {
        "overall": _attempt_group_summary(attempts),
        "by_fold": _group_attempts(
            attempts,
            "fold",
            expected_labels=(str(value) for value in folds),
        ),
        "by_symbol": _group_attempts(
            attempts, "symbol", expected_labels=symbols
        ),
        "by_side": _group_attempts(
            attempts, "side", expected_labels=("LONG", "SHORT")
        ),
        "by_fill_status": _group_attempts(
            attempts, "fill_status", expected_labels=FILL_NAMES
        ),
        "by_outcome": _group_attempts(
            attempts,
            "outcome",
            expected_labels=(
                "NO_FILL",
                "PARTIAL_UNWIND",
                *EXECUTION_OUTCOME_NAMES,
            ),
        ),
        "by_horizon_minutes": _group_attempts(
            attempts,
            "horizon_minutes",
            expected_labels=(str(horizon_minutes),),
        ),
        "by_reference_order_notional_usdt": _group_attempts(
            attempts,
            "order_notional_usdt",
            expected_labels=(f"{order_notional_usdt:.12g}",),
        ),
    }


def run_execution_evaluation(
    execution_dataset: str | Path,
    output_root: str | Path,
    *,
    config: AppConfig,
    order_notional_usdt: float,
    minimum_free_bytes: int = 0,
) -> ExecutionEvaluationResult:
    """Fit fill/outcome models and replay one maker order across six pairs."""

    parameters = replace(
        evaluation_parameters(config),
        # V3 outcomes start at the observed proxy fill price, so charging the
        # V2 decision-to-entry adverse-selection allowance would double count it.
        entry_adverse_selection_bps=0.0,
    )
    LOGGER.info("Validating execution research dataset at %s", execution_dataset)
    dataset = validate_execution_research_dataset(execution_dataset)
    if set(dataset.symbols) != set(config.bybit.symbols):
        missing = sorted(set(config.bybit.symbols) - set(dataset.symbols))
        unexpected = sorted(set(dataset.symbols) - set(config.bybit.symbols))
        raise EvaluationError(
            "execution dataset symbols do not match the configured universe: "
            f"missing={missing}, unexpected={unexpected}"
        )
    destination_root = Path(output_root).expanduser().resolve()
    if dataset.root == destination_root or dataset.root.is_relative_to(destination_root):
        raise EvaluationError(
            "execution evaluation output must not contain its research input"
        )
    if destination_root.is_relative_to(dataset.root):
        raise EvaluationError(
            "execution evaluation output must not be inside its research input"
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    if minimum_free_bytes < 0:
        raise EvaluationError("minimum_free_bytes must be non-negative")
    if shutil.disk_usage(destination_root).free < minimum_free_bytes:
        raise EvaluationError("insufficient free space for execution evaluation")
    if not np.isfinite(order_notional_usdt) or order_notional_usdt <= 0:
        raise EvaluationError("order_notional_usdt must be positive and finite")

    environment = _environment_payload()
    parameter_payload = {
        **parameters.to_dict(),
        "reference_order_notional_usdt": order_notional_usdt,
        "fill_model_classes": list(FILL_NAMES),
        "post_fill_outcome_classes": list(EXECUTION_OUTCOME_NAMES),
        "partial_fill_policy": (
            "cancel_residual_and_taker_unwind_filled_fraction"
        ),
        "entry_adverse_selection_handling": (
            "embedded_in_observed_proxy_fill_price_not_charged_twice"
        ),
    }
    parameter_fingerprint = _sha256_json(parameter_payload)
    input_payload = {
        "execution_evaluation_schema_version": (
            EXECUTION_EVALUATION_SCHEMA_VERSION
        ),
        "execution_dataset_id": dataset.execution_dataset_id,
        "execution_input_fingerprint": dataset.input_fingerprint,
        "execution_output_fingerprint": dataset.output_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "environment": environment,
    }
    input_fingerprint = _sha256_json(input_payload)
    experiment_id = (
        f"execution-backtest-v{EXECUTION_EVALUATION_SCHEMA_VERSION}-"
        f"{input_fingerprint[:16]}"
    )
    final_path = destination_root / experiment_id
    if final_path.exists():
        return _existing_result(
            final_path,
            experiment_id=experiment_id,
            input_fingerprint=input_fingerprint,
        )

    staging = destination_root / f".{experiment_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        data = prepare_execution_evaluation_data(
            dataset,
            horizon_minutes=parameters.horizon_minutes,
            order_notional_usdt=order_notional_usdt,
        )
        _release_preparation_memory(data)
        folds = build_execution_temporal_folds(data, parameters)
        data_span_days = (
            int(np.max(data.decision_at_ns))
            - int(np.min(data.decision_at_ns))
            + 60 * 1_000_000_000
        ) / NS_PER_DAY
        data_mode = (
            "walk_forward"
            if all(fold.mode == "walk_forward" for fold in folds)
            else "technical_smoke"
        )
        source_coverage_days = len(dataset.partition_dates)
        history_gate_met = (
            source_coverage_days >= parameters.acceptance_minimum_days
        )

        prediction_batches: list[ExecutionPredictionBatch] = []
        fold_reports: list[dict[str, object]] = []
        model_files: list[Path] = []
        importance_by_component: dict[str, list[NDArray[np.float64]]] = defaultdict(
            list
        )
        model_metadata: dict[str, dict[str, object]] = {}
        models_root = staging / "models"
        models_root.mkdir()

        for fold in folds:
            LOGGER.info(
                "Starting execution fold %d/%d (%s)",
                fold.fold,
                len(folds),
                fold.mode,
            )
            calibration_split = build_execution_calibration_split(
                data, fold, parameters
            )
            fit = calibration_split.fit_indices
            calibration = calibration_split.calibration_indices
            test = fold.test_indices
            fit_full = fit[data.outcome_y[fit] >= 0]
            calibration_full = calibration[data.outcome_y[calibration] >= 0]
            test_full_positions = np.flatnonzero(data.outcome_y[test] >= 0).astype(
                np.int64
            )

            timeout_estimate = timeout_return_estimate(data, fit, test)
            partial_estimate = partial_fraction_estimate(data, fit, test)
            fill_outputs = _model_map(
                fit_fold_models(
                    _matrix_view(data, fit),
                    data.fill_y[fit],
                    _matrix_view(data, calibration),
                    _matrix_view(data, test),
                    parameters,
                    class_names=FILL_NAMES,
                )
            )
            outcome_outputs = _model_map(
                fit_fold_models(
                    _matrix_view(data, fit_full),
                    data.outcome_y[fit_full],
                    _matrix_view(data, calibration_full),
                    _matrix_view(data, test),
                    parameters,
                    class_names=EXECUTION_OUTCOME_NAMES,
                )
            )

            fold_model_reports: dict[str, object] = {}
            for base_model in ("class_prior", "logistic", "lightgbm"):
                fill_output = fill_outputs[base_model]
                outcome_output = outcome_outputs[base_model]
                variants: dict[
                    str,
                    tuple[
                        NDArray[np.float64],
                        NDArray[np.float64],
                        NDArray[np.float64],
                        NDArray[np.float64],
                    ],
                ] = {
                    "raw": (
                        fill_output.calibration_probabilities,
                        fill_output.probabilities,
                        outcome_output.calibration_probabilities,
                        outcome_output.probabilities,
                    )
                }
                calibrator_report: dict[str, object] | None = None
                if base_model != "class_prior":
                    fill_calibrator = fit_probability_calibrator(
                        fill_output.calibration_probabilities,
                        data.fill_y[calibration],
                        class_names=FILL_NAMES,
                    )
                    outcome_calibrator = fit_probability_calibrator(
                        outcome_output.calibration_probabilities,
                        data.outcome_y[calibration_full],
                        class_names=EXECUTION_OUTCOME_NAMES,
                    )
                    variants["calibrated"] = (
                        fill_calibrator.transform(
                            fill_output.calibration_probabilities
                        ),
                        fill_calibrator.transform(fill_output.probabilities),
                        outcome_calibrator.transform(
                            outcome_output.calibration_probabilities
                        ),
                        outcome_calibrator.transform(
                            outcome_output.probabilities
                        ),
                    )
                    calibrator_report = {
                        "fill": fill_calibrator.to_dict(),
                        "post_fill_outcome": outcome_calibrator.to_dict(),
                    }

                variant_reports: dict[str, object] = {}
                for variant, (
                    fill_calibration_probabilities,
                    fill_test_probabilities,
                    outcome_calibration_probabilities,
                    outcome_test_probabilities,
                ) in variants.items():
                    model_name = (
                        "class_prior"
                        if base_model == "class_prior"
                        else f"{base_model}_{variant}"
                    )
                    expected = expected_execution_net_bps(
                        data,
                        test,
                        fill_test_probabilities,
                        outcome_test_probabilities,
                        timeout_estimate,
                        partial_estimate,
                        parameters,
                    )
                    prediction_batches.append(
                        ExecutionPredictionBatch(
                            model_name=model_name,
                            fold=fold.fold,
                            row_indices=test,
                            fill_probabilities=fill_test_probabilities,
                            outcome_probabilities=outcome_test_probabilities,
                            expected_net_bps=expected,
                        )
                    )
                    model_metadata[model_name] = {
                        "base_model": base_model,
                        "probability_variant": variant,
                        "fill_model": "all maker-order candidates",
                        "post_fill_outcome_model": (
                            "conditional on observed FULL_FILL only"
                        ),
                        "feature_profile": "causal_decision_features_full",
                    }
                    variant_reports[variant] = {
                        "fill_classification": {
                            "calibration_window": classification_metrics(
                                data.fill_y[calibration],
                                fill_calibration_probabilities,
                                class_names=FILL_NAMES,
                            ),
                            "outer_test": classification_metrics(
                                data.fill_y[test],
                                fill_test_probabilities,
                                class_names=FILL_NAMES,
                            ),
                        },
                        "post_fill_outcome_classification": {
                            "calibration_window": (
                                _conditional_classification_metrics(
                                    data.outcome_y[calibration_full],
                                    outcome_calibration_probabilities,
                                )
                            ),
                            "outer_test": _conditional_classification_metrics(
                                data.outcome_y[test][test_full_positions],
                                outcome_test_probabilities[test_full_positions],
                            ),
                        },
                    }

                if fill_output.model_text is not None:
                    fill_model_path = (
                        models_root
                        / f"fold-{fold.fold:02d}-{base_model}-fill.txt"
                    )
                    fill_model_path.write_text(
                        fill_output.model_text, encoding="utf-8", newline="\n"
                    )
                    model_files.append(fill_model_path)
                if outcome_output.model_text is not None:
                    outcome_model_path = (
                        models_root
                        / f"fold-{fold.fold:02d}-{base_model}-outcome.txt"
                    )
                    outcome_model_path.write_text(
                        outcome_output.model_text,
                        encoding="utf-8",
                        newline="\n",
                    )
                    model_files.append(outcome_model_path)
                if fill_output.feature_importance is not None:
                    importance_by_component[f"{base_model}_fill"].append(
                        fill_output.feature_importance
                    )
                if outcome_output.feature_importance is not None:
                    importance_by_component[f"{base_model}_outcome"].append(
                        outcome_output.feature_importance
                    )

                fold_model_reports[base_model] = {
                    "training_rows": {
                        "fill_available": fill_output.training_rows_available,
                        "fill_used": fill_output.training_rows_used,
                        "post_fill_outcome_available": (
                            outcome_output.training_rows_available
                        ),
                        "post_fill_outcome_used": outcome_output.training_rows_used,
                    },
                    "calibrators": calibrator_report,
                    "probability_variants": variant_reports,
                }
            fold_reports.append(
                {
                    **fold.to_dict(),
                    "nested_calibration": calibration_split.to_dict(),
                    "fit_full_fill_rows": len(fit_full),
                    "calibration_full_fill_rows": len(calibration_full),
                    "test_full_fill_rows": len(test_full_positions),
                    "models": fold_model_reports,
                }
            )
            release_unused_process_memory()

        model_names = sorted({batch.model_name for batch in prediction_batches})
        aggregate_models: dict[str, object] = {}
        attempt_paths: list[tuple[Path, int]] = []
        attempts_root = staging / "attempts"
        attempts_root.mkdir()
        fill_importance = _importance_report(
            importance_by_component["lightgbm_fill"], data.feature_names
        )
        outcome_importance = _importance_report(
            importance_by_component["lightgbm_outcome"], data.feature_names
        )
        for model_name in model_names:
            LOGGER.info("Replaying one-position policy for %s", model_name)
            combined = combine_execution_prediction_batches(
                prediction_batches, model_name=model_name
            )
            full_positions = np.flatnonzero(
                data.outcome_y[combined.row_indices] >= 0
            ).astype(np.int64)
            fill_metrics = classification_metrics(
                data.fill_y[combined.row_indices],
                combined.fill_probabilities,
                class_names=FILL_NAMES,
            )
            outcome_metrics = _conditional_classification_metrics(
                data.outcome_y[combined.row_indices][full_positions],
                combined.outcome_probabilities[full_positions],
            )
            backtest_summary, attempts = run_execution_one_position_backtest(
                data, combined, parameters
            )
            top_importance: dict[str, object] = {
                "fill_model": [],
                "post_fill_outcome_model": [],
            }
            if model_metadata[model_name]["base_model"] == "lightgbm":
                top_importance = {
                    "fill_model": fill_importance,
                    "post_fill_outcome_model": outcome_importance,
                }
            aggregate_models[model_name] = {
                "contract": model_metadata[model_name],
                "fill_classification": fill_metrics,
                "post_fill_outcome_classification": outcome_metrics,
                "execution_aware_backtest": backtest_summary,
                "selected_attempt_diagnostics": _attempt_diagnostics(
                    attempts,
                    symbols=data.symbols,
                    folds=(int(value) for value in np.unique(combined.folds)),
                    horizon_minutes=data.horizon_minutes,
                    order_notional_usdt=data.order_notional_usdt,
                ),
                "top_feature_importance": top_importance,
            }
            attempt_path = attempts_root / f"{model_name}.parquet"
            pq.write_table(
                pa.Table.from_pylist(attempts, schema=EXECUTION_ATTEMPT_SCHEMA),
                attempt_path,
                version="2.6",
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
                write_page_checksum=True,
            )
            attempt_paths.append((attempt_path, len(attempts)))

        report: dict[str, object] = {
            "execution_evaluation_schema_version": (
                EXECUTION_EVALUATION_SCHEMA_VERSION
            ),
            "experiment_id": experiment_id,
            "input_fingerprint": input_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "environment": environment,
            "execution_research_dataset": {
                "execution_dataset_id": dataset.execution_dataset_id,
                "source_dataset_id": dataset.source_dataset_id,
                "input_fingerprint": dataset.input_fingerprint,
                "output_fingerprint": dataset.output_fingerprint,
                "symbols": list(dataset.symbols),
                "partition_dates": list(dataset.partition_dates),
                "feature_rows": dataset.feature_rows,
                "execution_label_rows": dataset.label_rows,
                "verified_parquet_files": len(
                    dataset.feature_paths + dataset.label_paths
                ),
            },
            "parameters": parameter_payload,
            "selected_execution_scenario": {
                "horizon_minutes": parameters.horizon_minutes,
                "reference_order_notional_usdt": order_notional_usdt,
                "reference_equity_at_max_notional_usdt": (
                    order_notional_usdt / parameters.max_notional_fraction
                ),
                "note": (
                    "fill labels are valid for the selected reference notional; "
                    "normalized PnL uses the configured risk/notional caps"
                ),
            },
            "prepared_rows": data.rows,
            "feature_count": len(data.feature_names),
            "feature_names": list(data.feature_names),
            "excluded_rows": {
                "ambiguous_full_fills": data.excluded_ambiguous_full_fills,
                "unpriced_full_fills": data.excluded_unpriced_full_fills,
            },
            "pre_registered_comparisons": {
                "model_families": ["class_prior", "logistic", "lightgbm"],
                "probability_variants": ["raw", "calibrated"],
                "primary_candidate_model": "lightgbm_calibrated",
                "selection_rule": (
                    "maximum combined expected net bps per decision minute"
                ),
                "fill_target": list(FILL_NAMES),
                "post_fill_outcome_target": list(EXECUTION_OUTCOME_NAMES),
            },
            "fold_count": len(folds),
            "folds": fold_reports,
            "models": aggregate_models,
            "data_gate": {
                "mode": data_mode,
                "data_span_days": data_span_days,
                "source_coverage_days": source_coverage_days,
                "required_days_for_execution_model_review": (
                    parameters.acceptance_minimum_days
                ),
                "minimum_history_met": history_gate_met,
                "eligible_for_execution_model_review": (
                    history_gate_met and len(folds) >= 3
                ),
                "eligible_for_profitability_conclusion": False,
                "reason": (
                    "public market-data replay approximates visible queue depletion; "
                    "real queue priority, exchange acknowledgements, latency, and live "
                    "slippage remain unobserved"
                ),
            },
            "scope": {
                "bybit_access": "public-read-only",
                "order_submission": False,
                "real_exchange_orders_observed": False,
                "maker_fill_modeled": True,
                "maker_fill_is_public_data_proxy": True,
                "visible_queue_depletion_modeled": True,
                "real_queue_position_observed": False,
                "partial_fill_modeled": True,
                "partial_fill_policy": (
                    "cancel_residual_and_taker_unwind_filled_fraction"
                ),
                "post_fill_outcome_modeled_separately": True,
                "one_position_across_all_symbols_enforced": True,
                "openai_tokens": 0,
            },
        }
        report_path = staging / "report.json"
        _write_json(report_path, report)

        descriptors = [_output_descriptor(staging, report_path)]
        descriptors.extend(_output_descriptor(staging, path) for path in model_files)
        descriptors.extend(
            _output_descriptor(staging, path, rows=rows)
            for path, rows in attempt_paths
        )
        descriptors.sort(key=lambda item: cast(str, item["path"]))
        output_fingerprint = _sha256_json(descriptors)
        manifest: dict[str, object] = {
            "execution_evaluation_schema_version": (
                EXECUTION_EVALUATION_SCHEMA_VERSION
            ),
            "experiment_id": experiment_id,
            "input_fingerprint": input_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "execution_dataset_id": dataset.execution_dataset_id,
            "execution_output_fingerprint": dataset.output_fingerprint,
            "environment": environment,
            "output_fingerprint": output_fingerprint,
            "output_file_count": len(descriptors),
            "files": descriptors,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, final_path)
        LOGGER.info("Execution-aware evaluation ready at %s", final_path)
        return ExecutionEvaluationResult(
            experiment_id=experiment_id,
            experiment_path=final_path,
            manifest_path=final_path / "manifest.json",
            report_path=final_path / "report.json",
            data_mode=data_mode,
            data_span_days=data_span_days,
            folds=len(folds),
            reused=False,
        )
    except Exception:
        if staging.is_dir() and staging.parent == destination_root:
            shutil.rmtree(staging, ignore_errors=True)
        raise
