from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import uuid
from collections import defaultdict
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
from tradingbot.research.backtest import (
    combine_prediction_batches,
    expected_net_returns_bps,
    run_one_position_backtest,
)
from tradingbot.research.contracts import PRICE_RESEARCH_PROFILE
from tradingbot.research.diagnostics import trade_diagnostics
from tradingbot.research.evaluation_contracts import (
    EVALUATION_SCHEMA_VERSION,
    FEATURE_PROFILES,
    NS_PER_DAY,
    EvaluationError,
    EvaluationParameters,
    EvaluationResult,
    PredictionBatch,
    PreparedData,
)
from tradingbot.research.evaluation_dataset import (
    build_symbol_quality_gate,
    feature_profile_indices,
    prepare_evaluation_data,
    validate_research_dataset,
)
from tradingbot.research.models import (
    classification_metrics,
    fit_fold_models,
    fit_probability_calibrator,
    timeout_return_estimate,
)
from tradingbot.research.splits import build_calibration_split, build_temporal_folds

LOGGER = logging.getLogger(__name__)

TRADE_SCHEMA = pa.schema(
    [
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("fold", pa.int16(), nullable=False),
        pa.field("decision_at_ns", pa.int64(), nullable=False),
        pa.field("exit_at_ns", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
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
        pa.field("equity_before", pa.float64(), nullable=False),
        pa.field("equity_after", pa.float64(), nullable=False),
    ],
    metadata={
        b"tradingbot.evaluation_schema_version": str(EVALUATION_SCHEMA_VERSION).encode(
            "ascii"
        ),
        b"tradingbot.execution_assumption": b"conditional_entry_no_maker_fill_model",
    },
)


def evaluation_parameters(config: AppConfig) -> EvaluationParameters:
    selected = config.evaluation
    return EvaluationParameters(
        horizon_minutes=selected.horizon_minutes,
        embargo_minutes=selected.embargo_minutes,
        minimum_train_days=selected.minimum_train_days,
        test_days=selected.test_days,
        maximum_folds=selected.maximum_folds,
        acceptance_minimum_days=selected.acceptance_minimum_days,
        minimum_train_rows=selected.minimum_train_rows,
        minimum_test_rows=selected.minimum_test_rows,
        calibration_days=selected.calibration_days,
        minimum_calibration_rows=selected.minimum_calibration_rows,
        minimum_symbol_coverage_fraction=(
            selected.minimum_symbol_coverage_fraction
        ),
        maker_fee_bps=selected.maker_fee_bps,
        taker_fee_bps=selected.taker_fee_bps,
        entry_adverse_selection_bps=selected.entry_adverse_selection_bps,
        stop_slippage_bps=selected.stop_slippage_bps,
        timeout_slippage_bps=selected.timeout_slippage_bps,
        minimum_expected_net_bps=selected.minimum_expected_net_bps,
        lightgbm_estimators=selected.lightgbm_estimators,
        lightgbm_learning_rate=selected.lightgbm_learning_rate,
        lightgbm_num_leaves=selected.lightgbm_num_leaves,
        lightgbm_min_child_samples=selected.lightgbm_min_child_samples,
        training_threads=selected.training_threads,
        random_seed=selected.random_seed,
        max_notional_fraction=config.risk.max_notional_fraction,
        max_planned_risk_fraction=config.risk.max_planned_risk_fraction,
        rolling_24h_loss_fraction=config.risk.rolling_24h_loss_fraction,
    )


def _matrix_view(
    data: PreparedData,
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
) -> NDArray[np.float32]:
    if len(columns) == data.x.shape[1]:
        return data.x[rows]
    return data.x[np.ix_(rows, columns)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
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


def _output_descriptor(root: Path, path: Path, *, rows: int | None = None) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def _existing_result(
    path: Path, *, experiment_id: str, input_fingerprint: str
) -> EvaluationResult:
    try:
        manifest_raw: object = json.loads((path / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("existing evaluation manifest is unreadable") from exc
    if not isinstance(manifest_raw, dict):
        raise EvaluationError("existing evaluation manifest must be an object")
    manifest = cast(dict[str, Any], manifest_raw)
    if manifest.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("existing evaluation uses another schema version")
    if manifest.get("experiment_id") != experiment_id or path.name != experiment_id:
        raise EvaluationError("existing evaluation ID is inconsistent")
    if manifest.get("input_fingerprint") != input_fingerprint:
        raise EvaluationError("existing evaluation was built from another input")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EvaluationError("existing evaluation manifest has no output files")
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
            raise EvaluationError(f"existing evaluation output is corrupted: {relative}")
        descriptor = {"path": relative_text, "bytes": size, "sha256": digest}
        rows = item.get("rows")
        if rows is not None:
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise EvaluationError(f"files[{index}].rows is invalid")
            if actual.suffix == ".parquet" and pq.ParquetFile(actual).metadata.num_rows != rows:
                raise EvaluationError(f"existing Parquet row count is wrong: {relative}")
            descriptor["rows"] = rows
        descriptors.append(descriptor)
    if _sha256_json(sorted(descriptors, key=lambda item: cast(str, item["path"]))) != (
        manifest.get("output_fingerprint")
    ):
        raise EvaluationError("existing evaluation output fingerprint is inconsistent")
    report_path = path / "report.json"
    try:
        report_raw: object = json.loads(report_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("existing evaluation report is unreadable") from exc
    if not isinstance(report_raw, dict):
        raise EvaluationError("existing evaluation report must be an object")
    report = cast(dict[str, Any], report_raw)
    gate = report.get("data_gate")
    if not isinstance(gate, dict):
        raise EvaluationError("existing evaluation report has no data gate")
    mode = gate.get("mode")
    span_days = gate.get("data_span_days")
    fold_count = report.get("fold_count")
    if not isinstance(mode, str):
        raise EvaluationError("existing evaluation data gate mode is invalid")
    if isinstance(span_days, bool) or not isinstance(span_days, (int, float)):
        raise EvaluationError("existing evaluation data span is invalid")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise EvaluationError("existing evaluation fold count is invalid")
    return EvaluationResult(
        experiment_id=experiment_id,
        experiment_path=path,
        manifest_path=path / "manifest.json",
        report_path=report_path,
        data_mode=mode,
        data_span_days=float(span_days),
        folds=fold_count,
        reused=True,
    )


def run_offline_evaluation(
    research_dataset: str | Path,
    output_root: str | Path,
    *,
    config: AppConfig,
    minimum_free_bytes: int = 0,
) -> EvaluationResult:
    """Train baselines and LightGBM, then run a conditional-entry market backtest."""

    parameters = evaluation_parameters(config)
    dataset = validate_research_dataset(research_dataset)
    destination_root = Path(output_root).expanduser().resolve()
    if dataset.root == destination_root or dataset.root.is_relative_to(destination_root):
        raise EvaluationError("evaluation output must not contain the research input")
    if destination_root.is_relative_to(dataset.root):
        raise EvaluationError("evaluation output must not be inside the research input")
    destination_root.mkdir(parents=True, exist_ok=True)
    if minimum_free_bytes < 0:
        raise EvaluationError("minimum_free_bytes must be non-negative")
    if shutil.disk_usage(destination_root).free < minimum_free_bytes:
        raise EvaluationError("insufficient free space for offline evaluation")

    environment = _environment_payload()
    parameter_payload = parameters.to_dict()
    parameter_fingerprint = _sha256_json(parameter_payload)
    input_payload = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "research_dataset_id": dataset.research_dataset_id,
        "research_input_fingerprint": dataset.input_fingerprint,
        "research_output_fingerprint": dataset.output_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "environment": environment,
    }
    input_fingerprint = _sha256_json(input_payload)
    experiment_id = f"backtest-v{EVALUATION_SCHEMA_VERSION}-{input_fingerprint[:16]}"
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
        eligible_symbols, symbol_quality_gate = build_symbol_quality_gate(
            dataset,
            minimum_coverage_fraction=(
                parameters.minimum_symbol_coverage_fraction
            ),
        )
        data = prepare_evaluation_data(
            dataset,
            horizon_minutes=parameters.horizon_minutes,
            allowed_symbols=eligible_symbols,
        )
        profile_columns = {
            profile: feature_profile_indices(data.feature_names, profile)
            for profile in FEATURE_PROFILES
        }
        profile_feature_names = {
            profile: tuple(data.feature_names[int(index)] for index in columns)
            for profile, columns in profile_columns.items()
        }
        folds = build_temporal_folds(data, parameters)
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
        source_coverage_days: int | None = None
        if dataset.research_profile == PRICE_RESEARCH_PROFILE:
            source = dataset.manifest.get("source")
            raw_days = source.get("days") if isinstance(source, dict) else None
            if isinstance(raw_days, int) and not isinstance(raw_days, bool) and raw_days > 0:
                source_coverage_days = raw_days
        gate_coverage_days = (
            float(source_coverage_days)
            if source_coverage_days is not None
            else data_span_days
        )
        history_gate_met = gate_coverage_days >= parameters.acceptance_minimum_days

        prediction_batches: list[PredictionBatch] = []
        fold_reports: list[dict[str, object]] = []
        model_files: list[Path] = []
        importance_by_model: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
        importance_names: dict[str, tuple[str, ...]] = {}
        model_metadata: dict[str, dict[str, object]] = {}
        models_root = staging / "models"
        models_root.mkdir()
        for fold in folds:
            calibration_split = build_calibration_split(data, fold, parameters)
            fit = calibration_split.fit_indices
            calibration = calibration_split.calibration_indices
            test = fold.test_indices
            timeout_estimate = timeout_return_estimate(
                y_train=data.y[fit],
                returns_train=data.outcome_return_bps[fit],
                symbols_train=data.symbol_codes[fit],
                sides_train=data.side_codes[fit],
                symbols_test=data.symbol_codes[test],
                sides_test=data.side_codes[test],
            )
            profile_reports: dict[str, object] = {}
            for profile in FEATURE_PROFILES:
                columns = profile_columns[profile]
                outputs = fit_fold_models(
                    _matrix_view(data, fit, columns),
                    data.y[fit],
                    _matrix_view(data, calibration, columns),
                    _matrix_view(data, test, columns),
                    parameters,
                )
                profile_model_reports: dict[str, object] = {}
                for output in outputs:
                    if output.name == "class_prior" and profile != "full":
                        continue
                    if output.name == "class_prior":
                        model_name = "class_prior"
                        expected = expected_net_returns_bps(
                            data,
                            test,
                            output.probabilities,
                            timeout_estimate,
                            parameters,
                        )
                        prediction_batches.append(
                            PredictionBatch(
                                model_name=model_name,
                                fold=fold.fold,
                                row_indices=test,
                                probabilities=output.probabilities,
                                expected_net_bps=expected,
                            )
                        )
                        model_metadata[model_name] = {
                            "base_model": output.name,
                            "feature_profile": None,
                            "probability_variant": "raw",
                        }
                        profile_model_reports[output.name] = {
                            "raw": {
                                "calibration_window": classification_metrics(
                                    data.y[calibration],
                                    output.calibration_probabilities,
                                ),
                                "outer_test": classification_metrics(
                                    data.y[test], output.probabilities
                                ),
                            }
                        }
                        continue

                    calibrator = fit_probability_calibrator(
                        output.calibration_probabilities,
                        data.y[calibration],
                    )
                    calibrated_calibration = calibrator.transform(
                        output.calibration_probabilities
                    )
                    calibrated_test = calibrator.transform(output.probabilities)
                    probability_variants = {
                        "raw": output.probabilities,
                        "calibrated": calibrated_test,
                    }
                    for variant, probabilities in probability_variants.items():
                        model_name = f"{output.name}_{profile}_{variant}"
                        expected = expected_net_returns_bps(
                            data,
                            test,
                            probabilities,
                            timeout_estimate,
                            parameters,
                        )
                        prediction_batches.append(
                            PredictionBatch(
                                model_name=model_name,
                                fold=fold.fold,
                                row_indices=test,
                                probabilities=probabilities,
                                expected_net_bps=expected,
                            )
                        )
                        model_metadata[model_name] = {
                            "base_model": output.name,
                            "feature_profile": profile,
                            "probability_variant": variant,
                        }
                        if output.feature_importance is not None:
                            importance_by_model[model_name].append(
                                output.feature_importance
                            )
                            importance_names[model_name] = profile_feature_names[
                                profile
                            ]
                    profile_model_reports[output.name] = {
                        "calibrator": calibrator.to_dict(),
                        "raw": {
                            "calibration_window": classification_metrics(
                                data.y[calibration],
                                output.calibration_probabilities,
                            ),
                            "outer_test": classification_metrics(
                                data.y[test], output.probabilities
                            ),
                        },
                        "calibrated": {
                            "calibration_window": classification_metrics(
                                data.y[calibration], calibrated_calibration
                            ),
                            "outer_test": classification_metrics(
                                data.y[test], calibrated_test
                            ),
                        },
                    }
                    if output.model_text is not None:
                        model_path = (
                            models_root
                            / f"fold-{fold.fold:02d}-{output.name}-{profile}.txt"
                        )
                        model_path.write_text(
                            output.model_text, encoding="utf-8", newline="\n"
                        )
                        model_files.append(model_path)
                profile_reports[profile] = {
                    "feature_count": len(columns),
                    "models": profile_model_reports,
                }
            fold_reports.append(
                {
                    **fold.to_dict(),
                    "nested_calibration": calibration_split.to_dict(),
                    "feature_profiles": profile_reports,
                }
            )

        model_names = sorted({batch.model_name for batch in prediction_batches})
        aggregate_models: dict[str, object] = {}
        trade_paths: list[tuple[Path, int]] = []
        trades_root = staging / "trades"
        trades_root.mkdir()
        for model_name in model_names:
            combined = combine_prediction_batches(
                prediction_batches, model_name=model_name
            )
            aggregate_metrics = classification_metrics(
                data.y[combined.row_indices], combined.probabilities
            )
            backtest_summary, trades = run_one_position_backtest(
                data, combined, parameters
            )
            importance_report: list[dict[str, object]] = []
            if importance_by_model[model_name]:
                mean_importance = np.mean(
                    np.stack(importance_by_model[model_name]), axis=0
                )
                total_importance = float(np.sum(mean_importance))
                normalized = (
                    mean_importance
                    if total_importance <= 0
                    else mean_importance / total_importance
                )
                ranked = np.argsort(-normalized)[:25]
                names = importance_names[model_name]
                importance_report = [
                    {
                        "feature": names[int(index)],
                        "gain_fraction": float(normalized[int(index)]),
                    }
                    for index in ranked
                ]
            aggregate_models[model_name] = {
                "contract": model_metadata[model_name],
                "classification": aggregate_metrics,
                "conditional_entry_backtest": backtest_summary,
                "selected_trade_diagnostics": trade_diagnostics(trades),
                "top_feature_importance": importance_report,
            }
            trade_path = trades_root / f"{model_name}.parquet"
            pq.write_table(
                pa.Table.from_pylist(trades, schema=TRADE_SCHEMA),
                trade_path,
                version="2.6",
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
                write_page_checksum=True,
            )
            trade_paths.append((trade_path, len(trades)))

        report: dict[str, object] = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "input_fingerprint": input_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "environment": environment,
            "research_dataset": {
                "research_dataset_id": dataset.research_dataset_id,
                "research_profile": dataset.research_profile,
                "source_dataset_id": dataset.source_dataset_id,
                "input_fingerprint": dataset.input_fingerprint,
                "output_fingerprint": dataset.output_fingerprint,
                "symbols": list(dataset.symbols),
                "eligible_symbols": list(data.symbols),
                "feature_rows": dataset.feature_rows,
                "label_rows": dataset.label_rows,
                "verified_parquet_files": len(
                    dataset.feature_paths + dataset.label_paths
                ),
            },
            "parameters": parameter_payload,
            "prepared_rows": data.rows,
            "model_feature_counts": {
                profile: len(names)
                for profile, names in profile_feature_names.items()
            },
            "pre_registered_comparisons": {
                "feature_profiles": list(FEATURE_PROFILES),
                "probability_variants": ["raw", "calibrated"],
                "horizon_minutes": parameters.horizon_minutes,
                "primary_candidate_model": "lightgbm_full_calibrated",
                "selection_rule": "maximum expected net bps per decision minute",
            },
            "excluded_rows": {
                "ambiguous": data.excluded_ambiguous_rows,
                "unpriced": data.excluded_unpriced_rows,
            },
            "fold_count": len(folds),
            "folds": fold_reports,
            "models": aggregate_models,
            "data_gate": {
                "mode": data_mode,
                "data_span_days": data_span_days,
                "source_coverage_days": source_coverage_days,
                "gate_coverage_days": gate_coverage_days,
                "required_days_for_market_model_review": (
                    parameters.acceptance_minimum_days
                ),
                "minimum_history_met": history_gate_met,
                "eligible_for_market_model_review": (
                    history_gate_met and len(folds) >= 3
                ),
                "eligible_for_profitability_conclusion": False,
                "symbol_quality": symbol_quality_gate,
                "reason": (
                    (
                        "price-only history also lacks order book, spread, funding, and "
                        "open interest; maker fill, queue position, and partial execution "
                        "are not modeled"
                    )
                    if dataset.research_profile == PRICE_RESEARCH_PROFILE
                    else (
                        "maker fill, queue position, and partial execution are not modeled"
                    )
                ),
            },
            "scope": {
                "bybit_access": "public-read-only",
                "research_profile": dataset.research_profile,
                "order_submission": False,
                "conditional_entry_assumption": True,
                "maker_fill_modeled": False,
                "queue_position_modeled": False,
                "partial_fill_modeled": False,
                "funding_history_available": (
                    dataset.research_profile != PRICE_RESEARCH_PROFILE
                ),
                "openai_tokens": 0,
            },
        }
        report_path = staging / "report.json"
        _write_json(report_path, report)

        descriptors = [_output_descriptor(staging, report_path)]
        descriptors.extend(_output_descriptor(staging, path) for path in model_files)
        descriptors.extend(
            _output_descriptor(staging, path, rows=rows) for path, rows in trade_paths
        )
        descriptors.sort(key=lambda item: cast(str, item["path"]))
        output_fingerprint = _sha256_json(descriptors)
        manifest: dict[str, object] = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "input_fingerprint": input_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "research_dataset_id": dataset.research_dataset_id,
            "research_output_fingerprint": dataset.output_fingerprint,
            "environment": environment,
            "output_fingerprint": output_fingerprint,
            "output_file_count": len(descriptors),
            "files": descriptors,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, final_path)
        LOGGER.info("Offline evaluation ready at %s", final_path)
        return EvaluationResult(
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
