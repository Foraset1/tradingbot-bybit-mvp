"""Build and verify an immutable deployment bundle for read-only shadowing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from tradingbot import __version__
from tradingbot.research.evaluation_contracts import EvaluationError
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_EVALUATION_SCHEMA_VERSION,
    EXECUTION_OUTCOME_NAMES,
    FILL_NAMES,
)
from tradingbot.research.execution_evaluation_dataset import (
    validate_execution_research_dataset,
)

SHADOW_BUNDLE_SCHEMA_VERSION: Final = 1
PRIMARY_MODEL: Final = "lightgbm_calibrated"


class ShadowBundleError(RuntimeError):
    """Raised when a shadow bundle cannot be proven safe and immutable."""


@dataclass(frozen=True, slots=True)
class ShadowBundle:
    root: Path
    bundle_id: str
    bundle_fingerprint: str
    manifest: dict[str, Any]
    contract: dict[str, Any]
    fill_model_path: Path
    outcome_model_path: Path
    reused: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "bundle_path": self.root.as_posix(),
            "bundle_fingerprint": self.bundle_fingerprint,
            "manifest_path": (self.root / "manifest.json").as_posix(),
            "contract_path": (self.root / "bundle.json").as_posix(),
            "data_mode": self.contract["data_gate"]["mode"],
            "engineering_only": self.contract["scope"]["engineering_only"],
            "reused": self.reused,
        }


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
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(rendered)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ShadowBundleError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowBundleError(f"{label} must be a non-empty string")
    return value


def _plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowBundleError(f"{label} must be an integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowBundleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ShadowBundleError(f"{label} must be finite")
    return result


def _safe_relative(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_string(value, label))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ShadowBundleError(f"{label} must be a safe relative path")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowBundleError(f"{label} is unreadable: {path}") from exc
    return _object(parsed, label)


def _validate_descriptors(
    root: Path, raw_files: object, *, label: str
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    if not isinstance(raw_files, list) or not raw_files:
        raise ShadowBundleError(f"{label}.files must be a non-empty array")
    descriptors: list[dict[str, object]] = []
    paths: dict[str, Path] = {}
    for index, raw in enumerate(raw_files):
        item = _object(raw, f"{label}.files[{index}]")
        relative = _safe_relative(item.get("path"), f"{label}.files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in paths:
            raise ShadowBundleError(f"{label} contains duplicate path {relative_text}")
        actual = root.joinpath(*relative.parts).resolve()
        size = _plain_int(item.get("bytes"), f"{label}.files[{index}].bytes")
        digest = _string(item.get("sha256"), f"{label}.files[{index}].sha256")
        if (
            size < 0
            or len(digest) != 64
            or not actual.is_file()
            or not actual.is_relative_to(root)
            or actual.stat().st_size != size
            or _sha256_file(actual) != digest
        ):
            raise ShadowBundleError(f"{label} file failed verification: {relative_text}")
        descriptor: dict[str, object] = {
            "path": relative_text,
            "bytes": size,
            "sha256": digest,
        }
        rows = item.get("rows")
        if rows is not None:
            row_count = _plain_int(rows, f"{label}.files[{index}].rows")
            if row_count < 0:
                raise ShadowBundleError(f"{label} row count cannot be negative")
            descriptor["rows"] = row_count
        descriptors.append(descriptor)
        paths[relative_text] = actual
    return descriptors, paths


def _load_execution_evaluation(
    evaluation_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Path]]:
    root = Path(evaluation_path).expanduser().resolve()
    if not root.is_dir():
        raise ShadowBundleError(f"execution evaluation does not exist: {root}")
    manifest = _load_json(root / "manifest.json", "execution evaluation manifest")
    report = _load_json(root / "report.json", "execution evaluation report")
    if (
        manifest.get("execution_evaluation_schema_version")
        != EXECUTION_EVALUATION_SCHEMA_VERSION
        or report.get("execution_evaluation_schema_version")
        != EXECUTION_EVALUATION_SCHEMA_VERSION
    ):
        raise ShadowBundleError("unsupported execution evaluation schema")
    experiment_id = _string(manifest.get("experiment_id"), "experiment_id")
    if root.name != experiment_id or report.get("experiment_id") != experiment_id:
        raise ShadowBundleError("execution experiment identity is inconsistent")
    descriptors, paths = _validate_descriptors(
        root, manifest.get("files"), label="execution evaluation"
    )
    expected_output = _sha256_json(
        sorted(descriptors, key=lambda item: cast(str, item["path"]))
    )
    if manifest.get("output_fingerprint") != expected_output:
        raise ShadowBundleError("execution evaluation output fingerprint is invalid")
    report_descriptor = paths.get("report.json")
    if report_descriptor != (root / "report.json").resolve():
        raise ShadowBundleError("execution evaluation does not authenticate report.json")
    if report.get("input_fingerprint") != manifest.get("input_fingerprint"):
        raise ShadowBundleError("execution evaluation input fingerprint is inconsistent")
    return root, manifest, report, paths


@dataclass(slots=True)
class _EstimateAccumulator:
    total: float = 0.0
    count: int = 0
    side_totals: dict[str, float] | None = None
    side_counts: dict[str, int] | None = None
    group_totals: dict[str, float] | None = None
    group_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.side_totals = {}
        self.side_counts = {}
        self.group_totals = {}
        self.group_counts = {}

    def observe(self, symbol: str, side: str, value: float) -> None:
        if not math.isfinite(value):
            return
        assert self.side_totals is not None
        assert self.side_counts is not None
        assert self.group_totals is not None
        assert self.group_counts is not None
        group = f"{symbol}|{side}"
        self.total += value
        self.count += 1
        self.side_totals[side] = self.side_totals.get(side, 0.0) + value
        self.side_counts[side] = self.side_counts.get(side, 0) + 1
        self.group_totals[group] = self.group_totals.get(group, 0.0) + value
        self.group_counts[group] = self.group_counts.get(group, 0) + 1

    def result(self, *, minimum_group_rows: int, clip: bool = False) -> dict[str, object]:
        if self.count == 0:
            raise ShadowBundleError("calibration-fit window has no execution estimate rows")
        assert self.side_totals is not None
        assert self.side_counts is not None
        assert self.group_totals is not None
        assert self.group_counts is not None

        def selected(value: float) -> float:
            return min(1.0, max(0.0, value)) if clip else value

        return {
            "global": selected(self.total / self.count),
            "by_side": {
                side: selected(total / self.side_counts[side])
                for side, total in sorted(self.side_totals.items())
            },
            "by_symbol_side": {
                group: selected(total / self.group_counts[group])
                for group, total in sorted(self.group_totals.items())
                if self.group_counts[group] >= minimum_group_rows
            },
            "observations": self.count,
            "minimum_symbol_side_rows": minimum_group_rows,
            "fallback_order": ["symbol_side", "side", "global"],
        }


def _selected_scenario(report: dict[str, Any]) -> tuple[int, float]:
    scenario = _object(report.get("selected_execution_scenario"), "selected scenario")
    horizon = _plain_int(scenario.get("horizon_minutes"), "horizon_minutes")
    notional = _finite_float(
        scenario.get("reference_order_notional_usdt"),
        "reference_order_notional_usdt",
    )
    if horizon <= 0 or notional <= 0:
        raise ShadowBundleError("selected execution scenario must be positive")
    return horizon, notional


def _fit_estimates(
    label_paths: tuple[Path, ...],
    *,
    fit_purge_cutoff_ns: int,
    horizon_minutes: int,
    order_notional_usdt: float,
) -> dict[str, object]:
    timeout = _EstimateAccumulator()
    partial = _EstimateAccumulator()
    columns = [
        "symbol",
        "side",
        "horizon_minutes",
        "order_notional_usdt",
        "entry_window_end_ns",
        "position_end_ns",
        "fill_status",
        "fill_fraction",
        "outcome",
        "outcome_return_bps",
    ]
    for path in label_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
            table = pa.Table.from_batches([batch])
            scenario_mask = pc.and_(
                pc.equal(table["horizon_minutes"], pa.scalar(horizon_minutes)),
                pc.equal(
                    table["order_notional_usdt"], pa.scalar(order_notional_usdt)
                ),
            )
            table = table.filter(scenario_mask)
            for row in table.to_pylist():
                item = cast(dict[str, object], row)
                fill = str(item["fill_status"])
                label_end_raw = (
                    item["position_end_ns"]
                    if fill == "FULL_FILL"
                    else item["entry_window_end_ns"]
                )
                if label_end_raw is None or int(cast(int, label_end_raw)) > fit_purge_cutoff_ns:
                    continue
                symbol = str(item["symbol"])
                side = str(item["side"])
                if fill == "PARTIAL_FILL" and item["fill_fraction"] is not None:
                    partial.observe(symbol, side, float(cast(float, item["fill_fraction"])))
                if (
                    fill == "FULL_FILL"
                    and item["outcome"] == "TIMEOUT"
                    and item["outcome_return_bps"] is not None
                ):
                    timeout.observe(
                        symbol,
                        side,
                        float(cast(float, item["outcome_return_bps"])),
                    )
    return {
        "timeout_return_bps": timeout.result(minimum_group_rows=20),
        "partial_fill_fraction": partial.result(
            minimum_group_rows=10, clip=True
        ),
    }


def _calibrators(report: dict[str, Any], fold_index: int) -> dict[str, Any]:
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ShadowBundleError("execution report has no folds")
    matching = [
        _object(item, "fold")
        for item in folds
        if isinstance(item, dict) and item.get("fold") == fold_index
    ]
    if len(matching) != 1:
        raise ShadowBundleError("selected execution fold is not unique")
    models = _object(matching[0].get("models"), "fold.models")
    lightgbm = _object(models.get("lightgbm"), "fold.models.lightgbm")
    calibrators = _object(lightgbm.get("calibrators"), "LightGBM calibrators")
    for component, expected_classes in (
        ("fill", FILL_NAMES),
        ("post_fill_outcome", EXECUTION_OUTCOME_NAMES),
    ):
        value = _object(calibrators.get(component), f"calibrators.{component}")
        prior = _object(value.get("class_prior"), f"{component}.class_prior")
        if set(prior) != set(expected_classes):
            raise ShadowBundleError(f"{component} calibrator classes are invalid")
        _finite_float(value.get("temperature"), f"{component}.temperature")
        _finite_float(value.get("prior_weight"), f"{component}.prior_weight")
    return calibrators


def _output_descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def validate_shadow_bundle(bundle_path: str | Path) -> ShadowBundle:
    root = Path(bundle_path).expanduser().resolve()
    if not root.is_dir():
        raise ShadowBundleError(f"shadow bundle does not exist: {root}")
    manifest = _load_json(root / "manifest.json", "shadow bundle manifest")
    contract = _load_json(root / "bundle.json", "shadow bundle contract")
    if (
        manifest.get("shadow_bundle_schema_version") != SHADOW_BUNDLE_SCHEMA_VERSION
        or contract.get("shadow_bundle_schema_version") != SHADOW_BUNDLE_SCHEMA_VERSION
    ):
        raise ShadowBundleError("unsupported shadow bundle schema")
    bundle_id = _string(manifest.get("bundle_id"), "bundle_id")
    if root.name != bundle_id or contract.get("bundle_id") != bundle_id:
        raise ShadowBundleError("shadow bundle identity is inconsistent")
    input_fingerprint = _string(
        manifest.get("input_fingerprint"), "input_fingerprint"
    )
    if (
        len(input_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in input_fingerprint)
        or contract.get("input_fingerprint") != input_fingerprint
        or bundle_id
        != f"shadow-bundle-v{SHADOW_BUNDLE_SCHEMA_VERSION}-{input_fingerprint[:16]}"
    ):
        raise ShadowBundleError("shadow bundle input identity is inconsistent")
    descriptors, paths = _validate_descriptors(
        root, manifest.get("files"), label="shadow bundle"
    )
    fingerprint = _sha256_json(
        sorted(descriptors, key=lambda item: cast(str, item["path"]))
    )
    if manifest.get("bundle_fingerprint") != fingerprint:
        raise ShadowBundleError("shadow bundle fingerprint is inconsistent")
    if paths.get("bundle.json") != (root / "bundle.json").resolve():
        raise ShadowBundleError("shadow manifest does not authenticate bundle.json")
    scope = _object(contract.get("scope"), "scope")
    required_scope = {
        "bybit_access": "public-read-only",
        "order_submission": False,
        "trading_credentials_allowed": False,
        "eligible_for_trading": False,
        "openai_tokens": 0,
    }
    for key, expected in required_scope.items():
        if scope.get(key) != expected:
            raise ShadowBundleError(f"unsafe shadow scope: {key}")
    data_gate = _object(contract.get("data_gate"), "data_gate")
    review_eligible = data_gate.get("eligible_for_execution_model_review")
    if not isinstance(review_eligible, bool) or scope.get("engineering_only") is not (
        not review_eligible
    ):
        raise ShadowBundleError("shadow engineering-only scope contradicts data gate")
    models = _object(contract.get("models"), "models")
    fill_relative = _safe_relative(models.get("fill"), "models.fill").as_posix()
    outcome_relative = _safe_relative(
        models.get("post_fill_outcome"), "models.post_fill_outcome"
    ).as_posix()
    try:
        fill_path = paths[fill_relative]
        outcome_path = paths[outcome_relative]
    except KeyError as exc:
        raise ShadowBundleError("shadow model is not authenticated by manifest") from exc
    return ShadowBundle(
        root=root,
        bundle_id=bundle_id,
        bundle_fingerprint=fingerprint,
        manifest=manifest,
        contract=contract,
        fill_model_path=fill_path,
        outcome_model_path=outcome_path,
        reused=False,
    )


def build_shadow_bundle(
    *,
    execution_evaluation: str | Path,
    execution_dataset: str | Path,
    output_root: str | Path,
    allow_technical_smoke: bool = False,
) -> ShadowBundle:
    """Freeze one already-evaluated fold; this function never retrains a model."""

    evaluation_root, evaluation_manifest, report, evaluation_paths = (
        _load_execution_evaluation(execution_evaluation)
    )
    try:
        dataset = validate_execution_research_dataset(execution_dataset)
    except EvaluationError as exc:
        raise ShadowBundleError(f"execution dataset failed validation: {exc}") from exc
    research = _object(
        report.get("execution_research_dataset"), "execution_research_dataset"
    )
    if (
        research.get("execution_dataset_id") != dataset.execution_dataset_id
        or research.get("input_fingerprint") != dataset.input_fingerprint
        or research.get("output_fingerprint") != dataset.output_fingerprint
        or evaluation_manifest.get("execution_dataset_id")
        != dataset.execution_dataset_id
    ):
        raise ShadowBundleError("execution evaluation and dataset provenance differ")
    comparisons = _object(
        report.get("pre_registered_comparisons"), "pre_registered_comparisons"
    )
    if comparisons.get("primary_candidate_model") != PRIMARY_MODEL:
        raise ShadowBundleError("execution evaluation primary model is unsupported")
    gate = _object(report.get("data_gate"), "data_gate")
    mode = _string(gate.get("mode"), "data_gate.mode")
    review_eligibility = gate.get("eligible_for_execution_model_review")
    if not isinstance(review_eligibility, bool):
        raise ShadowBundleError("data gate review eligibility must be boolean")
    engineering_only = not review_eligibility
    if engineering_only and not allow_technical_smoke:
        raise ShadowBundleError(
            "unreviewed evaluation requires --allow-engineering-only "
            "(or its --allow-technical-smoke compatibility alias)"
        )
    if mode not in {"technical_smoke", "walk_forward"}:
        raise ShadowBundleError(f"unsupported execution data mode: {mode}")
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ShadowBundleError("execution evaluation has no frozen fold")
    fold_numbers = sorted(
        _plain_int(_object(item, "fold").get("fold"), "fold.fold")
        for item in folds
    )
    selected_fold = fold_numbers[-1]
    fold_report = next(
        _object(item, "fold")
        for item in folds
        if isinstance(item, dict) and item.get("fold") == selected_fold
    )
    nested = _object(fold_report.get("nested_calibration"), "nested_calibration")
    fit_cutoff = _plain_int(
        nested.get("fit_purge_cutoff_ns"), "fit_purge_cutoff_ns"
    )
    horizon, notional = _selected_scenario(report)
    fill_source_name = f"models/fold-{selected_fold:02d}-lightgbm-fill.txt"
    outcome_source_name = f"models/fold-{selected_fold:02d}-lightgbm-outcome.txt"
    try:
        fill_source = evaluation_paths[fill_source_name]
        outcome_source = evaluation_paths[outcome_source_name]
    except KeyError as exc:
        raise ShadowBundleError("selected fold LightGBM model file is missing") from exc
    estimates = _fit_estimates(
        dataset.label_paths,
        fit_purge_cutoff_ns=fit_cutoff,
        horizon_minutes=horizon,
        order_notional_usdt=notional,
    )
    calibrators = _calibrators(report, selected_fold)
    raw_features = report.get("feature_names")
    raw_symbols = research.get("symbols")
    if (
        not isinstance(raw_features, list)
        or not raw_features
        or not all(isinstance(item, str) and item for item in raw_features)
        or not isinstance(raw_symbols, list)
        or tuple(raw_symbols) != dataset.symbols
    ):
        raise ShadowBundleError("execution model feature/symbol contract is invalid")
    parameters = _object(report.get("parameters"), "parameters")
    execution_parameters = _object(dataset.manifest.get("parameters"), "parameters")
    parameter_copy = {
        key: value for key, value in execution_parameters.items() if key != "fingerprint"
    }
    input_payload = {
        "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
        "package_version": __version__,
        "execution_experiment_id": report["experiment_id"],
        "execution_evaluation_output_fingerprint": evaluation_manifest[
            "output_fingerprint"
        ],
        "execution_dataset_id": dataset.execution_dataset_id,
        "execution_dataset_output_fingerprint": dataset.output_fingerprint,
        "selected_fold": selected_fold,
        "primary_model": PRIMARY_MODEL,
    }
    input_fingerprint = _sha256_json(input_payload)
    bundle_id = f"shadow-bundle-v{SHADOW_BUNDLE_SCHEMA_VERSION}-{input_fingerprint[:16]}"
    destination_root = Path(output_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    final_path = destination_root / bundle_id
    if final_path.exists():
        existing = validate_shadow_bundle(final_path)
        if existing.contract.get("input_fingerprint") != input_fingerprint:
            raise ShadowBundleError(
                "existing shadow bundle collides with a different frozen input"
            )
        return ShadowBundle(
            root=existing.root,
            bundle_id=existing.bundle_id,
            bundle_fingerprint=existing.bundle_fingerprint,
            manifest=existing.manifest,
            contract=existing.contract,
            fill_model_path=existing.fill_model_path,
            outcome_model_path=existing.outcome_model_path,
            reused=True,
        )

    staging = destination_root / f".{bundle_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        models_root = staging / "models"
        models_root.mkdir()
        fill_target = models_root / "fill.txt"
        outcome_target = models_root / "post-fill-outcome.txt"
        shutil.copyfile(fill_source, fill_target)
        shutil.copyfile(outcome_source, outcome_target)
        contract: dict[str, object] = {
            "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "input_fingerprint": input_fingerprint,
            "created_by_package_version": __version__,
            "source": {
                **input_payload,
                "execution_evaluation_path_name": evaluation_root.name,
            },
            "model": {
                "name": PRIMARY_MODEL,
                "family": "lightgbm",
                "probability_variant": "calibrated",
                "selected_fold": selected_fold,
                "feature_names": list(cast(list[str], raw_features)),
                "fill_classes": list(FILL_NAMES),
                "post_fill_outcome_classes": list(EXECUTION_OUTCOME_NAMES),
                "calibrators": calibrators,
                "execution_estimates": estimates,
            },
            "models": {
                "fill": "models/fill.txt",
                "post_fill_outcome": "models/post-fill-outcome.txt",
            },
            "universe": {
                "symbols": list(dataset.symbols),
                "one_position_across_all_symbols": True,
            },
            "scenario": {
                "horizon_minutes": horizon,
                "reference_order_notional_usdt": notional,
            },
            "research_parameters": parameter_copy,
            "evaluation_parameters": parameters,
            "data_gate": gate,
            "scope": {
                "bybit_access": "public-read-only",
                "order_submission": False,
                "trading_credentials_allowed": False,
                "real_exchange_orders_observed": False,
                "maker_fill_is_public_data_proxy": True,
                "real_queue_position_observed": False,
                "engineering_only": engineering_only,
                "eligible_for_trading": False,
                "openai_tokens": 0,
            },
        }
        _write_json(staging / "bundle.json", contract)
        descriptors = [
            _output_descriptor(staging, staging / "bundle.json"),
            _output_descriptor(staging, fill_target),
            _output_descriptor(staging, outcome_target),
        ]
        descriptors.sort(key=lambda item: cast(str, item["path"]))
        bundle_fingerprint = _sha256_json(descriptors)
        manifest = {
            "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "input_fingerprint": input_fingerprint,
            "bundle_fingerprint": bundle_fingerprint,
            "output_file_count": len(descriptors),
            "files": descriptors,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, final_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_shadow_bundle(final_path)
