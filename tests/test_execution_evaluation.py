from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.cli import main
from tradingbot.config import AppConfig, load_config
from tradingbot.research.contracts import (
    EXECUTION_FEATURE_SCHEMA,
    EXECUTION_LABEL_SCHEMA,
    EXECUTION_RESEARCH_PROFILE,
    EXECUTION_RESEARCH_SCHEMA_VERSION,
)
from tradingbot.research.evaluation_contracts import (
    NS_PER_MINUTE,
    EvaluationError,
)
from tradingbot.research.evaluator import evaluation_parameters
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_EVALUATION_SCHEMA_VERSION,
    EXECUTION_OUTCOME_NAMES,
    FILL_NAMES,
    ExecutionEvaluationResult,
)
from tradingbot.research.execution_evaluation_dataset import (
    prepare_execution_evaluation_data,
    validate_execution_research_dataset,
)
from tradingbot.research.execution_evaluator import run_execution_evaluation
from tradingbot.research.execution_splits import (
    build_execution_calibration_split,
    build_execution_temporal_folds,
)
from tradingbot.research.models import fit_probability_calibrator

BASE_NS = 1_774_137_600 * 1_000_000_000


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Avoid Python 3.13's sandbox-incompatible Windows 0o700 temp ACL."""

    base = Path.cwd() / ".test-tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _default_row(schema: pa.Schema) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_string(field.type):
            row[field.name] = "fixture"
        elif pa.types.is_integer(field.type):
            row[field.name] = 1
        elif pa.types.is_floating(field.type):
            row[field.name] = 1.0
        elif pa.types.is_boolean(field.type):
            row[field.name] = True
        else:  # pragma: no cover - the immutable schemas are intentionally narrow
            raise AssertionError(f"unsupported fixture field: {field}")
    return row


def _feature_row(
    symbol: str,
    symbol_index: int,
    decision_index: int,
    *,
    step_minutes: int,
) -> dict[str, object]:
    decision_ns = BASE_NS + decision_index * step_minutes * NS_PER_MINUTE
    row = _default_row(EXECUTION_FEATURE_SCHEMA)
    mid = 100.0 + symbol_index * 20 + decision_index / 100
    row.update(
        {
            "research_schema_version": EXECUTION_RESEARCH_SCHEMA_VERSION,
            "decision_id": f"fixture:{symbol}:{decision_ns}",
            "source_dataset_id": "canonical-v1-execution-fixture",
            "symbol": symbol,
            "decision_at_ns": decision_ns,
            "decision_at_ms": decision_ns // 1_000_000,
            "decision_utc_date": datetime.fromtimestamp(
                decision_ns / 1_000_000_000, tz=UTC
            ).date().isoformat(),
            "book_received_at_ns": decision_ns - 500_000_000,
            "ticker_received_at_ns": decision_ns - 500_000_000,
            "latest_kline_received_at_ns": decision_ns - NS_PER_MINUTE,
            "latest_trade_received_at_ns": decision_ns - 100_000_000,
            "reference_mid_price": mid,
            "best_bid_price": mid - 0.01,
            "best_ask_price": mid + 0.01,
            "close_price": mid,
            "mark_price": mid,
            "index_price": mid - 0.005,
            "return_1m_fraction": ((decision_index % 11) - 5) / 10_000,
            "return_3m_fraction": ((decision_index % 13) - 6) / 8_000,
            "return_5m_fraction": ((decision_index % 17) - 8) / 7_000,
            "return_15m_fraction": ((decision_index % 19) - 9) / 5_000,
            "return_60m_fraction": ((decision_index % 23) - 11) / 3_000,
            "trade_imbalance_1m": ((decision_index % 9) - 4) / 5,
            "book_imbalance_5": ((decision_index % 7) - 3) / 4,
            "funding_rate": 0.0001,
            "minutes_to_funding": float(30 + decision_index % 120),
        }
    )
    return row


def _label_row(
    feature: dict[str, object],
    symbol_index: int,
    decision_index: int,
    side_index: int,
) -> dict[str, object]:
    decision_ns = cast(int, feature["decision_at_ns"])
    side = "LONG" if side_index == 0 else "SHORT"
    fill_status = FILL_NAMES[
        (decision_index + symbol_index + side_index) % len(FILL_NAMES)
    ]
    entry_window_end_ns = decision_ns + 30 * 1_000_000_000
    row = _default_row(EXECUTION_LABEL_SCHEMA)
    row.update(
        {
            "execution_research_schema_version": (
                EXECUTION_RESEARCH_SCHEMA_VERSION
            ),
            "decision_id": feature["decision_id"],
            "source_dataset_id": feature["source_dataset_id"],
            "symbol": feature["symbol"],
            "decision_at_ns": decision_ns,
            "decision_utc_date": feature["decision_utc_date"],
            "side": side,
            "horizon_minutes": 15,
            "order_notional_usdt": 50.0,
            "submitted_at_ns": decision_ns + 250_000_000,
            "activation_at_ns": decision_ns + 500_000_000,
            "activation_delay_ms": 500.0,
            "entry_window_end_ns": entry_window_end_ns,
            "entry_limit_price": 100.0,
            "order_size_base": 0.5,
            "activation_best_bid_price": 99.99,
            "activation_best_ask_price": 100.01,
            "post_only_valid": True,
            "queue_ahead_size_base": 1.0,
            "queue_required_size_base": 1.5,
            "entry_window_trade_count": 5,
            "contra_trade_count": 3,
            "contra_volume_at_entry_price_base": 1.5,
            "fill_status": fill_status,
            "stop_distance_bps": 10.0,
            "take_profit_distance_bps": 15.0,
            "stop_price": 99.9 if side == "LONG" else 100.1,
            "take_profit_price": 100.15 if side == "LONG" else 99.85,
            "future_trade_count": 10,
        }
    )
    nullable_fields = (
        "first_fill_at_ns",
        "full_fill_at_ns",
        "full_fill_event_ts_ms",
        "full_fill_sequence",
        "full_fill_trade_price",
        "time_to_full_fill_ms",
        "position_end_ns",
        "hit_at_ns",
        "hit_event_ts_ms",
        "hit_sequence",
        "hit_trade_price",
        "time_from_fill_to_hit_ms",
        "timeout_price",
        "outcome_return_bps",
    )
    row.update(dict.fromkeys(nullable_fields))
    if fill_status == "NO_FILL":
        row.update(
            {
                "fill_fraction": 0.0,
                "filled_size_base": 0.0,
                "outcome": "NO_FILL",
                "resolution": "entry_window_expired",
            }
        )
    elif fill_status == "PARTIAL_FILL":
        row.update(
            {
                "fill_fraction": 0.5,
                "filled_size_base": 0.25,
                "first_fill_at_ns": decision_ns + 1_000_000_000,
                "outcome": "PARTIAL_FILL",
                "resolution": "entry_window_partial",
            }
        )
    else:
        full_fill_at_ns = decision_ns + 1_000_000_000
        outcome = EXECUTION_OUTCOME_NAMES[
            (decision_index // 3 + symbol_index + side_index)
            % len(EXECUTION_OUTCOME_NAMES)
        ]
        position_end_ns = full_fill_at_ns + 15 * NS_PER_MINUTE
        outcome_return = {
            "SL_FIRST": -10.0,
            "TIMEOUT": 2.0 if side == "LONG" else -2.0,
            "TP_FIRST": 15.0,
        }[outcome]
        hit_at_ns = (
            None
            if outcome == "TIMEOUT"
            else full_fill_at_ns + 5 * NS_PER_MINUTE
        )
        row.update(
            {
                "fill_fraction": 1.0,
                "filled_size_base": 0.5,
                "first_fill_at_ns": full_fill_at_ns,
                "full_fill_at_ns": full_fill_at_ns,
                "full_fill_event_ts_ms": full_fill_at_ns // 1_000_000,
                "full_fill_sequence": decision_index + 1,
                "full_fill_trade_price": 100.0,
                "time_to_full_fill_ms": 1_000.0,
                "position_end_ns": position_end_ns,
                "outcome": outcome,
                "hit_at_ns": hit_at_ns,
                "hit_event_ts_ms": (
                    None if hit_at_ns is None else hit_at_ns // 1_000_000
                ),
                "hit_sequence": (
                    None if hit_at_ns is None else decision_index + 2
                ),
                "hit_trade_price": None if hit_at_ns is None else 100.0,
                "time_from_fill_to_hit_ms": (
                    None if hit_at_ns is None else 5 * 60 * 1_000.0
                ),
                "timeout_price": 100.02 if outcome == "TIMEOUT" else None,
                "outcome_return_bps": outcome_return,
                "resolution": "resolved",
            }
        )
    return row


def execution_fixture(
    tmp_path: Path,
    *,
    decisions: int = 300,
    step_minutes: int = 10,
) -> Path:
    symbols = ("BTCUSDT", "ETHUSDT")
    features_by_partition: dict[tuple[str, str], list[dict[str, object]]] = {}
    labels_by_partition: dict[tuple[str, str], list[dict[str, object]]] = {}
    for decision_index in range(decisions):
        for symbol_index, symbol in enumerate(symbols):
            feature = _feature_row(
                symbol,
                symbol_index,
                decision_index,
                step_minutes=step_minutes,
            )
            partition = cast(str, feature["decision_utc_date"])
            features_by_partition.setdefault((symbol, partition), []).append(feature)
            for side_index in range(2):
                labels_by_partition.setdefault((symbol, partition), []).append(
                    _label_row(feature, symbol_index, decision_index, side_index)
                )

    source_manifest_text = json.dumps({"fixture": True}, sort_keys=True)
    source_manifest_sha = hashlib.sha256(source_manifest_text.encode()).hexdigest()
    source_dataset_id = "archive-catalog-fixture"
    source_output_fingerprint = "2" * 64
    parameter_payload = {
        "position_horizons_minutes": [15],
        "order_notionals_usdt": [50.0],
    }
    parameter_fingerprint = _sha256_json(parameter_payload)
    builder = {
        "package_version": "fixture",
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
    }
    input_fingerprint = _sha256_json(
        {
            "execution_research_schema_version": (
                EXECUTION_RESEARCH_SCHEMA_VERSION
            ),
            "research_profile": EXECUTION_RESEARCH_PROFILE,
            "package_version": builder["package_version"],
            "pyarrow_version": builder["pyarrow_version"],
            "numpy_version": builder["numpy_version"],
            "source_dataset_id": source_dataset_id,
            "source_manifest_sha256": source_manifest_sha,
            "source_output_fingerprint": source_output_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
        }
    )
    root = tmp_path / f"execution-research-v1-{input_fingerprint[:16]}"
    root.mkdir(parents=True)
    source_manifest = root / "source-manifest.json"
    source_manifest.write_text(source_manifest_text, encoding="utf-8")
    descriptors: list[dict[str, object]] = []
    row_totals = {"features": 0, "execution_labels": 0}
    for table_name, schema, partitions in (
        ("features", EXECUTION_FEATURE_SCHEMA, features_by_partition),
        ("execution_labels", EXECUTION_LABEL_SCHEMA, labels_by_partition),
    ):
        for (symbol, partition), rows in sorted(partitions.items()):
            relative = Path(
                f"table={table_name}/symbol={symbol}/date={partition}/part-00000.parquet"
            )
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                path,
                version="2.6",
                compression="zstd",
                write_page_checksum=True,
            )
            descriptors.append(
                {
                    "path": relative.as_posix(),
                    "table": table_name,
                    "symbol": symbol,
                    "date": partition,
                    "rows": len(rows),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            row_totals[table_name] += len(rows)

    partition_dates = sorted(
        {partition for _, partition in features_by_partition}
    )
    manifest = {
        "execution_research_schema_version": EXECUTION_RESEARCH_SCHEMA_VERSION,
        "execution_dataset_id": root.name,
        "research_profile": EXECUTION_RESEARCH_PROFILE,
        "input_fingerprint": input_fingerprint,
        "builder": builder,
        "output_fingerprint": _sha256_json(
            sorted(descriptors, key=lambda item: cast(str, item["path"]))
        ),
        "scope": {
            "real_exchange_orders_observed": False,
            "maker_fill_is_proxy": True,
            "partial_fills_retained": True,
            "eligible_for_fill_model_training": True,
            "eligible_for_profitability_conclusion": False,
        },
        "causality": {
            "feature_rule": "received_at_ns <= decision_at_ns",
            "partial_fill_class": "PARTIAL_FILL",
            "no_fill_class": "NO_FILL",
        },
        "schemas": {
            "features": _schema_manifest(EXECUTION_FEATURE_SCHEMA),
            "execution_labels": _schema_manifest(EXECUTION_LABEL_SCHEMA),
        },
        "source": {
            "dataset_id": source_dataset_id,
            "output_fingerprint": source_output_fingerprint,
            "symbols": list(symbols),
            "partition_dates": partition_dates,
            "manifest_copy": source_manifest.name,
            "manifest_sha256": source_manifest_sha,
        },
        "parameters": {
            **parameter_payload,
            "fingerprint": parameter_fingerprint,
        },
        "output_file_count": len(descriptors),
        "output_rows": row_totals,
        "files": descriptors,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _small_config(config_path: Path) -> AppConfig:
    config = load_config(config_path)
    return replace(
        config,
        bybit=replace(config.bybit, symbols=("BTCUSDT", "ETHUSDT")),
        evaluation=replace(
            config.evaluation,
            horizon_minutes=15,
            embargo_minutes=15,
            minimum_train_rows=100,
            minimum_test_rows=50,
            calibration_days=1,
            minimum_calibration_rows=30,
            minimum_expected_net_bps=-1_000.0,
            lightgbm_estimators=5,
            lightgbm_min_child_samples=5,
            logistic_max_training_rows=500,
            training_threads=1,
        ),
    )


def test_validates_prepares_and_rejects_corrupted_execution_dataset(
    tmp_path: Path,
) -> None:
    root = execution_fixture(tmp_path, decisions=120)
    dataset = validate_execution_research_dataset(root)
    data = prepare_execution_evaluation_data(
        dataset, horizon_minutes=15, order_notional_usdt=50.0
    )

    assert dataset.symbols == ("BTCUSDT", "ETHUSDT")
    assert data.rows == 120 * 2 * 2
    assert set(np.unique(data.fill_y)) == {0, 1, 2}
    assert set(np.unique(data.outcome_y[data.outcome_y >= 0])) == {0, 1, 2}
    assert "activation_delay_ms" not in data.feature_names
    assert "queue_ahead_size_base" not in data.feature_names
    assert all(data.label_end_ns > data.decision_at_ns)

    manifest_path = root / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    tampered = json.loads(manifest_text)
    tampered["parameters"]["order_notionals_usdt"] = [50.0, 100.0]
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(EvaluationError, match="parameter fingerprint"):
        validate_execution_research_dataset(root)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    label_path = next(root.glob("table=execution_labels/**/*.parquet"))
    content = label_path.read_bytes()
    label_path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(EvaluationError, match="corrupted"):
        validate_execution_research_dataset(root)


def test_execution_splits_purge_fill_and_post_fill_labels(
    config_path: Path,
    tmp_path: Path,
) -> None:
    root = execution_fixture(tmp_path, decisions=400, step_minutes=10)
    dataset = validate_execution_research_dataset(root)
    data = prepare_execution_evaluation_data(
        dataset, horizon_minutes=15, order_notional_usdt=50.0
    )
    parameters = evaluation_parameters(_small_config(config_path))
    folds = build_execution_temporal_folds(data, parameters)
    split = build_execution_calibration_split(data, folds[0], parameters)

    assert folds[0].mode == "technical_smoke"
    assert np.intersect1d(folds[0].train_indices, folds[0].test_indices).size == 0
    assert np.intersect1d(split.fit_indices, split.calibration_indices).size == 0
    assert np.max(data.label_end_ns[split.fit_indices]) <= split.fit_purge_cutoff_ns
    assert np.max(data.label_end_ns[folds[0].train_indices]) < folds[0].test_start_ns


def test_fill_probability_calibration_supports_execution_classes() -> None:
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    y_true = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)

    calibrator = fit_probability_calibrator(
        probabilities, y_true, class_names=FILL_NAMES
    )
    transformed = calibrator.transform(probabilities)

    assert calibrator.class_names == FILL_NAMES
    assert set(cast(dict[str, float], calibrator.to_dict()["class_prior"])) == set(
        FILL_NAMES
    )
    np.testing.assert_allclose(np.sum(transformed, axis=1), 1.0)


def test_execution_evaluation_is_immutable_and_explicitly_not_profit_ready(
    config_path: Path,
    tmp_path: Path,
) -> None:
    root = execution_fixture(tmp_path / "input", decisions=800, step_minutes=5)
    config = _small_config(config_path)
    output_root = tmp_path / "execution-evaluations"

    first = run_execution_evaluation(
        root,
        output_root,
        config=config,
        order_notional_usdt=50.0,
        minimum_free_bytes=0,
    )
    second = run_execution_evaluation(
        root,
        output_root,
        config=config,
        order_notional_usdt=50.0,
        minimum_free_bytes=0,
    )
    report = json.loads(first.report_path.read_text(encoding="utf-8"))

    assert first.reused is False
    assert second.reused is True
    assert report["data_gate"]["mode"] == "technical_smoke"
    assert report["data_gate"]["eligible_for_profitability_conclusion"] is False
    assert report["scope"]["maker_fill_modeled"] is True
    assert report["scope"]["real_queue_position_observed"] is False
    assert report["scope"]["one_position_across_all_symbols_enforced"] is True
    assert report["parameters"]["entry_adverse_selection_bps"] == 0.0
    assert report["parameters"]["entry_adverse_selection_handling"] == (
        "embedded_in_observed_proxy_fill_price_not_charged_twice"
    )
    assert set(report["models"]) == {
        "class_prior",
        "logistic_raw",
        "logistic_calibrated",
        "lightgbm_raw",
        "lightgbm_calibrated",
    }
    assert report["folds"][0]["nested_calibration"]["calibration_rows"] > 0
    assert report["models"]["lightgbm_calibrated"][
        "post_fill_outcome_classification"
    ]["rows"] > 0
    diagnostics = report["models"]["lightgbm_calibrated"][
        "selected_attempt_diagnostics"
    ]
    assert set(diagnostics["by_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(diagnostics["by_side"]) == {"LONG", "SHORT"}

    for attempt_path in first.experiment_path.glob("attempts/*.parquet"):
        table = pq.read_table(attempt_path)
        decisions = table.column("decision_at_ns").to_pylist()
        exits = table.column("exit_at_ns").to_pylist()
        assert all(
            decisions[index] >= exits[index - 1]
            for index in range(1, len(decisions))
        )
        assert "probability_full_fill" in table.column_names
        assert "realized_equity_return_fraction" in table.column_names

    target = first.experiment_path / "attempts" / "lightgbm_calibrated.parquet"
    target.unlink()
    with pytest.raises(EvaluationError, match="corrupted"):
        run_execution_evaluation(
            root,
            output_root,
            config=config,
            order_notional_usdt=50.0,
            minimum_free_bytes=0,
        )


def test_run_execution_backtest_cli_prints_reproducible_summary(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "execution-research-v1-fixture"
    output_root = tmp_path / "execution-evaluations"

    def fake_evaluation(
        execution_dataset: str | Path,
        output_root: str | Path,
        *,
        config: AppConfig,
        order_notional_usdt: float,
        minimum_free_bytes: int,
    ) -> ExecutionEvaluationResult:
        assert Path(execution_dataset) == dataset
        assert Path(output_root) == output_root_path
        assert config.evaluation.horizon_minutes == 30
        assert order_notional_usdt == 100.0
        assert minimum_free_bytes == 1
        experiment = output_root_path / "execution-backtest-v1-fixture"
        return ExecutionEvaluationResult(
            experiment_id=experiment.name,
            experiment_path=experiment,
            manifest_path=experiment / "manifest.json",
            report_path=experiment / "report.json",
            data_mode="technical_smoke",
            data_span_days=28.0,
            folds=1,
            reused=False,
        )

    output_root_path = output_root
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "1")
    monkeypatch.setattr(
        "tradingbot.research.execution_evaluator.run_execution_evaluation",
        fake_evaluation,
    )
    main(
        [
            "--config",
            str(config_path),
            "run-execution-backtest",
            "--execution-dataset",
            str(dataset),
            "--output-root",
            str(output_root),
            "--horizon-minutes",
            "30",
            "--order-notional-usdt",
            "100",
        ]
    )

    summary: dict[str, object] = json.loads(capsys.readouterr().out)
    assert summary["execution_evaluation_schema_version"] == (
        EXECUTION_EVALUATION_SCHEMA_VERSION
    )
    assert summary["data_mode"] == "technical_smoke"
    assert summary["reused"] is False
