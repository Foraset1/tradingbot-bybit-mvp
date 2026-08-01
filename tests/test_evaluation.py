from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.cli import main
from tradingbot.config import load_config
from tradingbot.research.contracts import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    RESEARCH_SCHEMA_VERSION,
)
from tradingbot.research.evaluation_contracts import (
    NS_PER_MINUTE,
    OUTCOME_NAMES,
    EvaluationError,
    EvaluationResult,
)
from tradingbot.research.evaluation_dataset import (
    prepare_evaluation_data,
    validate_research_dataset,
)
from tradingbot.research.evaluator import (
    evaluation_parameters,
    run_offline_evaluation,
)
from tradingbot.research.splits import build_temporal_folds

BASE_NS = 1_774_137_600 * 1_000_000_000


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


def _feature_row(symbol: str, symbol_index: int, minute: int) -> dict[str, object]:
    decision_ns = BASE_NS + minute * NS_PER_MINUTE
    row: dict[str, object] = {}
    for field in FEATURE_SCHEMA:
        if pa.types.is_string(field.type):
            row[field.name] = "fixture"
        elif pa.types.is_integer(field.type):
            row[field.name] = 1
        elif pa.types.is_floating(field.type):
            row[field.name] = 1.0
        else:  # pragma: no cover - the schema contract is intentionally narrow
            raise AssertionError(f"unsupported fixture field: {field}")
    row.update(
        {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "decision_id": f"fixture:{symbol}:{decision_ns}",
            "source_dataset_id": "canonical-v1-synthetic",
            "symbol": symbol,
            "decision_at_ns": decision_ns,
            "decision_at_ms": decision_ns // 1_000_000,
            "decision_utc_date": datetime.fromtimestamp(
                decision_ns / 1_000_000_000, tz=UTC
            ).date().isoformat(),
            "book_received_at_ns": decision_ns - 500_000_000,
            "ticker_received_at_ns": decision_ns - 500_000_000,
            "latest_kline_received_at_ns": decision_ns - 1_000_000_000,
            "latest_trade_received_at_ns": decision_ns - 100_000_000,
            "reference_mid_price": 100.0 + symbol_index * 10 + minute / 100,
            "best_bid_price": 99.99 + symbol_index * 10 + minute / 100,
            "best_ask_price": 100.01 + symbol_index * 10 + minute / 100,
            "close_price": 100.0 + symbol_index * 10 + minute / 100,
            "mark_price": 100.0 + symbol_index * 10 + minute / 100,
            "index_price": 99.995 + symbol_index * 10 + minute / 100,
            "return_1m_fraction": ((minute % 11) - 5) / 10_000,
            "return_3m_fraction": ((minute % 13) - 6) / 8_000,
            "return_5m_fraction": ((minute % 17) - 8) / 7_000,
            "return_15m_fraction": ((minute % 19) - 9) / 5_000,
            "return_60m_fraction": ((minute % 23) - 11) / 3_000,
            "trade_imbalance_1m": ((minute % 9) - 4) / 5,
            "book_imbalance_5": ((minute % 7) - 3) / 4,
            "funding_rate": 0.0001,
            "minutes_to_funding": float(30 + minute % 120),
        }
    )
    return row


def _label_rows(
    feature: dict[str, object], symbol_index: int, minute: int
) -> list[dict[str, object]]:
    decision_ns = cast(int, feature["decision_at_ns"])
    entry = cast(float, feature["reference_mid_price"])
    rows: list[dict[str, object]] = []
    for side_index, side in enumerate(("LONG", "SHORT")):
        outcome = OUTCOME_NAMES[(minute + symbol_index + side_index) % len(OUTCOME_NAMES)]
        if outcome == "SL_FIRST":
            gross = -20.0
            hit_at_ns: int | None = decision_ns + 15 * NS_PER_MINUTE
        elif outcome == "TP_FIRST":
            gross = 30.0
            hit_at_ns = decision_ns + 20 * NS_PER_MINUTE
        else:
            direction = 1.0 if side == "LONG" else -1.0
            gross = direction * ((minute % 9) - 4)
            hit_at_ns = None
        rows.append(
            {
                "research_schema_version": RESEARCH_SCHEMA_VERSION,
                "decision_id": feature["decision_id"],
                "source_dataset_id": feature["source_dataset_id"],
                "symbol": feature["symbol"],
                "decision_at_ns": decision_ns,
                "decision_utc_date": feature["decision_utc_date"],
                "side": side,
                "horizon_minutes": 60,
                "label_end_ns": decision_ns + 60 * NS_PER_MINUTE,
                "entry_reference_price": entry,
                "stop_distance_bps": 20.0,
                "take_profit_distance_bps": 30.0,
                "stop_price": entry * (0.998 if side == "LONG" else 1.002),
                "take_profit_price": entry * (1.003 if side == "LONG" else 0.997),
                "outcome": outcome,
                "hit_at_ns": hit_at_ns,
                "hit_event_ts_ms": None if hit_at_ns is None else hit_at_ns // 1_000_000,
                "hit_sequence": None if hit_at_ns is None else minute + 1,
                "hit_trade_price": None if hit_at_ns is None else entry * (1 + gross / 10_000),
                "time_to_hit_ms": (
                    None if hit_at_ns is None else (hit_at_ns - decision_ns) / 1_000_000
                ),
                "timeout_price": (
                    entry * (1 + gross / 10_000) if hit_at_ns is None else None
                ),
                "outcome_return_bps": gross,
                "future_trade_count": 100,
                "resolution": (
                    "complete_horizon_no_barrier"
                    if hit_at_ns is None
                    else "public_trade_received_event_sequence"
                ),
            }
        )
    return rows


def research_fixture(tmp_path: Path, *, decisions: int = 720) -> Path:
    dataset_id = "research-v1-synthetic"
    root = tmp_path / dataset_id
    root.mkdir(parents=True)
    symbols = ("BTCUSDT", "ETHUSDT")
    files: list[dict[str, object]] = []
    totals = {"features": 0, "labels": 0}
    outcome_counts: dict[str, int] = {name: 0 for name in OUTCOME_NAMES}
    for symbol_index, symbol in enumerate(symbols):
        feature_rows = [
            _feature_row(symbol, symbol_index, minute) for minute in range(decisions)
        ]
        label_rows = [
            row
            for minute, feature in enumerate(feature_rows)
            for row in _label_rows(feature, symbol_index, minute)
        ]
        for row in label_rows:
            outcome = cast(str, row["outcome"])
            outcome_counts[outcome] += 1
        date = cast(str, feature_rows[0]["decision_utc_date"])
        for table_name, rows, schema in (
            ("features", feature_rows, FEATURE_SCHEMA),
            ("labels", label_rows, LABEL_SCHEMA),
        ):
            relative = (
                Path(f"table={table_name}")
                / f"symbol={symbol}"
                / f"date={date}"
                / "part-00000.parquet"
            )
            path = root / relative
            path.parent.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                path,
                version="2.6",
                compression="zstd",
                compression_level=3,
                write_page_checksum=True,
            )
            files.append(
                {
                    "path": relative.as_posix(),
                    "table": table_name,
                    "symbol": symbol,
                    "date": date,
                    "rows": len(rows),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            totals[table_name] += len(rows)

    source_manifest = {
        "dataset_schema_version": 1,
        "dataset_id": "canonical-v1-synthetic",
        "output_fingerprint": "1" * 64,
    }
    source_path = root / "source-manifest.json"
    source_path.write_text(
        json.dumps(source_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    sorted_files = sorted(files, key=lambda item: cast(str, item["path"]))
    manifest = {
        "research_schema_version": RESEARCH_SCHEMA_VERSION,
        "research_dataset_id": dataset_id,
        "input_fingerprint": "2" * 64,
        "source": {
            "dataset_id": "canonical-v1-synthetic",
            "manifest_copy": "source-manifest.json",
            "manifest_sha256": _sha256(source_path),
            "output_fingerprint": "1" * 64,
            "symbols": list(symbols),
            "bytes": 1,
        },
        "parameters": {"label_horizons_minutes": [60], "fingerprint": "3" * 64},
        "causality": {
            "feature_rule": "received_at_ns <= decision_at_ns",
            "label_rule": "decision_at_ns < trade.received_at_ns <= label_end_ns",
            "execution_labels_included": False,
            "maker_fill_claimed": False,
        },
        "schemas": {
            "features": _schema_manifest(FEATURE_SCHEMA),
            "labels": _schema_manifest(LABEL_SCHEMA),
        },
        "label_outcomes": outcome_counts,
        "labels_by_horizon": {"60m": totals["labels"]},
        "quality_by_symbol": {},
        "output_rows": totals,
        "output_file_count": len(sorted_files),
        "output_fingerprint": _sha256_json(sorted_files),
        "files": sorted_files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_validates_and_prepares_causal_evaluation_rows(tmp_path: Path) -> None:
    root = research_fixture(tmp_path)

    dataset = validate_research_dataset(root)
    prepared = prepare_evaluation_data(dataset, horizon_minutes=60)

    assert dataset.symbols == ("BTCUSDT", "ETHUSDT")
    assert prepared.rows == 720 * 2 * 2
    assert prepared.x.shape[1] == len(prepared.feature_names)
    assert prepared.x.shape[1] > 80
    assert set(prepared.y) == {0, 1, 2}
    assert prepared.excluded_ambiguous_rows == 0
    assert prepared.excluded_unpriced_rows == 0


def test_temporal_smoke_split_purges_future_labels(
    config_path: Path, tmp_path: Path
) -> None:
    dataset = validate_research_dataset(research_fixture(tmp_path))
    prepared = prepare_evaluation_data(dataset, horizon_minutes=60)
    parameters = evaluation_parameters(load_config(config_path))

    folds = build_temporal_folds(prepared, parameters)

    assert len(folds) == 1
    assert folds[0].mode == "technical_smoke"
    assert max(prepared.label_end_ns[folds[0].train_indices]) <= (
        folds[0].test_start_ns - parameters.embargo_minutes * NS_PER_MINUTE
    )
    assert not set(folds[0].train_indices).intersection(folds[0].test_indices)


def test_rejects_corrupted_research_input(tmp_path: Path) -> None:
    root = research_fixture(tmp_path, decisions=20)
    feature_path = next(root.glob("table=features/**/*.parquet"))
    content = feature_path.read_bytes()
    feature_path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

    with pytest.raises(EvaluationError, match="corrupted"):
        validate_research_dataset(root)


def test_offline_evaluation_is_idempotent_and_explicitly_smoke_only(
    config_path: Path, tmp_path: Path
) -> None:
    research_root = research_fixture(tmp_path / "input")
    config = load_config(config_path)
    config = replace(
        config,
        evaluation=replace(
            config.evaluation,
            lightgbm_estimators=25,
            lightgbm_min_child_samples=10,
            training_threads=1,
        ),
    )
    output_root = tmp_path / "evaluations"

    first = run_offline_evaluation(
        research_root, output_root, config=config, minimum_free_bytes=0
    )
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    second = run_offline_evaluation(
        research_root, output_root, config=config, minimum_free_bytes=0
    )

    assert first.reused is False
    assert second.reused is True
    assert second.experiment_id == first.experiment_id
    assert report["data_gate"]["mode"] == "technical_smoke"
    assert report["data_gate"]["minimum_history_met"] is False
    assert report["data_gate"]["eligible_for_profitability_conclusion"] is False
    assert set(report["models"]) == {"class_prior", "logistic", "lightgbm"}
    assert report["scope"]["maker_fill_modeled"] is False
    for trade_path in first.experiment_path.glob("trades/*.parquet"):
        table = pq.read_table(trade_path)
        decisions = table.column("decision_at_ns").to_pylist()
        exits = table.column("exit_at_ns").to_pylist()
        assert all(
            decisions[index] >= exits[index - 1]
            for index in range(1, len(decisions))
        )

    (first.experiment_path / "trades" / "lightgbm.parquet").unlink()
    with pytest.raises(EvaluationError, match="corrupted"):
        run_offline_evaluation(
            research_root, output_root, config=config, minimum_free_bytes=0
        )


def test_run_backtest_cli_prints_reproducible_summary(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    research_root = tmp_path / "research-v1-fixture"
    expected_output_root = tmp_path / "evaluations"

    def fake_evaluation(
        research_dataset: str | Path,
        output_root: str | Path,
        *,
        config: object,
        minimum_free_bytes: int,
    ) -> EvaluationResult:
        assert Path(research_dataset) == research_root
        assert Path(output_root) == expected_output_root
        assert config is not None
        assert minimum_free_bytes == 1
        experiment = expected_output_root / "backtest-v1-fixture"
        return EvaluationResult(
            experiment_id=experiment.name,
            experiment_path=experiment,
            manifest_path=experiment / "manifest.json",
            report_path=experiment / "report.json",
            data_mode="technical_smoke",
            data_span_days=3.0,
            folds=1,
            reused=False,
        )

    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "1")
    monkeypatch.setattr(
        "tradingbot.research.evaluator.run_offline_evaluation", fake_evaluation
    )

    main(
        [
            "--config",
            str(config_path),
            "run-backtest",
            "--research-dataset",
            str(research_root),
            "--output-root",
            str(expected_output_root),
        ]
    )

    summary: dict[str, object] = json.loads(capsys.readouterr().out)
    assert summary["evaluation_schema_version"] == 1
    assert summary["data_mode"] == "technical_smoke"
    assert summary["reused"] is False
