from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.cli import _parser
from tradingbot.market.records import MarketRecord
from tradingbot.research.contracts import ExecutionResearchParameters
from tradingbot.research.execution_evaluation_contracts import (
    ExecutionResearchDataset,
)
from tradingbot.shadow import bundle as bundle_module
from tradingbot.shadow.bundle import (
    SHADOW_BUNDLE_SCHEMA_VERSION,
    ShadowBundleError,
    validate_shadow_bundle,
)
from tradingbot.shadow.journal import ShadowJournal
from tradingbot.shadow.live import LiveMarketWindow
from tradingbot.shadow.model import ShadowScorer
from tradingbot.shadow.runtime import reject_trading_credentials


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_fingerprint(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _calibrator(classes: tuple[str, ...]) -> dict[str, object]:
    return {
        "method": "temperature_plus_prior_shrinkage_grid",
        "temperature": 1.0,
        "prior_weight": 0.0,
        "class_prior": {name: 1 / len(classes) for name in classes},
        "calibration_log_loss_before": 1.0,
        "calibration_log_loss_after": 1.0,
    }


def _write_test_bundle(tmp_path: Path) -> Path:
    input_fingerprint = "e" * 64
    root = tmp_path / f"shadow-bundle-v1-{input_fingerprint[:16]}"
    models = root / "models"
    models.mkdir(parents=True)
    x = np.asarray(
        [
            [0.1, 10, 15, 1, 1],
            [0.2, 11, 16, -1, 1],
            [0.3, 12, 18, 1, 1],
            [0.4, 13, 19, -1, 1],
            [0.5, 14, 21, 1, 1],
            [0.6, 15, 22, -1, 1],
            [0.7, 16, 24, 1, 1],
            [0.8, 17, 25, -1, 1],
            [0.9, 18, 27, 1, 1],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    for name in ("fill.txt", "post-fill-outcome.txt"):
        classifier = lightgbm.LGBMClassifier(
            n_estimators=2,
            num_leaves=3,
            min_child_samples=1,
            verbosity=-1,
            random_state=7,
        )
        classifier.fit(x, y)
        classifier.booster_.save_model(str(models / name))
    contract: dict[str, object] = {
        "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
        "bundle_id": root.name,
        "input_fingerprint": input_fingerprint,
        "data_gate": {
            "mode": "technical_smoke",
            "eligible_for_execution_model_review": False,
        },
        "scope": {
            "bybit_access": "public-read-only",
            "order_submission": False,
            "trading_credentials_allowed": False,
            "eligible_for_trading": False,
            "engineering_only": True,
            "openai_tokens": 0,
        },
        "models": {
            "fill": "models/fill.txt",
            "post_fill_outcome": "models/post-fill-outcome.txt",
        },
        "model": {
            "feature_names": [
                "spread_bps",
                "stop_distance_bps",
                "take_profit_distance_bps",
                "side_direction",
                "symbol_BTCUSDT",
            ],
            "calibrators": {
                "fill": _calibrator(("NO_FILL", "PARTIAL_FILL", "FULL_FILL")),
                "post_fill_outcome": _calibrator(
                    ("SL_FIRST", "TIMEOUT", "TP_FIRST")
                ),
            },
            "execution_estimates": {
                "timeout_return_bps": {
                    "global": 0.0,
                    "by_side": {},
                    "by_symbol_side": {},
                },
                "partial_fill_fraction": {
                    "global": 0.5,
                    "by_side": {},
                    "by_symbol_side": {},
                },
            },
        },
        "universe": {
            "symbols": ["BTCUSDT"],
            "one_position_across_all_symbols": True,
        },
        "scenario": {
            "horizon_minutes": 15,
            "reference_order_notional_usdt": 50.0,
        },
        "evaluation_parameters": {
            "maker_fee_bps": 2.0,
            "taker_fee_bps": 5.5,
            "stop_slippage_bps": 2.0,
            "timeout_slippage_bps": 2.0,
            "minimum_expected_net_bps": 0.0,
            "max_notional_fraction": 0.05,
            "max_planned_risk_fraction": 0.007,
            "rolling_24h_loss_fraction": 0.01,
        },
    }
    bundle_path = root / "bundle.json"
    bundle_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = [bundle_path, models / "fill.txt", models / "post-fill-outcome.txt"]
    descriptors = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ),
        key=lambda item: item["path"],
    )
    manifest = {
        "shadow_bundle_schema_version": SHADOW_BUNDLE_SCHEMA_VERSION,
        "bundle_id": root.name,
        "input_fingerprint": input_fingerprint,
        "bundle_fingerprint": _json_fingerprint(descriptors),
        "files": descriptors,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_shadow_bundle_verifies_models_and_scores_without_order_path(
    tmp_path: Path,
) -> None:
    bundle = validate_shadow_bundle(_write_test_bundle(tmp_path))
    scorer = ShadowScorer(bundle)

    candidate = scorer.score(
        {
            "decision_id": "decision-1",
            "decision_at_ns": 1_000_000_000,
            "spread_bps": 0.5,
            "funding_rate": 0.0,
            "minutes_to_funding": 120.0,
        },
        symbol="BTCUSDT",
        side="LONG",
        entry_limit_price=100.0,
        stop_distance_bps=20.0,
        take_profit_distance_bps=30.0,
        stop_price=99.8,
        take_profit_price=100.3,
    )

    assert sum(candidate.fill_probabilities) == pytest.approx(1.0)
    assert sum(candidate.outcome_probabilities) == pytest.approx(1.0)
    assert 0 < candidate.notional_fraction <= 0.05
    assert bundle.contract["scope"]["order_submission"] is False


def test_shadow_bundle_rejects_tampered_model(tmp_path: Path) -> None:
    root = _write_test_bundle(tmp_path)
    (root / "models" / "fill.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ShadowBundleError, match="failed verification"):
        validate_shadow_bundle(root)


def test_shadow_bundle_requires_explicit_technical_smoke_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_root = tmp_path / "execution-backtest-v1-test"
    evaluation_root.mkdir()
    dataset_root = tmp_path / "execution-research-v1-test"
    dataset_root.mkdir()
    dataset = ExecutionResearchDataset(
        root=dataset_root,
        execution_dataset_id=dataset_root.name,
        source_dataset_id="source-v1-test",
        input_fingerprint="a" * 64,
        output_fingerprint="b" * 64,
        symbols=("BTCUSDT",),
        partition_dates=("2026-08-01",),
        feature_paths=(),
        label_paths=(),
        feature_rows=1,
        label_rows=1,
        manifest={},
    )
    manifest = {
        "execution_dataset_id": dataset.execution_dataset_id,
        "output_fingerprint": "c" * 64,
    }
    report = {
        "execution_research_dataset": {
            "execution_dataset_id": dataset.execution_dataset_id,
            "input_fingerprint": dataset.input_fingerprint,
            "output_fingerprint": dataset.output_fingerprint,
            "symbols": ["BTCUSDT"],
        },
        "pre_registered_comparisons": {
            "primary_candidate_model": "lightgbm_calibrated"
        },
        "data_gate": {
            "mode": "technical_smoke",
            "eligible_for_execution_model_review": False,
        },
    }
    monkeypatch.setattr(
        bundle_module,
        "_load_execution_evaluation",
        lambda _path: (evaluation_root, manifest, report, {}),
    )
    monkeypatch.setattr(
        bundle_module,
        "validate_execution_research_dataset",
        lambda _path: dataset,
    )

    with pytest.raises(ShadowBundleError, match="--allow-technical-smoke"):
        bundle_module.build_shadow_bundle(
            execution_evaluation=evaluation_root,
            execution_dataset=dataset_root,
            output_root=tmp_path / "bundles",
        )


def test_shadow_bundle_freezes_selected_fold_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_root = tmp_path / "execution-backtest-v1-test"
    model_root = evaluation_root / "models"
    model_root.mkdir(parents=True)
    fill_model = model_root / "fold-01-lightgbm-fill.txt"
    outcome_model = model_root / "fold-01-lightgbm-outcome.txt"
    fill_model.write_text("exact frozen fill model\n", encoding="utf-8")
    outcome_model.write_text("exact frozen outcome model\n", encoding="utf-8")
    dataset_root = tmp_path / "execution-research-v1-test"
    dataset_root.mkdir()
    research_parameters = {
        "decision_interval_seconds": 60,
        "decision_offset_seconds": 5,
        "kline_history_minutes": 60,
        "max_orderbook_age_ms": 2500,
        "max_ticker_age_ms": 2500,
        "position_horizons_minutes": [15],
        "volatility_lookback_minutes": 60,
        "stop_volatility_multiple": 1.0,
        "take_profit_multiple": 1.5,
        "minimum_stop_bps": 10.0,
        "maximum_stop_bps": 250.0,
        "order_notionals_usdt": [50.0],
        "submission_latency_ms": 250,
        "activation_max_delay_ms": 2500,
        "entry_ttl_seconds": 30,
        "queue_ahead_multiplier": 1.0,
        "maximum_continuity_gap_ms": 90_000,
        "fingerprint": "d" * 64,
    }
    dataset = ExecutionResearchDataset(
        root=dataset_root,
        execution_dataset_id=dataset_root.name,
        source_dataset_id="source-v1-test",
        input_fingerprint="a" * 64,
        output_fingerprint="b" * 64,
        symbols=("BTCUSDT",),
        partition_dates=("2026-08-01",),
        feature_paths=(),
        label_paths=(),
        feature_rows=1,
        label_rows=1,
        manifest={"parameters": research_parameters},
    )
    evaluation_manifest = {
        "execution_dataset_id": dataset.execution_dataset_id,
        "output_fingerprint": "c" * 64,
    }
    calibrators = {
        "fill": _calibrator(("NO_FILL", "PARTIAL_FILL", "FULL_FILL")),
        "post_fill_outcome": _calibrator(("SL_FIRST", "TIMEOUT", "TP_FIRST")),
    }
    report = {
        "experiment_id": evaluation_root.name,
        "execution_research_dataset": {
            "execution_dataset_id": dataset.execution_dataset_id,
            "input_fingerprint": dataset.input_fingerprint,
            "output_fingerprint": dataset.output_fingerprint,
            "symbols": ["BTCUSDT"],
        },
        "pre_registered_comparisons": {
            "primary_candidate_model": "lightgbm_calibrated"
        },
        "data_gate": {
            "mode": "technical_smoke",
            "eligible_for_execution_model_review": False,
        },
        "folds": [
            {
                "fold": 1,
                "nested_calibration": {"fit_purge_cutoff_ns": 123},
                "models": {"lightgbm": {"calibrators": calibrators}},
            }
        ],
        "selected_execution_scenario": {
            "horizon_minutes": 15,
            "reference_order_notional_usdt": 50.0,
        },
        "feature_names": [
            "spread_bps",
            "stop_distance_bps",
            "take_profit_distance_bps",
            "side_direction",
            "symbol_BTCUSDT",
        ],
        "parameters": {
            "maker_fee_bps": 2.0,
            "taker_fee_bps": 5.5,
            "stop_slippage_bps": 3.0,
            "timeout_slippage_bps": 1.0,
            "minimum_expected_net_bps": 1.0,
            "max_notional_fraction": 0.05,
            "max_planned_risk_fraction": 0.007,
            "rolling_24h_loss_fraction": 0.01,
        },
    }
    estimates = {
        "timeout_return_bps": {
            "global": 0.0,
            "by_side": {},
            "by_symbol_side": {},
        },
        "partial_fill_fraction": {
            "global": 0.5,
            "by_side": {},
            "by_symbol_side": {},
        },
    }
    monkeypatch.setattr(
        bundle_module,
        "_load_execution_evaluation",
        lambda _path: (
            evaluation_root,
            evaluation_manifest,
            report,
            {
                "models/fold-01-lightgbm-fill.txt": fill_model,
                "models/fold-01-lightgbm-outcome.txt": outcome_model,
            },
        ),
    )
    monkeypatch.setattr(
        bundle_module,
        "validate_execution_research_dataset",
        lambda _path: dataset,
    )
    monkeypatch.setattr(bundle_module, "_fit_estimates", lambda *args, **kwargs: estimates)

    first = bundle_module.build_shadow_bundle(
        execution_evaluation=evaluation_root,
        execution_dataset=dataset_root,
        output_root=tmp_path / "bundles",
        allow_technical_smoke=True,
    )
    second = bundle_module.build_shadow_bundle(
        execution_evaluation=evaluation_root,
        execution_dataset=dataset_root,
        output_root=tmp_path / "bundles",
        allow_technical_smoke=True,
    )

    assert first.fill_model_path.read_bytes() == fill_model.read_bytes()
    assert first.outcome_model_path.read_bytes() == outcome_model.read_bytes()
    assert first.contract["model"]["selected_fold"] == 1
    assert first.contract["scope"]["eligible_for_trading"] is False
    assert first.reused is False
    assert second.reused is True
    assert second.bundle_fingerprint == first.bundle_fingerprint


def test_shadow_execution_estimates_use_only_calibration_fit_window(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "execution-labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "horizon_minutes": 15,
                    "order_notional_usdt": 50.0,
                    "entry_window_end_ns": 100,
                    "position_end_ns": 900,
                    "fill_status": "FULL_FILL",
                    "fill_fraction": 1.0,
                    "outcome": "TIMEOUT",
                    "outcome_return_bps": 4.0,
                },
                {
                    "symbol": "BTCUSDT",
                    "side": "SHORT",
                    "horizon_minutes": 15,
                    "order_notional_usdt": 50.0,
                    "entry_window_end_ns": 800,
                    "position_end_ns": None,
                    "fill_status": "PARTIAL_FILL",
                    "fill_fraction": 0.4,
                    "outcome": "PARTIAL_FILL",
                    "outcome_return_bps": None,
                },
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "horizon_minutes": 15,
                    "order_notional_usdt": 50.0,
                    "entry_window_end_ns": 100,
                    "position_end_ns": 1200,
                    "fill_status": "FULL_FILL",
                    "fill_fraction": 1.0,
                    "outcome": "TIMEOUT",
                    "outcome_return_bps": -100.0,
                },
            ]
        ),
        labels,
    )

    estimates = bundle_module._fit_estimates(
        (labels,),
        fit_purge_cutoff_ns=1000,
        horizon_minutes=15,
        order_notional_usdt=50.0,
    )

    assert estimates["timeout_return_bps"]["global"] == 4.0
    assert estimates["partial_fill_fraction"]["global"] == 0.4


def test_shadow_journal_is_hash_chained_restartable_and_single_writer(
    tmp_path: Path,
) -> None:
    bundle = validate_shadow_bundle(_write_test_bundle(tmp_path))
    journal_root = tmp_path / "journals"
    first = ShadowJournal(journal_root, run_id="run-1", bundle=bundle)
    first.append("decision_cycle", {"value": 1}, recorded_at_ns=1_800_000_000_000_000_000)
    with pytest.raises(ShadowBundleError, match="another writer"):
        ShadowJournal(journal_root, run_id="run-1", bundle=bundle)
    first.close()

    with ShadowJournal(journal_root, run_id="run-1", bundle=bundle) as resumed:
        assert resumed.sequence == 1
        resumed.append(
            "run_stopped", {"value": 2}, recorded_at_ns=1_800_000_001_000_000_000
        )
        assert resumed.sequence == 2

    event_path = next((journal_root / "run-1").glob("events-*.jsonl"))
    original = event_path.read_text(encoding="utf-8")
    event_path.write_text(original.replace('"value":1', '"value":9'), encoding="utf-8")
    with pytest.raises(ShadowBundleError, match="journal chain failed"):
        ShadowJournal(journal_root, run_id="run-1", bundle=bundle)


def test_shadow_refuses_any_trading_credentials() -> None:
    reject_trading_credentials({})
    with pytest.raises(ShadowBundleError, match="BYBIT_API_KEY"):
        reject_trading_credentials({"BYBIT_API_KEY": "must-not-be-present"})


def test_live_window_clears_all_rows_on_websocket_session_change() -> None:
    window = LiveMarketWindow(("BTCUSDT",), retention_minutes=70)
    first = MarketRecord(
        kind="ticker",
        symbol="BTCUSDT",
        exchange_ts_ms=1,
        received_at_ns=1_000_000,
        payload={},
        session_id="session-a",
    )
    second = MarketRecord(
        kind="ticker",
        symbol="BTCUSDT",
        exchange_ts_ms=2,
        received_at_ns=2_000_000,
        payload={},
        session_id="session-b",
    )

    assert window.accept(first) is None
    transition = window.accept(second)

    assert transition is not None
    assert transition.previous_session_id == "session-a"
    assert transition.new_session_id == "session-b"
    assert window.current_session_id == "session-b"
    assert len(window._rows["BTCUSDT"]["ticker"]) == 1


def test_live_window_builds_features_only_from_records_available_at_decision() -> None:
    window = LiveMarketWindow(("BTCUSDT",), retention_minutes=70)
    minute_seconds = 1_800_000_000 // 60 * 60
    decision_at_ns = (minute_seconds + 5) * 1_000_000_000
    session = "continuous-session"
    bids = [[str(100.0 - index * 0.01), "1.0"] for index in range(50)]
    asks = [[str(100.1 + index * 0.01), "1.0"] for index in range(50)]
    for offset in range(-61, 1):
        current_seconds = minute_seconds + offset * 60
        received_ns = (current_seconds + 4) * 1_000_000_000
        window.accept(
            MarketRecord(
                "orderbook",
                "BTCUSDT",
                current_seconds * 1000,
                received_ns,
                {
                    "matching_engine_ts_ms": current_seconds * 1000,
                    "update_id": offset + 1000,
                    "sequence": offset + 2000,
                    "bids": bids,
                    "asks": asks,
                },
                session_id=session,
            )
        )
        window.accept(
            MarketRecord(
                "ticker",
                "BTCUSDT",
                current_seconds * 1000,
                received_ns,
                {
                    "markPrice": "100.05",
                    "indexPrice": "100.00",
                    "openInterest": str(10_000 + offset),
                    "fundingRate": "0.0001",
                    "nextFundingTime": str((minute_seconds + 3600) * 1000),
                },
                session_id=session,
            )
        )
        window.accept(
            MarketRecord(
                "trades",
                "BTCUSDT",
                current_seconds * 1000,
                (current_seconds + 3) * 1_000_000_000,
                [
                    {
                        "T": current_seconds * 1000,
                        "i": f"trade-{offset}",
                        "S": "Buy" if offset % 2 else "Sell",
                        "p": str(100 + offset * 0.01),
                        "v": "0.1",
                        "seq": offset + 3000,
                    }
                ],
                session_id=session,
            )
        )
        if offset < 0:
            start_ms = current_seconds * 1000
            close = 100 + offset * 0.01
            window.accept(
                MarketRecord(
                    "kline_1",
                    "BTCUSDT",
                    (current_seconds + 60) * 1000,
                    (current_seconds + 61) * 1_000_000_000,
                    {
                        "interval": "1",
                        "start": start_ms,
                        "end": start_ms + 59_999,
                        "open": str(close - 0.02),
                        "high": str(close + 0.05),
                        "low": str(close - 0.05),
                        "close": str(close),
                        "volume": "10",
                        "turnover": str(close * 10),
                        "confirm": True,
                    },
                    session_id=session,
                )
            )

    features, skipped = window.features_at(
        decision_at_ns, ExecutionResearchParameters()
    )

    assert skipped == {}
    assert set(features) == {"BTCUSDT"}
    assert features["BTCUSDT"]["book_age_ms"] == pytest.approx(1000.0)
    assert features["BTCUSDT"]["ticker_age_ms"] == pytest.approx(1000.0)
    assert features["BTCUSDT"]["relative_return_60m_fraction"] == 0.0


def test_shadow_cli_exposes_build_validate_and_run_commands() -> None:
    parser = _parser()
    build = parser.parse_args(
        [
            "build-shadow-bundle",
            "--execution-evaluation",
            "evaluation",
            "--execution-dataset",
            "dataset",
            "--allow-technical-smoke",
        ]
    )
    validate = parser.parse_args(["validate-shadow-bundle", "--bundle", "bundle"])
    run = parser.parse_args(
        ["shadow", "--bundle", "bundle", "--run-id", "shadow-live"]
    )

    assert build.command == "build-shadow-bundle"
    assert build.allow_technical_smoke is True
    assert validate.command == "validate-shadow-bundle"
    assert run.command == "shadow"
    assert run.run_id == "shadow-live"
