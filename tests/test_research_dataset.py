from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.cli import main
from tradingbot.data.audit import AUDIT_REPORT_SCHEMA_VERSION
from tradingbot.data.canonical import (
    KLINE_SCHEMA,
    ORDERBOOK_SCHEMA,
    TICKER_SCHEMA,
    TRADES_SCHEMA,
    AuditedInputFile,
    DatasetFile,
    _audited_files_fingerprint,
    _dataset_files_fingerprint,
)
from tradingbot.research.builder import (
    NS_PER_SECOND,
    _OrderBookSeries,
    _TradeSeries,
    build_research_dataset,
)
from tradingbot.research.contracts import (
    ExecutionResearchParameters,
    ResearchBuildError,
)
from tradingbot.research.execution_builder import (
    _activation_snapshot,
    _execution_label_rows,
    _maker_fill,
    _maker_fills,
    build_execution_research_dataset,
)

BASE_TS_MS = 1_800_000_000_000
MINUTE_MS = 60_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_audit(root: Path, canonical_root: Path) -> tuple[Path, str]:
    raw_root = root / "raw"
    raw_file = (
        raw_root
        / "orderbook"
        / "BTCUSDT"
        / "2026"
        / "07"
        / "20"
        / "part-00000.jsonl"
    )
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("{}\n", encoding="utf-8", newline="\n")
    audited_file = AuditedInputFile(
        path=raw_file.relative_to(raw_root).as_posix(),
        bytes=raw_file.stat().st_size,
        lines=1,
        records=1,
        sha256=_sha256(raw_file),
    )
    input_fingerprint = _audited_files_fingerprint([audited_file])
    report: dict[str, object] = {
        "audit_report_schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "dataset_root": raw_root.as_posix(),
        "readiness": {"ok": True, "strict": True, "reasons": []},
        "errors": [],
        "warnings": [],
        "partial_file_count": 0,
        "partial_files": [],
        "missing_expected_streams": [],
        "short_streams": [],
        "files": [
            {
                "path": audited_file.path,
                "bytes": audited_file.bytes,
                "lines": audited_file.lines,
                "records": audited_file.records,
                "sha256": audited_file.sha256,
            }
        ],
        "file_count": 1,
        "input_fingerprint": input_fingerprint,
        "totals": {"bytes": audited_file.bytes, "records": 1},
        "policy": {
            "expected_symbols": ["BTCUSDT"],
            "kline_intervals": ["1"],
        },
        "streams": {},
    }
    audit_path = canonical_root / "source-audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit_path, input_fingerprint


def _common_row(
    kind: str, exchange_ts_ms: int, received_at_ns: int, source_line: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "bybit",
        "session_id": "research-fixture",
        "exchange_ts_ms": exchange_ts_ms,
        "received_at_ns": received_at_ns,
        "source_path": f"{kind}/BTCUSDT/fixture.jsonl",
        "source_line": source_line,
    }


def _fixture_price(minute: int) -> float:
    return 100.0 * (1.0002**minute)


def _write_canonical_table(
    canonical_root: Path,
    kind: str,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> DatasetFile:
    date = datetime.fromtimestamp(BASE_TS_MS / 1_000, tz=UTC).date().isoformat()
    relative = (
        Path("market")
        / f"kind={kind}"
        / "symbol=BTCUSDT"
        / f"date={date}"
        / "part-00000.parquet"
    )
    path = canonical_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        version="2.6",
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
        data_page_version="1.0",
        write_page_index=True,
        write_page_checksum=True,
    )
    return DatasetFile(
        path=relative.as_posix(),
        kind=kind,
        symbol="BTCUSDT",
        date=date,
        rows=len(rows),
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def canonical_fixture(tmp_path: Path) -> Path:
    placeholder = tmp_path / "canonical-placeholder"
    placeholder.mkdir()
    _, input_fingerprint = _write_source_audit(tmp_path, placeholder)
    dataset_id = f"canonical-v1-{input_fingerprint[:16]}"
    canonical_root = tmp_path / dataset_id
    placeholder.rename(canonical_root)
    audit_path = canonical_root / "source-audit.json"

    orderbooks: list[dict[str, object]] = []
    tickers: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    klines: list[dict[str, object]] = []
    for minute in range(127):
        snapshot_ms = BASE_TS_MS + minute * MINUTE_MS + 4_000
        received_at_ns = snapshot_ms * 1_000_000
        mid = _fixture_price(minute)
        bid_prices = [mid * (1 - 0.00005 - level * 0.00001) for level in range(50)]
        ask_prices = [mid * (1 + 0.00005 + level * 0.00001) for level in range(50)]
        sizes = [1.0 + level / 100 for level in range(50)]
        orderbooks.append(
            {
                **_common_row("orderbook", snapshot_ms, received_at_ns, minute + 1),
                "matching_engine_ts_ms": snapshot_ms - 1,
                "update_id": minute + 1,
                "sequence": minute + 1,
                "bid_prices": bid_prices,
                "bid_sizes": sizes,
                "ask_prices": ask_prices,
                "ask_sizes": list(reversed(sizes)),
            }
        )
        tickers.append(
            {
                **_common_row("ticker", snapshot_ms, received_at_ns, minute + 1),
                "last_price": mid,
                "index_price": mid * 0.9999,
                "mark_price": mid,
                "bid_price": bid_prices[0],
                "bid_size": sizes[0],
                "ask_price": ask_prices[0],
                "ask_size": sizes[-1],
                "open_interest": 10_000.0 + minute * 10,
                "open_interest_value": (10_000.0 + minute * 10) * mid,
                "funding_rate": 0.0001,
                "next_funding_time_ms": BASE_TS_MS + 8 * 60 * MINUTE_MS,
                "funding_interval_hours": 8,
                "volume_24h": 1_000.0,
                "turnover_24h": 100_000.0,
                "price_24h_fraction": 0.01,
                "high_price_24h": mid * 1.02,
                "low_price_24h": mid * 0.98,
                "previous_price_1h": _fixture_price(max(0, minute - 60)),
                "previous_price_24h": mid * 0.99,
                "tick_direction": "PlusTick",
            }
        )
        trades.append(
            {
                **_common_row("trades", snapshot_ms, received_at_ns, minute + 1),
                "event_ts_ms": snapshot_ms - 2,
                "trade_id": f"trade-{minute:04d}",
                "side": "Buy" if minute % 2 == 0 else "Sell",
                "price": mid,
                "size": 1.0 + minute / 1_000,
                "tick_direction": "PlusTick",
                "sequence": minute + 1,
                "is_block_trade": False,
                "is_rpi_trade": False,
            }
        )
        if minute < 126:
            start_ms = BASE_TS_MS + minute * MINUTE_MS
            open_price = _fixture_price(minute)
            close_price = _fixture_price(minute + 1)
            klines.append(
                {
                    **_common_row(
                        "kline_1",
                        start_ms + MINUTE_MS + 4_000,
                        (start_ms + MINUTE_MS + 4_000) * 1_000_000,
                        minute + 1,
                    ),
                    "interval": "1",
                    "start_ms": start_ms,
                    "end_ms": start_ms + MINUTE_MS - 1,
                    "open": open_price,
                    "high": close_price * 1.00005,
                    "low": open_price * 0.99995,
                    "close": close_price,
                    "volume": 10.0 + minute / 10,
                    "turnover": (10.0 + minute / 10) * close_price,
                    "payload_sha256": hashlib.sha256(
                        f"kline-{minute}".encode()
                    ).hexdigest(),
                }
            )

    files = [
        _write_canonical_table(
            canonical_root, "orderbook", orderbooks, ORDERBOOK_SCHEMA
        ),
        _write_canonical_table(canonical_root, "ticker", tickers, TICKER_SCHEMA),
        _write_canonical_table(canonical_root, "trades", trades, TRADES_SCHEMA),
        _write_canonical_table(canonical_root, "kline_1", klines, KLINE_SCHEMA),
    ]
    rows_by_kind = Counter({item.kind: item.rows for item in files})
    audit_sha256 = _sha256(audit_path)
    manifest: dict[str, object] = {
        "dataset_schema_version": 1,
        "dataset_id": dataset_id,
        "source": {
            "audit_report_sha256": audit_sha256,
            "input_fingerprint": input_fingerprint,
            "file_count": 1,
            "records": 1,
            "expected_symbols": ["BTCUSDT"],
            "kline_intervals": ["1"],
        },
        "canonicalization": {
            "source_kline_rows": len(klines),
            "canonical_kline_rows": len(klines),
            "collapsed_kline_rows": 0,
        },
        "output_rows_by_kind": dict(sorted(rows_by_kind.items())),
        "output_file_count": len(files),
        "output_fingerprint": _dataset_files_fingerprint(files),
        "files": [item.to_dict() for item in sorted(files, key=lambda item: item.path)],
    }
    (canonical_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return canonical_root


def _read_output(dataset_path: Path, table_name: str) -> pa.Table:
    paths = sorted(dataset_path.glob(f"table={table_name}/**/*.parquet"))
    assert paths
    return pa.concat_tables([pq.ParquetFile(path).read() for path in paths])


def test_builds_causal_research_dataset_and_is_idempotent(tmp_path: Path) -> None:
    canonical = canonical_fixture(tmp_path)
    output_root = tmp_path / "research"
    first = build_research_dataset(canonical, output_root)

    assert first.reused is False
    assert first.feature_rows > 0
    assert first.label_rows > 0
    assert first.output_files == 2

    features = _read_output(first.dataset_path, "features").to_pylist()
    labels = _read_output(first.dataset_path, "labels").to_pylist()
    assert features
    assert labels
    for row in features:
        decision = row["decision_at_ns"]
        assert row["book_received_at_ns"] <= decision
        assert row["ticker_received_at_ns"] <= decision
        assert row["latest_kline_received_at_ns"] <= decision
        assert row["latest_trade_received_at_ns"] <= decision
        assert row["btc_return_5m_fraction"] == row["return_5m_fraction"]
        assert row["relative_return_5m_fraction"] == 0.0
    for row in labels:
        assert row["label_end_ns"] > row["decision_at_ns"]
        if row["hit_at_ns"] is not None:
            assert row["decision_at_ns"] < row["hit_at_ns"] <= row["label_end_ns"]
        assert row["outcome"] in {"TP_FIRST", "SL_FIRST", "TIMEOUT", "AMBIGUOUS"}

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["causality"]["feature_rule"] == (
        "received_at_ns <= decision_at_ns"
    )
    assert manifest["causality"]["execution_labels_included"] is False
    assert manifest["label_outcomes"]["SL_FIRST"] > 0
    assert manifest["label_outcomes"]["TIMEOUT"] > 0

    second = build_research_dataset(canonical, output_root)
    assert second.reused is True
    assert second.output_fingerprint == first.output_fingerprint

    independent = build_research_dataset(canonical, tmp_path / "independent")
    assert independent.output_fingerprint == first.output_fingerprint
    assert independent.manifest_path.read_bytes() == first.manifest_path.read_bytes()


def test_rejects_corrupted_existing_research_output(tmp_path: Path) -> None:
    canonical = canonical_fixture(tmp_path)
    output_root = tmp_path / "research"
    result = build_research_dataset(canonical, output_root)
    feature_path = next(result.dataset_path.glob("table=features/**/*.parquet"))
    original = feature_path.read_bytes()
    feature_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))

    with pytest.raises(ResearchBuildError, match="corrupted"):
        build_research_dataset(canonical, output_root)


def test_equal_trade_order_key_is_ambiguous() -> None:
    decision_at_ns = BASE_TS_MS * 1_000_000
    shared_hit_ns = decision_at_ns + 1_000_000_000
    coverage_ns = decision_at_ns + 5_000_000_000
    table = pa.table(
        {
            "received_at_ns": [shared_hit_ns, shared_hit_ns, coverage_ns],
            "event_ts_ms": [BASE_TS_MS + 1_000, BASE_TS_MS + 1_000, BASE_TS_MS + 5_000],
            "sequence": [100, 100, 101],
            "side": ["Buy", "Sell", "Buy"],
            "price": [102.0, 98.0, 100.0],
            "size": [1.0, 1.0, 1.0],
        }
    )
    result = _TradeSeries(table).barrier_outcome(
        decision_at_ns=decision_at_ns,
        label_end_ns=coverage_ns,
        side="LONG",
        entry_price=100.0,
        stop_price=99.0,
        take_profit_price=101.0,
        stop_distance_bps=100.0,
        take_profit_distance_bps=100.0,
    )

    assert result is not None
    assert result.outcome == "AMBIGUOUS"
    assert result.resolution == "equal_received_event_sequence_key"


def test_build_research_cli_prints_summary(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = canonical_fixture(tmp_path)
    output_root = tmp_path / "research"
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "1")

    main(
        [
            "--config",
            str(config_path),
            "build-research",
            "--dataset",
            str(canonical),
            "--output-root",
            str(output_root),
        ]
    )

    summary: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert summary["research_schema_version"] == 1
    assert summary["feature_rows"] > 0
    assert summary["label_rows"] > 0
    assert summary["reused"] is False


def test_execution_parameters_reject_optimistic_or_unbounded_assumptions() -> None:
    defaults = ExecutionResearchParameters()
    defaults.validate()

    with pytest.raises(ResearchBuildError, match="at least 1.0"):
        replace(defaults, queue_ahead_multiplier=0.99).validate()
    with pytest.raises(ResearchBuildError, match="unique, positive"):
        replace(defaults, order_notionals_usdt=(100.0, 50.0)).validate()
    with pytest.raises(ResearchBuildError, match="one-hour"):
        replace(defaults, position_horizons_minutes=(15, 90)).validate()
    with pytest.raises(ResearchBuildError, match="must remain 60"):
        replace(defaults, volatility_lookback_minutes=30).validate()
    with pytest.raises(ResearchBuildError, match="extend beyond"):
        replace(
            defaults,
            submission_latency_ms=500,
            activation_max_delay_ms=2_500,
            entry_ttl_seconds=3,
        ).validate()


def test_activation_snapshot_uses_observed_queue_and_post_only_guard() -> None:
    submitted_at_ns = BASE_TS_MS * 1_000_000
    table = pa.table(
        {
            "received_at_ns": [submitted_at_ns + 100_000_000],
            "bid_prices": [[100.0, 99.5]],
            "bid_sizes": [[3.0, 4.0]],
            "ask_prices": [[101.0, 101.5]],
            "ask_sizes": [[5.0, 6.0]],
        }
    )
    books = _OrderBookSeries(table)

    exact, reason = _activation_snapshot(
        books,
        submitted_at_ns=submitted_at_ns,
        activation_max_delay_ms=500,
        side="LONG",
        entry_price=100.0,
    )
    assert reason is None
    assert exact is not None
    assert exact.post_only_valid is True
    assert exact.queue_ahead_size == 3.0

    inside, reason = _activation_snapshot(
        books,
        submitted_at_ns=submitted_at_ns,
        activation_max_delay_ms=500,
        side="LONG",
        entry_price=100.5,
    )
    assert reason is None
    assert inside is not None
    assert inside.queue_ahead_size == 0.0

    crossing, reason = _activation_snapshot(
        books,
        submitted_at_ns=submitted_at_ns,
        activation_max_delay_ms=500,
        side="LONG",
        entry_price=101.0,
    )
    assert reason is None
    assert crossing is not None
    assert crossing.post_only_valid is False
    assert crossing.queue_ahead_size is None


def _fill_trade_series(
    *,
    prices: list[float],
    sizes: list[float],
    sides: list[str],
) -> tuple[_TradeSeries, int, int]:
    activation_at_ns = BASE_TS_MS * 1_000_000
    received = [
        activation_at_ns + (index + 1) * 1_000_000_000
        for index in range(len(prices))
    ]
    table = pa.table(
        {
            "received_at_ns": received,
            "event_ts_ms": [BASE_TS_MS + (index + 1) * 1_000 for index in range(len(prices))],
            "sequence": list(range(1, len(prices) + 1)),
            "side": sides,
            "price": prices,
            "size": sizes,
        }
    )
    return _TradeSeries(table), activation_at_ns, received[-1]


def test_maker_fill_requires_queue_and_full_order_volume() -> None:
    trades, activation_at_ns, end_ns = _fill_trade_series(
        prices=[100.0, 100.0, 100.0, 101.0],
        sizes=[1.0, 0.25, 0.25, 1.0],
        sides=["Sell", "Sell", "Sell", "Buy"],
    )
    result = _maker_fill(
        trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=end_ns,
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=1.0,
        order_size=0.5,
        queue_ahead_multiplier=1.0,
    )

    assert result is not None
    assert result.status == "FULL_FILL"
    assert result.resolution == "visible_queue_depleted_by_public_trades"
    assert result.first_fill_index == 1
    assert result.full_fill_index == 2
    assert result.filled_size == 0.5
    assert result.contra_volume_at_entry_price == 1.5


def test_bulk_maker_fill_scans_one_window_for_multiple_sizes() -> None:
    trades, activation_at_ns, end_ns = _fill_trade_series(
        prices=[100.0, 100.0, 101.0],
        sizes=[1.25, 0.25, 1.0],
        sides=["Sell", "Sell", "Buy"],
    )

    results = _maker_fills(
        trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=end_ns,
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=1.0,
        order_sizes=(0.25, 1.0),
        queue_ahead_multiplier=1.0,
    )

    assert results is not None
    small, large = results
    assert small.status == "FULL_FILL"
    assert small.full_fill_index == 0
    assert large.status == "PARTIAL_FILL"
    assert large.filled_size == pytest.approx(0.5)


def test_maker_fill_ignores_block_and_rpi_prints() -> None:
    activation_at_ns = BASE_TS_MS * 1_000_000
    received = [
        activation_at_ns + NS_PER_SECOND,
        activation_at_ns + 2 * NS_PER_SECOND,
        activation_at_ns + 3 * NS_PER_SECOND,
    ]
    trades = _TradeSeries(
        pa.table(
            {
                "received_at_ns": received,
                "event_ts_ms": [BASE_TS_MS + 1_000, BASE_TS_MS + 2_000, BASE_TS_MS + 3_000],
                "sequence": [1, 2, 3],
                "side": ["Sell", "Sell", "Buy"],
                "price": [99.0, 99.0, 101.0],
                "size": [10.0, 10.0, 1.0],
                "is_block_trade": [True, False, False],
                "is_rpi_trade": [False, True, False],
            }
        )
    )

    result = _maker_fill(
        trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=received[-1],
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=0.0,
        order_size=1.0,
        queue_ahead_multiplier=1.0,
    )

    assert result is not None
    assert result.status == "NO_FILL"
    assert result.contra_trade_count == 0

    market_outcome = trades.barrier_outcome(
        decision_at_ns=activation_at_ns,
        label_end_ns=received[-1],
        side="LONG",
        entry_price=100.0,
        stop_price=99.5,
        take_profit_price=102.0,
        stop_distance_bps=50.0,
        take_profit_distance_bps=200.0,
    )
    execution_outcome = trades.barrier_outcome(
        decision_at_ns=activation_at_ns,
        label_end_ns=received[-1],
        side="LONG",
        entry_price=100.0,
        stop_price=99.5,
        take_profit_price=102.0,
        stop_distance_bps=50.0,
        take_profit_distance_bps=200.0,
        execution_eligible_only=True,
    )
    assert market_outcome is not None
    assert market_outcome.outcome == "SL_FIRST"
    assert execution_outcome is not None
    assert execution_outcome.outcome == "TIMEOUT"
    assert execution_outcome.timeout_price == 101.0


def test_maker_fill_retains_partial_and_price_through_cases() -> None:
    partial_trades, activation_at_ns, end_ns = _fill_trade_series(
        prices=[100.0, 100.0, 101.0],
        sizes=[1.0, 0.4, 1.0],
        sides=["Sell", "Sell", "Buy"],
    )
    partial = _maker_fill(
        partial_trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=end_ns,
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=1.0,
        order_size=1.0,
        queue_ahead_multiplier=1.0,
    )
    assert partial is not None
    assert partial.status == "PARTIAL_FILL"
    assert partial.filled_size == pytest.approx(0.4)

    through_trades, activation_at_ns, end_ns = _fill_trade_series(
        prices=[99.5, 101.0],
        sizes=[0.01, 1.0],
        sides=["Sell", "Buy"],
    )
    through = _maker_fill(
        through_trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=end_ns,
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=100.0,
        order_size=1.0,
        queue_ahead_multiplier=1.0,
    )
    assert through is not None
    assert through.status == "FULL_FILL"
    assert through.resolution == "public_trade_through_entry_price"
    assert through.full_fill_index == 0

    expiry_trades, activation_at_ns, expiry_ns = _fill_trade_series(
        prices=[100.0],
        sizes=[2.0],
        sides=["Sell"],
    )
    at_expiry = _maker_fill(
        expiry_trades,
        activation_at_ns=activation_at_ns,
        entry_window_end_ns=expiry_ns,
        side="LONG",
        entry_price=100.0,
        queue_ahead_size=0.0,
        order_size=1.0,
        queue_ahead_multiplier=1.0,
    )
    assert at_expiry is not None
    assert at_expiry.status == "NO_FILL"


def test_barrier_evaluation_keeps_later_trade_with_same_receive_time() -> None:
    received_at_ns = BASE_TS_MS * 1_000_000
    trades = _TradeSeries(
        pa.table(
            {
                "received_at_ns": [received_at_ns, received_at_ns],
                "event_ts_ms": [BASE_TS_MS, BASE_TS_MS + 1],
                "sequence": [10, 11],
                "side": ["Sell", "Buy"],
                "price": [100.0, 102.0],
                "size": [1.0, 1.0],
            }
        )
    )

    outcome = trades.barrier_outcome(
        decision_at_ns=received_at_ns,
        label_end_ns=received_at_ns,
        side="LONG",
        entry_price=100.0,
        stop_price=99.0,
        take_profit_price=101.0,
        stop_distance_bps=100.0,
        take_profit_distance_bps=100.0,
        start_after_index=0,
    )

    assert outcome is not None
    assert outcome.outcome == "TP_FIRST"
    assert outcome.hit_index == 1
    assert outcome.future_trade_count == 1


def test_execution_label_starts_market_outcome_after_full_fill() -> None:
    decision_at_ns = BASE_TS_MS * 1_000_000
    books = _OrderBookSeries(
        pa.table(
            {
                "received_at_ns": [decision_at_ns + 100_000_000],
                "bid_prices": [[100.0, 99.5]],
                "bid_sizes": [[1.0, 2.0]],
                "ask_prices": [[101.0, 101.5]],
                "ask_sizes": [[1.0, 2.0]],
            }
        )
    )
    full_fill_at_ns = decision_at_ns + NS_PER_SECOND
    horizon_end_ns = full_fill_at_ns + 15 * 60 * NS_PER_SECOND
    trades = _TradeSeries(
        pa.table(
            {
                "received_at_ns": [
                    full_fill_at_ns,
                    full_fill_at_ns + NS_PER_SECOND,
                    horizon_end_ns,
                ],
                "event_ts_ms": [
                    BASE_TS_MS + 1_000,
                    BASE_TS_MS + 2_000,
                    BASE_TS_MS + 901_000,
                ],
                "sequence": [1, 2, 3],
                "side": ["Sell", "Buy", "Buy"],
                "price": [100.0, 102.0, 100.0],
                "size": [1.5, 1.0, 1.0],
            }
        )
    )
    feature: dict[str, object] = {
        "decision_id": "decision-1",
        "decision_at_ns": decision_at_ns,
        "decision_utc_date": "2027-01-15",
        "best_bid_price": 100.0,
        "best_ask_price": 101.0,
        "realized_volatility_60m_fraction": 0.001,
    }

    labels = _execution_label_rows(
        source_dataset_id="source-1",
        symbol="BTCUSDT",
        feature=feature,
        books=books,
        trades=trades,
        parameters=ExecutionResearchParameters(
            position_horizons_minutes=(15,),
            order_notionals_usdt=(50.0,),
            submission_latency_ms=0,
            activation_max_delay_ms=500,
            entry_ttl_seconds=30,
        ),
        quality=Counter(),
    )

    long_label = next(row for row in labels if row["side"] == "LONG")
    assert long_label["fill_status"] == "FULL_FILL"
    assert long_label["full_fill_at_ns"] == full_fill_at_ns
    assert long_label["outcome"] == "TP_FIRST"
    assert long_label["hit_at_ns"] == full_fill_at_ns + NS_PER_SECOND


def test_builds_immutable_execution_research_dataset(tmp_path: Path) -> None:
    canonical = canonical_fixture(tmp_path)
    output_root = tmp_path / "execution-research"
    parameters = ExecutionResearchParameters(
        position_horizons_minutes=(15,),
        order_notionals_usdt=(50.0,),
        submission_latency_ms=0,
        activation_max_delay_ms=60_000,
        entry_ttl_seconds=65,
    )
    first = build_execution_research_dataset(
        canonical,
        output_root,
        parameters=parameters,
    )

    assert first.reused is False
    assert first.feature_rows > 0
    assert first.execution_label_rows > 0
    assert first.output_files == 2
    features = _read_output(first.dataset_path, "features").to_pylist()
    labels = _read_output(first.dataset_path, "execution_labels").to_pylist()
    assert features
    assert labels
    assert {row["horizon_minutes"] for row in labels} == {15}
    assert {row["order_notional_usdt"] for row in labels} == {50.0}
    assert {row["fill_status"] for row in labels} <= {
        "NO_FILL",
        "PARTIAL_FILL",
        "FULL_FILL",
    }
    for row in labels:
        assert row["decision_at_ns"] < row["activation_at_ns"]
        assert 0.0 <= row["fill_fraction"] <= 1.0
        if row["fill_status"] == "FULL_FILL":
            assert row["full_fill_at_ns"] is not None
            assert row["outcome"] in {
                "TP_FIRST",
                "SL_FIRST",
                "TIMEOUT",
                "AMBIGUOUS",
            }

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["research_profile"] == "execution_microstructure_v1"
    assert manifest["scope"]["maker_fill_is_proxy"] is True
    assert manifest["scope"]["eligible_for_profitability_conclusion"] is False
    assert manifest["processing"] == {
        "maximum_source_partitions_loaded_per_symbol": 3,
        "mode": "utc_day_with_adjacent_context",
        "output_partition_count": 1,
    }
    assert manifest["output_rows"]["execution_labels"] == len(labels)

    second = build_execution_research_dataset(
        canonical,
        output_root,
        parameters=parameters,
    )
    assert second.reused is True
    assert second.output_fingerprint == first.output_fingerprint


def test_build_execution_research_cli_prints_summary(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = canonical_fixture(tmp_path)
    output_root = tmp_path / "execution-cli"
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "1")

    main(
        [
            "--config",
            str(config_path),
            "build-execution-research",
            "--dataset",
            str(canonical),
            "--output-root",
            str(output_root),
            "--horizon-minutes",
            "15",
            "--order-notional-usdt",
            "50",
            "--submission-latency-ms",
            "0",
            "--activation-max-delay-ms",
            "60000",
            "--entry-ttl-seconds",
            "65",
        ]
    )

    summary: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert summary["execution_research_schema_version"] == 1
    assert summary["research_profile"] == "execution_microstructure_v1"
    assert summary["feature_rows"] > 0
    assert summary["execution_label_rows"] > 0
    assert summary["reused"] is False
