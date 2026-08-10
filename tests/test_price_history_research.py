from __future__ import annotations

import gzip
import io
import json
import math
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.data import bybit_history
from tradingbot.data.bybit_history import import_bybit_history_range
from tradingbot.research.contracts import PRICE_RESEARCH_PROFILE
from tradingbot.research.evaluation_dataset import (
    prepare_evaluation_data,
    validate_research_dataset,
)
from tradingbot.research.price_history_builder import (
    _SecondSeries,
    build_price_research_dataset,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._url = url

    def geturl(self) -> str:
        return self._url


def _payload(symbol: str, partition_date: str) -> bytes:
    parsed = datetime.fromisoformat(f"{partition_date}T00:00:00+00:00")
    day_number = (parsed.date() - date(2026, 1, 1)).days
    start_seconds = int(parsed.timestamp())
    lines = [
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional,RPI"
    ]
    symbol_offset = 0.0 if symbol == "BTCUSDT" else 20.0
    for minute in range(1_440):
        global_minute = day_number * 1_440 + minute
        price = (
            100.0
            + symbol_offset
            + 1.5 * math.sin(global_minute / 23)
            + 0.25 * math.sin(global_minute / 3)
        )
        timestamp = f"{start_seconds + minute * 60 + 59}.125"
        side = "Buy" if minute % 3 else "Sell"
        lines.append(
            f"{timestamp},{symbol},{side},0.5,{price:.6f},PlusTick,"
            f"{symbol}-{partition_date}-{minute},0,0.5,0,0"
        )
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


def _history_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    payloads = {
        (symbol, partition): _payload(symbol, partition)
        for partition in ("2026-08-01", "2026-08-02")
        for symbol in ("BTCUSDT", "ETHUSDT")
    }

    def fake_open(url: str, timeout_seconds: int) -> _FakeResponse:
        assert timeout_seconds == 10
        parsed = urlparse(url)
        symbol = Path(parsed.path).parent.name
        partition = (
            Path(parsed.path).name.removesuffix(".csv.gz").removeprefix(symbol)
        )
        return _FakeResponse(payloads[(symbol, partition)], url)

    monkeypatch.setattr(bybit_history, "_open_url", fake_open)
    result = import_bybit_history_range(
        history_root=tmp_path / "history",
        start_date="2026-08-01",
        end_date="2026-08-02",
        symbols=("BTCUSDT", "ETHUSDT"),
        request_timeout_seconds=10,
        download_attempts=1,
    )
    return result.catalog_path


def test_builds_and_prepares_price_history_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _history_fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "research"

    first = build_price_research_dataset(
        catalog,
        output_root,
        start_date="2026-08-01",
        end_date="2026-08-02",
    )
    second = build_price_research_dataset(
        catalog,
        output_root,
        start_date="2026-08-01",
        end_date="2026-08-02",
    )

    assert first.reused is False
    assert second.reused is True
    assert second.output_fingerprint == first.output_fingerprint
    assert first.feature_rows > 4_000
    assert first.label_rows > first.feature_rows * 6

    manifest = json.loads(first.manifest_path.read_bytes())
    assert manifest["research_profile"] == PRICE_RESEARCH_PROFILE
    assert manifest["source"]["days"] == 2
    assert manifest["source_capabilities"]["funding"] is False
    assert manifest["causality"] == {
        "barrier_resolution": "one_second_bars",
        "decision_grid": "UTC epoch aligned",
        "execution_labels_included": False,
        "feature_rule": "available_at_ns <= decision_at_ns",
        "label_rule": (
            "decision_at_ns < trade_bar_1s.available_at_ns <= label_end_ns"
        ),
        "maker_fill_claimed": False,
        "same_bar_double_cross": "AMBIGUOUS",
    }

    feature_path = next(first.dataset_path.glob("table=features/**/*.parquet"))
    features = pq.read_table(feature_path).to_pylist()
    assert features
    for row in features:
        assert row["latest_minute_bar_available_at_ns"] <= row["decision_at_ns"]
        assert row["latest_second_bar_available_at_ns"] <= row["decision_at_ns"]
    btc_path = next(
        first.dataset_path.glob("table=features/symbol=BTCUSDT/**/*.parquet")
    )
    btc = pq.read_table(btc_path).to_pylist()
    assert all(row["relative_return_60m_fraction"] == 0.0 for row in btc)

    dataset = validate_research_dataset(first.dataset_path)
    prepared = prepare_evaluation_data(dataset, horizon_minutes=60)
    assert dataset.research_profile == PRICE_RESEARCH_PROFILE
    assert prepared.rows > 0
    assert prepared.x.shape[1] == len(prepared.feature_names)
    assert prepared.x.shape[1] < 80
    assert np.all(np.isnan(prepared.funding_rate))
    assert np.all(np.isnan(prepared.minutes_to_funding))


def test_same_second_double_barrier_cross_is_ambiguous() -> None:
    decision_ns = 1_800_000_000_000_000_000
    table = pa.table(
        {
            "interval_seconds": [1, 1],
            "start_ms": [1_800_000_001_000, 1_800_000_002_000],
            "end_ms": [1_800_000_001_999, 1_800_000_002_999],
            "available_at_ns": [decision_ns + 2_000_000_000, decision_ns + 3_000_000_000],
            "high": [102.0, 100.0],
            "low": [98.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [2.0, 1.0],
            "turnover": [200.0, 100.0],
            "buy_volume": [1.0, 1.0],
            "sell_volume": [1.0, 0.0],
            "trade_count": [2, 1],
        }
    )
    outcome = _SecondSeries(table).barrier_outcome(
        decision_at_ns=decision_ns,
        label_end_ns=decision_ns + 10_000_000_000,
        side="LONG",
        entry_price=100.0,
        stop_price=99.0,
        take_profit_price=101.0,
        stop_distance_bps=100.0,
        take_profit_distance_bps=100.0,
    )

    assert outcome.outcome == "AMBIGUOUS"
    assert outcome.resolution == "same_one_second_bar_crossed_both_barriers"
