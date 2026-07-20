from __future__ import annotations

from tradingbot.config import MarketConfig
from tradingbot.market.normalizer import MarketNormalizer


def market_config() -> MarketConfig:
    return MarketConfig(
        orderbook_depth=50,
        orderbook_snapshot_ms=1000,
        ticker_snapshot_ms=1000,
        kline_intervals=("1", "5", "15"),
        collect_orderbook=True,
        collect_trades=True,
        collect_tickers=True,
        collect_klines=True,
    )


def test_reconstructs_and_samples_orderbook() -> None:
    normalizer = MarketNormalizer(market_config())
    snapshot = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "cts": 999,
        "data": {
            "u": 10,
            "seq": 20,
            "b": [["100", "1"], ["99", "2"]],
            "a": [["102", "1"], ["101", "3"]],
        },
    }

    records = normalizer.process(snapshot, received_at_ns=1_000_000_000)

    assert len(records) == 1
    assert records[0].payload["bids"] == [["100", "1"], ["99", "2"]]
    assert records[0].payload["asks"] == [["101", "3"], ["102", "1"]]

    too_early = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 1500,
        "cts": 1499,
        "data": {"u": 11, "seq": 21, "b": [["100", "0"], ["98", "4"]], "a": []},
    }
    assert normalizer.process(too_early, received_at_ns=1_500_000_000) == []

    due = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 2100,
        "cts": 2099,
        "data": {"u": 12, "seq": 22, "b": [], "a": [["101", "2.5"]]},
    }
    records = normalizer.process(due, received_at_ns=2_100_000_000)

    assert len(records) == 1
    assert records[0].payload["bids"] == [["99", "2"], ["98", "4"]]
    assert records[0].payload["asks"][0] == ["101", "2.5"]
    assert records[0].payload["update_id"] == 12


def test_orderbook_ignores_non_increasing_delta_ids_without_requiring_contiguity() -> None:
    normalizer = MarketNormalizer(market_config())
    snapshot = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {"u": 10, "seq": 20, "b": [["100", "1"]], "a": [["101", "1"]]},
    }
    assert len(normalizer.process(snapshot, received_at_ns=1_000_000_000)) == 1

    stale_update_id = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 2000,
        "data": {"u": 10, "seq": 21, "b": [["99", "2"]], "a": []},
    }
    stale_sequence = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 2100,
        "data": {"u": 11, "seq": 20, "b": [["98", "2"]], "a": []},
    }
    assert normalizer.process(stale_update_id, received_at_ns=2_000_000_000) == []
    assert normalizer.process(stale_sequence, received_at_ns=2_100_000_000) == []

    skipped_update_id = {
        "topic": "orderbook.50.BTCUSDT",
        "type": "delta",
        "ts": 2200,
        "data": {"u": 15, "seq": 25, "b": [["97", "2"]], "a": []},
    }
    records = normalizer.process(skipped_update_id, received_at_ns=2_200_000_000)

    assert len(records) == 1
    assert records[0].payload["bids"] == [["100", "1"], ["97", "2"]]
    assert records[0].payload["update_id"] == 15
    assert records[0].payload["sequence"] == 25


def test_orderbook_snapshot_resets_ids_and_rejects_invalid_decimal_levels() -> None:
    normalizer = MarketNormalizer(market_config())
    first_snapshot = {
        "topic": "orderbook.50.ETHUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {"u": 100, "seq": 200, "b": [["3000", "1"]], "a": [["3001", "1"]]},
    }
    assert len(normalizer.process(first_snapshot, received_at_ns=1_000_000_000)) == 1

    reset_snapshot = {
        "topic": "orderbook.50.ETHUSDT",
        "type": "snapshot",
        "ts": 2000,
        "data": {
            "u": 5,
            "seq": 10,
            "b": [
                ["2999", "2"],
                ["NaN", "1"],
                ["Infinity", "1"],
                ["0", "1"],
                ["-1", "1"],
                ["2998", "-1"],
                ["2997", "NaN"],
            ],
            "a": [["3002", "0"], ["3003", "Infinity"], ["3004", "3"]],
        },
    }
    records = normalizer.process(reset_snapshot, received_at_ns=2_000_000_000)

    assert len(records) == 1
    assert records[0].payload["bids"] == [["2999", "2"]]
    assert records[0].payload["asks"] == [["3004", "3"]]
    assert records[0].payload["update_id"] == 5
    assert records[0].payload["sequence"] == 10


def test_ignores_orderbook_delta_before_snapshot() -> None:
    normalizer = MarketNormalizer(market_config())
    message = {
        "topic": "orderbook.50.ETHUSDT",
        "type": "delta",
        "data": {"u": 2, "seq": 3, "b": [["3000", "1"]], "a": []},
    }

    assert normalizer.process(message, received_at_ns=1_000_000_000) == []


def test_requires_new_snapshot_after_connection_reset() -> None:
    normalizer = MarketNormalizer(market_config())
    snapshot = {
        "topic": "orderbook.50.ETHUSDT",
        "type": "snapshot",
        "data": {"u": 10, "seq": 20, "b": [["3000", "1"]], "a": []},
    }
    assert len(normalizer.process(snapshot, received_at_ns=1_000_000_000)) == 1

    normalizer.reset_connection_state()
    stale_delta = {
        "topic": "orderbook.50.ETHUSDT",
        "type": "delta",
        "data": {"u": 11, "seq": 21, "b": [["3001", "1"]], "a": []},
    }

    assert normalizer.process(stale_delta, received_at_ns=2_000_000_000) == []


def test_merges_and_samples_ticker_deltas() -> None:
    normalizer = MarketNormalizer(market_config())
    snapshot = {
        "topic": "tickers.SOLUSDT",
        "type": "snapshot",
        "ts": 1000,
        "data": {"lastPrice": "180", "openInterest": "1000"},
    }
    assert len(normalizer.process(snapshot, received_at_ns=1_000_000_000)) == 1

    delta = {
        "topic": "tickers.SOLUSDT",
        "type": "delta",
        "ts": 1500,
        "data": {"lastPrice": "181"},
    }
    assert normalizer.process(delta, received_at_ns=1_500_000_000) == []

    due = {
        "topic": "tickers.SOLUSDT",
        "type": "delta",
        "ts": 2100,
        "data": {"fundingRate": "0.0001"},
    }
    records = normalizer.process(due, received_at_ns=2_100_000_000)

    assert records[0].payload == {
        "lastPrice": "181",
        "openInterest": "1000",
        "fundingRate": "0.0001",
    }


def test_emits_trade_batch_and_only_closed_klines() -> None:
    normalizer = MarketNormalizer(market_config())
    trade = {
        "topic": "publicTrade.XRPUSDT",
        "ts": 1000,
        "data": [{"T": 999, "p": "0.60", "v": "10", "S": "Buy"}],
    }
    trade_records = normalizer.process(trade, received_at_ns=1_000_000_000)
    assert len(trade_records) == 1
    assert trade_records[0].kind == "trades"
    assert trade_records[0].payload[0]["S"] == "Buy"
    assert trade_records[0].schema_version == 1
    assert trade_records[0].source == "bybit"
    assert trade_records[0].session_id is None
    assert trade_records[0].to_dict()["schema_version"] == 1
    assert trade_records[0].to_dict()["source"] == "bybit"
    assert trade_records[0].to_dict()["session_id"] is None

    candles = {
        "topic": "kline.1.XRPUSDT",
        "ts": 2000,
        "data": [
            {"interval": "1", "timestamp": 1000, "confirm": False, "close": "0.61"},
            {
                "interval": "1",
                "start": 1000,
                "timestamp": 0,
                "confirm": True,
                "close": "0.60",
            },
        ],
    }
    candle_records = normalizer.process(candles, received_at_ns=2_000_000_000)

    assert len(candle_records) == 1
    assert candle_records[0].kind == "kline_1"
    assert candle_records[0].exchange_ts_ms == 2000
    assert candle_records[0].payload["start"] == 1000
    assert candle_records[0].payload["timestamp"] == 0
