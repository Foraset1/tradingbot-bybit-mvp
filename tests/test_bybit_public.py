from __future__ import annotations

from tradingbot.market.bybit_public import (
    BybitPublicCollector,
    CollectorStats,
    _topic_data_is_stale,
)


def test_decode_accepts_json_object_as_text_or_bytes() -> None:
    expected = {"topic": "tickers.BTCUSDT", "data": {"lastPrice": "1"}}
    assert BybitPublicCollector._decode(
        '{"topic":"tickers.BTCUSDT","data":{"lastPrice":"1"}}'
    ) == expected
    assert BybitPublicCollector._decode(b'{"op":"pong"}') == {"op": "pong"}


def test_decode_rejects_malformed_or_non_object_json() -> None:
    assert BybitPublicCollector._decode("not json") is None
    assert BybitPublicCollector._decode("[]") is None


def test_stats_track_cumulative_and_current_connection_topic_coverage() -> None:
    stats = CollectorStats(started_at_ms=100)
    stats.connection_opened(connected_at_ms=1_000)

    stats.observe_websocket_message(
        {"topic": "tickers.BTCUSDT", "ts": 1_900}, received_at_ns=2_000_000_000
    )
    stats.observe_websocket_message(
        {"topic": "tickers.BTCUSDT", "ts": 3_200}, received_at_ns=3_000_000_000
    )
    stats.observe_websocket_message({"op": "pong"}, received_at_ns=3_100_000_000)
    stats.pings_sent += 1
    stats.subscription_confirmed = True

    snapshot = stats.snapshot(queue_size=7)
    assert snapshot["current_session_id"] == "100-1"
    assert snapshot["messages_by_topic"] == {"tickers.BTCUSDT": 2}
    assert snapshot["current_connection_messages_by_topic"] == {"tickers.BTCUSDT": 2}
    assert snapshot["last_topic_message_at_ms"] == {"tickers.BTCUSDT": 3_000}
    assert snapshot["pings_sent"] == 1
    assert snapshot["pongs_received"] == 1
    assert snapshot["subscription_confirmed"] is True
    assert snapshot["raw_message_latency_ms"] == {
        "count": 2,
        "total": -100,
        "min": -200,
        "max": 100,
        "avg": -50.0,
        "negative_samples": 1,
    }

    stats.connection_opened(connected_at_ms=4_000)
    assert stats.current_session_id == "100-2"
    assert stats.subscription_confirmed is False
    assert stats.current_connection_messages_by_topic == {}
    assert stats.messages_by_topic == {"tickers.BTCUSDT": 2}


def test_topic_watchdog_is_not_reset_by_non_topic_frames() -> None:
    last_topic_message_at = 10.0
    assert not _topic_data_is_stale(last_topic_message_at, 69.9, 60.0)
    assert _topic_data_is_stale(last_topic_message_at, 70.0, 60.0)
