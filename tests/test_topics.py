from __future__ import annotations

from pathlib import Path

from tradingbot.config import load_config
from tradingbot.market.topics import build_public_topics


def test_builds_six_topics_per_symbol(config_path: Path) -> None:
    config = load_config(config_path)
    topics = build_public_topics(config)

    assert len(topics) == 36
    assert len(set(topics)) == 36
    assert topics[:6] == (
        "orderbook.50.BTCUSDT",
        "publicTrade.BTCUSDT",
        "tickers.BTCUSDT",
        "kline.1.BTCUSDT",
        "kline.5.BTCUSDT",
        "kline.15.BTCUSDT",
    )
    assert topics[-1] == "kline.15.LINKUSDT"
