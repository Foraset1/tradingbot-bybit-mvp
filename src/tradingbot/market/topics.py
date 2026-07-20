from __future__ import annotations

from tradingbot.config import AppConfig


def build_public_topics(config: AppConfig) -> tuple[str, ...]:
    """Build deterministic Bybit V5 public subscription topics."""
    topics: list[str] = []
    for symbol in config.bybit.symbols:
        if config.market.collect_orderbook:
            topics.append(f"orderbook.{config.market.orderbook_depth}.{symbol}")
        if config.market.collect_trades:
            topics.append(f"publicTrade.{symbol}")
        if config.market.collect_tickers:
            topics.append(f"tickers.{symbol}")
        if config.market.collect_klines:
            topics.extend(
                f"kline.{interval}.{symbol}" for interval in config.market.kline_intervals
            )
    return tuple(topics)

