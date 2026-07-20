from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from tradingbot.config import MarketConfig
from tradingbot.market.records import MarketRecord


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return default


def _valid_level(level: Any) -> tuple[str, str] | None:
    if not isinstance(level, list) or len(level) < 2:
        return None
    price, size = str(level[0]), str(level[1])
    try:
        decimal_price = Decimal(price)
        decimal_size = Decimal(size)
    except InvalidOperation:
        return None
    if (
        not decimal_price.is_finite()
        or not decimal_size.is_finite()
        or decimal_price <= 0
        or decimal_size < 0
    ):
        return None
    return price, size


@dataclass(slots=True)
class OrderBookState:
    bids: dict[str, str] = field(default_factory=dict)
    asks: dict[str, str] = field(default_factory=dict)
    initialized: bool = False
    update_id: int = 0
    sequence: int = 0
    matching_engine_ts_ms: int = 0

    @staticmethod
    def _apply_side(target: dict[str, str], levels: Any) -> None:
        if not isinstance(levels, list):
            return
        for raw_level in levels:
            level = _valid_level(raw_level)
            if level is None:
                continue
            price, size = level
            if Decimal(size) == 0:
                target.pop(price, None)
            else:
                target[price] = size

    def apply(self, message_type: str, data: dict[str, Any], cts_ms: int) -> bool:
        update_id = _as_int(data.get("u"))
        sequence = _as_int(data.get("seq"))
        is_snapshot = message_type == "snapshot" or update_id == 1
        if is_snapshot:
            self.bids.clear()
            self.asks.clear()
            self.initialized = True
        elif (
            not self.initialized
            or update_id <= self.update_id
            or sequence <= self.sequence
        ):
            return False

        self._apply_side(self.bids, data.get("b"))
        self._apply_side(self.asks, data.get("a"))
        self.update_id = update_id
        self.sequence = sequence
        self.matching_engine_ts_ms = cts_ms
        return True

    def snapshot(self, depth: int) -> dict[str, Any]:
        bids = sorted(self.bids.items(), key=lambda item: Decimal(item[0]), reverse=True)[:depth]
        asks = sorted(self.asks.items(), key=lambda item: Decimal(item[0]))[:depth]
        return {
            "bids": [[price, size] for price, size in bids],
            "asks": [[price, size] for price, size in asks],
            "update_id": self.update_id,
            "sequence": self.sequence,
            "matching_engine_ts_ms": self.matching_engine_ts_ms,
        }


class MarketNormalizer:
    """Convert Bybit messages into compact, stable records for later research."""

    def __init__(self, config: MarketConfig) -> None:
        self._config = config
        self._orderbooks: dict[str, OrderBookState] = {}
        self._tickers: dict[str, dict[str, Any]] = {}
        self._last_emit_ns: dict[tuple[str, str], int] = {}

    def reset_connection_state(self) -> None:
        """Discard state that cannot safely survive a WebSocket reconnect."""
        self._orderbooks.clear()
        self._tickers.clear()
        self._last_emit_ns.clear()

    def _due(self, kind: str, symbol: str, now_ns: int, interval_ms: int) -> bool:
        key = (kind, symbol)
        previous = self._last_emit_ns.get(key)
        if previous is not None and now_ns - previous < interval_ms * 1_000_000:
            return False
        self._last_emit_ns[key] = now_ns
        return True

    @staticmethod
    def _symbol_from_topic(topic: str) -> str:
        return topic.rsplit(".", maxsplit=1)[-1].upper()

    def process(self, message: dict[str, Any], received_at_ns: int) -> list[MarketRecord]:
        topic_value = message.get("topic")
        if not isinstance(topic_value, str):
            return []
        topic = topic_value
        symbol = self._symbol_from_topic(topic)
        exchange_ts_ms = _as_int(message.get("ts"), received_at_ns // 1_000_000)

        if topic.startswith("orderbook."):
            return self._orderbook(message, symbol, exchange_ts_ms, received_at_ns)
        if topic.startswith("publicTrade."):
            return self._trades(message, symbol, exchange_ts_ms, received_at_ns)
        if topic.startswith("tickers."):
            return self._ticker(message, symbol, exchange_ts_ms, received_at_ns)
        if topic.startswith("kline."):
            return self._klines(message, symbol, exchange_ts_ms, received_at_ns)
        return []

    def _orderbook(
        self,
        message: dict[str, Any],
        symbol: str,
        exchange_ts_ms: int,
        received_at_ns: int,
    ) -> list[MarketRecord]:
        data = message.get("data")
        if not isinstance(data, dict):
            return []
        state = self._orderbooks.setdefault(symbol, OrderBookState())
        message_type = str(message.get("type", "delta"))
        if not state.apply(message_type, data, _as_int(message.get("cts"))):
            return []
        if message_type != "snapshot" and not self._due(
            "orderbook", symbol, received_at_ns, self._config.orderbook_snapshot_ms
        ):
            return []
        self._last_emit_ns[("orderbook", symbol)] = received_at_ns
        return [
            MarketRecord(
                kind="orderbook",
                symbol=symbol,
                exchange_ts_ms=exchange_ts_ms,
                received_at_ns=received_at_ns,
                payload=state.snapshot(self._config.orderbook_depth),
            )
        ]

    @staticmethod
    def _trades(
        message: dict[str, Any],
        symbol: str,
        exchange_ts_ms: int,
        received_at_ns: int,
    ) -> list[MarketRecord]:
        data = message.get("data")
        if not isinstance(data, list):
            return []
        trades = [dict(item) for item in data if isinstance(item, dict)]
        if not trades:
            return []
        return [
            MarketRecord(
                kind="trades",
                symbol=symbol,
                exchange_ts_ms=exchange_ts_ms,
                received_at_ns=received_at_ns,
                payload=trades,
            )
        ]

    def _ticker(
        self,
        message: dict[str, Any],
        symbol: str,
        exchange_ts_ms: int,
        received_at_ns: int,
    ) -> list[MarketRecord]:
        data = message.get("data")
        if not isinstance(data, dict):
            return []
        message_type = str(message.get("type", "delta"))
        if message_type == "snapshot" or symbol not in self._tickers:
            self._tickers[symbol] = {}
        state = self._tickers[symbol]
        state.update(data)
        if message_type != "snapshot" and not self._due(
            "ticker", symbol, received_at_ns, self._config.ticker_snapshot_ms
        ):
            return []
        self._last_emit_ns[("ticker", symbol)] = received_at_ns
        return [
            MarketRecord(
                kind="ticker",
                symbol=symbol,
                exchange_ts_ms=exchange_ts_ms,
                received_at_ns=received_at_ns,
                payload=dict(state),
            )
        ]

    @staticmethod
    def _klines(
        message: dict[str, Any],
        symbol: str,
        exchange_ts_ms: int,
        received_at_ns: int,
    ) -> list[MarketRecord]:
        data = message.get("data")
        if not isinstance(data, list):
            return []
        records: list[MarketRecord] = []
        for item in data:
            if not isinstance(item, dict) or item.get("confirm") is not True:
                continue
            candle = dict(item)
            records.append(
                MarketRecord(
                    kind=f"kline_{candle.get('interval', 'unknown')}",
                    symbol=symbol,
                    exchange_ts_ms=exchange_ts_ms,
                    received_at_ns=received_at_ns,
                    payload=candle,
                )
            )
        return records
