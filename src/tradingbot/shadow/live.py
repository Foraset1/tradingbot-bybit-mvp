"""Bounded causal market window shared by live shadow inference and replay labels."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from tradingbot.data.canonical import (
    KLINE_SCHEMA,
    ORDERBOOK_SCHEMA,
    TICKER_SCHEMA,
    TRADES_SCHEMA,
    canonical_rows_from_market_record,
)
from tradingbot.market.records import MarketRecord
from tradingbot.research.builder import (
    NS_PER_MILLISECOND,
    NS_PER_SECOND,
    _add_btc_context,
    _decision_id,
    _KlineSeries,
    _OrderBookSeries,
    _TickerSeries,
    _time_features,
    _TradeSeries,
    _utc_date_from_ns,
)
from tradingbot.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    ExecutionResearchParameters,
    ResearchBuildError,
)
from tradingbot.research.execution_builder import (
    _ContinuityGuard,
    _execution_label_rows,
)
from tradingbot.shadow.bundle import ShadowBundleError


@dataclass(frozen=True, slots=True)
class SessionTransition:
    previous_session_id: str
    new_session_id: str


@dataclass(frozen=True, slots=True)
class SymbolSeries:
    books: _OrderBookSeries
    ticker: _TickerSeries
    klines: _KlineSeries
    trades: _TradeSeries


class LiveMarketWindow:
    """Retain only one WebSocket session and enough rows for H30 settlement."""

    def __init__(self, symbols: tuple[str, ...], *, retention_minutes: int) -> None:
        if not symbols or "BTCUSDT" not in symbols:
            raise ShadowBundleError("live shadow universe must include BTCUSDT")
        if retention_minutes < 65:
            raise ShadowBundleError("live shadow retention must cover feature warm-up")
        self.symbols = symbols
        self.symbol_set = frozenset(symbols)
        self.retention_ns = retention_minutes * 60 * NS_PER_SECOND
        self.current_session_id: str | None = None
        self.latest_received_at_ns: int | None = None
        self._source_line = 0
        self._rows: dict[str, dict[str, deque[dict[str, Any]]]] = {
            symbol: {
                "orderbook": deque(),
                "ticker": deque(),
                "trades": deque(),
                "kline_1": deque(),
            }
            for symbol in symbols
        }

    def _clear(self) -> None:
        for by_kind in self._rows.values():
            for rows in by_kind.values():
                rows.clear()
        self.latest_received_at_ns = None

    def accept(self, record: MarketRecord) -> SessionTransition | None:
        if record.symbol not in self.symbol_set:
            raise ShadowBundleError(f"collector emitted unconfigured symbol {record.symbol}")
        if record.session_id is None:
            raise ShadowBundleError("live public record has no WebSocket session ID")
        transition: SessionTransition | None = None
        if self.current_session_id is None:
            self.current_session_id = record.session_id
        elif record.session_id != self.current_session_id:
            transition = SessionTransition(self.current_session_id, record.session_id)
            self._clear()
            self.current_session_id = record.session_id
        if record.kind in {"kline_5", "kline_15"}:
            return transition
        by_kind = self._rows[record.symbol]
        if record.kind not in by_kind:
            raise ShadowBundleError(f"unsupported live record kind {record.kind}")
        self._source_line += 1
        rows = canonical_rows_from_market_record(
            record,
            source_path="<shadow-public-websocket>",
            source_line=self._source_line,
        )
        by_kind[record.kind].extend(rows)
        self.latest_received_at_ns = max(
            record.received_at_ns,
            self.latest_received_at_ns or record.received_at_ns,
        )
        self._trim(self.latest_received_at_ns - self.retention_ns)
        return transition

    def _trim(self, cutoff_ns: int) -> None:
        for by_kind in self._rows.values():
            for rows in by_kind.values():
                while rows and int(rows[0]["received_at_ns"]) < cutoff_ns:
                    rows.popleft()

    @staticmethod
    def _causal_rows(
        rows: deque[dict[str, Any]], at_ns: int
    ) -> list[dict[str, Any]]:
        return [row for row in rows if int(row["received_at_ns"]) <= at_ns]

    def series(self, symbol: str, *, at_ns: int) -> SymbolSeries:
        if symbol not in self.symbol_set:
            raise ShadowBundleError(f"unknown live shadow symbol {symbol}")
        source = self._rows[symbol]
        orderbook_rows = self._causal_rows(source["orderbook"], at_ns)
        ticker_rows = self._causal_rows(source["ticker"], at_ns)
        trade_rows = self._causal_rows(source["trades"], at_ns)
        raw_klines = self._causal_rows(source["kline_1"], at_ns)
        latest_klines: dict[int, dict[str, Any]] = {}
        for row in raw_klines:
            start_ms = int(row["start_ms"])
            previous = latest_klines.get(start_ms)
            if previous is None or int(row["received_at_ns"]) > int(
                previous["received_at_ns"]
            ):
                latest_klines[start_ms] = row
        if not orderbook_rows or not ticker_rows or not trade_rows or not latest_klines:
            raise ResearchBuildError(f"{symbol} live window is incomplete")
        return SymbolSeries(
            books=_OrderBookSeries(
                pa.Table.from_pylist(orderbook_rows, schema=ORDERBOOK_SCHEMA)
            ),
            ticker=_TickerSeries(
                pa.Table.from_pylist(ticker_rows, schema=TICKER_SCHEMA)
            ),
            klines=_KlineSeries(
                pa.Table.from_pylist(
                    list(latest_klines.values()), schema=KLINE_SCHEMA
                )
            ),
            trades=_TradeSeries(
                pa.Table.from_pylist(trade_rows, schema=TRADES_SCHEMA)
            ),
        )

    def features_at(
        self,
        decision_at_ns: int,
        parameters: ExecutionResearchParameters,
    ) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
        features: dict[str, list[dict[str, object]]] = defaultdict(list)
        skipped: dict[str, str] = {}
        feature_start_ns = decision_at_ns - (
            parameters.kline_history_minutes + 1
        ) * 60 * NS_PER_SECOND
        for symbol in self.symbols:
            try:
                series = self.series(symbol, at_ns=decision_at_ns)
            except ResearchBuildError:
                skipped[symbol] = "incomplete_live_window"
                continue
            if not series.books.has_continuous_coverage(
                feature_start_ns,
                decision_at_ns,
                parameters.maximum_continuity_gap_ms,
            ):
                skipped[symbol] = "discontinuous_orderbook_feature_window"
                continue
            book, reason = series.books.features_at(
                decision_at_ns, parameters.max_orderbook_age_ms
            )
            if book is None:
                skipped[symbol] = str(reason)
                continue
            ticker, reason = series.ticker.features_at(
                decision_at_ns, parameters.max_ticker_age_ms
            )
            if ticker is None:
                skipped[symbol] = str(reason)
                continue
            kline, reason = series.klines.features_at(
                decision_at_ns, parameters.kline_history_minutes
            )
            if kline is None:
                skipped[symbol] = str(reason)
                continue
            feature: dict[str, object] = {
                "research_schema_version": RESEARCH_SCHEMA_VERSION,
                "decision_id": _decision_id(
                    "shadow-live-v1", symbol, decision_at_ns
                ),
                "source_dataset_id": "shadow-live-v1",
                "symbol": symbol,
                "decision_at_ns": decision_at_ns,
                "decision_at_ms": decision_at_ns // NS_PER_MILLISECOND,
                "decision_utc_date": _utc_date_from_ns(decision_at_ns),
            }
            feature.update(book)
            feature.update(ticker)
            feature.update(kline)
            feature.update(series.trades.features_at(decision_at_ns))
            feature.update(_time_features(decision_at_ns))
            features[symbol].append(feature)
        if not features.get("BTCUSDT"):
            return {}, {**skipped, "*": "btc_context_unavailable"}
        _add_btc_context(features)
        flattened = {symbol: rows[0] for symbol, rows in features.items()}
        return flattened, skipped

    def execution_label(
        self,
        *,
        feature: dict[str, object],
        side: str,
        parameters: ExecutionResearchParameters,
        at_ns: int,
    ) -> tuple[dict[str, object] | None, dict[str, int]]:
        symbol = str(feature["symbol"])
        try:
            series = self.series(symbol, at_ns=at_ns)
        except ResearchBuildError:
            return None, {"settlement_incomplete_live_window": 1}
        quality: Counter[str] = Counter()
        continuity = _ContinuityGuard(
            books=series.books,
            klines=series.klines,
            maximum_gap_ms=parameters.maximum_continuity_gap_ms,
        )
        labels = _execution_label_rows(
            source_dataset_id="shadow-live-v1",
            symbol=symbol,
            feature=feature,
            books=series.books,
            trades=series.trades,
            parameters=parameters,
            quality=quality,
            continuity=continuity,
        )
        scenario = [
            row
            for row in labels
            if row["side"] == side
            and row["horizon_minutes"] == parameters.position_horizons_minutes[0]
            and row["order_notional_usdt"] == parameters.order_notionals_usdt[0]
        ]
        if len(scenario) != 1:
            return None, dict(quality)
        return scenario[0], dict(quality)
