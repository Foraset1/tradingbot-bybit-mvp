"""Live public-data Shadow Mode with no private API or order path."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import signal
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from tradingbot.config import AppConfig
from tradingbot.market.bybit_public import BybitPublicCollector, CollectorStats
from tradingbot.market.records import MarketRecord
from tradingbot.research.builder import NS_PER_SECOND
from tradingbot.research.contracts import ExecutionResearchParameters
from tradingbot.research.execution_builder import _barrier_prices
from tradingbot.shadow.bundle import (
    ShadowBundle,
    ShadowBundleError,
    validate_shadow_bundle,
)
from tradingbot.shadow.journal import ShadowJournal
from tradingbot.shadow.live import LiveMarketWindow, SessionTransition
from tradingbot.shadow.model import ShadowCandidate, ShadowScorer

NS_PER_MINUTE: Final = 60 * NS_PER_SECOND
NS_PER_DAY: Final = 24 * 60 * NS_PER_MINUTE
TRADING_CREDENTIAL_ENV_VARS: Final = (
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
    "BYBIT_API_SECRET_KEY",
    "BYBIT_SECRET",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
)
SETTLEMENT_GRACE_SECONDS: Final = 90
FROZEN_PUBLIC_WS_URL: Final = "wss://stream.bybit.com/v5/public/linear"


@dataclass(slots=True)
class PendingShadowCandidate:
    candidate: ShadowCandidate
    feature: dict[str, object]
    settle_after_ns: int


def reject_trading_credentials(environment: dict[str, str] | None = None) -> None:
    selected = os.environ if environment is None else environment
    present = sorted(name for name in TRADING_CREDENTIAL_ENV_VARS if selected.get(name))
    if present:
        raise ShadowBundleError(
            "Shadow Mode refuses trading credentials in its environment: "
            + ", ".join(present)
        )


def default_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"shadow-{stamp}-{uuid.uuid4().hex[:8]}"


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ShadowBundleError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowBundleError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowBundleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ShadowBundleError(f"{label} must be finite")
    return result


def _research_parameters(bundle: ShadowBundle) -> ExecutionResearchParameters:
    raw = _object(bundle.contract.get("research_parameters"), "research_parameters")
    scenario = _object(bundle.contract.get("scenario"), "scenario")
    horizon = _integer(scenario.get("horizon_minutes"), "horizon_minutes")
    notional = _number(
        scenario.get("reference_order_notional_usdt"), "order_notional_usdt"
    )
    parameters = ExecutionResearchParameters(
        decision_interval_seconds=_integer(
            raw.get("decision_interval_seconds"), "decision_interval_seconds"
        ),
        decision_offset_seconds=_integer(
            raw.get("decision_offset_seconds"), "decision_offset_seconds"
        ),
        kline_history_minutes=_integer(
            raw.get("kline_history_minutes"), "kline_history_minutes"
        ),
        max_orderbook_age_ms=_integer(
            raw.get("max_orderbook_age_ms"), "max_orderbook_age_ms"
        ),
        max_ticker_age_ms=_integer(
            raw.get("max_ticker_age_ms"), "max_ticker_age_ms"
        ),
        position_horizons_minutes=(horizon,),
        volatility_lookback_minutes=_integer(
            raw.get("volatility_lookback_minutes"),
            "volatility_lookback_minutes",
        ),
        stop_volatility_multiple=_number(
            raw.get("stop_volatility_multiple"), "stop_volatility_multiple"
        ),
        take_profit_multiple=_number(
            raw.get("take_profit_multiple"), "take_profit_multiple"
        ),
        minimum_stop_bps=_number(raw.get("minimum_stop_bps"), "minimum_stop_bps"),
        maximum_stop_bps=_number(raw.get("maximum_stop_bps"), "maximum_stop_bps"),
        order_notionals_usdt=(notional,),
        submission_latency_ms=_integer(
            raw.get("submission_latency_ms"), "submission_latency_ms"
        ),
        activation_max_delay_ms=_integer(
            raw.get("activation_max_delay_ms"), "activation_max_delay_ms"
        ),
        entry_ttl_seconds=_integer(
            raw.get("entry_ttl_seconds"), "entry_ttl_seconds"
        ),
        queue_ahead_multiplier=_number(
            raw.get("queue_ahead_multiplier"), "queue_ahead_multiplier"
        ),
        maximum_continuity_gap_ms=_integer(
            raw.get("maximum_continuity_gap_ms"), "maximum_continuity_gap_ms"
        ),
    )
    parameters.validate()
    return parameters


class ShadowRuntime:
    def __init__(
        self,
        *,
        config: AppConfig,
        bundle: ShadowBundle,
        journal: ShadowJournal,
        processing_delay_ms: int,
    ) -> None:
        if processing_delay_ms < 0 or processing_delay_ms > 10_000:
            raise ShadowBundleError("processing delay must be between 0 and 10000 ms")
        universe = _object(bundle.contract.get("universe"), "universe")
        symbols_raw = universe.get("symbols")
        if not isinstance(symbols_raw, list) or not all(
            isinstance(value, str) for value in symbols_raw
        ):
            raise ShadowBundleError("shadow universe is invalid")
        symbols = tuple(cast(list[str], symbols_raw))
        if symbols != config.bybit.symbols:
            raise ShadowBundleError(
                "configured Bybit symbols must exactly match the frozen bundle order"
            )
        if config.risk.max_open_positions != 1:
            raise ShadowBundleError("Shadow Mode requires exactly one global position")
        if config.bybit.public_ws_url != FROZEN_PUBLIC_WS_URL:
            raise ShadowBundleError(
                "Shadow Mode requires the frozen Bybit production linear public endpoint"
            )
        if (
            config.market.orderbook_depth != 50
            or config.market.orderbook_snapshot_ms != 1000
            or config.market.ticker_snapshot_ms != 1000
            or "1" not in config.market.kline_intervals
            or not config.market.collect_orderbook
            or not config.market.collect_trades
            or not config.market.collect_tickers
            or not config.market.collect_klines
        ):
            raise ShadowBundleError(
                "configured live market profile differs from frozen model inputs"
            )
        self.config = config
        self.bundle = bundle
        self.journal = journal
        self.parameters = _research_parameters(bundle)
        self.scorer = ShadowScorer(bundle)
        retention_minutes = max(
            70,
            self.parameters.kline_history_minutes
            + self.parameters.position_horizons_minutes[0]
            + 5,
        )
        self.window = LiveMarketWindow(
            symbols,
            retention_minutes=retention_minutes,
        )
        self.processing_delay_ns = processing_delay_ms * 1_000_000
        self.stats = CollectorStats()
        self.pending: PendingShadowCandidate | None = None
        self.last_decision_at_ns = -1
        self.equity = 1.0
        self.realized_24h: deque[tuple[int, float]] = deque()
        self.rolling_24h_return = 0.0
        self.decisions = 0
        self.selected = 0
        self.settled = 0
        self.skipped = 0
        self.session_resets = 0
        self.stop_reason: str | None = None
        self._restore_journal_state()

    def _restore_journal_state(self) -> None:
        unresolved: dict[str, dict[str, Any]] = {}
        for event in self.journal.events:
            payload = _object(event.get("payload"), "journal event payload")
            event_type = event.get("event_type")
            if event_type == "candidate_selected":
                candidate = _object(payload.get("candidate"), "selected candidate")
                unresolved[str(candidate["decision_id"])] = payload
            elif event_type in {"candidate_settled", "candidate_unresolved"}:
                decision_id = str(payload.get("decision_id"))
                unresolved.pop(decision_id, None)
                if event_type == "candidate_settled":
                    realized = _number(
                        payload.get("realized_equity_return_fraction"),
                        "realized return",
                    )
                    settled_at_ns = _integer(
                        payload.get("settled_at_ns"), "settled_at_ns"
                    )
                    self.realized_24h.append((settled_at_ns, realized))
                    self.equity = _number(payload.get("equity_after"), "equity_after")
        now_ns = time.time_ns()
        while self.realized_24h and self.realized_24h[0][0] < now_ns - NS_PER_DAY:
            self.realized_24h.popleft()
        self.rolling_24h_return = sum(value for _, value in self.realized_24h)
        for payload in unresolved.values():
            candidate = _object(payload.get("candidate"), "selected candidate")
            self.journal.append(
                "candidate_unresolved",
                {
                    "decision_id": str(candidate["decision_id"]),
                    "reason": "restart_without_complete_public_market_replay",
                    "realized_equity_return_fraction": 0.0,
                },
            )

    async def sink(self, record: MarketRecord) -> None:
        transition = self.window.accept(record)
        if transition is not None:
            self._on_session_transition(transition)

    def _on_session_transition(self, transition: SessionTransition) -> None:
        self.session_resets += 1
        if self.pending is not None:
            self._mark_unresolved("websocket_session_reset")
        self.journal.append(
            "websocket_session_reset",
            {
                "previous_session_id": transition.previous_session_id,
                "new_session_id": transition.new_session_id,
                "buffers_cleared": True,
                "warmup_restarted": True,
            },
        )

    def _expire_rolling(self, at_ns: int) -> None:
        while self.realized_24h and self.realized_24h[0][0] < at_ns - NS_PER_DAY:
            _, value = self.realized_24h.popleft()
            self.rolling_24h_return -= value

    def _loss_guard_active(self) -> bool:
        return self.rolling_24h_return <= -self.scorer.rolling_24h_loss_fraction

    def _mark_unresolved(self, reason: str) -> None:
        if self.pending is None:
            return
        self.journal.append(
            "candidate_unresolved",
            {
                "decision_id": self.pending.candidate.decision_id,
                "reason": reason,
                "realized_equity_return_fraction": 0.0,
            },
        )
        self.pending = None

    def _settlement_pnl(
        self,
        candidate: ShadowCandidate,
        feature: dict[str, object],
        label: dict[str, object],
    ) -> tuple[float, float, float, float, float]:
        evaluation = _object(
            self.bundle.contract.get("evaluation_parameters"),
            "evaluation_parameters",
        )
        maker = _number(evaluation.get("maker_fee_bps"), "maker fee")
        taker = _number(evaluation.get("taker_fee_bps"), "taker fee")
        stop_slippage = _number(
            evaluation.get("stop_slippage_bps"), "stop slippage"
        )
        timeout_slippage = _number(
            evaluation.get("timeout_slippage_bps"), "timeout slippage"
        )
        fill_status = str(label["fill_status"])
        fee_bps = 0.0
        slippage_bps = 0.0
        funding_bps = 0.0
        net_bps = 0.0
        realized_notional_fraction = 1.0
        if fill_status == "PARTIAL_FILL":
            fraction = _number(label.get("fill_fraction"), "fill_fraction")
            fee_bps = maker + taker
            slippage_bps = timeout_slippage
            net_bps = -fee_bps - slippage_bps
            realized_notional_fraction = fraction
        elif fill_status == "FULL_FILL":
            outcome = str(label["outcome"])
            gross = _number(label.get("outcome_return_bps"), "outcome_return_bps")
            fee_bps = maker + (maker if outcome == "TP_FIRST" else taker)
            if outcome == "SL_FIRST":
                slippage_bps = stop_slippage
            elif outcome == "TIMEOUT":
                slippage_bps = timeout_slippage
            minutes = feature.get("minutes_to_funding")
            rate = feature.get("funding_rate")
            if isinstance(minutes, (int, float)) and isinstance(rate, (int, float)):
                minutes_float = float(minutes)
                rate_float = float(rate)
                exit_at_ns_raw = label.get("hit_at_ns") or label.get("position_end_ns")
                if (
                    math.isfinite(minutes_float)
                    and math.isfinite(rate_float)
                    and exit_at_ns_raw is not None
                ):
                    funding_at_ns = candidate.decision_at_ns + int(
                        minutes_float * NS_PER_MINUTE
                    )
                    if funding_at_ns <= int(cast(int, exit_at_ns_raw)):
                        direction = 1.0 if candidate.side == "LONG" else -1.0
                        funding_bps = direction * rate_float * 10_000
            net_bps = gross - fee_bps - slippage_bps - funding_bps
        elif fill_status != "NO_FILL":
            raise ShadowBundleError(f"unknown proxy fill status: {fill_status}")
        equity_return = (
            candidate.notional_fraction
            * realized_notional_fraction
            * net_bps
            / 10_000
        )
        return fee_bps, slippage_bps, funding_bps, net_bps, equity_return

    def _try_settle(self, now_ns: int) -> None:
        if self.pending is None or now_ns < self.pending.settle_after_ns:
            return
        if (
            self.window.latest_received_at_ns is None
            or self.window.latest_received_at_ns < self.pending.settle_after_ns
        ):
            return
        pending = self.pending
        label, quality = self.window.execution_label(
            feature=pending.feature,
            side=pending.candidate.side,
            parameters=self.parameters,
            at_ns=now_ns,
        )
        if label is None:
            self.journal.append(
                "candidate_unresolved",
                {
                    "decision_id": pending.candidate.decision_id,
                    "reason": "proxy_label_unavailable_or_discontinuous",
                    "quality": quality,
                    "realized_equity_return_fraction": 0.0,
                },
            )
            self.pending = None
            return
        if label["fill_status"] == "FULL_FILL" and (
            label.get("outcome") not in {"SL_FIRST", "TIMEOUT", "TP_FIRST"}
            or label.get("outcome_return_bps") is None
        ):
            self.journal.append(
                "candidate_unresolved",
                {
                    "decision_id": pending.candidate.decision_id,
                    "reason": "ambiguous_or_unpriced_full_fill_proxy",
                    "proxy_outcome": label.get("outcome"),
                    "proxy_resolution": label.get("resolution"),
                    "quality": quality,
                    "realized_equity_return_fraction": 0.0,
                },
            )
            self.pending = None
            return
        fee, slippage, funding, net_bps, equity_return = self._settlement_pnl(
            pending.candidate, pending.feature, label
        )
        equity_before = self.equity
        self.equity *= 1.0 + equity_return
        settled_at_ns = now_ns
        self.realized_24h.append((settled_at_ns, equity_return))
        self.rolling_24h_return += equity_return
        self.settled += 1
        self.journal.append(
            "candidate_settled",
            {
                "decision_id": pending.candidate.decision_id,
                "settled_at_ns": settled_at_ns,
                "public_execution_proxy": {
                    key: label.get(key)
                    for key in (
                        "fill_status",
                        "fill_fraction",
                        "full_fill_at_ns",
                        "outcome",
                        "hit_at_ns",
                        "position_end_ns",
                        "outcome_return_bps",
                        "resolution",
                    )
                },
                "fee_bps": fee,
                "slippage_bps": slippage,
                "funding_cost_bps": funding,
                "net_return_bps": net_bps,
                "realized_equity_return_fraction": equity_return,
                "equity_before": equity_before,
                "equity_after": self.equity,
                "quality": quality,
                "real_exchange_order": False,
            },
        )
        self.pending = None

    def _decision_candidates(
        self, decision_at_ns: int
    ) -> tuple[list[ShadowCandidate], dict[str, str], dict[str, dict[str, object]]]:
        features, skipped = self.window.features_at(decision_at_ns, self.parameters)
        candidates: list[ShadowCandidate] = []
        for symbol, feature in features.items():
            for side in ("LONG", "SHORT"):
                entry = _number(
                    feature[
                        "best_bid_price" if side == "LONG" else "best_ask_price"
                    ],
                    "entry price",
                )
                stop, take_profit, stop_price, take_profit_price = _barrier_prices(
                    feature,
                    self.parameters,
                    side=side,
                    horizon_minutes=self.parameters.position_horizons_minutes[0],
                    entry_price=entry,
                )
                candidates.append(
                    self.scorer.score(
                        feature,
                        symbol=symbol,
                        side=side,
                        entry_limit_price=entry,
                        stop_distance_bps=stop,
                        take_profit_distance_bps=take_profit,
                        stop_price=stop_price,
                        take_profit_price=take_profit_price,
                    )
                )
        return candidates, skipped, features

    def evaluate(self, decision_at_ns: int) -> None:
        self._try_settle(time.time_ns())
        self._expire_rolling(decision_at_ns)
        self.last_decision_at_ns = decision_at_ns
        self.decisions += 1
        candidates, feature_skips, features = self._decision_candidates(decision_at_ns)
        eligible = [
            value
            for value in candidates
            if value.expected_net_bps >= self.scorer.minimum_expected_net_bps
        ]
        selected: ShadowCandidate | None = None
        skip_reason: str | None = None
        if self.pending is not None:
            skip_reason = "position_or_order_already_open"
        elif self._loss_guard_active():
            skip_reason = "rolling_24h_loss_limit"
        elif not candidates:
            skip_reason = "warmup_or_data_quality"
        elif not eligible:
            skip_reason = "no_positive_expected_value_candidate"
        else:
            symbol_rank = {
                symbol: index for index, symbol in enumerate(self.scorer.symbols)
            }
            selected = min(
                eligible,
                key=lambda value: (
                    -value.expected_net_bps,
                    symbol_rank[value.symbol],
                    0 if value.side == "LONG" else 1,
                ),
            )
        cycle_payload: dict[str, object] = {
            "decision_at_ns": decision_at_ns,
            "session_id": self.window.current_session_id,
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "minimum_expected_net_bps": self.scorer.minimum_expected_net_bps,
            "feature_skips": feature_skips,
            "causal_features": features,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "selected_decision_id": None if selected is None else selected.decision_id,
            "skip_reason": skip_reason,
            "rolling_24h_return_fraction": self.rolling_24h_return,
            "simulated_equity": self.equity,
            "order_submission": False,
        }
        self.journal.append("decision_cycle", cycle_payload)
        if selected is None:
            self.skipped += 1
            return
        feature = features[selected.symbol]
        settle_after_ns = (
            decision_at_ns
            + self.parameters.entry_ttl_seconds * NS_PER_SECOND
            + self.parameters.position_horizons_minutes[0] * NS_PER_MINUTE
            + SETTLEMENT_GRACE_SECONDS * NS_PER_SECOND
        )
        self.pending = PendingShadowCandidate(
            candidate=selected,
            feature=feature,
            settle_after_ns=settle_after_ns,
        )
        self.selected += 1
        self.journal.append(
            "candidate_selected",
            {
                "candidate": selected.to_dict(),
                "feature": feature,
                "settle_after_ns": settle_after_ns,
                "conservative_lock_policy": (
                    "global slot remains locked through entry TTL, full horizon, "
                    "and settlement grace"
                ),
                "real_exchange_order": False,
            },
        )

    def health_payload(self, status: str) -> dict[str, object]:
        disk = shutil.disk_usage(self.journal.root)
        return {
            "status": status,
            "updated_at_ns": time.time_ns(),
            "bundle_id": self.bundle.bundle_id,
            "bundle_fingerprint": self.bundle.bundle_fingerprint,
            "data_mode": self.bundle.contract["data_gate"]["mode"],
            "engineering_only": self.bundle.contract["scope"]["engineering_only"],
            "session_id": self.window.current_session_id,
            "last_market_record_at_ns": self.window.latest_received_at_ns,
            "last_decision_at_ns": (
                None if self.last_decision_at_ns < 0 else self.last_decision_at_ns
            ),
            "pending_decision_id": (
                None if self.pending is None else self.pending.candidate.decision_id
            ),
            "decisions": self.decisions,
            "selected": self.selected,
            "settled": self.settled,
            "skipped": self.skipped,
            "session_resets": self.session_resets,
            "stop_reason": self.stop_reason,
            "simulated_equity": self.equity,
            "rolling_24h_return_fraction": self.rolling_24h_return,
            "collector": self.stats.snapshot(0),
            "disk": {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "minimum_free_bytes": self.config.storage.min_free_bytes,
                "above_guard": disk.free >= self.config.storage.min_free_bytes,
            },
            "scope": {
                "bybit_access": "public-read-only",
                "order_submission": False,
                "trading_credentials_allowed": False,
            },
        }


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(signum, lambda _signum, _frame: stop_event.set())


async def _stop_after(seconds: float, stop_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        stop_event.set()


async def _decision_worker(
    runtime: ShadowRuntime,
    stop_event: asyncio.Event,
) -> None:
    interval_ns = runtime.parameters.decision_interval_seconds * NS_PER_SECOND
    offset_ns = runtime.parameters.decision_offset_seconds * NS_PER_SECOND
    while not stop_event.is_set():
        now_ns = time.time_ns()
        runtime._try_settle(now_ns)
        decision_at_ns = ((now_ns - offset_ns) // interval_ns) * interval_ns + offset_ns
        ready_at_ns = decision_at_ns + runtime.processing_delay_ns
        if decision_at_ns > runtime.last_decision_at_ns and now_ns >= ready_at_ns:
            runtime.evaluate(decision_at_ns)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=0.5)


async def _health_worker(
    runtime: ShadowRuntime,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        health = runtime.health_payload("running")
        disk = _object(health["disk"], "health.disk")
        if _integer(disk["free_bytes"], "free_bytes") < _integer(
            disk["minimum_free_bytes"], "minimum_free_bytes"
        ):
            runtime.journal.append(
                "disk_guard_triggered",
                {
                    "free_bytes": disk["free_bytes"],
                    "minimum_free_bytes": disk["minimum_free_bytes"],
                    "order_submission": False,
                },
            )
            runtime.stop_reason = "disk_guard"
            runtime.journal.write_health(runtime.health_payload("error_disk_guard"))
            stop_event.set()
            return
        runtime.journal.write_health(health)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=5.0)


async def run_shadow_mode(
    *,
    config: AppConfig,
    bundle_path: str | Path,
    shadow_root: str | Path,
    run_id: str | None = None,
    run_seconds: float | None = None,
    processing_delay_ms: int = 750,
) -> dict[str, object]:
    reject_trading_credentials()
    if run_seconds is not None and run_seconds <= 0:
        raise ShadowBundleError("shadow run duration must be positive")
    bundle = validate_shadow_bundle(bundle_path)
    selected_run_id = default_run_id() if run_id is None else run_id
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    with ShadowJournal(
        shadow_root,
        run_id=selected_run_id,
        bundle=bundle,
    ) as journal:
        runtime = ShadowRuntime(
            config=config,
            bundle=bundle,
            journal=journal,
            processing_delay_ms=processing_delay_ms,
        )
        journal.append(
            "run_resumed" if journal.sequence else "run_started",
            {
                "bundle_id": bundle.bundle_id,
                "bundle_fingerprint": bundle.bundle_fingerprint,
                "data_mode": bundle.contract["data_gate"]["mode"],
                "engineering_only": bundle.contract["scope"]["engineering_only"],
                "warmup_minutes": runtime.parameters.kline_history_minutes + 1,
                "order_submission": False,
            },
        )
        collector_config = replace(config)
        collector = BybitPublicCollector(
            collector_config,
            runtime.sink,
            runtime.stats,
        )
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(collector.run(stop_event), name="shadow-public-collector")
                tasks.create_task(
                    _decision_worker(runtime, stop_event), name="shadow-decision-worker"
                )
                tasks.create_task(
                    _health_worker(runtime, stop_event), name="shadow-health-worker"
                )
                if run_seconds is not None:
                    tasks.create_task(
                        _stop_after(run_seconds, stop_event), name="shadow-automatic-stop"
                    )
        finally:
            stop_event.set()
            if runtime.stop_reason is None:
                runtime.stop_reason = "graceful_shutdown"
            runtime._mark_unresolved(f"shadow_shutdown_{runtime.stop_reason}")
            journal.append(
                "run_stopped",
                {
                    "decisions": runtime.decisions,
                    "selected": runtime.selected,
                    "settled": runtime.settled,
                    "skipped": runtime.skipped,
                    "stop_reason": runtime.stop_reason,
                    "simulated_equity": runtime.equity,
                    "order_submission": False,
                },
            )
            journal.write_health(runtime.health_payload("stopped"))
        return {
            "ok": True,
            "run_id": selected_run_id,
            "run_path": journal.root.as_posix(),
            "bundle_id": bundle.bundle_id,
            "data_mode": bundle.contract["data_gate"]["mode"],
            "engineering_only": bundle.contract["scope"]["engineering_only"],
            "decisions": runtime.decisions,
            "selected": runtime.selected,
            "settled": runtime.settled,
            "order_submission": False,
        }
