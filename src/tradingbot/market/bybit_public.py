from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

from tradingbot.config import AppConfig
from tradingbot.market.normalizer import MarketNormalizer
from tradingbot.market.records import MarketRecord
from tradingbot.market.topics import build_public_topics

LOGGER = logging.getLogger(__name__)
RecordSink = Callable[[MarketRecord], Awaitable[None]]


def _topic_data_is_stale(
    last_topic_message_at: float, now: float, stale_connection_seconds: float
) -> bool:
    return now - last_topic_message_at >= stale_connection_seconds


@dataclass(slots=True)
class CollectorStats:
    started_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    status: str = "starting"
    connections: int = 0
    reconnects: int = 0
    websocket_messages: int = 0
    records_emitted: int = 0
    last_message_at_ms: int | None = None
    current_connection_last_message_at_ms: int | None = None
    last_error: str | None = None
    subscribed_topics: int = 0
    expected_topics: tuple[str, ...] = ()
    subscription_confirmed: bool = False
    pings_sent: int = 0
    pongs_received: int = 0
    current_session_id: str | None = None
    connected_at_ms: int | None = None
    records_by_kind: Counter[str] = field(default_factory=Counter)
    messages_by_topic: Counter[str] = field(default_factory=Counter)
    last_topic_message_at_ms: dict[str, int] = field(default_factory=dict)
    current_connection_messages_by_topic: Counter[str] = field(default_factory=Counter)
    current_connection_last_topic_message_at_ms: dict[str, int] = field(
        default_factory=dict
    )
    raw_message_latency_count: int = 0
    raw_message_latency_total_ms: int = 0
    raw_message_latency_min_ms: int | None = None
    raw_message_latency_max_ms: int | None = None
    raw_message_latency_negative_samples: int = 0
    queue_high_watermark: int = 0
    queue_full_events: int = 0
    fatal_error: bool = False
    collector_stopped: bool = False

    def connection_opened(self, connected_at_ms: int) -> None:
        if not self.fatal_error:
            self.status = "connected"
            self.last_error = None
        self.connections += 1
        self.subscription_confirmed = False
        self.current_session_id = f"{self.started_at_ms}-{self.connections}"
        self.connected_at_ms = connected_at_ms
        self.current_connection_last_message_at_ms = None
        self.current_connection_messages_by_topic.clear()
        self.current_connection_last_topic_message_at_ms.clear()

    def observe_websocket_message(
        self, message: dict[str, Any] | None, received_at_ns: int
    ) -> None:
        received_at_ms = received_at_ns // 1_000_000
        self.websocket_messages += 1
        self.last_message_at_ms = received_at_ms
        self.current_connection_last_message_at_ms = received_at_ms
        if message is None:
            return

        if message.get("op") == "pong" or message.get("ret_msg") == "pong":
            self.pongs_received += 1

        topic = message.get("topic")
        if isinstance(topic, str) and topic:
            self.messages_by_topic[topic] += 1
            self.last_topic_message_at_ms[topic] = received_at_ms
            self.current_connection_messages_by_topic[topic] += 1
            self.current_connection_last_topic_message_at_ms[topic] = received_at_ms

        exchange_ts_ms = self._raw_timestamp_ms(message.get("ts"))
        if exchange_ts_ms is not None:
            self._observe_raw_latency(received_at_ms - exchange_ts_ms)

    def observe_queue(self, queue_size: int, *, was_full: bool) -> None:
        self.queue_high_watermark = max(self.queue_high_watermark, queue_size)
        if was_full:
            self.queue_full_events += 1

    def mark_fatal_error(self, message: str) -> None:
        self.fatal_error = True
        self.status = "error"
        self.last_error = message

    def mark_stopped(self) -> None:
        self.collector_stopped = True
        if not self.fatal_error:
            self.status = "stopped"

    @staticmethod
    def _raw_timestamp_ms(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (OverflowError, ValueError):
                return None
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _observe_raw_latency(self, latency_ms: int) -> None:
        self.raw_message_latency_count += 1
        self.raw_message_latency_total_ms += latency_ms
        if latency_ms < 0:
            self.raw_message_latency_negative_samples += 1
        if self.raw_message_latency_min_ms is None:
            self.raw_message_latency_min_ms = latency_ms
        else:
            self.raw_message_latency_min_ms = min(self.raw_message_latency_min_ms, latency_ms)
        if self.raw_message_latency_max_ms is None:
            self.raw_message_latency_max_ms = latency_ms
        else:
            self.raw_message_latency_max_ms = max(self.raw_message_latency_max_ms, latency_ms)

    def snapshot(self, queue_size: int) -> dict[str, Any]:
        average_latency_ms = (
            self.raw_message_latency_total_ms / self.raw_message_latency_count
            if self.raw_message_latency_count
            else None
        )
        return {
            "started_at_ms": self.started_at_ms,
            "status": self.status,
            "connections": self.connections,
            "reconnects": self.reconnects,
            "websocket_messages": self.websocket_messages,
            "records_emitted": self.records_emitted,
            "last_message_at_ms": self.last_message_at_ms,
            "current_connection_last_message_at_ms": (
                self.current_connection_last_message_at_ms
            ),
            "last_error": self.last_error,
            "subscribed_topics": self.subscribed_topics,
            "expected_topics": list(self.expected_topics),
            "current_connection_missing_topics": sorted(
                set(self.expected_topics) - self.current_connection_messages_by_topic.keys()
            ),
            "subscription_confirmed": self.subscription_confirmed,
            "pings_sent": self.pings_sent,
            "pongs_received": self.pongs_received,
            "current_session_id": self.current_session_id,
            "connected_at_ms": self.connected_at_ms,
            "records_by_kind": dict(self.records_by_kind),
            "messages_by_topic": dict(self.messages_by_topic),
            "last_topic_message_at_ms": dict(self.last_topic_message_at_ms),
            "current_connection_messages_by_topic": dict(
                self.current_connection_messages_by_topic
            ),
            "current_connection_last_topic_message_at_ms": dict(
                self.current_connection_last_topic_message_at_ms
            ),
            "raw_message_latency_ms": {
                "count": self.raw_message_latency_count,
                "total": self.raw_message_latency_total_ms,
                "min": self.raw_message_latency_min_ms,
                "max": self.raw_message_latency_max_ms,
                "avg": average_latency_ms,
                "negative_samples": self.raw_message_latency_negative_samples,
            },
            "queue_size": queue_size,
            "queue_high_watermark": self.queue_high_watermark,
            "queue_full_events": self.queue_full_events,
            "fatal_error": self.fatal_error,
            "collector_stopped": self.collector_stopped,
            "updated_at_ms": int(time.time() * 1000),
        }


class BybitPublicCollector:
    def __init__(self, config: AppConfig, sink: RecordSink, stats: CollectorStats) -> None:
        self._config = config
        self._sink = sink
        self._stats = stats
        self._topics = build_public_topics(config)
        self._normalizer = MarketNormalizer(config.market)
        self._stats.subscribed_topics = len(self._topics)
        self._stats.expected_topics = tuple(self._topics)

    async def run(self, stop_event: asyncio.Event) -> None:
        delay = self._config.bybit.reconnect_min_seconds
        try:
            while not stop_event.is_set():
                try:
                    await self._run_connection(stop_event)
                    delay = self._config.bybit.reconnect_min_seconds
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # connection failures are expected and retried
                    if not self._stats.fatal_error:
                        self._stats.status = "reconnecting"
                        self._stats.last_error = f"{type(exc).__name__}: {exc}"
                    self._stats.reconnects += 1
                    LOGGER.warning("Bybit WebSocket disconnected: %s", self._stats.last_error)
                    jittered_delay = delay * random.uniform(0.8, 1.2)
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=jittered_delay)
                    delay = min(delay * 2, self._config.bybit.reconnect_max_seconds)
        finally:
            self._stats.mark_stopped()

    async def _run_connection(self, stop_event: asyncio.Event) -> None:
        LOGGER.info("Connecting to %s", self._config.bybit.public_ws_url)
        async with connect(
            self._config.bybit.public_ws_url,
            ping_interval=None,
            close_timeout=5,
            open_timeout=15,
            max_size=16 * 1024 * 1024,
            max_queue=2048,
        ) as websocket:
            # A fresh connection must rebuild snapshot/delta state from Bybit.
            self._normalizer.reset_connection_state()
            connected_at_ms = int(time.time() * 1000)
            self._stats.connection_opened(connected_at_ms)
            await websocket.send(
                json.dumps({"req_id": "market-data", "op": "subscribe", "args": self._topics})
            )
            heartbeat = asyncio.create_task(self._heartbeat(websocket, stop_event))
            last_topic_message_at = asyncio.get_running_loop().time()
            try:
                while not stop_event.is_set():
                    try:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    except TimeoutError:
                        stale_for = asyncio.get_running_loop().time() - last_topic_message_at
                        if _topic_data_is_stale(
                            last_topic_message_at,
                            asyncio.get_running_loop().time(),
                            self._config.bybit.stale_connection_seconds,
                        ):
                            raise TimeoutError(
                                "No topic data received for "
                                f"{stale_for:.1f} seconds"
                            ) from None
                        continue
                    received_at_ns = time.time_ns()
                    message = self._decode(raw_message)
                    self._stats.observe_websocket_message(message, received_at_ns)
                    frame_received_at = asyncio.get_running_loop().time()
                    if message is not None and isinstance(message.get("topic"), str):
                        last_topic_message_at = frame_received_at
                    elif _topic_data_is_stale(
                        last_topic_message_at,
                        frame_received_at,
                        self._config.bybit.stale_connection_seconds,
                    ):
                        raise TimeoutError(
                            "No topic data received for "
                            f"{frame_received_at - last_topic_message_at:.1f} seconds"
                        )
                    if message is None:
                        continue
                    if message.get("op") == "subscribe":
                        if message.get("success") is not True:
                            raise RuntimeError(f"Subscription rejected: {message}")
                        self._stats.subscription_confirmed = True
                        LOGGER.info("Subscribed to %d public topics", len(self._topics))
                        continue
                    if message.get("op") in {"pong", "ping"} or message.get("ret_msg") == "pong":
                        continue

                    session_id = self._stats.current_session_id
                    if session_id is None:  # pragma: no cover - connection_opened invariant
                        raise RuntimeError("Collector session is not initialized")
                    for record in self._normalizer.process(message, received_at_ns):
                        await self._sink(replace(record, session_id=session_id))
                        self._stats.records_emitted += 1
                        self._stats.records_by_kind[record.kind] += 1
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat(
        self, websocket: ClientConnection, stop_event: asyncio.Event
    ) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self._config.bybit.heartbeat_seconds
                )
            except TimeoutError:
                await websocket.send(json.dumps({"op": "ping"}))
                self._stats.pings_sent += 1

    @staticmethod
    def _decode(raw_message: str | bytes) -> dict[str, Any] | None:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            decoded = json.loads(raw_message)
        except (UnicodeDecodeError, json.JSONDecodeError):
            LOGGER.warning("Ignoring malformed WebSocket message")
            return None
        return decoded if isinstance(decoded, dict) else None
