from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tradingbot.health import health_worker
from tradingbot.market.bybit_public import CollectorStats
from tradingbot.market.records import MarketRecord


@pytest.mark.asyncio
async def test_final_health_is_stopped_with_an_empty_queue(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    health_path = tmp_path / "health.json"
    queue: asyncio.Queue[MarketRecord] = asyncio.Queue()
    stop_event = asyncio.Event()
    stats = CollectorStats()
    stats.mark_stopped()
    stop_event.set()

    await health_worker(
        stats,
        queue,
        health_path,
        data_root,
        min_free_bytes=1,
        stop_event=stop_event,
        interval_seconds=0.01,
    )

    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    assert payload["queue_size"] == 0
    assert payload["disk_free_bytes"] > 0
    assert payload["disk_total_bytes"] > 0
