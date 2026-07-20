from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

from tradingbot.market.bybit_public import CollectorStats
from tradingbot.market.records import MarketRecord


def write_health(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def health_snapshot(
    stats: CollectorStats,
    queue_size: int,
    data_root: Path,
) -> dict[str, Any]:
    usage = shutil.disk_usage(data_root)
    payload = stats.snapshot(queue_size)
    payload["disk_free_bytes"] = usage.free
    payload["disk_total_bytes"] = usage.total
    payload["disk_used_bytes"] = usage.used
    return payload


async def health_worker(
    stats: CollectorStats,
    queue: asyncio.Queue[MarketRecord],
    path: Path,
    data_root: Path,
    min_free_bytes: int,
    stop_event: asyncio.Event,
    interval_seconds: float = 5.0,
    shutdown_timeout_seconds: float = 10.0,
) -> None:
    while not stop_event.is_set():
        try:
            payload = health_snapshot(stats, queue.qsize(), data_root)
        except OSError as exc:
            stats.mark_fatal_error(f"Disk usage check failed: {type(exc).__name__}: {exc}")
            stop_event.set()
            break
        disk_free_bytes = payload["disk_free_bytes"]
        if not isinstance(disk_free_bytes, int):  # pragma: no cover - shutil contract
            raise TypeError("disk_free_bytes must be an integer")
        if disk_free_bytes < min_free_bytes:
            stats.mark_fatal_error(
                "Insufficient free disk space: "
                f"{disk_free_bytes} bytes available, {min_free_bytes} required"
            )
            stop_event.set()
            payload = health_snapshot(stats, queue.qsize(), data_root)
        write_health(path, payload)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)

    try:
        async with asyncio.timeout(shutdown_timeout_seconds):
            while not stats.collector_stopped:
                await asyncio.sleep(0.01)
            await queue.join()
    except TimeoutError:
        stats.mark_fatal_error(
            "Timed out waiting for the collector and storage queue to stop cleanly"
        )
    write_health(path, health_snapshot(stats, queue.qsize(), data_root))
