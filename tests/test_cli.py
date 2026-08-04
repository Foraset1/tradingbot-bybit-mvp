from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tradingbot.cli import _put_record, main
from tradingbot.market.bybit_public import CollectorStats
from tradingbot.market.records import MarketRecord


def test_validate_config_command_reports_read_only_mode(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--config", str(config_path), "validate-config"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert summary["mode"] == "public-read-only"
    assert summary["topics"] == 36
    assert summary["risk"]["max_open_positions"] == 1
    assert summary["archive"]["raw_retention_days"] == 7


def test_audit_data_command_prints_report_and_fails_when_streams_are_missing(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TRADINGBOT_HEALTH_PATH", str(tmp_path / "health.json"))
    missing_root = tmp_path / "missing-raw"

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--config",
                str(config_path),
                "audit-data",
                "--root",
                str(missing_root),
            ]
        )

    report = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 1
    assert report["readiness"]["ok"] is False
    assert "missing_expected_streams" in report["readiness"]["reasons"]


@pytest.mark.asyncio
async def test_queue_sink_tracks_high_watermark_and_full_events() -> None:
    queue: asyncio.Queue[MarketRecord] = asyncio.Queue(maxsize=1)
    stats = CollectorStats()
    first = MarketRecord("ticker", "BTCUSDT", 1, 1, {})
    second = MarketRecord("ticker", "ETHUSDT", 1, 1, {})
    await queue.put(first)

    pending = asyncio.create_task(_put_record(queue, stats, second))
    await asyncio.sleep(0)

    assert stats.queue_high_watermark == 1
    assert stats.queue_full_events == 1
    assert queue.get_nowait() is first
    queue.task_done()
    await pending
    assert queue.get_nowait() is second
    queue.task_done()
