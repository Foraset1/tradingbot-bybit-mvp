from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tradingbot.cli import _put_record, main
from tradingbot.data import archive as archive_module
from tradingbot.data import bybit_history
from tradingbot.data.archive import ArchiveError, RetentionCandidate
from tradingbot.data.bybit_history import HistoryRangeResult
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
    assert summary["history"]["profile"] == "price_futures_v1"
    assert summary["history"]["maximum_consecutive_trade_free_minutes"] == 5
    assert summary["history"]["retains_individual_trades"] is False


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


def test_import_history_command_writes_range_result(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history_root = tmp_path / "history"
    report_path = tmp_path / "reports" / "history.json"

    def fake_import(**kwargs: object) -> HistoryRangeResult:
        assert kwargs["history_root"] == history_root
        assert kwargs["start_date"] == "2026-07-01"
        assert kwargs["end_date"] == "2026-07-02"
        assert kwargs["symbols"] == ("BTCUSDT",)
        return HistoryRangeResult(
            start_date="2026-07-01",
            end_date="2026-07-02",
            history_root=history_root,
            catalog_path=history_root / "catalog.json",
            catalog_fingerprint="a" * 64,
            days=2,
            imported_days=2,
            reused_days=0,
            source_rows=123,
            output_rows=45,
            output_bytes=678,
        )

    monkeypatch.setattr(bybit_history, "import_bybit_history_range", fake_import)
    main(
        [
            "--config",
            str(config_path),
            "import-history",
            "--from-date",
            "2026-07-01",
            "--to-date",
            "2026-07-02",
            "--symbol",
            "BTCUSDT",
            "--history-root",
            str(history_root),
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["days"] == 2
    assert json.loads(report_path.read_bytes()) == payload


def test_archive_day_failure_is_machine_readable_on_stdout_and_disk(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "reports" / "archive-failure.json"

    def reject_archive(*args: object, **kwargs: object) -> None:
        raise ArchiveError(
            "partition failed archive policy",
            details={
                "archive_acceptance": {
                    "ok": False,
                    "reasons": ["unsupported_warning_codes"],
                }
            },
        )

    monkeypatch.setattr(archive_module, "archive_day", reject_archive)
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--config",
                str(config_path),
                "archive-day",
                "--date",
                "2026-07-20",
                "--output",
                str(report_path),
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 1
    assert payload["ok"] is False
    assert payload["partition_date"] == "2026-07-20"
    assert payload["archive_acceptance"]["reasons"] == [
        "unsupported_warning_codes"
    ]
    assert json.loads(report_path.read_bytes()) == payload


def test_apply_retention_command_prints_compact_summary_and_writes_receipt(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "raw"
    archive_root = tmp_path / "archive"
    raw_root.mkdir()
    archive_root.mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "receipt.json"
    fingerprint = "a" * 64

    def fake_apply(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["raw_root"] == raw_root
        assert kwargs["archive_root"] == archive_root
        assert kwargs["plan_path"] == plan_path
        assert kwargs["confirmed_plan_fingerprint"] == fingerprint
        assert kwargs["receipt_path"] == output.resolve()
        payload: dict[str, object] = {
            "retention_apply_schema_version": 1,
            "ok": True,
            "mode": "apply",
            "status": "complete",
            "deletion_performed": True,
            "plan_fingerprint": fingerprint,
            "delete_before_date": "2026-07-21",
            "planned_file_count": 1,
            "planned_bytes": 10,
            "deleted_files": [
                RetentionCandidate(
                    path="ticker/BTCUSDT/2026/07/20/part.jsonl",
                    partition_date="2026-07-20",
                    bytes=10,
                    sha256="b" * 64,
                ).to_dict()
            ],
            "deleted_file_count": 1,
            "deleted_bytes": 10,
            "receipt_fingerprint": "c" * 64,
        }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(archive_module, "apply_raw_retention", fake_apply)
    main(
        [
            "--config",
            str(config_path),
            "apply-retention",
            "--plan",
            str(plan_path),
            "--confirm-plan-fingerprint",
            fingerprint,
            "--root",
            str(raw_root),
            "--archive-root",
            str(archive_root),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    receipt = json.loads(output.read_bytes())
    assert summary["ok"] is True
    assert summary["deleted_file_count"] == 1
    assert "deleted_files" not in summary
    assert len(receipt["deleted_files"]) == 1


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
