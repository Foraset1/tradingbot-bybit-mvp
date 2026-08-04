from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.data import bybit_history
from tradingbot.data.bybit_history import (
    HISTORY_PROFILE,
    HistoryImportError,
    import_bybit_history_day,
    validate_history_day,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.headers = {
            "Content-Length": str(len(payload)),
            "ETag": '"fixture-etag"',
            "Last-Modified": "Mon, 03 Aug 2026 01:00:00 GMT",
        }
        self._url = url

    def geturl(self) -> str:
        return self._url


def _archive_payload(partition_date: str, *, minutes: int = 1_440) -> bytes:
    parsed = datetime.fromisoformat(f"{partition_date}T00:00:00+00:00")
    start_seconds = int(parsed.timestamp())
    lines = [
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional,RPI"
    ]
    for minute in range(minutes):
        timestamp = f"{start_seconds + minute * 60}.125"
        side = "Buy" if minute % 2 == 0 else "Sell"
        price = f"{100 + minute / 1000:.3f}"
        lines.append(f"{timestamp},BTCUSDT,{side},0.5,{price},PlusTick,fixture-{minute},0,0.5,0,0")
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


def _install_archive(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    opened: list[str] = []

    def fake_open(url: str, timeout_seconds: int) -> _FakeResponse:
        assert timeout_seconds == 10
        opened.append(url)
        return _FakeResponse(payload, url)

    monkeypatch.setattr(bybit_history, "_open_url", fake_open)
    return opened


def test_history_progress_logging_formats_large_row_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=bybit_history.LOGGER.name):
        bybit_history._log_history_progress("2026-08-02", "ETHUSDT", 1_000_000)

    assert caplog.messages == [
        "History progress 2026-08-02 ETHUSDT: 1,000,000 source trades"
    ]


def test_imports_streaming_bars_and_reuses_verified_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partition_date = "2026-08-02"
    payload = _archive_payload(partition_date)
    opened = _install_archive(monkeypatch, payload)
    root = tmp_path / "history"

    first = import_bybit_history_day(
        history_root=root,
        partition_date=partition_date,
        symbols=("BTCUSDT",),
        request_timeout_seconds=10,
        download_attempts=1,
    )

    assert first.reused is False
    assert first.output_files == 2
    assert first.output_rows_by_kind == {
        "trade_bar_1m": 1_440,
        "trade_bar_1s": 1_440,
    }
    assert first.source_rows == 1_440
    assert len(opened) == 1
    manifest = json.loads(first.manifest_path.read_bytes())
    assert manifest["dataset_profile"] == HISTORY_PROFILE
    assert manifest["coverage"]["individual_trades_retained"] is False
    assert manifest["coverage"]["orderbook_available"] is False
    assert manifest["coverage"]["synthetic_bars"] == 0
    assert manifest["sources"][0]["compressed_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["sources"][0]["source_retained"] is False
    assert (root / "catalog.json").is_file()

    minute_file = next(
        first.dataset_path / item["path"]
        for item in manifest["files"]
        if item["kind"] == "trade_bar_1m"
    )
    table = pq.read_table(minute_file)
    first_row = table.slice(0, 1).to_pylist()[0]
    day_start_ms = int(datetime(2026, 8, 2, tzinfo=UTC).timestamp() * 1_000)
    assert first_row["start_ms"] == day_start_ms
    assert first_row["available_at_ns"] == (day_start_ms + 61_000) * 1_000_000
    assert first_row["buy_volume"] == 0.5
    assert first_row["sell_volume"] == 0.0

    def unexpected_open(url: str, timeout_seconds: int) -> _FakeResponse:
        raise AssertionError(f"verified reuse must not download {url}/{timeout_seconds}")

    monkeypatch.setattr(bybit_history, "_open_url", unexpected_open)
    second = import_bybit_history_day(
        history_root=root,
        partition_date=partition_date,
        symbols=("BTCUSDT",),
        request_timeout_seconds=10,
        download_attempts=1,
    )

    assert second.reused is True
    assert second.output_fingerprint == first.output_fingerprint


def test_failed_incomplete_day_resumes_from_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partition_date = "2026-08-01"
    root = tmp_path / "history"
    _install_archive(monkeypatch, _archive_payload(partition_date, minutes=1_439))

    with pytest.raises(HistoryImportError, match="missing 1 trade minutes"):
        import_bybit_history_day(
            history_root=root,
            partition_date=partition_date,
            symbols=("BTCUSDT",),
            request_timeout_seconds=10,
            download_attempts=1,
        )

    assert not (root / f"day={partition_date}").exists()
    assert not (root / ".history-import.lock").exists()
    assert len(list(root.glob(f".day={partition_date}.*.staging"))) == 1

    _install_archive(monkeypatch, _archive_payload(partition_date))
    result = import_bybit_history_day(
        history_root=root,
        partition_date=partition_date,
        symbols=("BTCUSDT",),
        request_timeout_seconds=10,
        download_attempts=1,
    )

    assert result.reused is False
    assert result.output_rows_by_kind["trade_bar_1m"] == 1_440
    assert not list(root.glob(f".day={partition_date}.*.staging"))


def test_validation_detects_parquet_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partition_date = "2026-07-31"
    _install_archive(monkeypatch, _archive_payload(partition_date))
    result = import_bybit_history_day(
        history_root=tmp_path / "history",
        partition_date=partition_date,
        symbols=("BTCUSDT",),
        request_timeout_seconds=10,
        download_attempts=1,
    )
    manifest = json.loads(result.manifest_path.read_bytes())
    target = result.dataset_path / manifest["files"][0]["path"]
    with target.open("ab") as output:
        output.write(b"corrupt")

    with pytest.raises(HistoryImportError, match="size changed"):
        validate_history_day(result.dataset_path)


def test_rejects_redirect_away_from_official_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_payload("2026-07-30")

    def fake_open(url: str, timeout_seconds: int) -> _FakeResponse:
        return _FakeResponse(payload, "https://example.invalid/archive.csv.gz")

    monkeypatch.setattr(bybit_history, "_open_url", fake_open)
    with pytest.raises(HistoryImportError, match="redirected away"):
        import_bybit_history_day(
            history_root=tmp_path / "history",
            partition_date="2026-07-30",
            symbols=("BTCUSDT",),
            request_timeout_seconds=10,
            download_attempts=1,
        )
