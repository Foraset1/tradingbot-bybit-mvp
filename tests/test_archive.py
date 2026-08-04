from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from tradingbot.data.archive import ArchiveError, archive_day, plan_raw_retention
from tradingbot.research.builder import _load_archive_catalog_source

PARTITION = date(2026, 7, 20)
BASE_TS_MS = int(datetime(2026, 7, 20, 0, 0, tzinfo=UTC).timestamp() * 1_000)


def _record(
    kind: str,
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    exchange_ts_ms: int = BASE_TS_MS,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "bybit",
        "session_id": "archive-test",
        "kind": kind,
        "symbol": "BTCUSDT",
        "exchange_ts_ms": exchange_ts_ms,
        "received_at_ns": exchange_ts_ms * 1_000_000 + 1_000_000,
        "payload": payload,
    }


def _write_stream(root: Path, kind: str, records: list[dict[str, Any]]) -> Path:
    path = (
        root
        / kind
        / "BTCUSDT"
        / "2026"
        / "07"
        / "20"
        / f"part-{kind}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _raw_fixture(root: Path) -> None:
    _write_stream(
        root,
        "orderbook",
        [
            _record(
                "orderbook",
                {
                    "bids": [["100", "2"]],
                    "asks": [["101", "3"]],
                    "matching_engine_ts_ms": BASE_TS_MS,
                    "update_id": 10,
                    "sequence": 20,
                },
            )
        ],
    )
    _write_stream(
        root,
        "ticker",
        [
            _record(
                "ticker",
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "100",
                    "indexPrice": "100.1",
                    "markPrice": "100.2",
                    "bid1Price": "99.9",
                    "bid1Size": "5",
                    "ask1Price": "100.1",
                    "ask1Size": "6",
                    "openInterest": "1000",
                    "openInterestValue": "100000",
                    "fundingRate": "0.0001",
                    "nextFundingTime": str(BASE_TS_MS + 28_800_000),
                    "fundingIntervalHour": "8",
                    "volume24h": "500",
                    "turnover24h": "50000",
                    "price24hPcnt": "0.01",
                    "highPrice24h": "105",
                    "lowPrice24h": "95",
                    "prevPrice1h": "99",
                    "prevPrice24h": "98",
                    "tickDirection": "PlusTick",
                },
            )
        ],
    )
    _write_stream(
        root,
        "trades",
        [
            _record(
                "trades",
                [
                    {
                        "T": BASE_TS_MS,
                        "i": "trade-1",
                        "S": "Buy",
                        "p": "100",
                        "v": "0.5",
                        "L": "PlusTick",
                        "seq": 30,
                        "BT": False,
                        "RPI": False,
                        "s": "BTCUSDT",
                    }
                ],
            )
        ],
    )
    _write_stream(
        root,
        "kline_1",
        [
            _record(
                "kline_1",
                {
                    "start": BASE_TS_MS,
                    "end": BASE_TS_MS + 59_999,
                    "interval": "1",
                    "open": "100",
                    "high": "102",
                    "low": "99",
                    "close": "101",
                    "volume": "10",
                    "turnover": "1005",
                    "confirm": True,
                },
            )
        ],
    )


def _build_archive(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    archive = tmp_path / "archive"
    _raw_fixture(raw)
    result = archive_day(
        raw,
        archive,
        ["BTCUSDT"],
        ["1"],
        partition_date=PARTITION,
        minimum_duration_seconds=0,
        minimum_free_bytes=0,
        scratch_dir=tmp_path / "scratch",
        today_utc=date(2026, 7, 21),
    )
    assert result.raw_files == 4
    assert result.canonical_files == 4
    assert not result.reused
    return raw, archive


def test_daily_archive_is_committed_cataloged_and_idempotent(tmp_path: Path) -> None:
    raw, archive = _build_archive(tmp_path)
    first_manifest = json.loads(
        (archive / "days" / "date=2026-07-20" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    catalog = json.loads((archive / "catalog.json").read_text(encoding="utf-8"))

    assert first_manifest["partition_date"] == "2026-07-20"
    assert catalog["entry_count"] == 1
    assert catalog["entries"][0]["day_fingerprint"] == first_manifest["day_fingerprint"]

    reused = archive_day(
        raw,
        archive,
        ["BTCUSDT"],
        ["1"],
        partition_date="2026-07-20",
        minimum_duration_seconds=0,
        minimum_free_bytes=0,
        scratch_dir=tmp_path / "scratch",
        today_utc=date(2026, 7, 21),
    )

    assert reused.reused
    assert reused.day_fingerprint == first_manifest["day_fingerprint"]


def test_archive_rejects_current_day_and_overlapping_roots(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)

    with pytest.raises(ArchiveError, match="fully elapsed"):
        archive_day(
            raw,
            tmp_path / "archive",
            ["BTCUSDT"],
            ["1"],
            partition_date=PARTITION,
            minimum_duration_seconds=0,
            today_utc=PARTITION,
        )
    with pytest.raises(ArchiveError, match="must not overlap"):
        archive_day(
            raw,
            raw / "archive",
            ["BTCUSDT"],
            ["1"],
            partition_date=PARTITION,
            minimum_duration_seconds=0,
            today_utc=date(2026, 7, 21),
        )


def test_retention_rejects_future_policy_date(tmp_path: Path) -> None:
    raw, archive = _build_archive(tmp_path)

    with pytest.raises(ArchiveError, match="cannot be in the future"):
        plan_raw_retention(
            raw,
            archive,
            retention_days=7,
            as_of_date="2999-01-01",
        )


def test_retention_plan_verifies_raw_hashes_and_never_deletes(tmp_path: Path) -> None:
    raw, archive = _build_archive(tmp_path)
    active = (
        raw
        / "ticker"
        / "BTCUSDT"
        / "2026"
        / "07"
        / "28"
        / "part-active.jsonl.partial"
    )
    active.parent.mkdir(parents=True)
    active.write_text('{"active":true}\n', encoding="utf-8")

    plan = plan_raw_retention(
        raw,
        archive,
        retention_days=7,
        as_of_date="2026-07-28",
    )

    assert plan.safe_to_apply
    assert len(plan.candidates) == 4
    assert not plan.blockers
    assert plan.partial_files == (
        "ticker/BTCUSDT/2026/07/28/part-active.jsonl.partial",
    )
    assert active.exists()
    assert all((raw / Path(item.path)).exists() for item in plan.candidates)
    assert plan.to_dict()["deletion_performed"] is False

    ticker = next((raw / "ticker").rglob("*.jsonl"))
    ticker.write_bytes(ticker.read_bytes() + b" ")
    changed = plan_raw_retention(
        raw,
        archive,
        retention_days=7,
        as_of_date="2026-07-28",
    )

    assert not changed.safe_to_apply
    assert {item.code for item in changed.blockers} == {"raw_size_changed"}
    assert len(changed.candidates) == 3
    assert ticker.exists()


def test_retention_blocks_all_files_when_canonical_archive_is_corrupt(
    tmp_path: Path,
) -> None:
    raw, archive = _build_archive(tmp_path)
    parquet = next((archive / "canonical").rglob("*.parquet"))
    content = parquet.read_bytes()
    parquet.write_bytes(content[:-1] + bytes([content[-1] ^ 0x01]))

    plan = plan_raw_retention(
        raw,
        archive,
        retention_days=7,
        as_of_date="2026-07-28",
    )

    assert not plan.candidates
    assert len(plan.blockers) == 4
    assert {item.code for item in plan.blockers} == {"archive_not_verified"}
    assert all(path.exists() for path in raw.rglob("*.jsonl"))


def test_research_source_reads_verified_daily_archive_catalog(tmp_path: Path) -> None:
    _, archive = _build_archive(tmp_path)

    source = _load_archive_catalog_source(archive / "catalog.json")

    assert source.dataset_id.startswith("archive-catalog-v1-")
    assert source.symbols == ("BTCUSDT",)
    assert len(source.files) == 4
    assert source.paths("trades", "BTCUSDT")[0].is_file()
