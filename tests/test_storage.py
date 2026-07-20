from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingbot.config import StorageConfig
from tradingbot.market.records import MarketRecord
from tradingbot.storage.jsonl import SegmentedJsonlWriter


def storage_config(root: Path) -> StorageConfig:
    return StorageConfig(
        root=root,
        segment_seconds=300,
        segment_max_bytes=1024 * 1024,
        flush_seconds=1.0,
        queue_maxsize=100,
        min_free_bytes=1,
        health_path=root.parent / "health.json",
    )


def test_writes_partitioned_jsonl_and_finalizes_atomically(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    writer = SegmentedJsonlWriter(storage_config(root))
    record = MarketRecord(
        kind="ticker",
        symbol="BTCUSDT",
        exchange_ts_ms=1_784_505_600_000,
        received_at_ns=1_784_505_600_100_000_000,
        payload={"lastPrice": "118000"},
    )

    writer.write(record)
    assert len(list(root.rglob("*.jsonl.partial"))) == 1
    writer.close()

    completed = list(root.rglob("*.jsonl"))
    assert len(completed) == 1
    assert completed[0].relative_to(root).parts[:5] == (
        "ticker",
        "BTCUSDT",
        "2026",
        "07",
        "20",
    )
    payload = json.loads(completed[0].read_text(encoding="utf-8"))
    assert payload == record.to_dict()
    assert list(root.rglob("*.jsonl.partial")) == []


def test_recovers_partial_segment_on_start(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    partial = root / "interrupted.jsonl.partial"
    partial.write_text('{"valid":true}\n', encoding="utf-8")

    SegmentedJsonlWriter(storage_config(root)).close()

    recovered = root / "interrupted-recovered.jsonl"
    assert recovered.read_text(encoding="utf-8") == '{"valid":true}\n'
    assert not partial.exists()


def test_recovery_discards_only_incomplete_trailing_record(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    partial = root / "interrupted.jsonl.partial"
    partial.write_bytes(b'{"complete":true}\n{"incomplete":')

    SegmentedJsonlWriter(storage_config(root)).close()

    recovered = root / "interrupted-recovered.jsonl"
    assert recovered.read_bytes() == b'{"complete":true}\n'


def test_rejects_unsafe_path_components(tmp_path: Path) -> None:
    writer = SegmentedJsonlWriter(storage_config(tmp_path / "raw"))
    record = MarketRecord(
        kind="ticker",
        symbol="../BTCUSDT",
        exchange_ts_ms=1_784_505_600_000,
        received_at_ns=1,
        payload={},
    )

    with pytest.raises(ValueError, match="Unsafe storage path component"):
        writer.write(record)
    writer.close()
