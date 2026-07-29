from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tradingbot.cli import main
from tradingbot.data.audit import audit_dataset
from tradingbot.data.canonical import DatasetBuildError, build_canonical_dataset

BASE_TS_MS = 1_784_524_400_000
BASE_DATE = datetime.fromtimestamp(BASE_TS_MS / 1_000, tz=UTC).strftime("%Y/%m/%d")


def raw_record(
    kind: str,
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    exchange_ts_ms: int = BASE_TS_MS,
    received_at_ns: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "bybit",
        "session_id": "session-1",
        "kind": kind,
        "symbol": "BTCUSDT",
        "exchange_ts_ms": exchange_ts_ms,
        "received_at_ns": (
            exchange_ts_ms * 1_000_000 + 1_000_000
            if received_at_ns is None
            else received_at_ns
        ),
        "payload": payload,
    }


def write_stream(
    root: Path,
    kind: str,
    records: list[dict[str, Any]],
) -> Path:
    path = root / kind / "BTCUSDT" / BASE_DATE / f"part-{kind}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(rendered, encoding="utf-8", newline="\n")
    return path


def audited_input(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "raw"
    write_stream(
        root,
        "orderbook",
        [
            raw_record(
                "orderbook",
                {
                    "bids": [["100", "2"], ["99", "3"]],
                    "asks": [["101", "4"], ["102", "5"]],
                    "matching_engine_ts_ms": BASE_TS_MS,
                    "update_id": 10,
                    "sequence": 20,
                },
            )
        ],
    )
    write_stream(
        root,
        "ticker",
        [
            raw_record(
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
    write_stream(
        root,
        "trades",
        [
            raw_record(
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
                    },
                    {
                        "T": BASE_TS_MS + 1,
                        "i": "trade-2",
                        "S": "Sell",
                        "p": "99.9",
                        "v": "0.25",
                        "L": "MinusTick",
                        "seq": 31,
                        "BT": False,
                        "RPI": False,
                        "s": "BTCUSDT",
                    },
                ],
            )
        ],
    )
    start_ms = BASE_TS_MS - BASE_TS_MS % 60_000
    first_payload = {
        "start": start_ms,
        "end": start_ms + 59_999,
        "interval": "1",
        "open": "100",
        "high": "102",
        "low": "99",
        "close": "101",
        "volume": "10",
        "turnover": "1005",
        "confirm": True,
    }
    revised_payload = {
        **first_payload,
        "high": "103",
        "close": "102",
        "volume": "12",
        "turnover": "1210",
    }
    write_stream(
        root,
        "kline_1",
        [
            raw_record("kline_1", first_payload),
            raw_record(
                "kline_1",
                revised_payload,
                exchange_ts_ms=BASE_TS_MS + 5_000,
                received_at_ns=(BASE_TS_MS + 5_000) * 1_000_000 + 1_000_000,
            ),
        ],
    )

    report = audit_dataset(
        root=root,
        symbols=["BTCUSDT"],
        kline_intervals=["1"],
        strict=True,
        scratch_dir=tmp_path,
    ).to_dict()
    assert report["readiness"]["ok"] is True  # type: ignore[index]
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return root, audit_path, report


def parquet_table(dataset_path: Path, kind: str) -> Any:
    paths = sorted(dataset_path.glob(f"market/kind={kind}/**/*.parquet"))
    assert paths
    return pq.read_table(paths)


def test_builds_typed_canonical_dataset_and_is_idempotent(tmp_path: Path) -> None:
    root, audit_path, audit = audited_input(tmp_path)
    output_root = tmp_path / "datasets"

    first = build_canonical_dataset(
        audit_report=audit_path,
        source_root=root,
        output_root=output_root,
    )

    assert not first.reused
    assert first.input_fingerprint == audit["input_fingerprint"]
    assert first.source_files == 4
    assert first.source_records == 5
    assert first.output_rows_by_kind == {
        "kline_1": 1,
        "orderbook": 1,
        "ticker": 1,
        "trades": 2,
    }
    assert first.output_rows == 5
    assert first.output_files == 4

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonicalization"] == {
        "canonical_kline_rows": 1,
        "collapsed_kline_rows": 1,
        "equal_timestamp_different_payload": "error",
        "kline_key": ["symbol", "interval", "start_ms"],
        "reported_exact_redeliveries": 0,
        "reported_kline_revisions": 1,
        "selection": "maximum received_at_ns",
        "source_kline_rows": 2,
    }
    assert manifest["source"]["input_fingerprint"] == audit["input_fingerprint"]
    assert (first.dataset_path / "source-audit.json").read_bytes() == audit_path.read_bytes()

    kline = parquet_table(first.dataset_path, "kline_1").to_pylist()
    assert len(kline) == 1
    assert kline[0]["close"] == 102.0
    assert kline[0]["high"] == 103.0
    assert kline[0]["received_at_ns"] == (BASE_TS_MS + 5_000) * 1_000_000 + 1_000_000

    trades = parquet_table(first.dataset_path, "trades").to_pylist()
    assert [row["trade_id"] for row in trades] == ["trade-1", "trade-2"]
    assert [row["side"] for row in trades] == ["Buy", "Sell"]

    orderbook = parquet_table(first.dataset_path, "orderbook").to_pylist()
    assert orderbook[0]["bid_prices"] == [100.0, 99.0]
    assert orderbook[0]["ask_sizes"] == [4.0, 5.0]

    second = build_canonical_dataset(
        audit_report=audit_path,
        source_root=root,
        output_root=output_root,
    )
    assert second.reused
    assert second.dataset_path == first.dataset_path
    assert second.output_fingerprint == first.output_fingerprint

    independent = build_canonical_dataset(
        audit_report=audit_path,
        source_root=root,
        output_root=tmp_path / "independent-datasets",
    )
    assert independent.output_fingerprint == first.output_fingerprint
    assert independent.manifest_path.read_bytes() == first.manifest_path.read_bytes()


def test_rejects_source_changed_after_audit_and_cleans_staging(tmp_path: Path) -> None:
    root, audit_path, _ = audited_input(tmp_path)
    ticker = next(root.glob("ticker/**/*.jsonl"))
    content = ticker.read_text(encoding="utf-8")
    ticker.write_text(
        content.replace('"lastPrice":"100"', '"lastPrice":"109"'),
        encoding="utf-8",
        newline="\n",
    )
    output_root = tmp_path / "datasets"

    with pytest.raises(DatasetBuildError, match="SHA-256 changed"):
        build_canonical_dataset(
            audit_report=audit_path,
            source_root=root,
            output_root=output_root,
        )

    assert list(output_root.iterdir()) == []


def test_rejects_audit_that_is_not_strictly_ready(tmp_path: Path) -> None:
    root, audit_path, report = audited_input(tmp_path)
    report["readiness"]["ok"] = False
    report["readiness"]["reasons"] = ["validation_errors"]
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(DatasetBuildError, match="successful strict audit"):
        build_canonical_dataset(
            audit_report=audit_path,
            source_root=root,
            output_root=tmp_path / "datasets",
        )


def test_build_dataset_cli_prints_reproducible_summary(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, audit_path, report = audited_input(tmp_path)
    output_root = tmp_path / "datasets"
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "1")

    main(
        [
            "--config",
            str(config_path),
            "build-dataset",
            "--audit-report",
            str(audit_path),
            "--root",
            str(root),
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_schema_version"] == 1
    assert summary["input_fingerprint"] == report["input_fingerprint"]
    assert summary["output_rows"] == 5
    assert summary["reused"] is False
