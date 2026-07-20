from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingbot.data.audit import audit_dataset

BASE_TS_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def record(
    kind: str,
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    exchange_ts_ms: int = BASE_TS_MS,
    received_offset_ms: int = 5,
    session_id: str | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    if kind == "trades" and isinstance(payload, list):
        for trade in payload:
            trade.setdefault("p", "100")
            trade.setdefault("v", "1")
            trade.setdefault("S", "Buy")
    if kind == "ticker" and isinstance(payload, dict):
        payload.setdefault("bid1Price", "100")
        payload.setdefault("ask1Price", "101")
    if kind.startswith("kline_") and isinstance(payload, dict):
        start = int(payload.get("start", exchange_ts_ms))
        payload.setdefault("end", start + 59_999)
        payload.setdefault("confirm", True)
        payload.setdefault("open", "100")
        payload.setdefault("high", "102")
        payload.setdefault("low", "99")
        payload.setdefault("close", "101")
        payload.setdefault("volume", "10")
        payload.setdefault("turnover", "1000")
    result: dict[str, Any] = {
        "kind": kind,
        "symbol": "BTCUSDT",
        "exchange_ts_ms": exchange_ts_ms,
        "received_at_ns": (exchange_ts_ms + received_offset_ms) * 1_000_000,
        "payload": payload,
        "source": "bybit",
    }
    if not legacy:
        result["schema_version"] = 1
    if session_id is not None:
        result["session_id"] = session_id
    return result


def write_records(
    root: Path,
    kind: str,
    records: list[dict[str, Any]],
    *,
    suffix: str = "part-test.jsonl",
) -> Path:
    timestamp = records[0]["exchange_ts_ms"]
    partition = datetime.fromtimestamp(timestamp / 1_000, tz=UTC)
    path = (
        root
        / kind
        / "BTCUSDT"
        / f"{partition.year:04d}"
        / f"{partition.month:02d}"
        / f"{partition.day:02d}"
        / suffix
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
    path.write_text(encoded, encoding="utf-8")
    return path


def populate_complete_dataset(root: Path) -> None:
    write_records(
        root,
        "orderbook",
        [
            record(
                "orderbook",
                {"bids": [["100", "2"]], "asks": [["101", "3"]], "sequence": 10},
                session_id="one",
            ),
            record(
                "orderbook",
                {"bids": [["101", "2"]], "asks": [["102", "3"]], "sequence": 11},
                exchange_ts_ms=BASE_TS_MS + 60_000,
                session_id="one",
            ),
        ],
    )
    write_records(
        root,
        "ticker",
        [
            record("ticker", {"symbol": "BTCUSDT", "lastPrice": "100"}),
            record(
                "ticker",
                {"symbol": "BTCUSDT", "lastPrice": "101"},
                exchange_ts_ms=BASE_TS_MS + 60_000,
            ),
        ],
    )
    write_records(
        root,
        "trades",
        [
            record(
                "trades",
                [{"i": "trade-1", "T": BASE_TS_MS, "s": "BTCUSDT"}],
            ),
            record(
                "trades",
                [{"i": "trade-2", "T": BASE_TS_MS + 60_000, "s": "BTCUSDT"}],
                exchange_ts_ms=BASE_TS_MS + 60_000,
            ),
        ],
        suffix="part-recovered.jsonl",
    )
    write_records(
        root,
        "kline_1",
        [
            record("kline_1", {"start": BASE_TS_MS, "interval": "1"}),
            record(
                "kline_1",
                {"start": BASE_TS_MS + 60_000, "interval": "1"},
                exchange_ts_ms=BASE_TS_MS + 60_000,
            ),
        ],
    )


def issue_codes(report_errors: object) -> set[str]:
    return {item.code for item in report_errors}  # type: ignore[union-attr]


def test_complete_dataset_is_ready_and_report_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    populate_complete_dataset(root)

    first = audit_dataset(root, ["BTCUSDT"], ["1"], 60)
    second = audit_dataset(root, ["BTCUSDT"], ["1"], 60)

    assert first.ok
    assert first.to_dict() == second.to_dict()
    assert len(first.files) == 4
    assert any(item.recovered for item in first.files)
    assert first.streams["trades/BTCUSDT"].events == 2
    assert first.streams["ticker/BTCUSDT"].duration_seconds == 60
    assert first.streams["ticker/BTCUSDT"].mean_latency_ms == 5
    assert first.observed_duration_seconds == 60
    assert first.projected_bytes_per_day is not None
    json.dumps(first.to_dict())


def test_hashes_and_fingerprint_cover_recovered_but_ignore_partial(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    populate_complete_dataset(root)
    initial = audit_dataset(root, ["BTCUSDT"], ["1"])
    recovered = next(item for item in initial.files if item.recovered)
    recovered_path = root / Path(recovered.path)
    assert recovered.sha256 == hashlib.sha256(recovered_path.read_bytes()).hexdigest()

    partial = root / "trades" / "BTCUSDT" / "active.jsonl.partial"
    partial.write_text('{"ignored":true}\n', encoding="utf-8")
    with_partial = audit_dataset(root, ["BTCUSDT"], ["1"])

    assert with_partial.input_fingerprint == initial.input_fingerprint
    assert with_partial.partial_files == ("trades/BTCUSDT/active.jsonl.partial",)
    assert with_partial.ok
    assert not audit_dataset(root, ["BTCUSDT"], ["1"], strict=True).ok


def test_legacy_schema_is_warning_and_strict_threshold(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    populate_complete_dataset(root)
    path = next((root / "ticker").rglob("*.jsonl"))
    ticker_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for item in ticker_records:
        item.pop("schema_version")
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in ticker_records), encoding="utf-8"
    )

    relaxed = audit_dataset(root, ["BTCUSDT"], ["1"])
    strict = audit_dataset(root, ["BTCUSDT"], ["1"], strict=True)

    warning = next(item for item in relaxed.warnings if item.code == "legacy_schema_version")
    assert warning.count == 2
    assert relaxed.ok
    assert not strict.ok
    assert strict.readiness_reasons == ("strict_warnings",)


def test_detects_session_scoped_sequence_regression_and_crossed_book(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_records(
        root,
        "orderbook",
        [
            record(
                "orderbook",
                {"bids": [["100", "1"]], "asks": [["101", "1"]], "sequence": 50},
                session_id="old",
            ),
            record(
                "orderbook",
                {"bids": [["100", "1"]], "asks": [["101", "1"]], "sequence": 2},
                exchange_ts_ms=BASE_TS_MS + 1_000,
                session_id="new",
            ),
            record(
                "orderbook",
                {"bids": [["102", "1"]], "asks": [["101", "1"]], "sequence": 1},
                exchange_ts_ms=BASE_TS_MS + 2_000,
                session_id="new",
            ),
        ],
    )

    report = audit_dataset(root, ["BTCUSDT"], ["1"])

    assert "orderbook_sequence_regression" in issue_codes(report.errors)
    assert "crossed_orderbook" in issue_codes(report.errors)
    assert report.streams["orderbook/BTCUSDT"].orderbook_sequence_regressions == 1


def test_detects_duplicate_trade_and_kline_gap_and_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_records(
        root,
        "trades",
        [
            record("trades", [{"i": "same", "T": BASE_TS_MS}]),
            record(
                "trades",
                [{"i": "same", "T": BASE_TS_MS + 1_000}],
                exchange_ts_ms=BASE_TS_MS + 1_000,
            ),
        ],
    )
    write_records(
        root,
        "kline_1",
        [
            record("kline_1", {"start": BASE_TS_MS, "interval": "1"}),
            record(
                "kline_1",
                {"start": BASE_TS_MS, "interval": "1"},
                exchange_ts_ms=BASE_TS_MS + 1_000,
            ),
            record(
                "kline_1",
                {"start": BASE_TS_MS + 180_000, "interval": "1"},
                exchange_ts_ms=BASE_TS_MS + 180_000,
            ),
        ],
    )

    report = audit_dataset(root, ["BTCUSDT"], ["1"])
    codes = issue_codes(report.errors)

    assert "duplicate_trade_id" in codes
    assert "duplicate_kline" in codes
    kline = report.streams["kline_1/BTCUSDT"]
    assert kline.kline_gaps == 1
    assert kline.missing_klines == 2
    assert kline.max_kline_gap_ms == 180_000
    assert "kline_gap" in issue_codes(report.warnings)


def test_invalid_json_path_and_missing_duration_affect_readiness(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    populate_complete_dataset(root)
    malformed = root / "ticker" / "BTCUSDT" / "2024" / "01" / "01" / "bad.jsonl"
    malformed.write_bytes(b'{"broken":\n')

    report = audit_dataset(root, ["BTCUSDT"], ["1", "5"], 120)

    assert not report.ok
    assert "invalid_json" in issue_codes(report.errors)
    assert report.missing_expected_streams == ("kline_5/BTCUSDT",)
    assert "kline_1/BTCUSDT" in report.short_streams
    assert report.readiness_reasons == (
        "validation_errors",
        "missing_expected_streams",
        "minimum_duration_not_met",
    )


def test_schema_v1_requires_bybit_source(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    populate_complete_dataset(root)
    path = next((root / "ticker").rglob("*.jsonl"))
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    items[0].pop("source")
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")

    report = audit_dataset(root, ["BTCUSDT"], ["1"])

    assert "invalid_source" in issue_codes(report.errors)
    assert not report.ok
