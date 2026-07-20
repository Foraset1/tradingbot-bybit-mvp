"""Streaming integrity audit for normalized raw market-data JSONL files.

The auditor deliberately has no dependency on the collector configuration.  A caller
supplies the expected symbols and kline intervals, which makes an audit reproducible
for an immutable dataset snapshot.  Completed and recovered ``*.jsonl`` files are
read one line at a time.  Active ``*.jsonl.partial`` files are reported, but never
read or included in the input fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

RAW_RECORD_SCHEMA_VERSION: Final = 1
AUDIT_REPORT_SCHEMA_VERSION: Final = 1

_BASE_KINDS: Final = ("orderbook", "ticker", "trades")
_FIXED_KLINE_INTERVALS_MS: Final = {
    "D": 24 * 60 * 60 * 1_000,
    "W": 7 * 24 * 60 * 60 * 1_000,
}


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """A compact, possibly aggregated validation finding."""

    code: str
    message: str
    count: int
    path: str | None = None
    line: int | None = None
    stream: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "path": self.path,
            "line": self.line,
            "stream": self.stream,
        }


@dataclass(frozen=True, slots=True)
class AuditFile:
    """Content identity and parsing counts for one completed input segment."""

    path: str
    bytes: int
    sha256: str
    lines: int
    records: int
    recovered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "lines": self.lines,
            "records": self.records,
            "recovered": self.recovered,
        }


@dataclass(frozen=True, slots=True)
class StreamAudit:
    """Aggregated statistics for one ``kind/symbol`` stream."""

    kind: str
    symbol: str
    records: int
    events: int
    files: int
    bytes: int
    first_exchange_ts_ms: int | None
    last_exchange_ts_ms: int | None
    duration_seconds: float
    min_latency_ms: float | None
    max_latency_ms: float | None
    mean_latency_ms: float | None
    max_gap_ms: int
    duplicate_trade_ids: int
    orderbook_sequence_regressions: int
    duplicate_klines: int
    kline_gaps: int
    missing_klines: int
    max_kline_gap_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "symbol": self.symbol,
            "records": self.records,
            "events": self.events,
            "files": self.files,
            "bytes": self.bytes,
            "first_exchange_ts_ms": self.first_exchange_ts_ms,
            "last_exchange_ts_ms": self.last_exchange_ts_ms,
            "duration_seconds": self.duration_seconds,
            "latency_ms": {
                "min": self.min_latency_ms,
                "max": self.max_latency_ms,
                "mean": self.mean_latency_ms,
            },
            "max_gap_ms": self.max_gap_ms,
            "duplicate_trade_ids": self.duplicate_trade_ids,
            "orderbook_sequence_regressions": self.orderbook_sequence_regressions,
            "duplicate_klines": self.duplicate_klines,
            "kline_gaps": self.kline_gaps,
            "missing_klines": self.missing_klines,
            "max_kline_gap_ms": self.max_kline_gap_ms,
        }


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    """Deterministic, JSON-serializable result of :func:`audit_dataset`."""

    dataset_root: str
    expected_symbols: tuple[str, ...]
    kline_intervals: tuple[str, ...]
    input_fingerprint: str
    files: tuple[AuditFile, ...]
    partial_files: tuple[str, ...]
    streams: dict[str, StreamAudit]
    errors: tuple[AuditIssue, ...]
    warnings: tuple[AuditIssue, ...]
    missing_expected_streams: tuple[str, ...]
    short_streams: tuple[str, ...]
    minimum_duration_seconds: float
    strict: bool
    total_bytes: int
    total_lines: int
    total_records: int
    observed_duration_seconds: float
    projected_bytes_per_day: int | None
    ready: bool
    readiness_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether the dataset meets the requested readiness policy."""

        return self.ready

    def to_dict(self) -> dict[str, object]:
        """Return a stable structure accepted directly by ``json.dumps``."""

        return {
            "audit_report_schema_version": AUDIT_REPORT_SCHEMA_VERSION,
            "dataset_root": self.dataset_root,
            "input_fingerprint": self.input_fingerprint,
            "files": [item.to_dict() for item in self.files],
            "file_count": len(self.files),
            "partial_files": list(self.partial_files),
            "partial_file_count": len(self.partial_files),
            "streams": {
                name: self.streams[name].to_dict() for name in sorted(self.streams)
            },
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
            "missing_expected_streams": list(self.missing_expected_streams),
            "short_streams": list(self.short_streams),
            "totals": {
                "bytes": self.total_bytes,
                "lines": self.total_lines,
                "records": self.total_records,
                "observed_duration_seconds": self.observed_duration_seconds,
                "projected_bytes_per_day": self.projected_bytes_per_day,
            },
            "readiness": {
                "ok": self.ok,
                "strict": self.strict,
                "minimum_duration_seconds": self.minimum_duration_seconds,
                "reasons": list(self.readiness_reasons),
            },
            "policy": {
                "expected_symbols": list(self.expected_symbols),
                "kline_intervals": list(self.kline_intervals),
            },
        }


@dataclass(slots=True)
class _MutableIssue:
    code: str
    message: str
    count: int
    path: str | None
    line: int | None
    stream: str | None


class _Issues:
    def __init__(self) -> None:
        self._errors: dict[tuple[str, str | None], _MutableIssue] = {}
        self._warnings: dict[tuple[str, str | None], _MutableIssue] = {}

    def error(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        stream: str | None = None,
        count: int = 1,
    ) -> None:
        self._add(self._errors, code, message, path, line, stream, count)

    def warning(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        stream: str | None = None,
        count: int = 1,
    ) -> None:
        self._add(self._warnings, code, message, path, line, stream, count)

    @staticmethod
    def _add(
        target: dict[tuple[str, str | None], _MutableIssue],
        code: str,
        message: str,
        path: str | None,
        line: int | None,
        stream: str | None,
        count: int,
    ) -> None:
        key = (code, stream)
        issue = target.get(key)
        if issue is None:
            target[key] = _MutableIssue(code, message, count, path, line, stream)
        else:
            issue.count += count

    @staticmethod
    def _freeze(source: dict[tuple[str, str | None], _MutableIssue]) -> tuple[AuditIssue, ...]:
        return tuple(
            AuditIssue(
                code=item.code,
                message=item.message,
                count=item.count,
                path=item.path,
                line=item.line,
                stream=item.stream,
            )
            for _, item in sorted(
                source.items(), key=lambda pair: (pair[0][0], pair[0][1] or "")
            )
        )

    def errors(self) -> tuple[AuditIssue, ...]:
        return self._freeze(self._errors)

    def warnings(self) -> tuple[AuditIssue, ...]:
        return self._freeze(self._warnings)


@dataclass(slots=True)
class _MutableStream:
    kind: str
    symbol: str
    records: int = 0
    events: int = 0
    files: int = 0
    bytes: int = 0
    first_exchange_ts_ms: int | None = None
    last_exchange_ts_ms: int | None = None
    last_causal_received_at_ns: int | None = None
    last_causal_exchange_ts_ms: int | None = None
    latency_count: int = 0
    latency_sum_ns: int = 0
    min_latency_ns: int | None = None
    max_latency_ns: int | None = None
    max_gap_ms: int = 0
    duplicate_trade_ids: int = 0
    orderbook_sequence_regressions: int = 0
    duplicate_klines: int = 0
    kline_gaps: int = 0
    missing_klines: int = 0
    max_kline_gap_ms: int = 0

    def observe_time(self, exchange_ts_ms: int, received_at_ns: int) -> bool:
        if self.first_exchange_ts_ms is None or exchange_ts_ms < self.first_exchange_ts_ms:
            self.first_exchange_ts_ms = exchange_ts_ms
        if self.last_exchange_ts_ms is None or exchange_ts_ms > self.last_exchange_ts_ms:
            self.last_exchange_ts_ms = exchange_ts_ms

        latency_ns = received_at_ns - exchange_ts_ms * 1_000_000
        self.latency_count += 1
        self.latency_sum_ns += latency_ns
        if self.min_latency_ns is None or latency_ns < self.min_latency_ns:
            self.min_latency_ns = latency_ns
        if self.max_latency_ns is None or latency_ns > self.max_latency_ns:
            self.max_latency_ns = latency_ns

        causal = (
            self.last_causal_received_at_ns is None
            or received_at_ns >= self.last_causal_received_at_ns
        )
        if causal:
            if self.last_causal_exchange_ts_ms is not None:
                gap = exchange_ts_ms - self.last_causal_exchange_ts_ms
                if gap > self.max_gap_ms:
                    self.max_gap_ms = gap
            self.last_causal_received_at_ns = received_at_ns
            self.last_causal_exchange_ts_ms = exchange_ts_ms
        return causal

    @staticmethod
    def _millis(nanoseconds: int | None) -> float | None:
        if nanoseconds is None:
            return None
        return round(nanoseconds / 1_000_000, 6)

    def freeze(self) -> StreamAudit:
        first = self.first_exchange_ts_ms
        last = self.last_exchange_ts_ms
        duration = 0.0 if first is None or last is None else (last - first) / 1_000
        mean_latency = (
            None
            if self.latency_count == 0
            else round(self.latency_sum_ns / self.latency_count / 1_000_000, 6)
        )
        return StreamAudit(
            kind=self.kind,
            symbol=self.symbol,
            records=self.records,
            events=self.events,
            files=self.files,
            bytes=self.bytes,
            first_exchange_ts_ms=first,
            last_exchange_ts_ms=last,
            duration_seconds=duration,
            min_latency_ms=self._millis(self.min_latency_ns),
            max_latency_ms=self._millis(self.max_latency_ns),
            mean_latency_ms=mean_latency,
            max_gap_ms=self.max_gap_ms,
            duplicate_trade_ids=self.duplicate_trade_ids,
            orderbook_sequence_regressions=self.orderbook_sequence_regressions,
            duplicate_klines=self.duplicate_klines,
            kline_gaps=self.kline_gaps,
            missing_klines=self.missing_klines,
            max_kline_gap_ms=self.max_kline_gap_ms,
        )


@dataclass(slots=True)
class _OrderbookSequence:
    received_at_ns: int
    sequence: int


@dataclass(slots=True)
class _KlinePosition:
    received_at_ns: int
    start_ms: int


class _AuditState:
    def __init__(
        self,
        root: Path,
        symbols: frozenset[str],
        intervals: frozenset[str],
        database: sqlite3.Connection,
    ) -> None:
        self.root = root
        self.symbols = symbols
        self.intervals = intervals
        self.known_kinds = frozenset((*_BASE_KINDS, *(f"kline_{x}" for x in intervals)))
        self.issues = _Issues()
        self.streams: dict[str, _MutableStream] = {}
        self.database = database
        self.orderbook_sequences: dict[tuple[str, str], _OrderbookSequence] = {}
        self.kline_positions: dict[tuple[str, str], _KlinePosition] = {}

    def stream(self, kind: str, symbol: str) -> _MutableStream:
        name = _stream_name(kind, symbol)
        result = self.streams.get(name)
        if result is None:
            result = _MutableStream(kind=kind, symbol=symbol)
            self.streams[name] = result
        return result

    def process_record(
        self,
        raw: object,
        *,
        relative_path: str,
        path_parts: tuple[str, ...],
        line_number: int,
        line_bytes: int,
        streams_in_file: set[str],
    ) -> bool:
        if not isinstance(raw, dict):
            self.issues.error(
                "invalid_wrapper",
                "JSONL record must be an object",
                path=relative_path,
                line=line_number,
            )
            return False

        kind_value = raw.get("kind")
        symbol_value = raw.get("symbol")
        if not isinstance(kind_value, str) or not isinstance(symbol_value, str):
            self.issues.error(
                "invalid_wrapper_types",
                "kind and symbol must be strings",
                path=relative_path,
                line=line_number,
            )
            return False
        kind = kind_value
        symbol = symbol_value
        name = _stream_name(kind, symbol)
        stream = self.stream(kind, symbol)
        stream.records += 1
        stream.bytes += line_bytes
        streams_in_file.add(name)

        self._validate_schema_version(raw, relative_path, line_number, name)
        self._validate_wrapper_metadata(raw, relative_path, line_number, name)
        if kind not in self.known_kinds:
            self.issues.error(
                "unknown_kind",
                f"kind {kind!r} is not expected",
                path=relative_path,
                line=line_number,
                stream=name,
            )
        if symbol not in self.symbols:
            self.issues.error(
                "unknown_symbol",
                f"symbol {symbol!r} is not expected",
                path=relative_path,
                line=line_number,
                stream=name,
            )

        exchange_ts = raw.get("exchange_ts_ms")
        received_at = raw.get("received_at_ns")
        valid_timestamps = _positive_int(exchange_ts) and _positive_int(received_at)
        causal = False
        if not valid_timestamps:
            self.issues.error(
                "invalid_timestamps",
                "exchange_ts_ms and received_at_ns must be positive integers",
                path=relative_path,
                line=line_number,
                stream=name,
            )
        else:
            assert isinstance(exchange_ts, int)
            assert isinstance(received_at, int)
            causal = stream.observe_time(exchange_ts, received_at)
            if not causal:
                self.issues.warning(
                    "causal_order_regression",
                    "records are not ordered by received_at_ns",
                    path=relative_path,
                    line=line_number,
                    stream=name,
                )
            if received_at < exchange_ts * 1_000_000:
                self.issues.warning(
                    "negative_latency",
                    "received_at_ns precedes exchange_ts_ms",
                    path=relative_path,
                    line=line_number,
                    stream=name,
                )
            self._validate_partition_date(
                exchange_ts, relative_path, path_parts, line_number, name
            )

        self._validate_path(kind, symbol, relative_path, path_parts, line_number, name)
        payload = raw.get("payload")
        if kind == "orderbook":
            events = self._validate_orderbook(
                payload, raw, relative_path, line_number, name, symbol, received_at, causal
            )
        elif kind == "trades":
            events = self._validate_trades(
                payload, relative_path, line_number, name, symbol
            )
        elif kind == "ticker":
            events = self._validate_ticker(payload, relative_path, line_number, name, symbol)
        elif kind.startswith("kline_"):
            events = self._validate_kline(
                payload, relative_path, line_number, name, symbol, kind, received_at, causal
            )
        else:
            events = 0
            if not isinstance(payload, (dict, list)):
                self.issues.error(
                    "invalid_payload_type",
                    "payload must be an object or array",
                    path=relative_path,
                    line=line_number,
                    stream=name,
                )
        stream.events += events
        return True

    def _validate_schema_version(
        self, raw: dict[object, object], path: str, line: int, stream: str
    ) -> None:
        if "schema_version" not in raw:
            self.issues.warning(
                "legacy_schema_version",
                "record has no schema_version and is treated as legacy v0",
                path=path,
                line=line,
                stream=stream,
            )
            return
        version = raw["schema_version"]
        if not _plain_int(version) or version != RAW_RECORD_SCHEMA_VERSION:
            self.issues.error(
                "unsupported_schema_version",
                f"schema_version must be integer {RAW_RECORD_SCHEMA_VERSION}",
                path=path,
                line=line,
                stream=stream,
            )

    def _validate_wrapper_metadata(
        self, raw: dict[object, object], path: str, line: int, stream: str
    ) -> None:
        source = raw.get("source")
        if raw.get("schema_version") == RAW_RECORD_SCHEMA_VERSION and source != "bybit":
            self.issues.error(
                "invalid_source",
                "schema v1 records must have source='bybit'",
                path=path,
                line=line,
                stream=stream,
            )
        elif source is not None and (not isinstance(source, str) or not source):
            self.issues.error(
                "invalid_source",
                "source must be a non-empty string when present",
                path=path,
                line=line,
                stream=stream,
            )
        session_id = raw.get("session_id")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            self.issues.error(
                "invalid_session_id",
                "session_id must be a non-empty string when present",
                path=path,
                line=line,
                stream=stream,
            )

    def _validate_path(
        self,
        kind: str,
        symbol: str,
        path: str,
        parts: tuple[str, ...],
        line: int,
        stream: str,
    ) -> None:
        if len(parts) != 6:
            self.issues.error(
                "invalid_partition_path",
                "path must be kind/symbol/YYYY/MM/DD/file.jsonl",
                path=path,
                line=line,
                stream=stream,
            )
            return
        if parts[0] != kind or parts[1] != symbol:
            self.issues.error(
                "path_stream_mismatch",
                "path kind/symbol does not match the record wrapper",
                path=path,
                line=line,
                stream=stream,
            )

    def _validate_partition_date(
        self,
        exchange_ts_ms: int,
        path: str,
        parts: tuple[str, ...],
        line: int,
        stream: str,
    ) -> None:
        if len(parts) != 6:
            return
        expected = datetime.fromtimestamp(exchange_ts_ms / 1_000, tz=UTC)
        actual = parts[2:5]
        if actual != (f"{expected.year:04d}", f"{expected.month:02d}", f"{expected.day:02d}"):
            self.issues.error(
                "partition_date_mismatch",
                "path partition date does not match exchange_ts_ms",
                path=path,
                line=line,
                stream=stream,
            )

    def _validate_orderbook(
        self,
        payload: object,
        raw: dict[object, object],
        path: str,
        line: int,
        stream_name: str,
        symbol: str,
        received_at: object,
        causal: bool,
    ) -> int:
        if not isinstance(payload, dict):
            self.issues.error(
                "invalid_orderbook_payload",
                "orderbook payload must be an object",
                path=path,
                line=line,
                stream=stream_name,
            )
            return 0
        bids = self._validate_levels(payload.get("bids"), "bids", path, line, stream_name)
        asks = self._validate_levels(payload.get("asks"), "asks", path, line, stream_name)
        if bids is not None and asks is not None and bids and asks and max(bids) >= min(asks):
            self.issues.error(
                "crossed_orderbook",
                "best bid must be strictly below best ask",
                path=path,
                line=line,
                stream=stream_name,
            )

        sequence = payload.get("sequence")
        if not _nonnegative_int(sequence):
            self.issues.error(
                "invalid_orderbook_sequence",
                "orderbook sequence must be a non-negative integer",
                path=path,
                line=line,
                stream=stream_name,
            )
        elif _positive_int(received_at) and causal:
            session_value = raw.get("session_id")
            if session_value is None:
                session_id = "<legacy>"
            elif isinstance(session_value, str) and session_value:
                session_id = session_value
            else:
                session_id = "<invalid>"
            assert isinstance(sequence, int)
            assert isinstance(received_at, int)
            key = (symbol, session_id)
            previous = self.orderbook_sequences.get(key)
            if previous is not None and received_at >= previous.received_at_ns:
                if sequence <= previous.sequence:
                    self.streams[stream_name].orderbook_sequence_regressions += 1
                    self.issues.error(
                        "orderbook_sequence_regression",
                        "orderbook sequence did not advance within one collection session",
                        path=path,
                        line=line,
                        stream=stream_name,
                    )
                self.orderbook_sequences[key] = _OrderbookSequence(received_at, sequence)
            elif previous is None:
                self.orderbook_sequences[key] = _OrderbookSequence(received_at, sequence)
        return 1

    def _validate_levels(
        self,
        raw_levels: object,
        side: str,
        path: str,
        line: int,
        stream: str,
    ) -> list[float] | None:
        if not isinstance(raw_levels, list) or not raw_levels:
            self.issues.error(
                "empty_orderbook_side",
                f"orderbook {side} must be a non-empty array",
                path=path,
                line=line,
                stream=stream,
            )
            return None
        prices: list[float] = []
        invalid = 0
        for level in raw_levels:
            if not isinstance(level, list) or len(level) < 2:
                invalid += 1
                continue
            price = _finite_number(level[0])
            size = _finite_number(level[1])
            if price is None or price <= 0 or size is None or size < 0:
                invalid += 1
                continue
            prices.append(price)
        if invalid:
            self.issues.error(
                "invalid_orderbook_level",
                "orderbook levels require finite positive price and non-negative size",
                path=path,
                line=line,
                stream=stream,
                count=invalid,
            )
        return prices

    def _validate_trades(
        self, payload: object, path: str, line: int, stream: str, symbol: str
    ) -> int:
        if not isinstance(payload, list) or not payload:
            self.issues.error(
                "invalid_trades_payload",
                "trades payload must be a non-empty array",
                path=path,
                line=line,
                stream=stream,
            )
            return 0
        valid_events = 0
        for trade in payload:
            if not isinstance(trade, dict):
                self.issues.error(
                    "invalid_trade",
                    "each trade must be an object",
                    path=path,
                    line=line,
                    stream=stream,
                )
                continue
            trade_id = trade.get("i")
            trade_ts = trade.get("T")
            if not isinstance(trade_id, str) or not trade_id:
                self.issues.error(
                    "invalid_trade_id",
                    "trade id i must be a non-empty string",
                    path=path,
                    line=line,
                    stream=stream,
                )
            else:
                try:
                    self.database.execute(
                        "INSERT INTO trade_ids(symbol, trade_id) VALUES (?, ?)",
                        (symbol, trade_id),
                    )
                except sqlite3.IntegrityError:
                    self.streams[stream].duplicate_trade_ids += 1
                    self.issues.error(
                        "duplicate_trade_id",
                        f"trade id {trade_id!r} occurs more than once",
                        path=path,
                        line=line,
                        stream=stream,
                    )
            if not _positive_int(trade_ts):
                self.issues.error(
                    "invalid_trade_timestamp",
                    "trade timestamp T must be a positive integer",
                    path=path,
                    line=line,
                    stream=stream,
                )
            price = _finite_number(trade.get("p"))
            size = _finite_number(trade.get("v"))
            if price is None or price <= 0 or size is None or size <= 0:
                self.issues.error(
                    "invalid_trade_values",
                    "trade p and v must be finite positive numbers",
                    path=path,
                    line=line,
                    stream=stream,
                )
            if trade.get("S") not in {"Buy", "Sell"}:
                self.issues.error(
                    "invalid_trade_side",
                    "trade side S must be Buy or Sell",
                    path=path,
                    line=line,
                    stream=stream,
                )
            embedded_symbol = trade.get("s")
            if embedded_symbol is not None and embedded_symbol != symbol:
                self.issues.error(
                    "payload_symbol_mismatch",
                    "trade symbol does not match the wrapper symbol",
                    path=path,
                    line=line,
                    stream=stream,
                )
            valid_events += 1
        return valid_events

    def _validate_ticker(
        self, payload: object, path: str, line: int, stream: str, symbol: str
    ) -> int:
        if not isinstance(payload, dict):
            self.issues.error(
                "invalid_ticker_payload",
                "ticker payload must be an object",
                path=path,
                line=line,
                stream=stream,
            )
            return 0
        embedded_symbol = payload.get("symbol")
        if embedded_symbol is not None and embedded_symbol != symbol:
            self.issues.error(
                "payload_symbol_mismatch",
                "ticker symbol does not match the wrapper symbol",
                path=path,
                line=line,
                stream=stream,
            )
        positive_fields = ("lastPrice", "indexPrice", "markPrice", "bid1Price", "ask1Price")
        nonnegative_fields = (
            "bid1Size",
            "ask1Size",
            "openInterest",
            "openInterestValue",
            "volume24h",
            "turnover24h",
        )
        invalid_numeric = 0
        for key in positive_fields:
            if key in payload:
                number = _finite_number(payload[key])
                if number is None or number <= 0:
                    invalid_numeric += 1
        for key in nonnegative_fields:
            if key in payload:
                number = _finite_number(payload[key])
                if number is None or number < 0:
                    invalid_numeric += 1
        if "fundingRate" in payload and _finite_number(payload["fundingRate"]) is None:
            invalid_numeric += 1
        if invalid_numeric:
            self.issues.error(
                "invalid_ticker_values",
                "present ticker numeric fields must be finite and in their valid domain",
                path=path,
                line=line,
                stream=stream,
                count=invalid_numeric,
            )
        bid = _finite_number(payload.get("bid1Price"))
        ask = _finite_number(payload.get("ask1Price"))
        if bid is not None and ask is not None and bid >= ask:
            self.issues.error(
                "crossed_ticker",
                "ticker bid1Price must be strictly below ask1Price",
                path=path,
                line=line,
                stream=stream,
            )
        return 1

    def _validate_kline(
        self,
        payload: object,
        path: str,
        line: int,
        stream: str,
        symbol: str,
        kind: str,
        received_at: object,
        causal: bool,
    ) -> int:
        if not isinstance(payload, dict):
            self.issues.error(
                "invalid_kline_payload",
                "kline payload must be an object",
                path=path,
                line=line,
                stream=stream,
            )
            return 0
        interval = kind.removeprefix("kline_")
        payload_interval = payload.get("interval")
        if not isinstance(payload_interval, str) or payload_interval != interval:
            self.issues.error(
                "kline_interval_mismatch",
                "payload interval must match the stream kind",
                path=path,
                line=line,
                stream=stream,
            )
        if payload.get("confirm") is not True:
            self.issues.error(
                "unconfirmed_kline",
                "stored klines must have confirm=true",
                path=path,
                line=line,
                stream=stream,
            )
        start = payload.get("start")
        if not _nonnegative_int(start):
            self.issues.error(
                "invalid_kline_start",
                "kline start must be a non-negative integer",
                path=path,
                line=line,
                stream=stream,
            )
            return 1
        assert isinstance(start, int)
        end = payload.get("end")
        invalid_end = not _positive_int(end)
        if not invalid_end:
            assert isinstance(end, int)
            invalid_end = end < start
        if invalid_end:
            self.issues.error(
                "invalid_kline_end",
                "kline end must be a positive integer not before start",
                path=path,
                line=line,
                stream=stream,
            )
        invalid_prices = 0
        prices: dict[str, float] = {}
        for field in ("open", "high", "low", "close"):
            number = _finite_number(payload.get(field))
            if number is None or number <= 0:
                invalid_prices += 1
            else:
                prices[field] = number
        invalid_volume = 0
        for field in ("volume", "turnover"):
            number = _finite_number(payload.get(field))
            if number is None or number < 0:
                invalid_volume += 1
        if invalid_prices or invalid_volume:
            self.issues.error(
                "invalid_kline_values",
                "kline OHLC must be positive and volume/turnover non-negative finite numbers",
                path=path,
                line=line,
                stream=stream,
                count=invalid_prices + invalid_volume,
            )
        elif prices["low"] > min(prices["open"], prices["close"]):
            self.issues.error(
                "inconsistent_kline_prices",
                "kline low cannot exceed open or close",
                path=path,
                line=line,
                stream=stream,
            )
        elif prices["high"] < max(prices["open"], prices["close"]):
            self.issues.error(
                "inconsistent_kline_prices",
                "kline high cannot be below open or close",
                path=path,
                line=line,
                stream=stream,
            )
        try:
            self.database.execute(
                "INSERT INTO klines(symbol, interval, start_ms) VALUES (?, ?, ?)",
                (symbol, interval, start),
            )
        except sqlite3.IntegrityError:
            self.streams[stream].duplicate_klines += 1
            self.issues.error(
                "duplicate_kline",
                f"kline start {start} occurs more than once",
                path=path,
                line=line,
                stream=stream,
            )

        step = _kline_interval_ms(interval)
        if step is None:
            self.issues.warning(
                "variable_kline_interval",
                f"gap validation is unavailable for interval {interval!r}",
                path=path,
                line=line,
                stream=stream,
            )
            return 1
        if start % step != 0:
            self.issues.error(
                "misaligned_kline_start",
                "kline start is not aligned to its interval",
                path=path,
                line=line,
                stream=stream,
            )
        if _positive_int(received_at) and causal:
            assert isinstance(received_at, int)
            position_key = (symbol, interval)
            previous = self.kline_positions.get(position_key)
            if previous is None:
                self.kline_positions[position_key] = _KlinePosition(received_at, start)
            elif received_at >= previous.received_at_ns:
                delta = start - previous.start_ms
                if delta < 0:
                    self.issues.error(
                        "kline_start_regression",
                        "kline start regressed in causal order",
                        path=path,
                        line=line,
                        stream=stream,
                    )
                elif delta > step:
                    mutable = self.streams[stream]
                    mutable.kline_gaps += 1
                    mutable.max_kline_gap_ms = max(mutable.max_kline_gap_ms, delta)
                    missing = max(1, math.ceil(delta / step) - 1)
                    mutable.missing_klines += missing
                    self.issues.warning(
                        "kline_gap",
                        f"kline series has a gap of {delta} ms",
                        path=path,
                        line=line,
                        stream=stream,
                        count=missing,
                    )
                if start >= previous.start_ms:
                    self.kline_positions[position_key] = _KlinePosition(received_at, start)
        return 1


def _stream_name(kind: str, symbol: str) -> str:
    return f"{kind}/{symbol}"


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return _plain_int(value) and value > 0  # type: ignore[operator]


def _nonnegative_int(value: object) -> bool:
    return _plain_int(value) and value >= 0  # type: ignore[operator]


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _kline_interval_ms(interval: str) -> int | None:
    if interval in _FIXED_KLINE_INTERVALS_MS:
        return _FIXED_KLINE_INTERVALS_MS[interval]
    try:
        minutes = int(interval)
    except ValueError:
        return None
    return minutes * 60_000 if minutes > 0 else None


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _input_fingerprint(files: Sequence[AuditFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_database(database: sqlite3.Connection) -> None:
    database.execute(
        "CREATE TABLE trade_ids ("
        "symbol TEXT NOT NULL, trade_id TEXT NOT NULL, PRIMARY KEY(symbol, trade_id)"
        ") WITHOUT ROWID"
    )
    database.execute(
        "CREATE TABLE klines ("
        "symbol TEXT NOT NULL, interval TEXT NOT NULL, start_ms INTEGER NOT NULL, "
        "PRIMARY KEY(symbol, interval, start_ms)"
        ") WITHOUT ROWID"
    )


def _audit_file(path: Path, relative_path: str, state: _AuditState) -> AuditFile:
    digest = hashlib.sha256()
    size = 0
    lines = 0
    records = 0
    streams_in_file: set[str] = set()
    parts = tuple(Path(relative_path).parts)
    try:
        with path.open("rb") as handle:
            for lines, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    state.issues.error(
                        "blank_line",
                        "JSONL files must not contain blank lines",
                        path=relative_path,
                        line=lines,
                    )
                    continue
                try:
                    raw = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    state.issues.error(
                        "invalid_json",
                        f"line is not valid UTF-8 JSON: {detail}",
                        path=relative_path,
                        line=lines,
                    )
                    continue
                if state.process_record(
                    raw,
                    relative_path=relative_path,
                    path_parts=parts,
                    line_number=lines,
                    line_bytes=len(raw_line),
                    streams_in_file=streams_in_file,
                ):
                    records += 1
    except OSError as exc:
        state.issues.error(
            "file_read_error",
            f"could not read input file: {exc}",
            path=relative_path,
        )
    if lines == 0:
        state.issues.error(
            "empty_file",
            "completed JSONL file must contain at least one record",
            path=relative_path,
        )
    for name in streams_in_file:
        state.streams[name].files += 1
    return AuditFile(
        path=relative_path,
        bytes=size,
        sha256=digest.hexdigest(),
        lines=lines,
        records=records,
        recovered="-recovered" in path.stem,
    )


def _normalized_values(values: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    for value in normalized:
        if not value or "/" in value or "\\" in value:
            raise ValueError(f"invalid {label} value: {value!r}")
    return normalized


def audit_dataset(
    root: str | Path,
    symbols: Sequence[str],
    kline_intervals: Sequence[str],
    minimum_duration_seconds: float = 0,
    strict: bool = False,
    *,
    scratch_dir: str | Path | None = None,
) -> DatasetAuditReport:
    """Audit normalized raw data and return a deterministic readiness report.

    ``strict=False`` tolerates warnings such as legacy records without an explicit
    schema version or kline gaps.  Errors, missing streams, and streams shorter than
    ``minimum_duration_seconds`` always make the report not ready.  In strict mode,
    any warning (including active partial files) also makes it not ready.
    """

    if (
        isinstance(minimum_duration_seconds, bool)
        or not math.isfinite(minimum_duration_seconds)
        or minimum_duration_seconds < 0
    ):
        raise ValueError("minimum_duration_seconds must be finite and non-negative")
    expected_symbols = _normalized_values(symbols, "symbols")
    intervals = _normalized_values(kline_intervals, "kline_intervals")
    root_path = Path(root).resolve()
    expected = tuple(
        sorted(
            _stream_name(kind, symbol)
            for symbol in expected_symbols
            for kind in (*_BASE_KINDS, *(f"kline_{interval}" for interval in intervals))
        )
    )

    files: list[AuditFile] = []
    partial_files: tuple[str, ...] = ()
    scratch_parent = (
        Path(scratch_dir).resolve() if scratch_dir is not None else root_path.parent
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)
    database_path = scratch_parent / f".tradingbot-audit-{uuid.uuid4().hex}.sqlite3"
    database = sqlite3.connect(str(database_path))
    try:
        _prepare_database(database)
        state = _AuditState(
            root_path,
            frozenset(expected_symbols),
            frozenset(intervals),
            database,
        )
        if not root_path.is_dir():
            state.issues.error(
                "missing_dataset_root",
                "dataset root does not exist or is not a directory",
            )
        else:
            completed_paths = sorted(
                root_path.rglob("*.jsonl"), key=lambda item: _relative(item, root_path)
            )
            partial_paths = sorted(
                root_path.rglob("*.jsonl.partial"),
                key=lambda item: _relative(item, root_path),
            )
            partial_files = tuple(_relative(item, root_path) for item in partial_paths)
            if partial_files:
                state.issues.warning(
                    "partial_files_present",
                    "active partial JSONL files were ignored",
                    path=partial_files[0],
                    count=len(partial_files),
                )
            for path in completed_paths:
                relative_path = _relative(path, root_path)
                files.append(_audit_file(path, relative_path, state))
                database.commit()
            recovered_count = sum(item.recovered for item in files)
            if recovered_count:
                state.issues.warning(
                    "recovered_segments_present",
                    "dataset contains segments recovered after an unclean stop",
                    path=next(item.path for item in files if item.recovered),
                    count=recovered_count,
                )

        frozen_streams = {
            name: state.streams[name].freeze() for name in sorted(state.streams)
        }
        missing = tuple(name for name in expected if name not in frozen_streams)
        short = tuple(
            name
            for name in expected
            if name in frozen_streams
            and frozen_streams[name].duration_seconds < minimum_duration_seconds
        )
        errors = state.issues.errors()
        warnings = state.issues.warnings()
    finally:
        database.close()
        database_path.unlink(missing_ok=True)

    reasons: list[str] = []
    if errors:
        reasons.append("validation_errors")
    if missing:
        reasons.append("missing_expected_streams")
    if short:
        reasons.append("minimum_duration_not_met")
    if strict and warnings:
        reasons.append("strict_warnings")
    ready = not reasons
    total_bytes = sum(item.bytes for item in files)
    total_lines = sum(item.lines for item in files)
    total_records = sum(item.records for item in files)
    first_timestamps = [
        stream.first_exchange_ts_ms
        for stream in frozen_streams.values()
        if stream.first_exchange_ts_ms is not None
    ]
    last_timestamps = [
        stream.last_exchange_ts_ms
        for stream in frozen_streams.values()
        if stream.last_exchange_ts_ms is not None
    ]
    observed_duration_seconds = (
        0.0
        if not first_timestamps or not last_timestamps
        else (max(last_timestamps) - min(first_timestamps)) / 1_000
    )
    projected_bytes_per_day = (
        None
        if observed_duration_seconds <= 0
        else round(total_bytes * 86_400 / observed_duration_seconds)
    )
    return DatasetAuditReport(
        dataset_root=root_path.as_posix(),
        expected_symbols=expected_symbols,
        kline_intervals=intervals,
        input_fingerprint=_input_fingerprint(files),
        files=tuple(files),
        partial_files=partial_files,
        streams=frozen_streams,
        errors=errors,
        warnings=warnings,
        missing_expected_streams=missing,
        short_streams=short,
        minimum_duration_seconds=float(minimum_duration_seconds),
        strict=strict,
        total_bytes=total_bytes,
        total_lines=total_lines,
        total_records=total_records,
        observed_duration_seconds=observed_duration_seconds,
        projected_bytes_per_day=projected_bytes_per_day,
        ready=ready,
        readiness_reasons=tuple(reasons),
    )
