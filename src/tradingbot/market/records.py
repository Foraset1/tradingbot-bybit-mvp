from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MarketRecord:
    kind: str
    symbol: str
    exchange_ts_ms: int
    received_at_ns: int
    payload: dict[str, Any] | list[dict[str, Any]]
    schema_version: int = 1
    source: str = "bybit"
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
