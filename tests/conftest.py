from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def config_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("BYBIT_PUBLIC_WS_URL", raising=False)
    monkeypatch.delenv("TRADINGBOT_DATA_ROOT", raising=False)
    monkeypatch.delenv("TRADINGBOT_HISTORY_ROOT", raising=False)
    return Path(__file__).parents[1] / "config" / "tradingbot.toml"
