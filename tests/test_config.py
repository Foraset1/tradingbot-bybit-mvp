from __future__ import annotations

from pathlib import Path

import pytest

from tradingbot.config import ConfigError, load_config


def test_loads_expected_safe_defaults(config_path: Path) -> None:
    config = load_config(config_path)

    assert config.bybit.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "BNBUSDT",
        "LINKUSDT",
    )
    assert config.market.orderbook_depth == 50
    assert config.bybit.stale_connection_seconds == 60.0
    assert config.risk.max_open_positions == 1
    assert config.risk.max_hold_seconds == 3600
    assert config.risk.forced_exit_seconds == 3540
    assert config.storage.root == config_path.parents[1] / "data" / "raw"
    assert config.storage.min_free_bytes == 10 * 1024**3


def test_data_root_can_be_overridden(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "market-data"
    monkeypatch.setenv("TRADINGBOT_DATA_ROOT", str(override))

    assert load_config(config_path).storage.root == override


def test_health_path_can_be_overridden(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "soak-health.json"
    monkeypatch.setenv("TRADINGBOT_HEALTH_PATH", str(override))

    assert load_config(config_path).storage.health_path == override


def test_accepts_risk_values_below_mvp_caps(config_path: Path, tmp_path: Path) -> None:
    target = tmp_path / "config" / "tradingbot.toml"
    target.parent.mkdir()
    source = config_path.read_text(encoding="utf-8")
    replacements = {
        "max_notional_fraction = 0.05": "max_notional_fraction = 0.04",
        "target_risk_fraction = 0.005": "target_risk_fraction = 0.003",
        "max_planned_risk_fraction = 0.007": "max_planned_risk_fraction = 0.006",
        "rolling_24h_loss_fraction = 0.01": "rolling_24h_loss_fraction = 0.009",
        "max_hold_seconds = 3600": "max_hold_seconds = 3500",
        "soft_exit_seconds = 3300": "soft_exit_seconds = 3200",
        "forced_exit_seconds = 3540": "forced_exit_seconds = 3400",
    }
    for old, new in replacements.items():
        assert old in source
        source = source.replace(old, new)
    target.write_text(source, encoding="utf-8")

    config = load_config(target)

    assert config.risk.max_notional_fraction == 0.04
    assert config.risk.max_hold_seconds == 3500


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ('    "LINKUSDT",', '    "BTCUSDT",', "contains duplicates"),
        (
            "target_risk_fraction = 0.005",
            "target_risk_fraction = 0.008",
            "cannot exceed",
        ),
        ("max_open_positions = 1", "max_open_positions = 2", "exactly one"),
        ("forced_exit_seconds = 3540", "forced_exit_seconds = 3600", "Exit timers"),
        ("collect_trades = true", 'collect_trades = "true"', "must be a boolean"),
        (
            "heartbeat_seconds = 20",
            "heartbeat_seconds_renamed = 20",
            "Missing required value",
        ),
        (
            "stale_connection_seconds = 60.0",
            "stale_connection_seconds = 20.0",
            "must exceed heartbeat_seconds",
        ),
        (
            "max_notional_fraction = 0.05",
            "max_notional_fraction = 0.051",
            "safety cap",
        ),
        (
            "target_risk_fraction = 0.005",
            "target_risk_fraction = 0.006",
            "safety cap",
        ),
        (
            "max_planned_risk_fraction = 0.007",
            "max_planned_risk_fraction = 0.008",
            "safety cap",
        ),
        (
            "rolling_24h_loss_fraction = 0.01",
            "rolling_24h_loss_fraction = 0.011",
            "safety cap",
        ),
        (
            "max_hold_seconds = 3600",
            "max_hold_seconds = 3601",
            "safety cap",
        ),
        (
            "target_risk_fraction = 0.005",
            "target_risk_fraction = nan",
            "must be finite",
        ),
    ],
)
def test_rejects_unsafe_configuration(
    config_path: Path,
    tmp_path: Path,
    old: str,
    new: str,
    error: str,
) -> None:
    target = tmp_path / "config" / "tradingbot.toml"
    target.parent.mkdir()
    source = config_path.read_text(encoding="utf-8")
    assert old in source
    target.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=error):
        load_config(target)
