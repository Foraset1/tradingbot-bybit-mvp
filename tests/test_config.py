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
    assert config.storage.min_free_bytes == 15 * 1024**3
    assert config.archive.root == config_path.parents[1] / "data" / "archive"
    assert config.archive.raw_retention_days == 7
    assert config.archive.daily_minimum_duration_seconds == 82_800
    assert config.history.root == config_path.parents[1] / "data" / "history"
    assert config.history.public_base_url == "https://public.bybit.com/trading"
    assert config.history.assumed_latency_ms == 1_000
    assert config.history.maximum_missing_minutes == 5
    assert config.evaluation.horizon_minutes == 60
    assert config.evaluation.embargo_minutes == 60
    assert config.evaluation.acceptance_minimum_days == 365
    assert config.evaluation.calibration_days == 7
    assert config.evaluation.minimum_calibration_rows == 250
    assert config.evaluation.minimum_symbol_coverage_fraction == 0.95
    assert config.evaluation.training_threads == 4


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


def test_min_free_bytes_can_be_overridden(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", str(12 * 1024**3))

    assert load_config(config_path).storage.min_free_bytes == 12 * 1024**3


def test_archive_policy_can_be_overridden(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "daily-archive"
    monkeypatch.setenv("TRADINGBOT_ARCHIVE_ROOT", str(archive))
    monkeypatch.setenv("TRADINGBOT_RAW_RETENTION_DAYS", "5")

    config = load_config(config_path)

    assert config.archive.root == archive
    assert config.archive.raw_retention_days == 5


def test_history_root_can_be_overridden(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "official-history"
    monkeypatch.setenv("TRADINGBOT_HISTORY_ROOT", str(history))

    assert load_config(config_path).history.root == history


def test_rejects_invalid_min_free_bytes_override(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADINGBOT_MIN_FREE_BYTES", "12GB")

    with pytest.raises(ConfigError, match="TRADINGBOT_MIN_FREE_BYTES must be an integer"):
        load_config(config_path)


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
        "horizon_minutes = 60": "horizon_minutes = 30",
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
        (
            "embargo_minutes = 60",
            "embargo_minutes = 30",
            "cannot be shorter",
        ),
        (
            "maker_fee_bps = 2.0",
            "maker_fee_bps = -1.0",
            r"within \[0, 100\]",
        ),
        (
            "training_threads = 4",
            "training_threads = 0",
            "integer limits must be positive",
        ),
        (
            "calibration_days = 7",
            "calibration_days = 30",
            "must be shorter",
        ),
        (
            "minimum_symbol_coverage_fraction = 0.95",
            "minimum_symbol_coverage_fraction = 1.1",
            r"within \(0, 1\]",
        ),
        (
            "raw_retention_days = 7",
            "raw_retention_days = 0",
            "raw_retention_days must be positive",
        ),
        (
            'public_base_url = "https://public.bybit.com/trading"',
            'public_base_url = "https://example.com/trading"',
            "official",
        ),
        (
            "download_attempts = 3",
            "download_attempts = 0",
            "download_attempts",
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
