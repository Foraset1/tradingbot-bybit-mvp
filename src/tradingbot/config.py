from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Raised when the application configuration is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class BybitConfig:
    public_ws_url: str
    symbols: tuple[str, ...]
    heartbeat_seconds: int
    stale_connection_seconds: float
    reconnect_min_seconds: float
    reconnect_max_seconds: float


@dataclass(frozen=True, slots=True)
class MarketConfig:
    orderbook_depth: int
    orderbook_snapshot_ms: int
    ticker_snapshot_ms: int
    kline_intervals: tuple[str, ...]
    collect_orderbook: bool
    collect_trades: bool
    collect_tickers: bool
    collect_klines: bool


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    segment_seconds: int
    segment_max_bytes: int
    flush_seconds: float
    queue_maxsize: int
    min_free_bytes: int
    health_path: Path


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_notional_fraction: float
    target_risk_fraction: float
    max_planned_risk_fraction: float
    rolling_24h_loss_fraction: float
    max_open_positions: int
    max_hold_seconds: int
    soft_exit_seconds: int
    forced_exit_seconds: int
    entry_order_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    horizon_minutes: int
    embargo_minutes: int
    minimum_train_days: int
    test_days: int
    maximum_folds: int
    acceptance_minimum_days: int
    minimum_train_rows: int
    minimum_test_rows: int
    maker_fee_bps: float
    taker_fee_bps: float
    entry_adverse_selection_bps: float
    stop_slippage_bps: float
    timeout_slippage_bps: float
    minimum_expected_net_bps: float
    lightgbm_estimators: int
    lightgbm_learning_rate: float
    lightgbm_num_leaves: int
    lightgbm_min_child_samples: int
    training_threads: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    bybit: BybitConfig
    market: MarketConfig
    storage: StorageConfig
    risk: RiskConfig
    evaluation: EvaluationConfig
    source_path: Path


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{name}] table")
    return value


def _required(table: dict[str, Any], key: str, table_name: str) -> Any:
    if key not in table:
        raise ConfigError(f"Missing required value: {table_name}.{key}")
    return table[key]


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    return int(value)


def _environment_integer(name: str, default: Any, field: str) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return _integer(default, field)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ConfigError(f"{field} must be finite")
    return number


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _str_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be a non-empty list of strings")
    return tuple(value)


def _resolve_path(value: Any, field: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_ws_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "wss" or not parsed.hostname:
        raise ConfigError("bybit.public_ws_url must be a valid wss:// URL")


def _validate_symbols(symbols: tuple[str, ...]) -> None:
    if len(symbols) != len(set(symbols)):
        raise ConfigError("bybit.symbols contains duplicates")
    for symbol in symbols:
        if symbol != symbol.upper() or not symbol.endswith("USDT") or not symbol.isalnum():
            raise ConfigError(f"Unsupported symbol format: {symbol!r}")


def _validate_risk(config: RiskConfig) -> None:
    fractions = (
        config.max_notional_fraction,
        config.target_risk_fraction,
        config.max_planned_risk_fraction,
        config.rolling_24h_loss_fraction,
    )
    if any(value <= 0 or value > 1 for value in fractions):
        raise ConfigError("Risk fractions must be within (0, 1]")
    if config.target_risk_fraction > config.max_planned_risk_fraction:
        raise ConfigError("target_risk_fraction cannot exceed max_planned_risk_fraction")
    if config.max_planned_risk_fraction >= config.rolling_24h_loss_fraction:
        raise ConfigError("max_planned_risk_fraction must be below rolling_24h_loss_fraction")
    risk_caps = {
        "max_notional_fraction": (config.max_notional_fraction, 0.05),
        "target_risk_fraction": (config.target_risk_fraction, 0.005),
        "max_planned_risk_fraction": (config.max_planned_risk_fraction, 0.007),
        "rolling_24h_loss_fraction": (config.rolling_24h_loss_fraction, 0.01),
    }
    for name, (value, cap) in risk_caps.items():
        if value > cap:
            raise ConfigError(f"{name} cannot exceed the MVP safety cap of {cap}")
    if config.max_open_positions != 1:
        raise ConfigError("MVP requires exactly one open position")
    if config.max_hold_seconds > 3600:
        raise ConfigError("max_hold_seconds cannot exceed the MVP safety cap of 3600")
    if not 0 < config.soft_exit_seconds < config.forced_exit_seconds < config.max_hold_seconds:
        raise ConfigError("Exit timers must satisfy soft < forced < max hold")
    if config.entry_order_ttl_seconds <= 0:
        raise ConfigError("entry_order_ttl_seconds must be positive")


def _validate_evaluation(config: EvaluationConfig, risk: RiskConfig) -> None:
    if config.horizon_minutes not in {5, 15, 30, 60}:
        raise ConfigError("evaluation.horizon_minutes must be one of 5, 15, 30, or 60")
    if config.horizon_minutes * 60 > risk.max_hold_seconds:
        raise ConfigError("evaluation horizon cannot exceed risk.max_hold_seconds")
    if config.embargo_minutes < config.horizon_minutes:
        raise ConfigError("evaluation.embargo_minutes cannot be shorter than the horizon")
    positive_integers = {
        "minimum_train_days": config.minimum_train_days,
        "test_days": config.test_days,
        "maximum_folds": config.maximum_folds,
        "acceptance_minimum_days": config.acceptance_minimum_days,
        "minimum_train_rows": config.minimum_train_rows,
        "minimum_test_rows": config.minimum_test_rows,
        "lightgbm_estimators": config.lightgbm_estimators,
        "lightgbm_num_leaves": config.lightgbm_num_leaves,
        "lightgbm_min_child_samples": config.lightgbm_min_child_samples,
        "training_threads": config.training_threads,
    }
    if any(value <= 0 for value in positive_integers.values()):
        raise ConfigError("evaluation integer limits must be positive")
    if config.acceptance_minimum_days < (
        config.minimum_train_days + config.test_days
    ):
        raise ConfigError(
            "evaluation.acceptance_minimum_days must cover train and test windows"
        )
    costs = {
        "maker_fee_bps": config.maker_fee_bps,
        "taker_fee_bps": config.taker_fee_bps,
        "entry_adverse_selection_bps": config.entry_adverse_selection_bps,
        "stop_slippage_bps": config.stop_slippage_bps,
        "timeout_slippage_bps": config.timeout_slippage_bps,
        "minimum_expected_net_bps": config.minimum_expected_net_bps,
    }
    if any(value < 0 or value > 100 for value in costs.values()):
        raise ConfigError("evaluation costs and thresholds must be within [0, 100] bps")
    if not 0 < config.lightgbm_learning_rate <= 1:
        raise ConfigError("evaluation.lightgbm_learning_rate must be within (0, 1]")
    if config.random_seed < 0:
        raise ConfigError("evaluation.random_seed must be non-negative")


def load_config(path: str | Path) -> AppConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {source_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {source_path}: {exc}") from exc

    project_root = source_path.parent.parent
    bybit_raw = _table(raw, "bybit")
    market_raw = _table(raw, "market")
    storage_raw = _table(raw, "storage")
    risk_raw = _table(raw, "risk")
    evaluation_raw = _table(raw, "evaluation")

    default_ws_url = _string(
        _required(bybit_raw, "public_ws_url", "bybit"), "bybit.public_ws_url"
    )
    ws_url = os.getenv("BYBIT_PUBLIC_WS_URL", default_ws_url)
    symbols = _str_tuple(_required(bybit_raw, "symbols", "bybit"), "bybit.symbols")
    bybit = BybitConfig(
        public_ws_url=ws_url,
        symbols=symbols,
        heartbeat_seconds=_integer(
            _required(bybit_raw, "heartbeat_seconds", "bybit"),
            "bybit.heartbeat_seconds",
        ),
        stale_connection_seconds=_number(
            _required(bybit_raw, "stale_connection_seconds", "bybit"),
            "bybit.stale_connection_seconds",
        ),
        reconnect_min_seconds=_number(
            _required(bybit_raw, "reconnect_min_seconds", "bybit"),
            "bybit.reconnect_min_seconds",
        ),
        reconnect_max_seconds=_number(
            _required(bybit_raw, "reconnect_max_seconds", "bybit"),
            "bybit.reconnect_max_seconds",
        ),
    )

    market = MarketConfig(
        orderbook_depth=_integer(
            _required(market_raw, "orderbook_depth", "market"),
            "market.orderbook_depth",
        ),
        orderbook_snapshot_ms=_integer(
            _required(market_raw, "orderbook_snapshot_ms", "market"),
            "market.orderbook_snapshot_ms",
        ),
        ticker_snapshot_ms=_integer(
            _required(market_raw, "ticker_snapshot_ms", "market"),
            "market.ticker_snapshot_ms",
        ),
        kline_intervals=_str_tuple(
            _required(market_raw, "kline_intervals", "market"),
            "market.kline_intervals",
        ),
        collect_orderbook=_boolean(
            _required(market_raw, "collect_orderbook", "market"),
            "market.collect_orderbook",
        ),
        collect_trades=_boolean(
            _required(market_raw, "collect_trades", "market"),
            "market.collect_trades",
        ),
        collect_tickers=_boolean(
            _required(market_raw, "collect_tickers", "market"),
            "market.collect_tickers",
        ),
        collect_klines=_boolean(
            _required(market_raw, "collect_klines", "market"),
            "market.collect_klines",
        ),
    )

    default_data_root = _string(_required(storage_raw, "root", "storage"), "storage.root")
    data_root_value = os.getenv("TRADINGBOT_DATA_ROOT", default_data_root)
    default_health_path = _string(
        _required(storage_raw, "health_path", "storage"), "storage.health_path"
    )
    health_path_value = os.getenv("TRADINGBOT_HEALTH_PATH", default_health_path)
    storage = StorageConfig(
        root=_resolve_path(data_root_value, "storage.root", project_root),
        segment_seconds=_integer(
            _required(storage_raw, "segment_seconds", "storage"),
            "storage.segment_seconds",
        ),
        segment_max_bytes=_integer(
            _required(storage_raw, "segment_max_bytes", "storage"),
            "storage.segment_max_bytes",
        ),
        flush_seconds=_number(
            _required(storage_raw, "flush_seconds", "storage"),
            "storage.flush_seconds",
        ),
        queue_maxsize=_integer(
            _required(storage_raw, "queue_maxsize", "storage"),
            "storage.queue_maxsize",
        ),
        min_free_bytes=_environment_integer(
            "TRADINGBOT_MIN_FREE_BYTES",
            _required(storage_raw, "min_free_bytes", "storage"),
            "storage.min_free_bytes",
        ),
        health_path=_resolve_path(
            health_path_value,
            "storage.health_path",
            project_root,
        ),
    )

    risk = RiskConfig(
        max_notional_fraction=_number(
            _required(risk_raw, "max_notional_fraction", "risk"),
            "risk.max_notional_fraction",
        ),
        target_risk_fraction=_number(
            _required(risk_raw, "target_risk_fraction", "risk"),
            "risk.target_risk_fraction",
        ),
        max_planned_risk_fraction=_number(
            _required(risk_raw, "max_planned_risk_fraction", "risk"),
            "risk.max_planned_risk_fraction",
        ),
        rolling_24h_loss_fraction=_number(
            _required(risk_raw, "rolling_24h_loss_fraction", "risk"),
            "risk.rolling_24h_loss_fraction",
        ),
        max_open_positions=_integer(
            _required(risk_raw, "max_open_positions", "risk"),
            "risk.max_open_positions",
        ),
        max_hold_seconds=_integer(
            _required(risk_raw, "max_hold_seconds", "risk"),
            "risk.max_hold_seconds",
        ),
        soft_exit_seconds=_integer(
            _required(risk_raw, "soft_exit_seconds", "risk"),
            "risk.soft_exit_seconds",
        ),
        forced_exit_seconds=_integer(
            _required(risk_raw, "forced_exit_seconds", "risk"),
            "risk.forced_exit_seconds",
        ),
        entry_order_ttl_seconds=_integer(
            _required(risk_raw, "entry_order_ttl_seconds", "risk"),
            "risk.entry_order_ttl_seconds",
        ),
    )

    evaluation = EvaluationConfig(
        horizon_minutes=_integer(
            _required(evaluation_raw, "horizon_minutes", "evaluation"),
            "evaluation.horizon_minutes",
        ),
        embargo_minutes=_integer(
            _required(evaluation_raw, "embargo_minutes", "evaluation"),
            "evaluation.embargo_minutes",
        ),
        minimum_train_days=_integer(
            _required(evaluation_raw, "minimum_train_days", "evaluation"),
            "evaluation.minimum_train_days",
        ),
        test_days=_integer(
            _required(evaluation_raw, "test_days", "evaluation"),
            "evaluation.test_days",
        ),
        maximum_folds=_integer(
            _required(evaluation_raw, "maximum_folds", "evaluation"),
            "evaluation.maximum_folds",
        ),
        acceptance_minimum_days=_integer(
            _required(evaluation_raw, "acceptance_minimum_days", "evaluation"),
            "evaluation.acceptance_minimum_days",
        ),
        minimum_train_rows=_integer(
            _required(evaluation_raw, "minimum_train_rows", "evaluation"),
            "evaluation.minimum_train_rows",
        ),
        minimum_test_rows=_integer(
            _required(evaluation_raw, "minimum_test_rows", "evaluation"),
            "evaluation.minimum_test_rows",
        ),
        maker_fee_bps=_number(
            _required(evaluation_raw, "maker_fee_bps", "evaluation"),
            "evaluation.maker_fee_bps",
        ),
        taker_fee_bps=_number(
            _required(evaluation_raw, "taker_fee_bps", "evaluation"),
            "evaluation.taker_fee_bps",
        ),
        entry_adverse_selection_bps=_number(
            _required(
                evaluation_raw, "entry_adverse_selection_bps", "evaluation"
            ),
            "evaluation.entry_adverse_selection_bps",
        ),
        stop_slippage_bps=_number(
            _required(evaluation_raw, "stop_slippage_bps", "evaluation"),
            "evaluation.stop_slippage_bps",
        ),
        timeout_slippage_bps=_number(
            _required(evaluation_raw, "timeout_slippage_bps", "evaluation"),
            "evaluation.timeout_slippage_bps",
        ),
        minimum_expected_net_bps=_number(
            _required(evaluation_raw, "minimum_expected_net_bps", "evaluation"),
            "evaluation.minimum_expected_net_bps",
        ),
        lightgbm_estimators=_integer(
            _required(evaluation_raw, "lightgbm_estimators", "evaluation"),
            "evaluation.lightgbm_estimators",
        ),
        lightgbm_learning_rate=_number(
            _required(evaluation_raw, "lightgbm_learning_rate", "evaluation"),
            "evaluation.lightgbm_learning_rate",
        ),
        lightgbm_num_leaves=_integer(
            _required(evaluation_raw, "lightgbm_num_leaves", "evaluation"),
            "evaluation.lightgbm_num_leaves",
        ),
        lightgbm_min_child_samples=_integer(
            _required(evaluation_raw, "lightgbm_min_child_samples", "evaluation"),
            "evaluation.lightgbm_min_child_samples",
        ),
        training_threads=_integer(
            _required(evaluation_raw, "training_threads", "evaluation"),
            "evaluation.training_threads",
        ),
        random_seed=_integer(
            _required(evaluation_raw, "random_seed", "evaluation"),
            "evaluation.random_seed",
        ),
    )

    _validate_ws_url(bybit.public_ws_url)
    _validate_symbols(bybit.symbols)
    if bybit.heartbeat_seconds <= 0:
        raise ConfigError("heartbeat_seconds must be positive")
    if bybit.stale_connection_seconds <= bybit.heartbeat_seconds:
        raise ConfigError("stale_connection_seconds must exceed heartbeat_seconds")
    if not 0 < bybit.reconnect_min_seconds <= bybit.reconnect_max_seconds:
        raise ConfigError("Reconnect delays must satisfy 0 < min <= max")
    if market.orderbook_depth not in {1, 50, 200, 1000}:
        raise ConfigError("Unsupported linear orderbook depth")
    if market.orderbook_snapshot_ms <= 0 or market.ticker_snapshot_ms <= 0:
        raise ConfigError("Snapshot intervals must be positive")
    valid_intervals = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720"}
    if not set(market.kline_intervals).issubset(valid_intervals):
        raise ConfigError("Unsupported kline interval")
    if not any(
        (
            market.collect_orderbook,
            market.collect_trades,
            market.collect_tickers,
            market.collect_klines,
        )
    ):
        raise ConfigError("At least one market stream must be enabled")
    if storage.segment_seconds <= 0 or storage.segment_max_bytes <= 0:
        raise ConfigError("Storage segment limits must be positive")
    if (
        storage.flush_seconds <= 0
        or storage.queue_maxsize <= 0
        or storage.min_free_bytes <= 0
    ):
        raise ConfigError(
            "Storage flush interval, queue size, and minimum free bytes must be positive"
        )
    _validate_risk(risk)
    _validate_evaluation(evaluation, risk)

    return AppConfig(
        bybit=bybit,
        market=market,
        storage=storage,
        risk=risk,
        evaluation=evaluation,
        source_path=source_path,
    )
