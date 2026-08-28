"""Frozen LightGBM inference and execution-aware candidate scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import lightgbm
import numpy as np
from numpy.typing import NDArray

from tradingbot.research.evaluation_contracts import (
    DIRECT_FEATURE_COLUMNS,
    LOG1P_FEATURE_COLUMNS,
)
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_OUTCOME_NAMES,
    EXECUTION_OUTCOME_TO_INDEX,
    FILL_NAMES,
    FILL_TO_INDEX,
)
from tradingbot.shadow.bundle import ShadowBundle, ShadowBundleError


@dataclass(frozen=True, slots=True)
class ShadowCandidate:
    decision_id: str
    decision_at_ns: int
    symbol: str
    side: str
    entry_limit_price: float
    stop_distance_bps: float
    take_profit_distance_bps: float
    stop_price: float
    take_profit_price: float
    fill_probabilities: tuple[float, ...]
    outcome_probabilities: tuple[float, ...]
    timeout_return_estimate_bps: float
    partial_fill_fraction_estimate: float
    expected_net_bps: float
    notional_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "decision_at_ns": self.decision_at_ns,
            "symbol": self.symbol,
            "side": self.side,
            "entry_limit_price": self.entry_limit_price,
            "stop_distance_bps": self.stop_distance_bps,
            "take_profit_distance_bps": self.take_profit_distance_bps,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "fill_probabilities": dict(zip(FILL_NAMES, self.fill_probabilities, strict=True)),
            "post_fill_outcome_probabilities": dict(
                zip(
                    EXECUTION_OUTCOME_NAMES,
                    self.outcome_probabilities,
                    strict=True,
                )
            ),
            "timeout_return_estimate_bps": self.timeout_return_estimate_bps,
            "partial_fill_fraction_estimate": self.partial_fill_fraction_estimate,
            "expected_net_bps": self.expected_net_bps,
            "notional_fraction": self.notional_fraction,
        }


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ShadowBundleError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _number(value: object, label: str, *, allow_nan: bool = False) -> float:
    if value is None and allow_nan:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowBundleError(f"{label} must be numeric")
    result = float(value)
    if math.isinf(result) or (math.isnan(result) and not allow_nan):
        raise ShadowBundleError(f"{label} must be finite")
    return result


def _calibrate(
    raw: NDArray[np.float64],
    contract: dict[str, Any],
    classes: tuple[str, ...],
) -> NDArray[np.float64]:
    if raw.shape != (1, len(classes)):
        raise ShadowBundleError("LightGBM returned an invalid probability shape")
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or float(np.sum(raw)) <= 0:
        raise ShadowBundleError("LightGBM returned invalid probabilities")
    normalized = raw / np.sum(raw, axis=1, keepdims=True)
    temperature = _number(contract.get("temperature"), "calibrator.temperature")
    prior_weight = _number(contract.get("prior_weight"), "calibrator.prior_weight")
    if temperature <= 0 or not 0 <= prior_weight < 1:
        raise ShadowBundleError("calibrator parameters are outside their domain")
    prior_payload = _object(contract.get("class_prior"), "calibrator.class_prior")
    if set(prior_payload) != set(classes):
        raise ShadowBundleError("calibrator class prior is incompatible")
    prior = np.asarray(
        [_number(prior_payload[name], f"class_prior.{name}") for name in classes],
        dtype=np.float64,
    )
    if np.any(prior < 0) or not math.isclose(float(np.sum(prior)), 1.0, abs_tol=1e-8):
        raise ShadowBundleError("calibrator class prior is invalid")
    logits = np.log(np.clip(normalized, 1e-12, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= np.sum(scaled, axis=1, keepdims=True)
    blended = (1.0 - prior_weight) * scaled + prior_weight * prior[None, :]
    return blended / np.sum(blended, axis=1, keepdims=True)


def _estimate(contract: dict[str, Any], symbol: str, side: str) -> float:
    groups = _object(contract.get("by_symbol_side"), "estimate.by_symbol_side")
    sides = _object(contract.get("by_side"), "estimate.by_side")
    group = f"{symbol}|{side}"
    if group in groups:
        return _number(groups[group], f"estimate.{group}")
    if side in sides:
        return _number(sides[side], f"estimate.{side}")
    return _number(contract.get("global"), "estimate.global")


class ShadowScorer:
    """Load authenticated models and score one decision without side effects."""

    def __init__(self, bundle: ShadowBundle) -> None:
        self.bundle = bundle
        model = _object(bundle.contract.get("model"), "model")
        names = model.get("feature_names")
        if not isinstance(names, list) or not all(
            isinstance(item, str) and item for item in names
        ):
            raise ShadowBundleError("model feature names are invalid")
        self.feature_names = tuple(cast(list[str], names))
        self.fill_model = lightgbm.Booster(model_file=str(bundle.fill_model_path))
        self.outcome_model = lightgbm.Booster(model_file=str(bundle.outcome_model_path))
        for name, booster in (
            ("fill", self.fill_model),
            ("post-fill outcome", self.outcome_model),
        ):
            if booster.num_feature() != len(self.feature_names):
                raise ShadowBundleError(
                    f"{name} model expects {booster.num_feature()} features, "
                    f"bundle declares {len(self.feature_names)}"
                )
        calibrators = _object(model.get("calibrators"), "model.calibrators")
        self.fill_calibrator = _object(calibrators.get("fill"), "fill calibrator")
        self.outcome_calibrator = _object(
            calibrators.get("post_fill_outcome"), "outcome calibrator"
        )
        estimates = _object(model.get("execution_estimates"), "execution estimates")
        self.timeout_estimates = _object(
            estimates.get("timeout_return_bps"), "timeout estimates"
        )
        self.partial_estimates = _object(
            estimates.get("partial_fill_fraction"), "partial estimates"
        )
        universe = _object(bundle.contract.get("universe"), "universe")
        raw_symbols = universe.get("symbols")
        if not isinstance(raw_symbols, list) or not all(
            isinstance(item, str) and item for item in raw_symbols
        ):
            raise ShadowBundleError("bundle universe is invalid")
        self.symbols = tuple(cast(list[str], raw_symbols))
        self.symbol_set = frozenset(self.symbols)
        self.evaluation = _object(
            bundle.contract.get("evaluation_parameters"), "evaluation parameters"
        )

    def _feature_vector(
        self,
        feature: dict[str, object],
        *,
        symbol: str,
        side: str,
        stop_distance_bps: float,
        take_profit_distance_bps: float,
    ) -> NDArray[np.float32]:
        if symbol not in self.symbol_set or side not in {"LONG", "SHORT"}:
            raise ShadowBundleError("candidate symbol or side is outside the bundle")
        direct = frozenset(DIRECT_FEATURE_COLUMNS)
        log_columns = frozenset(LOG1P_FEATURE_COLUMNS)
        values: list[float] = []
        for name in self.feature_names:
            if name in direct:
                values.append(_number(feature.get(name), name, allow_nan=True))
            elif name.startswith("log1p_") and name.removeprefix("log1p_") in log_columns:
                source_name = name.removeprefix("log1p_")
                source = _number(feature.get(source_name), source_name, allow_nan=True)
                if math.isfinite(source) and source < 0:
                    raise ShadowBundleError(f"{source_name} cannot be negative")
                values.append(math.log1p(source) if math.isfinite(source) else math.nan)
            elif name == "stop_distance_bps":
                values.append(stop_distance_bps)
            elif name == "take_profit_distance_bps":
                values.append(take_profit_distance_bps)
            elif name == "side_direction":
                values.append(1.0 if side == "LONG" else -1.0)
            elif name.startswith("symbol_"):
                values.append(1.0 if name == f"symbol_{symbol}" else 0.0)
            else:
                raise ShadowBundleError(f"unsupported frozen feature: {name}")
        vector = np.asarray(values, dtype=np.float32)[None, :]
        if np.any(np.isinf(vector)):
            raise ShadowBundleError("candidate vector contains infinity")
        return vector

    def _expected_net_bps(
        self,
        *,
        feature: dict[str, object],
        side: str,
        stop_distance_bps: float,
        take_profit_distance_bps: float,
        fill: NDArray[np.float64],
        outcome: NDArray[np.float64],
        timeout_return_bps: float,
        partial_fraction: float,
    ) -> float:
        maker_fee = _number(self.evaluation.get("maker_fee_bps"), "maker fee")
        taker_fee = _number(self.evaluation.get("taker_fee_bps"), "taker fee")
        stop_slippage = _number(
            self.evaluation.get("stop_slippage_bps"), "stop slippage"
        )
        timeout_slippage = _number(
            self.evaluation.get("timeout_slippage_bps"), "timeout slippage"
        )
        sl = EXECUTION_OUTCOME_TO_INDEX["SL_FIRST"]
        timeout = EXECUTION_OUTCOME_TO_INDEX["TIMEOUT"]
        tp = EXECUTION_OUTCOME_TO_INDEX["TP_FIRST"]
        gross_if_full = (
            outcome[tp] * take_profit_distance_bps
            - outcome[sl] * stop_distance_bps
            + outcome[timeout] * timeout_return_bps
        )
        exit_fees = outcome[tp] * maker_fee + (
            outcome[sl] + outcome[timeout]
        ) * taker_fee
        slippage = outcome[sl] * stop_slippage + outcome[timeout] * timeout_slippage
        funding_rate = _number(
            feature.get("funding_rate"), "funding_rate", allow_nan=True
        )
        minutes_to_funding = _number(
            feature.get("minutes_to_funding"),
            "minutes_to_funding",
            allow_nan=True,
        )
        scenario = _object(self.bundle.contract.get("scenario"), "scenario")
        horizon = int(_number(scenario.get("horizon_minutes"), "horizon"))
        direction = 1.0 if side == "LONG" else -1.0
        funding_cost = 0.0
        if (
            math.isfinite(funding_rate)
            and math.isfinite(minutes_to_funding)
            and 0 <= minutes_to_funding <= horizon
        ):
            funding_cost = max(direction * funding_rate * 10_000, 0.0)
        net_if_full = (
            gross_if_full - maker_fee - exit_fees - slippage - funding_cost
        )
        partial_cost = partial_fraction * (
            maker_fee + taker_fee + timeout_slippage
        )
        return float(
            fill[FILL_TO_INDEX["FULL_FILL"]] * net_if_full
            - fill[FILL_TO_INDEX["PARTIAL_FILL"]] * partial_cost
        )

    def score(
        self,
        feature: dict[str, object],
        *,
        symbol: str,
        side: str,
        entry_limit_price: float,
        stop_distance_bps: float,
        take_profit_distance_bps: float,
        stop_price: float,
        take_profit_price: float,
    ) -> ShadowCandidate:
        vector = self._feature_vector(
            feature,
            symbol=symbol,
            side=side,
            stop_distance_bps=stop_distance_bps,
            take_profit_distance_bps=take_profit_distance_bps,
        )
        fill_raw = np.asarray(
            cast(object, self.fill_model.predict(vector)), dtype=np.float64
        )
        outcome_raw = np.asarray(
            cast(object, self.outcome_model.predict(vector)), dtype=np.float64
        )
        if fill_raw.ndim == 1:
            fill_raw = fill_raw[None, :]
        if outcome_raw.ndim == 1:
            outcome_raw = outcome_raw[None, :]
        fill = _calibrate(fill_raw, self.fill_calibrator, FILL_NAMES)[0]
        outcome = _calibrate(
            outcome_raw, self.outcome_calibrator, EXECUTION_OUTCOME_NAMES
        )[0]
        timeout_estimate = _estimate(self.timeout_estimates, symbol, side)
        partial_estimate = _estimate(self.partial_estimates, symbol, side)
        expected = self._expected_net_bps(
            feature=feature,
            side=side,
            stop_distance_bps=stop_distance_bps,
            take_profit_distance_bps=take_profit_distance_bps,
            fill=fill,
            outcome=outcome,
            timeout_return_bps=timeout_estimate,
            partial_fraction=partial_estimate,
        )
        maker_fee = _number(self.evaluation.get("maker_fee_bps"), "maker fee")
        taker_fee = _number(self.evaluation.get("taker_fee_bps"), "taker fee")
        stop_slippage = _number(
            self.evaluation.get("stop_slippage_bps"), "stop slippage"
        )
        stressed_loss_fraction = (
            stop_distance_bps + maker_fee + taker_fee + stop_slippage
        ) / 10_000
        maximum_notional = _number(
            self.evaluation.get("max_notional_fraction"), "max notional fraction"
        )
        maximum_risk = _number(
            self.evaluation.get("max_planned_risk_fraction"), "max risk fraction"
        )
        notional_fraction = min(maximum_notional, maximum_risk / stressed_loss_fraction)
        return ShadowCandidate(
            decision_id=str(feature["decision_id"]),
            decision_at_ns=int(cast(int, feature["decision_at_ns"])),
            symbol=symbol,
            side=side,
            entry_limit_price=entry_limit_price,
            stop_distance_bps=stop_distance_bps,
            take_profit_distance_bps=take_profit_distance_bps,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            fill_probabilities=tuple(float(value) for value in fill),
            outcome_probabilities=tuple(float(value) for value in outcome),
            timeout_return_estimate_bps=timeout_estimate,
            partial_fill_fraction_estimate=partial_estimate,
            expected_net_bps=expected,
            notional_fraction=notional_fraction,
        )

    @property
    def minimum_expected_net_bps(self) -> float:
        return _number(
            self.evaluation.get("minimum_expected_net_bps"),
            "minimum expected net bps",
        )

    @property
    def rolling_24h_loss_fraction(self) -> float:
        return _number(
            self.evaluation.get("rolling_24h_loss_fraction"),
            "rolling loss fraction",
        )
