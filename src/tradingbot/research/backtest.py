from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tradingbot.research.evaluation_contracts import (
    NS_PER_DAY,
    NS_PER_MINUTE,
    OUTCOME_NAMES,
    EvaluationError,
    EvaluationParameters,
    PredictionBatch,
    PreparedData,
)


@dataclass(frozen=True, slots=True)
class CombinedPredictions:
    row_indices: NDArray[np.int64]
    folds: NDArray[np.int16]
    probabilities: NDArray[np.float64]
    expected_net_bps: NDArray[np.float64]


def expected_net_returns_bps(
    data: PreparedData,
    row_indices: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    timeout_returns_bps: NDArray[np.float64],
    parameters: EvaluationParameters,
) -> NDArray[np.float64]:
    if probabilities.shape != (len(row_indices), len(OUTCOME_NAMES)):
        raise EvaluationError("expected-return probabilities have an invalid shape")
    if len(timeout_returns_bps) != len(row_indices):
        raise EvaluationError("timeout-return estimates have an invalid shape")
    sl_index = OUTCOME_NAMES.index("SL_FIRST")
    timeout_index = OUTCOME_NAMES.index("TIMEOUT")
    tp_index = OUTCOME_NAMES.index("TP_FIRST")
    gross = (
        probabilities[:, tp_index] * data.take_profit_distance_bps[row_indices]
        - probabilities[:, sl_index] * data.stop_distance_bps[row_indices]
        + probabilities[:, timeout_index] * timeout_returns_bps
    )
    exit_fees = (
        probabilities[:, tp_index] * parameters.maker_fee_bps
        + (probabilities[:, sl_index] + probabilities[:, timeout_index])
        * parameters.taker_fee_bps
    )
    slippage = (
        parameters.entry_adverse_selection_bps
        + probabilities[:, sl_index] * parameters.stop_slippage_bps
        + probabilities[:, timeout_index] * parameters.timeout_slippage_bps
    )
    funding_payment_bps = (
        data.side_codes[row_indices].astype(np.float64)
        * data.funding_rate[row_indices]
        * 10_000
    )
    funding_in_horizon = (
        np.isfinite(data.minutes_to_funding[row_indices])
        & (data.minutes_to_funding[row_indices] >= 0)
        & (data.minutes_to_funding[row_indices] <= parameters.horizon_minutes)
    )
    conservative_funding = np.where(
        funding_in_horizon,
        np.maximum(funding_payment_bps, 0.0) * probabilities[:, timeout_index],
        0.0,
    )
    return (
        gross
        - parameters.maker_fee_bps
        - exit_fees
        - slippage
        - conservative_funding
    )


def combine_prediction_batches(
    batches: list[PredictionBatch], *, model_name: str
) -> CombinedPredictions:
    selected = [batch for batch in batches if batch.model_name == model_name]
    if not selected:
        raise EvaluationError(f"no predictions were produced for {model_name}")
    row_indices = np.concatenate([batch.row_indices for batch in selected])
    probabilities = np.concatenate([batch.probabilities for batch in selected])
    expected = np.concatenate([batch.expected_net_bps for batch in selected])
    folds = np.concatenate(
        [np.full(len(batch.row_indices), batch.fold, dtype=np.int16) for batch in selected]
    )
    if len(np.unique(row_indices)) != len(row_indices):
        raise EvaluationError(f"{model_name} predictions contain duplicate test rows")
    return CombinedPredictions(
        row_indices=row_indices.astype(np.int64, copy=False),
        folds=folds,
        probabilities=probabilities.astype(np.float64, copy=False),
        expected_net_bps=expected.astype(np.float64, copy=False),
    )


def _position_notional_fraction(
    stop_distance_bps: float, parameters: EvaluationParameters
) -> float:
    stressed_loss_bps = (
        stop_distance_bps
        + parameters.maker_fee_bps
        + parameters.taker_fee_bps
        + parameters.entry_adverse_selection_bps
        + parameters.stop_slippage_bps
    )
    if stressed_loss_bps <= 0:
        raise EvaluationError("stressed position loss must be positive")
    risk_limited = parameters.max_planned_risk_fraction / (
        stressed_loss_bps / 10_000
    )
    return min(parameters.max_notional_fraction, risk_limited)


def _actual_costs(
    data: PreparedData, row: int, parameters: EvaluationParameters
) -> tuple[int, float, float, float, float]:
    outcome = OUTCOME_NAMES[int(data.y[row])]
    exit_at_ns = int(data.hit_at_ns[row])
    if exit_at_ns < 0:
        exit_at_ns = int(data.label_end_ns[row])
    exit_fee = (
        parameters.maker_fee_bps
        if outcome == "TP_FIRST"
        else parameters.taker_fee_bps
    )
    fee_bps = parameters.maker_fee_bps + exit_fee
    slippage_bps = parameters.entry_adverse_selection_bps
    if outcome == "SL_FIRST":
        slippage_bps += parameters.stop_slippage_bps
    elif outcome == "TIMEOUT":
        slippage_bps += parameters.timeout_slippage_bps

    funding_cost_bps = 0.0
    minutes = float(data.minutes_to_funding[row])
    rate = float(data.funding_rate[row])
    if math_is_finite(minutes) and math_is_finite(rate) and 0 <= minutes <= (
        parameters.horizon_minutes
    ):
        funding_at_ns = int(data.decision_at_ns[row] + minutes * NS_PER_MINUTE)
        if exit_at_ns >= funding_at_ns:
            funding_cost_bps = float(data.side_codes[row]) * rate * 10_000
    net_bps = (
        float(data.outcome_return_bps[row])
        - fee_bps
        - slippage_bps
        - funding_cost_bps
    )
    return exit_at_ns, fee_bps, slippage_bps, funding_cost_bps, net_bps


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def run_one_position_backtest(
    data: PreparedData,
    predictions: CombinedPredictions,
    parameters: EvaluationParameters,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Rank candidates while enforcing one position and a rolling 24h loss stop."""

    if len(predictions.row_indices) == 0:
        raise EvaluationError("cannot backtest an empty prediction set")
    order = np.lexsort(
        (
            data.side_codes[predictions.row_indices],
            data.symbol_codes[predictions.row_indices],
            data.decision_at_ns[predictions.row_indices],
        )
    )
    rows = predictions.row_indices[order]
    folds = predictions.folds[order]
    probabilities = predictions.probabilities[order]
    expected = predictions.expected_net_bps[order]

    equity = 1.0
    peak_equity = 1.0
    maximum_drawdown = 0.0
    active_until_ns = -1
    realized_24h: deque[tuple[int, float]] = deque()
    rolling_return = 0.0
    skip_counts: Counter[str] = Counter()
    trades: list[dict[str, object]] = []
    pnl_values: list[float] = []
    outcome_counts: Counter[str] = Counter()
    position = 0
    while position < len(rows):
        decision_ns = int(data.decision_at_ns[rows[position]])
        group_end = position + 1
        while (
            group_end < len(rows)
            and int(data.decision_at_ns[rows[group_end]]) == decision_ns
        ):
            group_end += 1

        while realized_24h and realized_24h[0][0] < decision_ns - NS_PER_DAY:
            _, expired_return = realized_24h.popleft()
            rolling_return -= expired_return
        if decision_ns < active_until_ns:
            skip_counts["position_already_open"] += 1
            position = group_end
            continue
        if rolling_return <= -parameters.rolling_24h_loss_fraction:
            skip_counts["rolling_24h_loss_limit"] += 1
            position = group_end
            continue

        group_expected = expected[position:group_end]
        eligible_offsets = np.flatnonzero(
            group_expected >= parameters.minimum_expected_net_bps
        )
        if len(eligible_offsets) == 0:
            skip_counts["below_expected_return_threshold"] += 1
            position = group_end
            continue
        best_offset = min(
            (int(offset) for offset in eligible_offsets),
            key=lambda offset: (
                -float(group_expected[offset]),
                int(data.symbol_codes[rows[position + offset]]),
                -int(data.side_codes[rows[position + offset]]),
            ),
        )
        eligible_expected = np.sort(group_expected[eligible_offsets])[::-1]
        expected_margin_to_second = (
            None
            if len(eligible_expected) < 2
            else float(eligible_expected[0] - eligible_expected[1])
        )
        selected_position = position + best_offset
        row = int(rows[selected_position])
        exit_at_ns, fee_bps, slippage_bps, funding_bps, net_bps = _actual_costs(
            data, row, parameters
        )
        notional_fraction = _position_notional_fraction(
            float(data.stop_distance_bps[row]), parameters
        )
        equity_before = equity
        equity_return = notional_fraction * net_bps / 10_000
        equity *= 1 + equity_return
        peak_equity = max(peak_equity, equity)
        maximum_drawdown = max(maximum_drawdown, 1 - equity / peak_equity)
        active_until_ns = exit_at_ns
        realized_24h.append((exit_at_ns, equity_return))
        rolling_return += equity_return
        pnl_values.append(equity_return)
        outcome = OUTCOME_NAMES[int(data.y[row])]
        outcome_counts[outcome] += 1
        trades.append(
            {
                "decision_id": str(data.decision_ids[row]),
                "fold": int(folds[selected_position]),
                "decision_at_ns": decision_ns,
                "exit_at_ns": exit_at_ns,
                "symbol": data.symbols[int(data.symbol_codes[row])],
                "side": "LONG" if int(data.side_codes[row]) == 1 else "SHORT",
                "outcome": outcome,
                "probability_sl_first": float(probabilities[selected_position, 0]),
                "probability_timeout": float(probabilities[selected_position, 1]),
                "probability_tp_first": float(probabilities[selected_position, 2]),
                "expected_net_bps": float(expected[selected_position]),
                "candidate_count": group_end - position,
                "eligible_candidate_count": len(eligible_offsets),
                "expected_margin_to_second_bps": expected_margin_to_second,
                "gross_return_bps": float(data.outcome_return_bps[row]),
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "funding_cost_bps": funding_bps,
                "net_return_bps": net_bps,
                "notional_fraction": notional_fraction,
                "equity_before": equity_before,
                "equity_after": equity,
            }
        )
        position = group_end

    positive = sum(value for value in pnl_values if value > 0)
    negative = -sum(value for value in pnl_values if value < 0)
    summary: dict[str, object] = {
        "candidate_rows": len(rows),
        "unique_decisions": int(len(np.unique(data.decision_at_ns[rows]))),
        "trades": len(trades),
        "total_equity_return_fraction": equity - 1,
        "ending_equity_multiple": equity,
        "maximum_drawdown_fraction": maximum_drawdown,
        "win_rate": (
            0.0 if not pnl_values else sum(value > 0 for value in pnl_values) / len(pnl_values)
        ),
        "profit_factor": None if negative == 0 else positive / negative,
        "average_net_return_bps": (
            None if not trades else float(np.mean([t["net_return_bps"] for t in trades]))
        ),
        "outcomes": dict(sorted(outcome_counts.items())),
        "skipped_decisions": dict(sorted(skip_counts.items())),
        "conditional_entry_assumption": True,
        "maker_fill_modeled": False,
        "queue_position_modeled": False,
    }
    return summary, trades
