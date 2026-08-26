from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tradingbot.research.evaluation_contracts import (
    NS_PER_DAY,
    NS_PER_MINUTE,
    EvaluationError,
    EvaluationParameters,
)
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_OUTCOME_NAMES,
    EXECUTION_OUTCOME_TO_INDEX,
    FILL_NAMES,
    FILL_TO_INDEX,
    ExecutionPredictionBatch,
    ExecutionPreparedData,
)


@dataclass(frozen=True, slots=True)
class CombinedExecutionPredictions:
    row_indices: NDArray[np.int64]
    folds: NDArray[np.int16]
    fill_probabilities: NDArray[np.float64]
    outcome_probabilities: NDArray[np.float64]
    expected_net_bps: NDArray[np.float64]


def _decision_id_text(value: np.bytes_) -> str:
    return bytes(value).decode("ascii")


def _group_estimate(
    *,
    observed: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
    symbols_train: NDArray[np.int16],
    sides_train: NDArray[np.int8],
    symbols_test: NDArray[np.int16],
    sides_test: NDArray[np.int8],
    minimum_group_rows: int = 20,
) -> NDArray[np.float64]:
    if not np.any(observed_mask):
        raise EvaluationError("training fold has no required execution observations")
    global_mean = float(np.mean(observed[observed_mask]))
    side_means: dict[int, float] = {}
    group_means: dict[tuple[int, int], float] = {}
    for side in (-1, 1):
        mask = observed_mask & (sides_train == side)
        if np.any(mask):
            side_means[side] = float(np.mean(observed[mask]))
    for symbol in np.unique(symbols_train):
        for side in (-1, 1):
            mask = (
                observed_mask
                & (symbols_train == symbol)
                & (sides_train == side)
            )
            if np.count_nonzero(mask) >= minimum_group_rows:
                group_means[(int(symbol), side)] = float(np.mean(observed[mask]))
    result = np.empty(len(symbols_test), dtype=np.float64)
    for index, (symbol, side) in enumerate(
        zip(symbols_test, sides_test, strict=True)
    ):
        result[index] = group_means.get(
            (int(symbol), int(side)),
            side_means.get(int(side), global_mean),
        )
    return result


def timeout_return_estimate(
    data: ExecutionPreparedData,
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    timeout_index = EXECUTION_OUTCOME_TO_INDEX["TIMEOUT"]
    observed_mask = data.outcome_y[train_indices] == timeout_index
    return _group_estimate(
        observed=data.outcome_return_bps[train_indices],
        observed_mask=observed_mask,
        symbols_train=data.symbol_codes[train_indices],
        sides_train=data.side_codes[train_indices],
        symbols_test=data.symbol_codes[test_indices],
        sides_test=data.side_codes[test_indices],
    )


def partial_fraction_estimate(
    data: ExecutionPreparedData,
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    partial_index = FILL_TO_INDEX["PARTIAL_FILL"]
    observed_mask = data.fill_y[train_indices] == partial_index
    estimates = _group_estimate(
        observed=data.fill_fraction[train_indices],
        observed_mask=observed_mask,
        symbols_train=data.symbol_codes[train_indices],
        sides_train=data.side_codes[train_indices],
        symbols_test=data.symbol_codes[test_indices],
        sides_test=data.side_codes[test_indices],
        minimum_group_rows=10,
    )
    return np.clip(estimates, 0.0, 1.0)


def expected_execution_net_bps(
    data: ExecutionPreparedData,
    row_indices: NDArray[np.int64],
    fill_probabilities: NDArray[np.float64],
    outcome_probabilities: NDArray[np.float64],
    timeout_returns_bps: NDArray[np.float64],
    partial_fractions: NDArray[np.float64],
    parameters: EvaluationParameters,
) -> NDArray[np.float64]:
    expected_fill_shape = (len(row_indices), len(FILL_NAMES))
    expected_outcome_shape = (len(row_indices), len(EXECUTION_OUTCOME_NAMES))
    if fill_probabilities.shape != expected_fill_shape:
        raise EvaluationError("fill probabilities have an invalid shape")
    if outcome_probabilities.shape != expected_outcome_shape:
        raise EvaluationError("outcome probabilities have an invalid shape")
    if len(timeout_returns_bps) != len(row_indices) or len(partial_fractions) != len(
        row_indices
    ):
        raise EvaluationError("execution expected-return estimates are incompatible")

    full_index = FILL_TO_INDEX["FULL_FILL"]
    partial_index = FILL_TO_INDEX["PARTIAL_FILL"]
    sl_index = EXECUTION_OUTCOME_TO_INDEX["SL_FIRST"]
    timeout_index = EXECUTION_OUTCOME_TO_INDEX["TIMEOUT"]
    tp_index = EXECUTION_OUTCOME_TO_INDEX["TP_FIRST"]
    gross_if_full = (
        outcome_probabilities[:, tp_index]
        * data.take_profit_distance_bps[row_indices]
        - outcome_probabilities[:, sl_index]
        * data.stop_distance_bps[row_indices]
        + outcome_probabilities[:, timeout_index] * timeout_returns_bps
    )
    exit_fees_if_full = (
        outcome_probabilities[:, tp_index] * parameters.maker_fee_bps
        + (
            outcome_probabilities[:, sl_index]
            + outcome_probabilities[:, timeout_index]
        )
        * parameters.taker_fee_bps
    )
    slippage_if_full = (
        outcome_probabilities[:, sl_index] * parameters.stop_slippage_bps
        + outcome_probabilities[:, timeout_index]
        * parameters.timeout_slippage_bps
    )
    funding_payment_bps = (
        data.side_codes[row_indices].astype(np.float64)
        * data.funding_rate[row_indices]
        * 10_000
    )
    funding_in_horizon = (
        np.isfinite(data.minutes_to_funding[row_indices])
        & (data.minutes_to_funding[row_indices] >= 0)
        & (data.minutes_to_funding[row_indices] <= data.horizon_minutes)
    )
    conservative_funding_if_full = np.where(
        funding_in_horizon,
        np.maximum(funding_payment_bps, 0.0),
        0.0,
    )
    net_if_full = (
        gross_if_full
        - parameters.maker_fee_bps
        - exit_fees_if_full
        - slippage_if_full
        - conservative_funding_if_full
    )
    partial_unwind_cost = partial_fractions * (
        parameters.maker_fee_bps
        + parameters.taker_fee_bps
        + parameters.timeout_slippage_bps
    )
    return (
        fill_probabilities[:, full_index] * net_if_full
        - fill_probabilities[:, partial_index] * partial_unwind_cost
    )


def combine_execution_prediction_batches(
    batches: list[ExecutionPredictionBatch],
    *,
    model_name: str,
) -> CombinedExecutionPredictions:
    selected = [batch for batch in batches if batch.model_name == model_name]
    if not selected:
        raise EvaluationError(f"no execution predictions exist for {model_name}")
    row_indices = np.concatenate([batch.row_indices for batch in selected])
    if len(np.unique(row_indices)) != len(row_indices):
        raise EvaluationError(f"{model_name} has duplicate execution test rows")
    return CombinedExecutionPredictions(
        row_indices=row_indices.astype(np.int64, copy=False),
        folds=np.concatenate(
            [
                np.full(len(batch.row_indices), batch.fold, dtype=np.int16)
                for batch in selected
            ]
        ),
        fill_probabilities=np.concatenate(
            [batch.fill_probabilities for batch in selected]
        ).astype(np.float64, copy=False),
        outcome_probabilities=np.concatenate(
            [batch.outcome_probabilities for batch in selected]
        ).astype(np.float64, copy=False),
        expected_net_bps=np.concatenate(
            [batch.expected_net_bps for batch in selected]
        ).astype(np.float64, copy=False),
    )


def _position_notional_fraction(
    stop_distance_bps: float,
    parameters: EvaluationParameters,
) -> float:
    stressed_loss_bps = (
        stop_distance_bps
        + parameters.maker_fee_bps
        + parameters.taker_fee_bps
        + parameters.stop_slippage_bps
    )
    if stressed_loss_bps <= 0:
        raise EvaluationError("stressed execution loss must be positive")
    risk_limited = parameters.max_planned_risk_fraction / (
        stressed_loss_bps / 10_000
    )
    return min(parameters.max_notional_fraction, risk_limited)


def _actual_full_fill_costs(
    data: ExecutionPreparedData,
    row: int,
    parameters: EvaluationParameters,
) -> tuple[int, float, float, float, float]:
    outcome_index = int(data.outcome_y[row])
    if not 0 <= outcome_index < len(EXECUTION_OUTCOME_NAMES):
        raise EvaluationError("full-fill row has no priced execution outcome")
    outcome = EXECUTION_OUTCOME_NAMES[outcome_index]
    exit_at_ns = int(data.hit_at_ns[row])
    if exit_at_ns < 0:
        exit_at_ns = int(data.position_end_ns[row])
    exit_fee = (
        parameters.maker_fee_bps
        if outcome == "TP_FIRST"
        else parameters.taker_fee_bps
    )
    fee_bps = parameters.maker_fee_bps + exit_fee
    slippage_bps = 0.0
    if outcome == "SL_FIRST":
        slippage_bps = parameters.stop_slippage_bps
    elif outcome == "TIMEOUT":
        slippage_bps = parameters.timeout_slippage_bps

    funding_cost_bps = 0.0
    minutes = float(data.minutes_to_funding[row])
    rate = float(data.funding_rate[row])
    if np.isfinite(minutes) and np.isfinite(rate) and 0 <= minutes <= (
        data.horizon_minutes
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


def run_execution_one_position_backtest(
    data: ExecutionPreparedData,
    predictions: CombinedExecutionPredictions,
    parameters: EvaluationParameters,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select one maker order per minute and honor its observed fill lifecycle."""

    if len(predictions.row_indices) == 0:
        raise EvaluationError("cannot backtest empty execution predictions")
    order = np.lexsort(
        (
            data.side_codes[predictions.row_indices],
            data.symbol_codes[predictions.row_indices],
            data.decision_at_ns[predictions.row_indices],
        )
    )
    rows = predictions.row_indices[order]
    folds = predictions.folds[order]
    fill_probabilities = predictions.fill_probabilities[order]
    outcome_probabilities = predictions.outcome_probabilities[order]
    expected = predictions.expected_net_bps[order]

    equity = 1.0
    peak_equity = 1.0
    maximum_drawdown = 0.0
    active_until_ns = -1
    realized_24h: deque[tuple[int, float]] = deque()
    rolling_return = 0.0
    skip_counts: Counter[str] = Counter()
    fill_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    attempts: list[dict[str, object]] = []
    pnl_values: list[float] = []
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
            skip_counts["position_or_order_already_open"] += 1
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
        ranked_expected = np.sort(group_expected[eligible_offsets])[::-1]
        expected_margin = (
            None
            if len(ranked_expected) < 2
            else float(ranked_expected[0] - ranked_expected[1])
        )
        selected_position = position + best_offset
        row = int(rows[selected_position])
        fill_status = FILL_NAMES[int(data.fill_y[row])]
        fill_counts[fill_status] += 1
        notional_fraction = _position_notional_fraction(
            float(data.stop_distance_bps[row]), parameters
        )
        equity_before = equity
        exit_at_ns = int(data.entry_window_end_ns[row])
        outcome = fill_status
        fee_bps = 0.0
        slippage_bps = 0.0
        funding_cost_bps = 0.0
        gross_return_bps = 0.0
        net_return_bps = 0.0
        realized_fraction = 0.0

        if fill_status == "FULL_FILL":
            (
                exit_at_ns,
                fee_bps,
                slippage_bps,
                funding_cost_bps,
                net_return_bps,
            ) = _actual_full_fill_costs(data, row, parameters)
            gross_return_bps = float(data.outcome_return_bps[row])
            outcome = EXECUTION_OUTCOME_NAMES[int(data.outcome_y[row])]
            outcome_counts[outcome] += 1
            realized_fraction = notional_fraction * net_return_bps / 10_000
            active_until_ns = exit_at_ns
        elif fill_status == "PARTIAL_FILL":
            fee_bps = parameters.maker_fee_bps + parameters.taker_fee_bps
            slippage_bps = parameters.timeout_slippage_bps
            net_return_bps = -(fee_bps + slippage_bps)
            realized_fraction = (
                notional_fraction
                * float(data.fill_fraction[row])
                * net_return_bps
                / 10_000
            )
            outcome = "PARTIAL_UNWIND"
            active_until_ns = exit_at_ns
        else:
            active_until_ns = exit_at_ns

        if realized_fraction:
            equity *= 1 + realized_fraction
            peak_equity = max(peak_equity, equity)
            maximum_drawdown = max(maximum_drawdown, 1 - equity / peak_equity)
            realized_24h.append((exit_at_ns, realized_fraction))
            rolling_return += realized_fraction
            pnl_values.append(realized_fraction)

        fill_probability = fill_probabilities[selected_position]
        outcome_probability = outcome_probabilities[selected_position]
        attempts.append(
            {
                "decision_id": _decision_id_text(data.decision_ids[row]),
                "fold": int(folds[selected_position]),
                "decision_at_ns": decision_ns,
                "exit_at_ns": exit_at_ns,
                "symbol": data.symbols[int(data.symbol_codes[row])],
                "side": "LONG" if int(data.side_codes[row]) == 1 else "SHORT",
                "horizon_minutes": data.horizon_minutes,
                "order_notional_usdt": data.order_notional_usdt,
                "fill_status": fill_status,
                "fill_fraction": float(data.fill_fraction[row]),
                "outcome": outcome,
                "probability_no_fill": float(
                    fill_probability[FILL_TO_INDEX["NO_FILL"]]
                ),
                "probability_partial_fill": float(
                    fill_probability[FILL_TO_INDEX["PARTIAL_FILL"]]
                ),
                "probability_full_fill": float(
                    fill_probability[FILL_TO_INDEX["FULL_FILL"]]
                ),
                "probability_sl_first": float(
                    outcome_probability[EXECUTION_OUTCOME_TO_INDEX["SL_FIRST"]]
                ),
                "probability_timeout": float(
                    outcome_probability[EXECUTION_OUTCOME_TO_INDEX["TIMEOUT"]]
                ),
                "probability_tp_first": float(
                    outcome_probability[EXECUTION_OUTCOME_TO_INDEX["TP_FIRST"]]
                ),
                "expected_net_bps": float(expected[selected_position]),
                "candidate_count": group_end - position,
                "eligible_candidate_count": len(eligible_offsets),
                "expected_margin_to_second_bps": expected_margin,
                "gross_return_bps": gross_return_bps,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "funding_cost_bps": funding_cost_bps,
                "net_return_bps": net_return_bps,
                "notional_fraction": notional_fraction,
                "realized_equity_return_fraction": (
                    0.0
                    if equity_before == 0
                    else equity / equity_before - 1.0
                ),
                "equity_before": equity_before,
                "equity_after": equity,
            }
        )
        position = group_end

    positive = sum(value for value in pnl_values if value > 0)
    negative = -sum(value for value in pnl_values if value < 0)
    full_trades = fill_counts["FULL_FILL"]
    summary: dict[str, object] = {
        "candidate_rows": len(rows),
        "unique_decisions": int(len(np.unique(data.decision_at_ns[rows]))),
        "order_attempts": len(attempts),
        "full_fill_trades": full_trades,
        "partial_unwinds": fill_counts["PARTIAL_FILL"],
        "no_fills": fill_counts["NO_FILL"],
        "observed_fill_statuses": dict(sorted(fill_counts.items())),
        "observed_full_fill_outcomes": dict(sorted(outcome_counts.items())),
        "total_equity_return_fraction": equity - 1,
        "ending_equity_multiple": equity,
        "maximum_drawdown_fraction": maximum_drawdown,
        "realized_position_win_rate": (
            0.0
            if not pnl_values
            else sum(value > 0 for value in pnl_values) / len(pnl_values)
        ),
        "profit_factor": None if negative == 0 else positive / negative,
        "average_full_fill_net_return_bps": (
            None
            if full_trades == 0
            else float(
                np.mean(
                    [
                        item["net_return_bps"]
                        for item in attempts
                        if item["fill_status"] == "FULL_FILL"
                    ]
                )
            )
        ),
        "skipped_decisions": dict(sorted(skip_counts.items())),
        "maker_fill_modeled": True,
        "visible_queue_modeled": True,
        "partial_fill_policy": "cancel_residual_and_taker_unwind_filled_fraction",
        "one_position_enforced": True,
    }
    return summary, attempts
