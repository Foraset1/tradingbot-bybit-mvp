from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tradingbot.research.evaluation_contracts import OUTCOME_NAMES

Trade = dict[str, object]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("trade diagnostic value must be numeric")
    return float(value)


def _float_values(trades: list[Trade], name: str) -> NDArray[np.float64]:
    return np.asarray([_number(trade[name]) for trade in trades], dtype=np.float64)


def _equity_return(trades: list[Trade], return_field: str) -> float:
    equity = 1.0
    for trade in trades:
        equity *= 1.0 + (
            _number(trade["notional_fraction"])
            * _number(trade[return_field])
            / 10_000
        )
    return equity - 1.0


def _summary(trades: list[Trade]) -> dict[str, object]:
    if not trades:
        return {
            "trades": 0,
            "average_expected_net_bps": None,
            "average_gross_return_bps": None,
            "average_net_return_bps": None,
            "actual_minus_expected_bps": None,
            "win_rate": None,
            "profit_factor": None,
            "total_equity_return_fraction": 0.0,
            "zero_cost_same_trades_equity_return_fraction": 0.0,
        }
    expected = _float_values(trades, "expected_net_bps")
    gross = _float_values(trades, "gross_return_bps")
    net = _float_values(trades, "net_return_bps")
    notionals = _float_values(trades, "notional_fraction")
    equity_returns = notionals * net / 10_000
    positive = float(np.sum(equity_returns[equity_returns > 0]))
    negative = float(-np.sum(equity_returns[equity_returns < 0]))
    return {
        "trades": len(trades),
        "average_expected_net_bps": float(np.mean(expected)),
        "average_gross_return_bps": float(np.mean(gross)),
        "average_fee_bps": float(np.mean(_float_values(trades, "fee_bps"))),
        "average_slippage_bps": float(
            np.mean(_float_values(trades, "slippage_bps"))
        ),
        "average_funding_cost_bps": float(
            np.mean(_float_values(trades, "funding_cost_bps"))
        ),
        "average_net_return_bps": float(np.mean(net)),
        "actual_minus_expected_bps": float(np.mean(net - expected)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": None if negative == 0 else positive / negative,
        "total_equity_return_fraction": _equity_return(trades, "net_return_bps"),
        "zero_cost_same_trades_equity_return_fraction": _equity_return(
            trades, "gross_return_bps"
        ),
        "outcomes": {
            name: sum(trade["outcome"] == name for trade in trades)
            for name in OUTCOME_NAMES
        },
    }


def _grouped(
    trades: list[Trade], key: Callable[[Trade], str]
) -> dict[str, dict[str, object]]:
    groups: defaultdict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade)
    return {name: _summary(groups[name]) for name in sorted(groups)}


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _correlation(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _expected_bins(trades: list[Trade]) -> dict[str, dict[str, object]]:
    boundaries = (2.0, 5.0, 10.0, 20.0)
    labels = ("below_2", "2_to_5", "5_to_10", "10_to_20", "20_plus")
    groups: dict[str, list[Trade]] = {label: [] for label in labels}
    for trade in trades:
        value = _number(trade["expected_net_bps"])
        index = int(np.searchsorted(boundaries, value, side="right"))
        groups[labels[index]].append(trade)
    return {label: _summary(groups[label]) for label in labels}


def _expected_deciles(trades: list[Trade]) -> dict[str, dict[str, object]]:
    if not trades:
        return {}
    ordered = sorted(trades, key=lambda trade: _number(trade["expected_net_bps"]))
    groups = np.array_split(np.asarray(ordered, dtype=object), 10)
    return {
        f"decile_{index:02d}": _summary(list(group))
        for index, group in enumerate(groups, start=1)
        if len(group)
    }


def trade_diagnostics(trades: list[Trade]) -> dict[str, object]:
    """Create machine-readable diagnostics for selected, non-independent trades."""

    expected = _float_values(trades, "expected_net_bps") if trades else np.array([])
    actual = _float_values(trades, "net_return_bps") if trades else np.array([])
    selected_probability_report: dict[str, object] = {}
    if trades:
        probability_fields = (
            "probability_sl_first",
            "probability_timeout",
            "probability_tp_first",
        )
        selected_probability_report = {
            name: {
                "mean_predicted_probability": float(
                    np.mean(_float_values(trades, probability_fields[index]))
                ),
                "observed_fraction": float(
                    np.mean([trade["outcome"] == name for trade in trades])
                ),
            }
            for index, name in enumerate(OUTCOME_NAMES)
        }
    candidate_counts = (
        _float_values(trades, "candidate_count") if trades else np.array([])
    )
    eligible_counts = (
        _float_values(trades, "eligible_candidate_count")
        if trades
        else np.array([])
    )
    margins = np.asarray(
        [
            _number(trade["expected_margin_to_second_bps"])
            for trade in trades
            if trade["expected_margin_to_second_bps"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "overall": _summary(trades),
        "by_fold": _grouped(trades, lambda trade: str(trade["fold"])),
        "by_symbol": _grouped(trades, lambda trade: str(trade["symbol"])),
        "by_side": _grouped(trades, lambda trade: str(trade["side"])),
        "by_outcome": _grouped(trades, lambda trade: str(trade["outcome"])),
        "by_expected_net_bin": _expected_bins(trades),
        "by_expected_net_decile": _expected_deciles(trades),
        "selected_probability_calibration": selected_probability_report,
        "expected_vs_actual": {
            "pearson_correlation": _correlation(expected, actual),
            "spearman_correlation": _correlation(_rank(expected), _rank(actual)),
            "actual_minus_expected_bps": (
                None if not trades else float(np.mean(actual - expected))
            ),
        },
        "selection": {
            "mean_candidates_per_selected_decision": (
                None if not trades else float(np.mean(candidate_counts))
            ),
            "mean_eligible_candidates_per_selected_decision": (
                None if not trades else float(np.mean(eligible_counts))
            ),
            "mean_winner_margin_to_second_bps": (
                None if len(margins) == 0 else float(np.mean(margins))
            ),
            "max_candidate_selection_bias_audited": True,
            "warning": (
                "candidate rows share the same market minute; selected-trade metrics "
                "must not be interpreted as independent classification samples"
            ),
        },
    }
