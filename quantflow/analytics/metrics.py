from __future__ import annotations

import numpy as np


def total_return(equity_curve: list[float]) -> float:
    """Total return as a fraction (0.3 = 30%)."""
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]


def cagr(equity_curve: list[float], periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    total = equity_curve[-1] / equity_curve[0]
    n_years = (len(equity_curve) - 1) / periods_per_year
    if n_years <= 0:
        return 0.0
    if total <= 0:
        return -1.0
    return total ** (1.0 / n_years) - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a negative fraction (-0.2 = 20% drawdown). Returns 0.0 if no drawdown."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    worst_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        if dd < worst_dd:
            worst_dd = dd
    return worst_dd


def sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    if len(equity_curve) < 2:
        return 0.0
    prices = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]
    if len(returns) == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    if len(equity_curve) < 2:
        return 0.0
    prices = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]
    if len(returns) == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        downside_std = 0.0
    else:
        downside_std = float(np.std(downside, ddof=1))

    if downside_std == 0:
        return float(np.mean(excess) * np.sqrt(periods_per_year)) if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def win_rate(pnls: list[float]) -> float:
    """Fraction of trades that were profitable."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_win_loss_ratio(pnls: list[float]) -> float:
    """Average winning trade / average losing trade."""
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    if not wins or not losses:
        return 0.0
    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))
