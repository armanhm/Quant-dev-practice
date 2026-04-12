# quantflow/analytics/tearsheet.py
from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from quantflow.backtest.engine import BacktestResult
from quantflow.analytics.metrics import (
    total_return,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    profit_factor,
    avg_win_loss_ratio,
)


def print_tearsheet(result: BacktestResult) -> dict[str, float]:
    """Print performance summary to console and return metrics dict."""
    pnls = [t.pnl for t in result.trades]

    metrics = {
        "Total Return": total_return(result.equity_curve),
        "CAGR": cagr(result.equity_curve),
        "Max Drawdown": max_drawdown(result.equity_curve),
        "Sharpe Ratio": sharpe_ratio(result.equity_curve),
        "Sortino Ratio": sortino_ratio(result.equity_curve),
        "Win Rate": win_rate(pnls),
        "Profit Factor": profit_factor(pnls),
        "Avg Win/Loss": avg_win_loss_ratio(pnls),
        "Total Trades": len(result.trades),
        "Final Equity": result.equity_curve[-1] if result.equity_curve else 0.0,
    }

    print("\n" + "=" * 50)
    print("       QUANTFLOW BACKTEST TEARSHEET")
    print("=" * 50)
    print(f"  Initial Capital:   ${result.initial_cash:>14,.2f}")
    print(f"  Final Equity:      ${metrics['Final Equity']:>14,.2f}")
    print("-" * 50)
    print(f"  Total Return:      {metrics['Total Return']:>14.2%}")
    print(f"  CAGR:              {metrics['CAGR']:>14.2%}")
    print(f"  Max Drawdown:      {metrics['Max Drawdown']:>14.2%}")
    print("-" * 50)
    print(f"  Sharpe Ratio:      {metrics['Sharpe Ratio']:>14.2f}")
    print(f"  Sortino Ratio:     {metrics['Sortino Ratio']:>14.2f}")
    print("-" * 50)
    print(f"  Total Trades:      {metrics['Total Trades']:>14d}")
    print(f"  Win Rate:          {metrics['Win Rate']:>14.2%}")
    print(f"  Profit Factor:     {metrics['Profit Factor']:>14.2f}")
    print(f"  Avg Win/Loss:      {metrics['Avg Win/Loss']:>14.2f}")
    print("=" * 50)

    return metrics


def plot_tearsheet(result: BacktestResult, save_path: str | None = None) -> None:
    """Plot equity curve, drawdown, and trade markers."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    dates = result.timestamps
    equity = np.array(result.equity_curve)
    benchmark = np.array(result.benchmark_equity)

    # --- Equity Curve ---
    ax1 = axes[0]
    ax1.plot(dates, equity, label="Strategy", color="#2196F3", linewidth=1.5)
    ax1.plot(dates, benchmark, label="Buy & Hold", color="#9E9E9E", linewidth=1, linestyle="--")
    ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Drawdown ---
    ax2 = axes[1]
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    ax2.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.3)
    ax2.plot(dates, drawdown, color="#F44336", linewidth=1)
    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {save_path}")
    else:
        plt.show()
