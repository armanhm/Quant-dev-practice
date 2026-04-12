import pytest
import numpy as np
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


class TestReturnMetrics:
    def test_total_return(self):
        equity = [100_000, 110_000, 120_000, 115_000, 130_000]
        assert total_return(equity) == pytest.approx(0.3)  # 30%

    def test_total_return_negative(self):
        equity = [100_000, 90_000, 80_000]
        assert total_return(equity) == pytest.approx(-0.2)

    def test_total_return_single_point(self):
        assert total_return([100_000]) == 0.0

    def test_cagr(self):
        # $100k -> $200k over 3 years
        equity = [100_000] + [0] * 755 + [200_000]  # ~3 years of daily bars
        result = cagr(equity, periods_per_year=252)
        assert 0.20 < result < 0.30  # roughly 26% CAGR


class TestRiskMetrics:
    def test_max_drawdown(self):
        equity = [100, 110, 105, 120, 100, 130]
        dd = max_drawdown(equity)
        # Peak was 120, trough was 100, drawdown = -20/120 = -16.67%
        assert dd == pytest.approx(-20.0 / 120.0, abs=0.001)

    def test_max_drawdown_no_drawdown(self):
        equity = [100, 110, 120, 130]
        assert max_drawdown(equity) == 0.0

    def test_sharpe_ratio(self):
        # Consistent positive returns
        equity = [100_000 + i * 100 for i in range(252)]
        sr = sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=252)
        assert sr > 0  # Positive and high since returns are very consistent

    def test_sharpe_ratio_flat(self):
        equity = [100_000] * 100
        sr = sharpe_ratio(equity)
        assert sr == 0.0

    def test_sortino_ratio(self):
        equity = [100_000 + i * 100 for i in range(252)]
        sr = sortino_ratio(equity, risk_free_rate=0.0, periods_per_year=252)
        assert sr > 0


class TestTradeMetrics:
    def test_win_rate(self):
        pnls = [100, -50, 200, -30, 150]
        assert win_rate(pnls) == pytest.approx(0.6)  # 3 wins out of 5

    def test_win_rate_all_wins(self):
        assert win_rate([100, 200, 50]) == pytest.approx(1.0)

    def test_win_rate_no_trades(self):
        assert win_rate([]) == 0.0

    def test_profit_factor(self):
        pnls = [100, -50, 200, -30]
        # Gross profit = 300, gross loss = 80
        assert profit_factor(pnls) == pytest.approx(300.0 / 80.0)

    def test_profit_factor_no_losses(self):
        assert profit_factor([100, 200]) == float("inf")

    def test_avg_win_loss_ratio(self):
        pnls = [100, -50, 200, -30]
        # Avg win = 150, avg loss = 40
        assert avg_win_loss_ratio(pnls) == pytest.approx(150.0 / 40.0)
