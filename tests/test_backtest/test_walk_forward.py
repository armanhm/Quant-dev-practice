# tests/test_backtest/test_walk_forward.py
import pytest
import pandas as pd
import random
from quantflow.core.models import Asset, AssetClass
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.walk_forward import WalkForward, WalkForwardResult


def make_price_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": [p - 1 for p in prices], "high": [p + 2 for p in prices],
         "low": [p - 2 for p in prices], "close": prices,
         "volume": [1e6] * len(prices)},
        index=dates,
    )


class TestWalkForward:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        random.seed(42)
        prices = [100.0]
        for _ in range(499):
            prices.append(prices[-1] * (1 + random.gauss(0.0003, 0.015)))
        self.data = {self.asset: make_price_data(prices)}

    def test_walk_forward_runs(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=self.data, train_bars=100, test_bars=50, step_bars=50,
        )
        result = wf.run()
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    def test_walk_forward_has_oos_metrics(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=self.data, train_bars=100, test_bars=50, step_bars=50,
        )
        result = wf.run()
        for window in result.windows:
            assert "best_params" in window
            assert "oos_sharpe" in window
            assert "is_sharpe" in window

    def test_walk_forward_multiple_windows(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=self.data, train_bars=100, test_bars=50, step_bars=50,
        )
        result = wf.run()
        assert len(result.windows) >= 2
