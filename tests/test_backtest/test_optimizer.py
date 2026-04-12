# tests/test_backtest/test_optimizer.py
import pytest
import pandas as pd
from quantflow.core.models import Asset, AssetClass
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.optimizer import ParameterSweep, SweepResult


def make_price_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": [p - 1 for p in prices], "high": [p + 2 for p in prices],
         "low": [p - 2 for p in prices], "close": prices,
         "volume": [1e6] * len(prices)},
        index=dates,
    )


class TestParameterSweep:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        prices = list(range(100, 200)) + list(range(200, 100, -1))
        self.data = {self.asset: make_price_data(prices)}

    def test_sweep_runs(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5], "slow_period": [10, 15]},
            data=self.data,
        )
        result = sweep.run()
        assert isinstance(result, SweepResult)
        assert len(result.results) == 4

    def test_sweep_returns_best_params(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5, 7], "slow_period": [10, 15]},
            data=self.data,
        )
        result = sweep.run(metric="sharpe_ratio")
        assert result.best_params is not None
        assert "fast_period" in result.best_params
        assert "slow_period" in result.best_params

    def test_sweep_each_result_has_metrics(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5], "slow_period": [10]},
            data=self.data,
        )
        result = sweep.run()
        for r in result.results:
            assert "params" in r
            assert "sharpe_ratio" in r
            assert "total_return" in r
