import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Direction
from quantflow.core.events import EventBus
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.engine import BacktestEngine, BacktestResult


def make_price_data(asset: Asset, prices: list[float], start_year: int = 2024) -> pd.DataFrame:
    """Helper to create a DataFrame of OHLCV data from a list of close prices."""
    dates = pd.date_range(
        start=f"{start_year}-01-01", periods=len(prices), freq="B", tz="UTC"
    )
    df = pd.DataFrame(
        {
            "open": [p - 1 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )
    return df


class TestBacktestEngine:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def test_backtest_runs_without_error(self):
        prices = list(range(100, 200))  # Steady uptrend, 100 bars
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=20)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_backtest_result_has_trades(self):
        # Uptrend then downtrend to generate at least one trade
        prices = list(range(100, 140)) + list(range(140, 100, -1))
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert len(result.trades) > 0

    def test_equity_curve_starts_at_initial_cash(self):
        prices = list(range(100, 150))
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(initial_cash=50_000.0)

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert result.equity_curve[0] == pytest.approx(50_000.0)

    def test_buy_and_hold_benchmark(self):
        prices = [100.0] * 10 + [200.0] * 10  # Doubles in price
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(initial_cash=100_000.0)

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=3, slow_period=5)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        # Benchmark should reflect the price change
        assert result.benchmark_equity[-1] > result.benchmark_equity[0]
