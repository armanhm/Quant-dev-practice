"""
Mean Reversion Backtest Example
================================
Run with: python -m quantflow.examples.mean_reversion

Backtests a Bollinger Band mean reversion strategy on SPY.
Uses Fixed Fractional position sizing (5% per trade).
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.portfolio.sizing import FixedFractional
from quantflow.portfolio.risk import RiskManager
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    asset = Asset(symbol="SPY", asset_class=AssetClass.EQUITY)

    print("Fetching SPY data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    print("\nRunning Mean Reversion backtest...")
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,
        commission_pct=0.0,
        position_sizer=FixedFractional(fraction=0.05),
        risk_manager=RiskManager(max_drawdown=0.25, max_position_pct=0.30),
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> MeanReversion:
        return MeanReversion(bus, assets, bb_period=20, num_std=2.0)

    result = engine.run(data={asset: df}, strategy_factory=strategy_factory)

    print_tearsheet(result)
    plot_tearsheet(result, save_path="mean_reversion_tearsheet.png")


if __name__ == "__main__":
    main()
