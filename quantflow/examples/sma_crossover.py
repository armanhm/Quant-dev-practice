# quantflow/examples/sma_crossover.py
"""
SMA Crossover Backtest Example
===============================
Run with: python -m quantflow.examples.sma_crossover

Backtests a Simple Moving Average crossover strategy on AAPL.
Fast SMA (10) crossing above Slow SMA (50) = buy signal.
Fast SMA (10) crossing below Slow SMA (50) = sell signal.
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    # 1. Define what we're trading
    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    # 2. Fetch historical data
    print("Fetching AAPL data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # 3. Run the backtest
    print("\nRunning SMA Crossover backtest...")
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,   # 0.05%
        commission_pct=0.0,     # Commission-free (modern brokers)
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> SMACrossover:
        return SMACrossover(bus, assets, fast_period=10, slow_period=50)

    result = engine.run(
        data={asset: df},
        strategy_factory=strategy_factory,
    )

    # 4. Print results
    print_tearsheet(result)

    # 5. Plot charts
    plot_tearsheet(result, save_path="sma_crossover_tearsheet.png")


if __name__ == "__main__":
    main()
