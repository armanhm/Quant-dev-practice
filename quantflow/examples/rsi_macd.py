"""
RSI + MACD Combo Backtest Example
===================================
Run with: python -m quantflow.examples.rsi_macd

Backtests an RSI + MACD combination strategy on MSFT.
Uses Kelly Criterion position sizing.
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.rsi_macd import RSIMACDCombo
from quantflow.portfolio.sizing import KellyCriterion
from quantflow.portfolio.risk import RiskManager
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    asset = Asset(symbol="MSFT", asset_class=AssetClass.EQUITY)

    print("Fetching MSFT data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    print("\nRunning RSI + MACD Combo backtest...")
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,
        commission_pct=0.0,
        position_sizer=KellyCriterion(win_rate=0.45, avg_win_loss_ratio=2.0, kelly_fraction=0.5),
        risk_manager=RiskManager(max_drawdown=0.20, max_position_pct=0.25),
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> RSIMACDCombo:
        return RSIMACDCombo(bus, assets, rsi_period=14, rsi_oversold=30, rsi_overbought=70)

    result = engine.run(data={asset: df}, strategy_factory=strategy_factory)

    print_tearsheet(result)
    plot_tearsheet(result, save_path="rsi_macd_tearsheet.png")


if __name__ == "__main__":
    main()
