"""Paper trading via Alpaca API."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.strategies.registry import get_strategy
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet


class PaperTrader:
    """Paper trading using Alpaca's REST API.

    Fetches recent bars from Alpaca and runs them through the
    backtest engine. This is 'paper trading as a special backtest'
    -- same strategies, same risk controls, just a live data source.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_bars(
        self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """Fetch recent bars from Alpaca."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Handle MultiIndex (symbol, timestamp)
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")

        return df

    def run(
        self,
        symbol: str,
        strategy_name: str,
        strategy_params: dict | None = None,
        days: int = 30,
        initial_cash: float = 100_000.0,
    ) -> None:
        """Run a paper trading session."""
        print(f"Fetching {days} days of {symbol} data from Alpaca...")
        df = self.fetch_bars(symbol, days=days)
        print(f"Got {len(df)} bars")

        if df.empty:
            print("No data available.")
            return

        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        strategy_cls = get_strategy(strategy_name)
        params = strategy_params or {}

        engine = BacktestEngine(initial_cash=initial_cash, slippage_pct=0.0005)

        def factory(bus: EventBus, assets: list[Asset]):
            return strategy_cls(event_bus=bus, assets=assets, **params)

        print(f"Running {strategy_name} on live data...")
        result = engine.run(data={asset: df}, strategy_factory=factory)
        print_tearsheet(result)
