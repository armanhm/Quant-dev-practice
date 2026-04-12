"""FRED (Federal Reserve Economic Data) fetcher."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from fredapi import Fred

from quantflow.core.models import Asset, AssetClass


class FREDFetcher:
    """Fetches economic indicator data from FRED."""

    def __init__(self, api_key: str) -> None:
        self._fred = Fred(api_key=api_key)

    def fetch_series(self, indicator: str, start: datetime, end: datetime) -> pd.DataFrame:
        series = self._fred.get_series(
            indicator,
            observation_start=start.strftime("%Y-%m-%d"),
            observation_end=end.strftime("%Y-%m-%d"),
        )
        if series.empty:
            return pd.DataFrame(columns=["value"])
        df = pd.DataFrame({"value": series.values}, index=series.index)
        df.index.name = "date"
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def fetch_ohlcv(self, asset: Asset, start: datetime, end: datetime, timeframe: str = "1d") -> pd.DataFrame:
        df = self.fetch_series(asset.symbol, start, end)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ohlcv = pd.DataFrame(
            {"open": df["value"], "high": df["value"], "low": df["value"], "close": df["value"], "volume": 0.0},
            index=df.index,
        )
        return ohlcv

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.MACRO]
