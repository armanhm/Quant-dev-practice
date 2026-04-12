from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from quantflow.core.models import Asset, AssetClass

TIMEFRAME_MAP = {
    "1d": "1d",
    "1h": "1h",
    "5m": "5m",
    "1m": "1m",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YahooFetcher:
    """Fetches OHLCV data from Yahoo Finance via yfinance."""

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        yf_interval = TIMEFRAME_MAP.get(timeframe, "1d")

        df = yf.download(
            asset.symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            return df

        # Normalize column names to lowercase
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]

        # Keep only OHLCV columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in expected_cols if c in df.columns]]

        # Ensure UTC timezone
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")

        return df

    def supported_asset_classes(self) -> list[AssetClass]:
        return [
            AssetClass.EQUITY,
            AssetClass.CRYPTO,
            AssetClass.FOREX,
            AssetClass.COMMODITY,
            AssetClass.INDEX,
        ]
