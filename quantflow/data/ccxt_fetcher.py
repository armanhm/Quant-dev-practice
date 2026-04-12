"""CCXT crypto data fetcher (Binance public API)."""
from __future__ import annotations

from datetime import datetime

import ccxt
import pandas as pd

from quantflow.core.models import Asset, AssetClass


TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
}


class CCXTFetcher:
    """Fetches crypto OHLCV data from Binance via CCXT. No API key needed."""

    def __init__(self, exchange_id: str = "binance") -> None:
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        ccxt_timeframe = TIMEFRAME_MAP.get(timeframe, "1d")
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_candles = []
        while since < end_ms:
            candles = self._exchange.fetch_ohlcv(
                asset.symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=1000,
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
        df = df[df.index <= end_ts]
        return df[["open", "high", "low", "close", "volume"]]

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.CRYPTO]
