"""DataManager: orchestrates data fetchers and cache."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from quantflow.core.models import Asset, AssetClass
from quantflow.data.cache import DataCache


class DataManager:
    """Central data access layer. Routes requests to fetchers and caches results."""

    def __init__(self, cache: DataCache, fetchers: list) -> None:
        self.cache = cache
        self._fetcher_map: dict[AssetClass, list] = {}
        for fetcher in fetchers:
            for asset_class in fetcher.supported_asset_classes():
                if asset_class not in self._fetcher_map:
                    self._fetcher_map[asset_class] = []
                self._fetcher_map[asset_class].append(fetcher)

    def get_ohlcv(self, asset: Asset, start: datetime, end: datetime, timeframe: str = "1d") -> pd.DataFrame:
        if self.cache.is_fresh(asset, start, end):
            cached = self.cache.get_ohlcv(asset, start, end)
            if cached is not None and not cached.empty:
                return cached

        fetchers = self._fetcher_map.get(asset.asset_class)
        if not fetchers:
            raise ValueError(f"No fetcher available for asset class {asset.asset_class.value}")

        last_error = None
        for fetcher in fetchers:
            try:
                df = fetcher.fetch_ohlcv(asset, start, end, timeframe)
                if not df.empty:
                    source = type(fetcher).__name__.lower().replace("fetcher", "")
                    self.cache.put_ohlcv(asset, df, source=source)
                    return df
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_macro(self, indicator: str, start: datetime, end: datetime) -> pd.DataFrame:
        cached = self.cache.get_macro(indicator, start, end)
        if cached is not None and not cached.empty:
            return cached

        fetchers = self._fetcher_map.get(AssetClass.MACRO)
        if not fetchers:
            raise ValueError("No fetcher available for macro data")

        for fetcher in fetchers:
            try:
                df = fetcher.fetch_series(indicator, start, end)
                if not df.empty:
                    self.cache.put_macro(indicator, df, source="fred")
                    return df
            except Exception:
                continue

        return pd.DataFrame(columns=["value"])
