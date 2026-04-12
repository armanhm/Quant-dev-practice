from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import pandas as pd

from quantflow.core.models import Asset, AssetClass


class DataFetcher(Protocol):
    """Protocol for all data source adapters."""

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data. Returns DataFrame with DatetimeIndex (UTC)
        and columns: [open, high, low, close, volume]."""
        ...

    def supported_asset_classes(self) -> list[AssetClass]:
        ...
