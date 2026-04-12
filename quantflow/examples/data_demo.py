"""
Data Infrastructure Demo
=========================
Run with: python -m quantflow.examples.data_demo

Demonstrates DataManager with SQLite caching:
- Fetches AAPL equity data (Yahoo Finance)
- Shows cache status
- Second fetch is instant (served from cache)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.data.cache import DataCache
from quantflow.data.manager import DataManager
from quantflow.data.yahoo_fetcher import YahooFetcher


def main():
    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])

    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    print("First fetch (from Yahoo Finance)...")
    t0 = time.time()
    df = manager.get_ohlcv(asset, start, end)
    t1 = time.time()
    print(f"  Got {len(df)} bars in {t1 - t0:.2f}s")

    print("\nSecond fetch (from SQLite cache)...")
    t0 = time.time()
    df = manager.get_ohlcv(asset, start, end)
    t1 = time.time()
    print(f"  Got {len(df)} bars in {t1 - t0:.4f}s")

    print("\nCached assets:")
    for info in cache.list_cached_assets():
        print(f"  {info['symbol']} ({info['asset_class']}): "
              f"{info['bar_count']} bars, {info['first_date'][:10]} to {info['last_date'][:10]}")

    cache.close()


if __name__ == "__main__":
    main()
