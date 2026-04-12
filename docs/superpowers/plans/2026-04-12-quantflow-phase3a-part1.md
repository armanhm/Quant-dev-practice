# QuantFlow Phase 3A Part 1: Data Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite data caching, FRED macro data fetcher, CCXT crypto data fetcher, and a DataManager that orchestrates all data sources through the cache -- so backtests are instant after the first fetch and the platform supports equities, crypto, and macro data from multiple sources.

**Architecture:** DataCache wraps SQLite for persistence. DataManager sits on top of all fetchers + cache, routing requests by asset class and serving from cache when fresh. FRED and CCXT fetchers implement the same DataFetcher protocol as YahooFetcher. DataManager is used before backtests (to fetch/cache data), not during them -- the engine still receives pre-fetched DataFrames.

**Tech Stack:** Python 3.11+, sqlite3 (stdlib), fredapi, ccxt, python-dotenv, pytest

---

## File Structure

```
quantflow/
    data/
        cache.py           -- NEW: SQLite DataCache
        manager.py         -- NEW: DataManager orchestrator
        fred_fetcher.py    -- NEW: FRED macro data fetcher
        ccxt_fetcher.py    -- NEW: CCXT crypto data fetcher
    examples/
        data_demo.py       -- NEW: Demo of DataManager with caching
config/
    example.env            -- NEW: Template for API keys
tests/
    test_data/
        test_cache.py      -- NEW
        test_manager.py    -- NEW
        test_fred_fetcher.py   -- NEW
        test_ccxt_fetcher.py   -- NEW
```

---

### Task 1: SQLite DataCache

**Files:**
- Create: `quantflow/data/cache.py`
- Create: `tests/test_data/test_cache.py`

- [ ] **Step 1: Write failing tests for DataCache**

```python
# tests/test_data/test_cache.py
import pytest
import os
import tempfile
import pandas as pd
from datetime import datetime, timezone, timedelta

from quantflow.core.models import Asset, AssetClass
from quantflow.data.cache import DataCache


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    return DataCache(db_path=db_path)


@pytest.fixture
def sample_asset():
    return Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "high": [155.0, 156.0, 157.0, 158.0, 159.0],
            "low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "close": [153.0, 154.0, 155.0, 156.0, 157.0],
            "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
        },
        index=dates,
    )


class TestDataCache:
    def test_put_and_get_ohlcv(self, cache, sample_asset, sample_df):
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        result = cache.get_ohlcv(
            sample_asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) == 5
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert result.iloc[0]["close"] == 153.0

    def test_get_ohlcv_empty(self, cache, sample_asset):
        result = cache.get_ohlcv(
            sample_asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        assert result is None

    def test_get_ohlcv_partial_range(self, cache, sample_asset, sample_df):
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        # Request only 3 days that overlap
        result = cache.get_ohlcv(
            sample_asset,
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 7, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) >= 2  # At least the days within range

    def test_is_fresh_true(self, cache, sample_asset, sample_df):
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        assert cache.is_fresh(
            sample_asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
            max_age_hours=24,
        ) is True

    def test_is_fresh_false_when_empty(self, cache, sample_asset):
        assert cache.is_fresh(
            sample_asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
            max_age_hours=24,
        ) is False

    def test_put_and_get_macro(self, cache):
        dates = pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")
        df = pd.DataFrame({"value": [2.5, 2.6, 2.7]}, index=dates)
        cache.put_macro("CPI", df, source="fred")
        result = cache.get_macro(
            "CPI",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) == 3
        assert "value" in result.columns

    def test_list_cached_assets(self, cache, sample_asset, sample_df):
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        assets = cache.list_cached_assets()
        assert len(assets) == 1
        assert assets[0]["symbol"] == "AAPL"
        assert assets[0]["asset_class"] == "equity"

    def test_upsert_does_not_duplicate(self, cache, sample_asset, sample_df):
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        cache.put_ohlcv(sample_asset, sample_df, source="yahoo")
        result = cache.get_ohlcv(
            sample_asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        assert len(result) == 5  # Not 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_cache.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement DataCache**

```python
# quantflow/data/cache.py
"""SQLite-backed data cache for market and macro data."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantflow.core.models import Asset


_DEFAULT_DB_PATH = Path.home() / ".quantflow" / "cache.db"


class DataCache:
    """Persistent SQLite cache for OHLCV and macro data."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(_DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS market_data (
                asset_symbol TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                source TEXT,
                fetched_at TEXT,
                PRIMARY KEY (asset_symbol, asset_class, timestamp)
            );
            CREATE TABLE IF NOT EXISTS macro_data (
                indicator TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL,
                source TEXT,
                fetched_at TEXT,
                PRIMARY KEY (indicator, date)
            );
        """)
        self._conn.commit()

    def put_ohlcv(self, asset: Asset, df: pd.DataFrame, source: str) -> None:
        if df.empty:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                asset.symbol,
                asset.asset_class.value,
                ts.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                source,
                now,
            ))
        self._conn.executemany(
            """INSERT OR REPLACE INTO market_data
               (asset_symbol, asset_class, timestamp, open, high, low, close, volume, source, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_ohlcv(
        self, asset: Asset, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        cursor = self._conn.execute(
            """SELECT timestamp, open, high, low, close, volume FROM market_data
               WHERE asset_symbol = ? AND asset_class = ?
               AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (asset.symbol, asset.asset_class.value, start.isoformat(), end.isoformat()),
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df

    def is_fresh(
        self, asset: Asset, start: datetime, end: datetime, max_age_hours: int = 24
    ) -> bool:
        cursor = self._conn.execute(
            """SELECT MIN(fetched_at) FROM market_data
               WHERE asset_symbol = ? AND asset_class = ?
               AND timestamp >= ? AND timestamp <= ?""",
            (asset.symbol, asset.asset_class.value, start.isoformat(), end.isoformat()),
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return False
        oldest_fetch = datetime.fromisoformat(row[0])
        age = datetime.now(timezone.utc) - oldest_fetch
        return age.total_seconds() < max_age_hours * 3600

    def put_macro(self, indicator: str, df: pd.DataFrame, source: str) -> None:
        if df.empty:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                indicator,
                ts.isoformat(),
                float(row["value"]),
                source,
                now,
            ))
        self._conn.executemany(
            """INSERT OR REPLACE INTO macro_data
               (indicator, date, value, source, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_macro(
        self, indicator: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        cursor = self._conn.execute(
            """SELECT date, value FROM macro_data
               WHERE indicator = ? AND date >= ? AND date <= ?
               ORDER BY date""",
            (indicator, start.isoformat(), end.isoformat()),
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        return df

    def list_cached_assets(self) -> list[dict]:
        cursor = self._conn.execute(
            """SELECT DISTINCT asset_symbol, asset_class,
               MIN(timestamp) as first_date, MAX(timestamp) as last_date,
               COUNT(*) as bar_count
               FROM market_data GROUP BY asset_symbol, asset_class"""
        )
        return [
            {
                "symbol": row[0],
                "asset_class": row[1],
                "first_date": row[2],
                "last_date": row[3],
                "bar_count": row[4],
            }
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_cache.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/data/cache.py tests/test_data/test_cache.py
git commit -m "feat: add SQLite DataCache for persistent OHLCV and macro data caching"
```

---

### Task 2: FRED Macro Data Fetcher

**Files:**
- Create: `quantflow/data/fred_fetcher.py`
- Create: `tests/test_data/test_fred_fetcher.py`

- [ ] **Step 1: Write failing tests (mocked -- no real API calls)**

```python
# tests/test_data/test_fred_fetcher.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from quantflow.core.models import Asset, AssetClass
from quantflow.data.fred_fetcher import FREDFetcher


class TestFREDFetcher:
    def setup_method(self):
        self.fetcher = FREDFetcher(api_key="test_key")

    def test_supported_asset_classes(self):
        classes = self.fetcher.supported_asset_classes()
        assert AssetClass.MACRO in classes
        assert len(classes) == 1

    @patch("quantflow.data.fred_fetcher.Fred")
    def test_fetch_series(self, mock_fred_class):
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred
        mock_fred.get_series.return_value = pd.Series(
            [2.5, 2.6, 2.7],
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
                name="date",
            ),
        )

        fetcher = FREDFetcher(api_key="test_key")
        result = fetcher.fetch_series(
            "CPI",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert len(result) == 3
        assert "value" in result.columns
        assert result.iloc[0]["value"] == 2.5

    @patch("quantflow.data.fred_fetcher.Fred")
    def test_fetch_ohlcv_wraps_series(self, mock_fred_class):
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred
        mock_fred.get_series.return_value = pd.Series(
            [2.5, 2.6],
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1), datetime(2024, 2, 1)],
                name="date",
            ),
        )

        fetcher = FREDFetcher(api_key="test_key")
        asset = Asset(symbol="CPI", asset_class=AssetClass.MACRO)
        result = fetcher.fetch_ohlcv(
            asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert result.iloc[0]["close"] == 2.5
        assert result.iloc[0]["volume"] == 0.0

    @patch("quantflow.data.fred_fetcher.Fred")
    def test_fetch_empty_series(self, mock_fred_class):
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred
        mock_fred.get_series.return_value = pd.Series(dtype=float)

        fetcher = FREDFetcher(api_key="test_key")
        result = fetcher.fetch_series(
            "INVALID",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert result.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_fred_fetcher.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement FREDFetcher**

```python
# quantflow/data/fred_fetcher.py
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

    def fetch_series(
        self, indicator: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
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

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        df = self.fetch_series(asset.symbol, start, end)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        ohlcv = pd.DataFrame(
            {
                "open": df["value"],
                "high": df["value"],
                "low": df["value"],
                "close": df["value"],
                "volume": 0.0,
            },
            index=df.index,
        )
        return ohlcv

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.MACRO]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_fred_fetcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/data/fred_fetcher.py tests/test_data/test_fred_fetcher.py
git commit -m "feat: add FRED macro data fetcher for economic indicators"
```

---

### Task 3: CCXT Crypto Data Fetcher

**Files:**
- Create: `quantflow/data/ccxt_fetcher.py`
- Create: `tests/test_data/test_ccxt_fetcher.py`

- [ ] **Step 1: Write failing tests (mocked)**

```python
# tests/test_data/test_ccxt_fetcher.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from quantflow.core.models import Asset, AssetClass
from quantflow.data.ccxt_fetcher import CCXTFetcher


class TestCCXTFetcher:
    def setup_method(self):
        self.asset = Asset(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)

    def test_supported_asset_classes(self):
        with patch("quantflow.data.ccxt_fetcher.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            fetcher = CCXTFetcher()
            classes = fetcher.supported_asset_classes()
            assert AssetClass.CRYPTO in classes
            assert len(classes) == 1

    @patch("quantflow.data.ccxt_fetcher.ccxt")
    def test_fetch_ohlcv_returns_correct_format(self, mock_ccxt):
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        # CCXT returns list of [timestamp_ms, open, high, low, close, volume]
        mock_exchange.fetch_ohlcv.return_value = [
            [1704153600000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5],
            [1704240000000, 42300.0, 43000.0, 42100.0, 42800.0, 120.3],
        ]

        fetcher = CCXTFetcher()
        result = fetcher.fetch_ohlcv(
            self.asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 2
        assert result.iloc[0]["close"] == 42300.0
        assert result.index.tzinfo is not None

    @patch("quantflow.data.ccxt_fetcher.ccxt")
    def test_fetch_ohlcv_empty(self, mock_ccxt):
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = []

        fetcher = CCXTFetcher()
        result = fetcher.fetch_ohlcv(
            self.asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        assert result.empty

    @patch("quantflow.data.ccxt_fetcher.ccxt")
    def test_timeframe_mapping(self, mock_ccxt):
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = []

        fetcher = CCXTFetcher()
        fetcher.fetch_ohlcv(
            self.asset,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            timeframe="1h",
        )

        call_args = mock_exchange.fetch_ohlcv.call_args
        assert call_args[1].get("timeframe", call_args[0][1] if len(call_args[0]) > 1 else None) == "1h" or "1h" in str(call_args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_ccxt_fetcher.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement CCXTFetcher**

```python
# quantflow/data/ccxt_fetcher.py
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
            since = candles[-1][0] + 1  # Next ms after last candle

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        # Filter to requested range
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        return df[["open", "high", "low", "close", "volume"]]

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.CRYPTO]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_ccxt_fetcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/data/ccxt_fetcher.py tests/test_data/test_ccxt_fetcher.py
git commit -m "feat: add CCXT crypto data fetcher for Binance OHLCV"
```

---

### Task 4: DataManager

**Files:**
- Create: `quantflow/data/manager.py`
- Create: `tests/test_data/test_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data/test_manager.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from quantflow.core.models import Asset, AssetClass
from quantflow.data.manager import DataManager
from quantflow.data.cache import DataCache


@pytest.fixture
def mock_cache(tmp_path):
    return DataCache(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "high": [155.0, 156.0, 157.0, 158.0, 159.0],
            "low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "close": [153.0, 154.0, 155.0, 156.0, 157.0],
            "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
        },
        index=dates,
    )


class TestDataManager:
    def test_get_ohlcv_fetches_and_caches(self, mock_cache, sample_df):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = sample_df
        mock_fetcher.supported_asset_classes.return_value = [AssetClass.EQUITY]

        manager = DataManager(cache=mock_cache, fetchers=[mock_fetcher])
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        result = manager.get_ohlcv(asset, start, end)

        assert len(result) == 5
        mock_fetcher.fetch_ohlcv.assert_called_once()

    def test_get_ohlcv_serves_from_cache(self, mock_cache, sample_df):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        # Pre-populate cache
        mock_cache.put_ohlcv(asset, sample_df, source="yahoo")

        mock_fetcher = MagicMock()
        mock_fetcher.supported_asset_classes.return_value = [AssetClass.EQUITY]

        manager = DataManager(cache=mock_cache, fetchers=[mock_fetcher])
        result = manager.get_ohlcv(asset, start, end)

        assert len(result) == 5
        mock_fetcher.fetch_ohlcv.assert_not_called()  # Should NOT fetch

    def test_routes_to_correct_fetcher(self, mock_cache, sample_df):
        equity_fetcher = MagicMock()
        equity_fetcher.fetch_ohlcv.return_value = sample_df
        equity_fetcher.supported_asset_classes.return_value = [AssetClass.EQUITY]

        crypto_fetcher = MagicMock()
        crypto_fetcher.fetch_ohlcv.return_value = sample_df
        crypto_fetcher.supported_asset_classes.return_value = [AssetClass.CRYPTO]

        manager = DataManager(cache=mock_cache, fetchers=[equity_fetcher, crypto_fetcher])

        equity_asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        crypto_asset = Asset(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        manager.get_ohlcv(equity_asset, start, end)
        equity_fetcher.fetch_ohlcv.assert_called_once()
        crypto_fetcher.fetch_ohlcv.assert_not_called()

        manager.get_ohlcv(crypto_asset, start, end)
        crypto_fetcher.fetch_ohlcv.assert_called_once()

    def test_no_fetcher_for_asset_class_raises(self, mock_cache):
        manager = DataManager(cache=mock_cache, fetchers=[])
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        with pytest.raises(ValueError, match="No fetcher"):
            manager.get_ohlcv(
                asset,
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 31, tzinfo=timezone.utc),
            )

    def test_get_macro(self, mock_cache):
        mock_fetcher = MagicMock()
        dates = pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")
        macro_df = pd.DataFrame({"value": [2.5, 2.6, 2.7]}, index=dates)
        mock_fetcher.fetch_series.return_value = macro_df
        mock_fetcher.supported_asset_classes.return_value = [AssetClass.MACRO]

        manager = DataManager(cache=mock_cache, fetchers=[mock_fetcher])
        result = manager.get_macro(
            "CPI",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_manager.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement DataManager**

```python
# quantflow/data/manager.py
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

    def get_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        # Check cache first
        if self.cache.is_fresh(asset, start, end):
            cached = self.cache.get_ohlcv(asset, start, end)
            if cached is not None and not cached.empty:
                return cached

        # Find a fetcher for this asset class
        fetchers = self._fetcher_map.get(asset.asset_class)
        if not fetchers:
            raise ValueError(
                f"No fetcher available for asset class {asset.asset_class.value}"
            )

        # Try fetchers in order (first one wins)
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

    def get_macro(
        self,
        indicator: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # Check cache
        cached = self.cache.get_macro(indicator, start, end)
        if cached is not None and not cached.empty:
            return cached

        # Find a MACRO fetcher
        fetchers = self._fetcher_map.get(AssetClass.MACRO)
        if not fetchers:
            raise ValueError("No fetcher available for macro data")

        for fetcher in fetchers:
            try:
                df = fetcher.fetch_series(indicator, start, end)
                if not df.empty:
                    self.cache.put_macro(indicator, df, source="fred")
                    return df
            except Exception as e:
                continue

        return pd.DataFrame(columns=["value"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_manager.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/data/manager.py tests/test_data/test_manager.py
git commit -m "feat: add DataManager to orchestrate fetchers and cache"
```

---

### Task 5: Update Dependencies and Config

**Files:**
- Modify: `pyproject.toml`
- Create: `config/example.env`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

Add to the `dependencies` list in `pyproject.toml`:
```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "yfinance>=0.2.18",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
    "fredapi>=0.5",
    "ccxt>=4.0",
    "python-dotenv>=1.0",
]
```

- [ ] **Step 2: Create example.env**

```bash
mkdir -p config
```

```
# config/example.env
# Copy this to .env in the project root and fill in your API keys

# FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here
```

- [ ] **Step 3: Install new dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed fredapi, ccxt, python-dotenv

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (existing + new)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config/example.env
git commit -m "chore: add fredapi, ccxt, python-dotenv dependencies and example.env"
```

---

### Task 6: Data Demo Example

**Files:**
- Create: `quantflow/examples/data_demo.py`

- [ ] **Step 1: Create the data demo example**

```python
# quantflow/examples/data_demo.py
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
    # Set up cache and manager
    cache = DataCache()  # Uses ~/.quantflow/cache.db
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])

    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    # First fetch (hits Yahoo Finance)
    print("First fetch (from Yahoo Finance)...")
    t0 = time.time()
    df = manager.get_ohlcv(asset, start, end)
    t1 = time.time()
    print(f"  Got {len(df)} bars in {t1 - t0:.2f}s")

    # Second fetch (from cache)
    print("\nSecond fetch (from SQLite cache)...")
    t0 = time.time()
    df = manager.get_ohlcv(asset, start, end)
    t1 = time.time()
    print(f"  Got {len(df)} bars in {t1 - t0:.4f}s")

    # Show cached assets
    print("\nCached assets:")
    for info in cache.list_cached_assets():
        print(f"  {info['symbol']} ({info['asset_class']}): "
              f"{info['bar_count']} bars, {info['first_date'][:10]} to {info['last_date'][:10]}")

    cache.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add quantflow/examples/data_demo.py
git commit -m "feat: add data infrastructure demo showing cached multi-source fetch"
```

---

## Phase 3A Part 1 Summary

After completing these 6 tasks, you will have:

- **SQLite DataCache** -- persistent caching for OHLCV and macro data
- **FRED fetcher** -- economic indicators (GDP, CPI, unemployment, VIX)
- **CCXT fetcher** -- crypto OHLCV from Binance with pagination
- **DataManager** -- orchestrates all fetchers through the cache, routes by asset class
- **Updated dependencies** -- fredapi, ccxt, python-dotenv
- **Data demo example** -- shows cache speedup

## What Comes Next (Phase 3A Part 2)

- Strategy registry
- Pairs Trading strategy
- Macro Regime strategy
- CompositeStrategy
- ParameterSweep + WalkForward optimization
- CLI via click
- Runnable examples for all new strategies
