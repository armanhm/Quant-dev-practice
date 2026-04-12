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

    def get_ohlcv(self, asset: Asset, start: datetime, end: datetime) -> pd.DataFrame | None:
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

    def is_fresh(self, asset: Asset, start: datetime, end: datetime, max_age_hours: int = 24) -> bool:
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
            rows.append((indicator, ts.isoformat(), float(row["value"]), source, now))
        self._conn.executemany(
            """INSERT OR REPLACE INTO macro_data (indicator, date, value, source, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_macro(self, indicator: str, start: datetime, end: datetime) -> pd.DataFrame | None:
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
            {"symbol": row[0], "asset_class": row[1], "first_date": row[2],
             "last_date": row[3], "bar_count": row[4]}
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
