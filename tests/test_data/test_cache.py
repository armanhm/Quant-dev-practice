# tests/test_data/test_cache.py
import pytest
import pandas as pd
from datetime import datetime, timezone

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
        result = cache.get_ohlcv(
            sample_asset,
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 7, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) >= 2

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
        assert len(result) == 5
