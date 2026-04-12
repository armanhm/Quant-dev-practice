# tests/test_data/test_manager.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock

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

        mock_cache.put_ohlcv(asset, sample_df, source="yahoo")

        mock_fetcher = MagicMock()
        mock_fetcher.supported_asset_classes.return_value = [AssetClass.EQUITY]

        manager = DataManager(cache=mock_cache, fetchers=[mock_fetcher])
        result = manager.get_ohlcv(asset, start, end)
        assert len(result) == 5
        mock_fetcher.fetch_ohlcv.assert_not_called()

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
            manager.get_ohlcv(asset, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc))

    def test_get_macro(self, mock_cache):
        mock_fetcher = MagicMock()
        dates = pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")
        macro_df = pd.DataFrame({"value": [2.5, 2.6, 2.7]}, index=dates)
        mock_fetcher.fetch_series.return_value = macro_df
        mock_fetcher.supported_asset_classes.return_value = [AssetClass.MACRO]

        manager = DataManager(cache=mock_cache, fetchers=[mock_fetcher])
        result = manager.get_macro("CPI", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc))
        assert len(result) == 3
