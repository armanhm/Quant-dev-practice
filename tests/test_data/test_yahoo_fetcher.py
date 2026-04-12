import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from quantflow.core.models import Asset, AssetClass
from quantflow.data.yahoo_fetcher import YahooFetcher


class TestYahooFetcher:
    def setup_method(self):
        self.fetcher = YahooFetcher()

    def test_supported_asset_classes(self):
        classes = self.fetcher.supported_asset_classes()
        assert AssetClass.EQUITY in classes
        assert AssetClass.CRYPTO in classes
        assert AssetClass.FOREX in classes

    @patch("quantflow.data.yahoo_fetcher.yf.download")
    def test_fetch_ohlcv_returns_correct_columns(self, mock_download):
        mock_df = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [155.0, 156.0],
                "Low": [149.0, 150.0],
                "Close": [153.0, 154.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 2, tzinfo=timezone.utc),
                 datetime(2024, 1, 3, tzinfo=timezone.utc)],
                name="Date",
            ),
        )
        mock_download.return_value = mock_df

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        result = self.fetcher.fetch_ohlcv(asset, start, end)

        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 2
        assert result.index.tzinfo is not None  # UTC-aware

    @patch("quantflow.data.yahoo_fetcher.yf.download")
    def test_fetch_ohlcv_empty_dataframe(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        asset = Asset(symbol="INVALID", asset_class=AssetClass.EQUITY)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        result = self.fetcher.fetch_ohlcv(asset, start, end)
        assert result.empty

    @patch("quantflow.data.yahoo_fetcher.yf.download")
    def test_fetch_ohlcv_normalizes_column_names(self, mock_download):
        mock_df = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 2, tzinfo=timezone.utc)],
                name="Date",
            ),
        )
        mock_download.return_value = mock_df

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        result = self.fetcher.fetch_ohlcv(asset, start, end)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
