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
            "CPI", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc),
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
            index=pd.DatetimeIndex([datetime(2024, 1, 1), datetime(2024, 2, 1)], name="date"),
        )
        fetcher = FREDFetcher(api_key="test_key")
        asset = Asset(symbol="CPI", asset_class=AssetClass.MACRO)
        result = fetcher.fetch_ohlcv(
            asset, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc),
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
            "INVALID", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert result.empty
