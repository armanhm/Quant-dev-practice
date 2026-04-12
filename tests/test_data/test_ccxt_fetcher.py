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
