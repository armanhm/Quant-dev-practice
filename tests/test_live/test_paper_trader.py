import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timezone


class TestPaperTrader:
    @patch("quantflow.live.paper_trader.StockHistoricalDataClient")
    def test_paper_trader_creates(self, mock_client_cls):
        from quantflow.live.paper_trader import PaperTrader
        trader = PaperTrader(api_key="test", secret_key="test")
        assert trader is not None

    @patch("quantflow.live.paper_trader.StockHistoricalDataClient")
    def test_fetch_bars_returns_dataframe(self, mock_client_cls):
        from quantflow.live.paper_trader import PaperTrader

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock the bars response
        mock_bars = MagicMock()
        dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        mock_df = pd.DataFrame(
            {"open": [150.0]*5, "high": [155.0]*5, "low": [149.0]*5,
             "close": [153.0]*5, "volume": [1e6]*5},
            index=dates,
        )
        mock_bars.df = mock_df
        mock_client.get_stock_bars.return_value = mock_bars

        trader = PaperTrader(api_key="test", secret_key="test")
        result = trader.fetch_bars("AAPL", days=30)
        assert len(result) == 5
        assert "close" in result.columns
