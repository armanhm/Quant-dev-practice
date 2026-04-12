# tests/test_data/test_indicators.py
import pytest
import math
from quantflow.data.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class TestSMA:
    def test_sma_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sma(data, period=3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_period_equals_length(self):
        data = [10.0, 20.0, 30.0]
        result = sma(data, period=3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(20.0)

    def test_sma_empty(self):
        assert sma([], period=3) == []


class TestEMA:
    def test_ema_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ema(data, period=3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_ema_empty(self):
        assert ema([], period=3) == []


class TestRSI:
    def test_rsi_overbought(self):
        data = list(range(50, 70))
        result = rsi(data, period=14)
        assert result[-1] > 70.0

    def test_rsi_oversold(self):
        data = list(range(70, 50, -1))
        result = rsi(data, period=14)
        assert result[-1] < 30.0

    def test_rsi_range(self):
        data = [100 + (i % 5) * (-1)**i for i in range(30)]
        result = rsi(data, period=14)
        valid = [v for v in result if not math.isnan(v)]
        assert all(0.0 <= v <= 100.0 for v in valid)

    def test_rsi_empty(self):
        assert rsi([], period=14) == []


class TestMACD:
    def test_macd_returns_three_lists(self):
        data = [100.0 + i * 0.5 for i in range(40)]
        macd_line, signal_line, histogram = macd(data)
        assert len(macd_line) == 40
        assert len(signal_line) == 40
        assert len(histogram) == 40

    def test_macd_uptrend_positive(self):
        data = [100.0 + i * 2.0 for i in range(40)]
        macd_line, signal_line, histogram = macd(data)
        valid_macd = [v for v in macd_line[-5:] if not math.isnan(v)]
        assert len(valid_macd) > 0
        assert all(v > 0 for v in valid_macd)

    def test_macd_empty(self):
        m, s, h = macd([])
        assert m == [] and s == [] and h == []


class TestBollingerBands:
    def test_bollinger_basic(self):
        data = [100.0] * 20 + [110.0]
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        assert middle[19] == pytest.approx(100.0)
        assert upper[19] == pytest.approx(100.0)
        assert lower[19] == pytest.approx(100.0)

    def test_bollinger_band_width(self):
        import random
        random.seed(42)
        data = [100.0 + random.gauss(0, 5) for _ in range(30)]
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        assert upper[-1] > middle[-1] > lower[-1]

    def test_bollinger_empty(self):
        u, m, l = bollinger_bands([], period=20)
        assert u == [] and m == [] and l == []


class TestATR:
    def test_atr_basic(self):
        highs = [110.0, 112.0, 115.0, 113.0, 116.0]
        lows = [100.0, 102.0, 105.0, 103.0, 106.0]
        closes = [105.0, 108.0, 110.0, 107.0, 112.0]
        result = atr(highs, lows, closes, period=3)
        assert len(result) == 5
        valid = [v for v in result if not math.isnan(v)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_atr_empty(self):
        assert atr([], [], [], period=14) == []
