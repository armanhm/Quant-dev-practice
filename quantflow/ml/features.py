# quantflow/ml/features.py
"""Feature pipeline: transforms OHLCV bars into ML-ready feature matrices."""
from __future__ import annotations

import numpy as np

from quantflow.core.models import Bar
from quantflow.data.indicators import sma, ema, rsi, macd, bollinger_bands, atr


def build_features(
    bars: list[Bar], lookback: int = 20, horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and labels from bar data.

    Features per row: returns (1d/5d/10d/20d), RSI(14), MACD histogram,
    Bollinger %B, ATR ratio, volume ratio, volatility (10d/20d).

    Labels: 1 if forward return > 0.1%, -1 if < -0.1%, else 0.
    """
    n = len(bars)
    min_required = max(lookback, 26) + horizon
    if n < min_required:
        return np.empty((0, 11)), np.empty(0)

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]

    rsi_vals = rsi(closes, period=14)
    macd_line, signal_line, macd_hist = macd(closes)
    bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=20)
    atr_vals = atr(highs, lows, closes, period=14)
    sma_20_vol = sma(volumes, period=20)

    returns_1d = [0.0] + [(closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0.0
                          for i in range(1, n)]

    features = []
    labels = []

    start = max(lookback, 26)
    end = n - horizon

    for i in range(start, end):
        if _any_nan(rsi_vals[i], macd_hist[i], bb_upper[i], bb_lower[i],
                    bb_middle[i], atr_vals[i], sma_20_vol[i]):
            continue

        ret_1d = returns_1d[i]
        ret_5d = (closes[i] - closes[i-5]) / closes[i-5] if i >= 5 and closes[i-5] != 0 else 0.0
        ret_10d = (closes[i] - closes[i-10]) / closes[i-10] if i >= 10 and closes[i-10] != 0 else 0.0
        ret_20d = (closes[i] - closes[i-20]) / closes[i-20] if i >= 20 and closes[i-20] != 0 else 0.0

        bb_range = bb_upper[i] - bb_lower[i]
        bb_pct_b = (closes[i] - bb_lower[i]) / bb_range if bb_range > 0 else 0.5
        atr_ratio = atr_vals[i] / closes[i] if closes[i] > 0 else 0.0
        vol_ratio = volumes[i] / sma_20_vol[i] if sma_20_vol[i] > 0 else 1.0

        vol_10d = np.std(returns_1d[max(0,i-9):i+1]) if i >= 10 else 0.0
        vol_20d = np.std(returns_1d[max(0,i-19):i+1]) if i >= 20 else 0.0

        row = [ret_1d, ret_5d, ret_10d, ret_20d, rsi_vals[i], macd_hist[i],
               bb_pct_b, atr_ratio, vol_ratio, vol_10d, vol_20d]
        features.append(row)

        forward_return = (closes[i + horizon] - closes[i]) / closes[i]
        if forward_return > 0.001:
            labels.append(1)
        elif forward_return < -0.001:
            labels.append(-1)
        else:
            labels.append(0)

    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.float64)


def time_series_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/test split. Never shuffles."""
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def _any_nan(*values) -> bool:
    return any(v != v for v in values)
