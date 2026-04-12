# quantflow/data/indicators.py
"""Technical indicator functions.

All functions take lists of floats and return lists of floats of the same length.
Positions where the indicator can't be computed (not enough data) are float('nan').
"""
from __future__ import annotations

import math


def sma(data: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if not data:
        return []
    result = [float("nan")] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1 : i + 1]) / period
    return result


def ema(data: list[float], period: int) -> list[float]:
    """Exponential Moving Average. Seeded with SMA of first 'period' values."""
    if not data:
        return []
    result = [float("nan")] * len(data)
    if len(data) < period:
        return result
    seed = sum(data[:period]) / period
    result[period - 1] = seed
    k = 2.0 / (period + 1)
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(data: list[float], period: int = 14) -> list[float]:
    """Relative Strength Index (Wilder's smoothed)."""
    if not data:
        return []
    result = [float("nan")] * len(data)
    if len(data) < period + 1:
        return result
    deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, len(data)):
        delta = deltas[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def macd(
    data: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    if not data:
        return [], [], []
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    macd_line = [float("nan")] * len(data)
    for i in range(len(data)):
        if not math.isnan(fast_ema[i]) and not math.isnan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]
    valid_macd = [v for v in macd_line if not math.isnan(v)]
    if len(valid_macd) < signal_period:
        signal_line = [float("nan")] * len(data)
        histogram = [float("nan")] * len(data)
        return macd_line, signal_line, histogram
    signal_ema = ema(valid_macd, signal_period)
    signal_line = [float("nan")] * len(data)
    j = 0
    for i in range(len(data)):
        if not math.isnan(macd_line[i]):
            if j < len(signal_ema):
                signal_line[i] = signal_ema[j]
            j += 1
    histogram = [float("nan")] * len(data)
    for i in range(len(data)):
        if not math.isnan(macd_line[i]) and not math.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, histogram


def bollinger_bands(
    data: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands: returns (upper, middle, lower)."""
    if not data:
        return [], [], []
    middle = sma(data, period)
    upper = [float("nan")] * len(data)
    lower = [float("nan")] * len(data)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, middle, lower


def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """Average True Range."""
    if not highs:
        return []
    n = len(highs)
    result = [float("nan")] * n
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    if n < period:
        return result
    result[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result
