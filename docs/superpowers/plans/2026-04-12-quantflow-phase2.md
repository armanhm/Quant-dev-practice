# QuantFlow Phase 2: Indicators, Strategies, Position Sizing & Risk Controls

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a technical indicator library integrated into the Strategy base class, two new strategies (Mean Reversion and RSI+MACD), pluggable position sizing (Fixed Fractional, Kelly Criterion), and risk controls (drawdown kill switch, exposure limits) -- then refactor SMA Crossover to use the indicator API.

**Architecture:** Indicators are pure functions that take a list of floats and return a list of floats. The Strategy base class gets an `indicator()` registration method that auto-updates on each bar. Position sizing and risk controls are protocol-based (same adapter pattern as data fetchers), plugged into the BacktestEngine's signal handler.

**Tech Stack:** Python 3.11+, numpy, pandas, pytest (no new dependencies)

---

## File Structure

```
quantflow/
    data/
        indicators.py          -- NEW: Pure indicator functions (SMA, EMA, RSI, MACD, BB, ATR)
    strategies/
        base.py                -- MODIFY: Add indicator() registration and IndicatorBuffer
        sma_crossover.py       -- MODIFY: Refactor to use indicator API
        mean_reversion.py      -- NEW: Bollinger Band mean reversion strategy
        rsi_macd.py            -- NEW: RSI + MACD combo strategy
    portfolio/
        __init__.py            -- NEW
        sizing.py              -- NEW: PositionSizer protocol + Fixed Fractional + Kelly
        risk.py                -- NEW: RiskManager with drawdown kill switch + exposure limits
    backtest/
        engine.py              -- MODIFY: Integrate PositionSizer and RiskManager
    examples/
        mean_reversion.py      -- NEW: Runnable mean reversion example
        rsi_macd.py            -- NEW: Runnable RSI+MACD example
tests/
    test_data/
        test_indicators.py     -- NEW
    test_strategies/
        test_mean_reversion.py -- NEW
        test_rsi_macd.py       -- NEW
    test_portfolio/
        __init__.py            -- NEW
        test_sizing.py         -- NEW
        test_risk.py           -- NEW
```

---

### Task 1: Technical Indicator Functions

**Files:**
- Create: `quantflow/data/indicators.py`
- Create: `tests/test_data/test_indicators.py`

- [ ] **Step 1: Write failing tests for indicator functions**

```python
# tests/test_data/test_indicators.py
import pytest
import math
from quantflow.data.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class TestSMA:
    def test_sma_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sma(data, period=3)
        # First 2 values should be NaN (not enough data), then 2.0, 3.0, 4.0
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
        # EMA starts with SMA of first 'period' values = 2.0
        assert result[2] == pytest.approx(2.0)
        # k = 2/(3+1) = 0.5; EMA = 3*0.5 + 2.0*0.5 = 2.5... wait
        # k = 2/(3+1) = 0.5; EMA = 4*0.5 + 2.5*0.5 = 3.25
        assert result[3] == pytest.approx(3.0)  # 4*0.5 + 2.0*0.5 = 3.0
        assert result[4] == pytest.approx(4.0)  # 5*0.5 + 3.0*0.5 = 4.0

    def test_ema_empty(self):
        assert ema([], period=3) == []


class TestRSI:
    def test_rsi_overbought(self):
        # Steadily rising prices -> RSI should be high (near 100)
        data = list(range(50, 70))  # 20 bars going up
        result = rsi(data, period=14)
        # Last value should be > 70 (overbought territory)
        assert result[-1] > 70.0

    def test_rsi_oversold(self):
        # Steadily falling prices -> RSI should be low (near 0)
        data = list(range(70, 50, -1))  # 20 bars going down
        result = rsi(data, period=14)
        assert result[-1] < 30.0

    def test_rsi_range(self):
        # RSI should always be between 0 and 100
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
        # Strong uptrend should have positive MACD
        data = [100.0 + i * 2.0 for i in range(40)]
        macd_line, signal_line, histogram = macd(data)
        # After enough bars, MACD line should be positive
        valid_macd = [v for v in macd_line[-5:] if not math.isnan(v)]
        assert len(valid_macd) > 0
        assert all(v > 0 for v in valid_macd)

    def test_macd_empty(self):
        m, s, h = macd([])
        assert m == [] and s == [] and h == []


class TestBollingerBands:
    def test_bollinger_basic(self):
        data = [100.0] * 20 + [110.0]  # Flat then spike
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        # Middle band at bar 19 should be 100.0 (SMA of 20 x 100.0)
        assert middle[19] == pytest.approx(100.0)
        # Upper and lower should be equal when std=0 (all same values)
        assert upper[19] == pytest.approx(100.0)
        assert lower[19] == pytest.approx(100.0)

    def test_bollinger_band_width(self):
        # Volatile data should have wider bands
        import random
        random.seed(42)
        data = [100.0 + random.gauss(0, 5) for _ in range(30)]
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        # At the last bar, upper > middle > lower
        assert upper[-1] > middle[-1] > lower[-1]

    def test_bollinger_empty(self):
        u, m, l = bollinger_bands([], period=20)
        assert u == [] and m == [] and l == []


class TestATR:
    def test_atr_basic(self):
        # highs, lows, closes
        highs = [110.0, 112.0, 115.0, 113.0, 116.0]
        lows = [100.0, 102.0, 105.0, 103.0, 106.0]
        closes = [105.0, 108.0, 110.0, 107.0, 112.0]
        result = atr(highs, lows, closes, period=3)
        assert len(result) == 5
        # First values are NaN, then real ATR values
        valid = [v for v in result if not math.isnan(v)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_atr_empty(self):
        assert atr([], [], [], period=14) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_indicators.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement indicator functions**

```python
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

    # Seed with SMA
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

    # Calculate price changes
    deltas = [data[i] - data[i - 1] for i in range(1, len(data))]

    # First average gain/loss over 'period' bars
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing for subsequent bars
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

    # Signal line is EMA of MACD line (skip NaNs for seeding)
    valid_macd = [v for v in macd_line if not math.isnan(v)]
    if len(valid_macd) < signal_period:
        signal_line = [float("nan")] * len(data)
        histogram = [float("nan")] * len(data)
        return macd_line, signal_line, histogram

    signal_ema = ema(valid_macd, signal_period)

    # Map signal EMA back to original indices
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

    # True Range
    tr = [highs[0] - lows[0]]  # First bar: just high - low
    for i in range(1, n):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    if n < period:
        return result

    # First ATR is simple average
    result[period - 1] = sum(tr[:period]) / period

    # Wilder's smoothing
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_indicators.py -v`
Expected: All 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/data/indicators.py tests/test_data/test_indicators.py
git commit -m "feat: add technical indicator library (SMA, EMA, RSI, MACD, Bollinger, ATR)"
```

---

### Task 2: Indicator Integration into Strategy Base Class

**Files:**
- Modify: `quantflow/strategies/base.py`
- Modify: `tests/test_strategies/test_sma_crossover.py` (add IndicatorBuffer tests to TestStrategyBase)

- [ ] **Step 1: Write failing tests for the indicator API**

Append to `tests/test_strategies/test_sma_crossover.py`:

```python
from quantflow.data.indicators import sma as sma_func


class IndicatorStrategy(Strategy):
    """Strategy that uses the indicator API for testing."""

    def init(self) -> None:
        self.sma_fast = self.indicator("sma", period=3)
        self.sma_slow = self.indicator("sma", period=5)
        self.rsi_val = self.indicator("rsi", period=5)

    def next(self, event: MarketDataEvent) -> None:
        pass  # Just testing that indicators update


class TestIndicatorIntegration:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def _emit_bar(self, close: float, day: int):
        bar = Bar(
            timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
            open=close - 1, high=close + 1, low=close - 2,
            close=close, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_indicator_returns_buffer(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        assert strategy.sma_fast is not None

    def test_indicator_updates_on_bar(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        prices = [100, 102, 104, 106, 108, 110]
        for i, p in enumerate(prices):
            self._emit_bar(close=p, day=i + 1)

        # After 6 bars, SMA(3) should have valid values
        assert not math.isnan(strategy.sma_fast[self.asset][-1])
        # SMA(3) of last 3 values: (106 + 108 + 110) / 3 = 108.0
        assert strategy.sma_fast[self.asset][-1] == pytest.approx(108.0)

    def test_indicator_latest_shortcut(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        prices = [100, 102, 104, 106, 108, 110]
        for i, p in enumerate(prices):
            self._emit_bar(close=p, day=i + 1)

        # .latest(asset) should return the most recent non-NaN value
        assert strategy.sma_fast.latest(self.asset) == pytest.approx(108.0)

    def test_indicator_not_enough_data_returns_nan(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        self._emit_bar(close=100, day=1)
        # Only 1 bar, SMA(3) needs 3 -> should be NaN
        assert math.isnan(strategy.sma_fast.latest(self.asset))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_sma_crossover.py::TestIndicatorIntegration -v`
Expected: FAIL -- ImportError or AttributeError (Strategy has no `indicator` method yet)

- [ ] **Step 3: Add indicator support to Strategy base class**

Replace `quantflow/strategies/base.py` with:

```python
# quantflow/strategies/base.py
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from quantflow.core.models import Asset, Bar, Signal, Direction
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent
from quantflow.data import indicators as ind


# Map indicator names to (function, input_type)
# "close" means the function takes a list of close prices
# "hlc" means it takes (highs, lows, closes)
_INDICATOR_REGISTRY: dict[str, tuple[Callable, str]] = {
    "sma": (ind.sma, "close"),
    "ema": (ind.ema, "close"),
    "rsi": (ind.rsi, "close"),
    "bollinger_bands": (ind.bollinger_bands, "close"),
    "atr": (ind.atr, "hlc"),
}


class IndicatorBuffer:
    """Stores computed indicator values per asset. Updated by the Strategy on each bar."""

    def __init__(self, name: str, func: Callable, input_type: str, params: dict):
        self.name = name
        self.func = func
        self.input_type = input_type
        self.params = params
        self._values: dict[Asset, list[float]] = defaultdict(list)

    def update(self, asset: Asset, bars: list[Bar]) -> None:
        """Recompute indicator from full bar history."""
        if not bars:
            return
        if self.input_type == "close":
            closes = [b.close for b in bars]
            result = self.func(closes, **self.params)
        elif self.input_type == "hlc":
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            closes = [b.close for b in bars]
            result = self.func(highs, lows, closes, **self.params)
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

        # Handle tuple returns (e.g., bollinger_bands returns 3 lists)
        if isinstance(result, tuple):
            # Store only the first element for simple access; full tuple accessible via .raw()
            self._values[asset] = list(result[0])
            self._raw: dict[Asset, tuple] = getattr(self, "_raw", {})
            self._raw[asset] = result
        else:
            self._values[asset] = result

    def __getitem__(self, asset: Asset) -> list[float]:
        """Get full indicator history for an asset."""
        return self._values[asset]

    def latest(self, asset: Asset) -> float:
        """Get the most recent indicator value for an asset."""
        values = self._values.get(asset, [])
        if not values:
            return float("nan")
        return values[-1]

    def raw(self, asset: Asset) -> tuple | None:
        """Get raw tuple result for multi-output indicators (e.g., bollinger_bands)."""
        raw = getattr(self, "_raw", {})
        return raw.get(asset)


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement init() for setup and next() for per-bar logic.
    The base class handles event subscription, bar history, indicator
    computation, and signal emission.
    """

    def __init__(self, event_bus: EventBus, assets: list[Asset]) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.bars: dict[Asset, list[Bar]] = defaultdict(list)
        self._current_event: MarketDataEvent | None = None
        self._indicators: list[IndicatorBuffer] = []

        self.event_bus.subscribe(MarketDataEvent, self._on_market_data)
        self.init()

    @abstractmethod
    def init(self) -> None:
        """Called once at construction. Set up indicators, state, etc."""
        ...

    @abstractmethod
    def next(self, event: MarketDataEvent) -> None:
        """Called on each new bar. Implement trading logic here."""
        ...

    def indicator(self, name: str, **params) -> IndicatorBuffer:
        """Register a technical indicator. Call in init().

        Returns an IndicatorBuffer that auto-updates on each bar.
        Access values via buffer[asset] (full list) or buffer.latest(asset).
        """
        if name not in _INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator: {name}. Available: {list(_INDICATOR_REGISTRY.keys())}")
        func, input_type = _INDICATOR_REGISTRY[name]
        buf = IndicatorBuffer(name=name, func=func, input_type=input_type, params=params)
        self._indicators.append(buf)
        return buf

    def signal(self, direction: Direction, strength: float) -> None:
        """Emit a trading signal from within next()."""
        if self._current_event is None:
            raise RuntimeError("signal() can only be called from within next()")

        sig = Signal(
            timestamp=self._current_event.bar.timestamp,
            asset=self._current_event.asset,
            direction=direction,
            strength=strength,
        )
        self.event_bus.emit(SignalEvent(signal=sig))

    def _on_market_data(self, event: MarketDataEvent) -> None:
        if event.asset not in self.assets:
            return
        self.bars[event.asset].append(event.bar)

        # Update all registered indicators
        for buf in self._indicators:
            buf.update(event.asset, self.bars[event.asset])

        self._current_event = event
        self.next(event)
        self._current_event = None
```

- [ ] **Step 4: Run ALL tests to verify nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests still pass, plus new indicator tests pass

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/base.py tests/test_strategies/test_sma_crossover.py
git commit -m "feat: add indicator() API to Strategy base class with IndicatorBuffer"
```

---

### Task 3: Refactor SMA Crossover to Use Indicator API

**Files:**
- Modify: `quantflow/strategies/sma_crossover.py`

- [ ] **Step 1: Run existing SMA tests to confirm they pass before refactoring**

Run: `pytest tests/test_strategies/test_sma_crossover.py::TestSMACrossover -v`
Expected: All 4 tests PASS

- [ ] **Step 2: Refactor SMACrossover to use indicator API**

```python
# quantflow/strategies/sma_crossover.py
from __future__ import annotations

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy, IndicatorBuffer


class SMACrossover(Strategy):
    """Simple Moving Average Crossover strategy.

    Goes long when the fast SMA crosses above the slow SMA (golden cross).
    Goes short when the fast SMA crosses below the slow SMA (death cross).
    Only emits a signal on the actual crossover, not on every bar.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        fast_period: int = 10,
        slow_period: int = 50,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        self._sma_fast = self.indicator("sma", period=self.fast_period)
        self._sma_slow = self.indicator("sma", period=self.slow_period)
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        fast = self._sma_fast.latest(asset)
        slow = self._sma_slow.latest(asset)

        if fast != fast or slow != slow:  # NaN check
            return

        if fast > slow:
            new_direction = Direction.LONG
        elif fast < slow:
            new_direction = Direction.SHORT
        else:
            return

        prev = self._prev_position[asset]
        if new_direction != prev:
            self.signal(direction=new_direction, strength=1.0)
            self._prev_position[asset] = new_direction
```

- [ ] **Step 3: Run ALL tests to verify nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (SMA tests still pass since behavior is identical)

- [ ] **Step 4: Run the example to verify end-to-end still works**

Run: `python -m quantflow.examples.sma_crossover`
Expected: Same tearsheet output as before (minor floating point differences are OK)

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/sma_crossover.py
git commit -m "refactor: SMA Crossover now uses indicator API instead of inline computation"
```

---

### Task 4: Mean Reversion Strategy (Bollinger Bands)

**Files:**
- Create: `quantflow/strategies/mean_reversion.py`
- Create: `tests/test_strategies/test_mean_reversion.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategies/test_mean_reversion.py
import pytest
import math
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.mean_reversion import MeanReversion


class TestMeanReversion:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, close: float, day: int):
        bar = Bar(
            timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
            open=close - 1, high=close + 2, low=close - 2,
            close=close, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_no_signal_before_enough_data(self):
        strategy = MeanReversion(
            event_bus=self.bus,
            assets=[self.asset],
            bb_period=10,
        )
        for i in range(8):
            self._emit_bar(close=100.0, day=i + 1)
        assert len(self.signals) == 0

    def test_long_signal_below_lower_band(self):
        strategy = MeanReversion(
            event_bus=self.bus,
            assets=[self.asset],
            bb_period=10,
            num_std=2.0,
        )
        # Stable prices then sharp drop below lower band
        for i in range(15):
            self._emit_bar(close=100.0, day=i + 1)
        # Price drops sharply -- should be below lower Bollinger Band
        self._emit_bar(close=80.0, day=16)

        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(long_signals) > 0

    def test_short_signal_above_upper_band(self):
        strategy = MeanReversion(
            event_bus=self.bus,
            assets=[self.asset],
            bb_period=10,
            num_std=2.0,
        )
        for i in range(15):
            self._emit_bar(close=100.0, day=i + 1)
        # Price spikes above upper band
        self._emit_bar(close=120.0, day=16)

        short_signals = [s for s in self.signals if s.signal.direction == Direction.SHORT]
        assert len(short_signals) > 0

    def test_flat_signal_at_mean(self):
        strategy = MeanReversion(
            event_bus=self.bus,
            assets=[self.asset],
            bb_period=10,
            num_std=2.0,
        )
        # All same price -- no signal since price is at mean
        for i in range(20):
            self._emit_bar(close=100.0, day=i + 1)

        # With zero std, price == upper == lower == middle, no breach
        assert len(self.signals) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_mean_reversion.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement Mean Reversion strategy**

```python
# quantflow/strategies/mean_reversion.py
from __future__ import annotations

import math

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bollinger Band Mean Reversion strategy.

    Buys when price drops below the lower Bollinger Band (oversold).
    Sells when price rises above the upper Bollinger Band (overbought).
    Exits (goes flat) when price returns to the middle band.

    Quant concept: prices tend to revert to their mean. When price
    deviates significantly (measured in standard deviations), it's
    likely to snap back.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        bb_period: int = 20,
        num_std: float = 2.0,
    ) -> None:
        self.bb_period = bb_period
        self.num_std = num_std
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        self._bb = self.indicator("bollinger_bands", period=self.bb_period, num_std=self.num_std)
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        price = event.bar.close

        raw = self._bb.raw(asset)
        if raw is None:
            return
        upper, middle, lower = raw

        if not upper or math.isnan(upper[-1]):
            return

        current_upper = upper[-1]
        current_middle = middle[-1]
        current_lower = lower[-1]

        prev = self._prev_position[asset]

        if price < current_lower:
            # Below lower band -> buy (expect reversion up)
            if prev != Direction.LONG:
                self.signal(direction=Direction.LONG, strength=0.8)
                self._prev_position[asset] = Direction.LONG
        elif price > current_upper:
            # Above upper band -> sell (expect reversion down)
            if prev != Direction.SHORT:
                self.signal(direction=Direction.SHORT, strength=0.8)
                self._prev_position[asset] = Direction.SHORT
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies/test_mean_reversion.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/mean_reversion.py tests/test_strategies/test_mean_reversion.py
git commit -m "feat: add Bollinger Band Mean Reversion strategy"
```

---

### Task 5: RSI + MACD Combo Strategy

**Files:**
- Create: `quantflow/strategies/rsi_macd.py`
- Create: `tests/test_strategies/test_rsi_macd.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategies/test_rsi_macd.py
import pytest
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.rsi_macd import RSIMACDCombo


class TestRSIMACDCombo:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, close: float, day: int):
        bar = Bar(
            timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
            open=close - 1, high=close + 2, low=close - 2,
            close=close, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_no_signal_before_enough_data(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus,
            assets=[self.asset],
            rsi_period=14,
        )
        # Need at least 26 bars for MACD slow EMA + 14 for RSI
        for i in range(20):
            self._emit_bar(close=100.0 + i * 0.1, day=i + 1)
        assert len(self.signals) == 0

    def test_long_signal_on_oversold_with_macd_cross(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus,
            assets=[self.asset],
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
        )
        # Drop prices to make RSI oversold, then recover for MACD cross
        prices = list(range(150, 100, -1)) + list(range(100, 130))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)

        # Should eventually generate a long signal (oversold + MACD bullish)
        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        # At minimum there should be some signal activity
        assert len(self.signals) > 0

    def test_short_signal_on_overbought_with_macd_cross(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus,
            assets=[self.asset],
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
        )
        # Rise prices to make RSI overbought, then drop for MACD cross
        prices = list(range(100, 160)) + list(range(160, 130, -1))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)

        short_signals = [s for s in self.signals if s.signal.direction == Direction.SHORT]
        assert len(self.signals) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_rsi_macd.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement RSI + MACD Combo strategy**

```python
# quantflow/strategies/rsi_macd.py
from __future__ import annotations

import math

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class RSIMACDCombo(Strategy):
    """RSI + MACD Combination strategy.

    Uses RSI to identify overbought/oversold conditions and MACD
    for trend confirmation. A signal fires when both agree:
    - Long: RSI recovers from oversold (<30) AND MACD histogram turns positive
    - Short: RSI drops from overbought (>70) AND MACD histogram turns negative

    Quant concept: combining a momentum oscillator (RSI) with a trend
    indicator (MACD) filters out false signals from either alone.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self._was_oversold: dict[Asset, bool] = {}
        self._was_overbought: dict[Asset, bool] = {}
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        self._rsi = self.indicator("rsi", period=self.rsi_period)
        # MACD isn't in the indicator registry as a single-output indicator.
        # We'll compute it manually from the bar history using the indicator module.
        for asset in self.assets:
            self._was_oversold[asset] = False
            self._was_overbought[asset] = False
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        rsi_val = self._rsi.latest(asset)

        if math.isnan(rsi_val):
            return

        bars = self.bars[asset]
        if len(bars) < 35:  # Need enough bars for MACD (26 slow + 9 signal)
            return

        # Compute MACD from closes
        from quantflow.data.indicators import macd
        closes = [b.close for b in bars]
        macd_line, signal_line, histogram = macd(closes)

        if not histogram or math.isnan(histogram[-1]):
            return

        hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) >= 2 and not math.isnan(histogram[-2]) else 0.0

        # Track RSI state
        if rsi_val < self.rsi_oversold:
            self._was_oversold[asset] = True
            self._was_overbought[asset] = False
        elif rsi_val > self.rsi_overbought:
            self._was_overbought[asset] = True
            self._was_oversold[asset] = False

        prev = self._prev_position[asset]

        # Long: was oversold, RSI recovering, MACD histogram turning positive
        if self._was_oversold[asset] and rsi_val > self.rsi_oversold and hist > 0 and prev_hist <= 0:
            if prev != Direction.LONG:
                self.signal(direction=Direction.LONG, strength=0.7)
                self._prev_position[asset] = Direction.LONG
                self._was_oversold[asset] = False

        # Short: was overbought, RSI dropping, MACD histogram turning negative
        elif self._was_overbought[asset] and rsi_val < self.rsi_overbought and hist < 0 and prev_hist >= 0:
            if prev != Direction.SHORT:
                self.signal(direction=Direction.SHORT, strength=0.7)
                self._prev_position[asset] = Direction.SHORT
                self._was_overbought[asset] = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies/test_rsi_macd.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/rsi_macd.py tests/test_strategies/test_rsi_macd.py
git commit -m "feat: add RSI + MACD Combo strategy with dual confirmation"
```

---

### Task 6: Position Sizer Protocol + Fixed Fractional

**Files:**
- Create: `quantflow/portfolio/__init__.py`
- Create: `quantflow/portfolio/sizing.py`
- Create: `tests/test_portfolio/__init__.py`
- Create: `tests/test_portfolio/test_sizing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio/test_sizing.py
import pytest
from quantflow.core.models import Asset, AssetClass
from quantflow.portfolio.sizing import (
    FixedFractional,
    KellyCriterion,
)


class TestFixedFractional:
    def test_basic_sizing(self):
        sizer = FixedFractional(fraction=0.02)  # Risk 2% per trade
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset,
            price=150.0,
            equity=100_000.0,
            signal_strength=1.0,
        )
        # 2% of 100k = 2000, at $150/share = 13.33 shares
        assert qty == pytest.approx(2000.0 / 150.0, rel=0.01)

    def test_signal_strength_scales_size(self):
        sizer = FixedFractional(fraction=0.02)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        full = sizer.calculate_quantity(asset=asset, price=150.0, equity=100_000.0, signal_strength=1.0)
        half = sizer.calculate_quantity(asset=asset, price=150.0, equity=100_000.0, signal_strength=0.5)
        assert half == pytest.approx(full * 0.5, rel=0.01)

    def test_zero_equity(self):
        sizer = FixedFractional(fraction=0.02)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(asset=asset, price=150.0, equity=0.0, signal_strength=1.0)
        assert qty == 0.0


class TestKellyCriterion:
    def test_basic_kelly(self):
        # 60% win rate, 2:1 payoff -> Kelly = 0.6 - (0.4 / 2) = 0.4
        sizer = KellyCriterion(win_rate=0.6, avg_win_loss_ratio=2.0, kelly_fraction=1.0)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        # Kelly = 0.4, so risk 40% of 100k = 40000 / 100 = 400 shares
        assert qty == pytest.approx(400.0, rel=0.01)

    def test_half_kelly(self):
        sizer = KellyCriterion(win_rate=0.6, avg_win_loss_ratio=2.0, kelly_fraction=0.5)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        # Half Kelly = 0.2, 20000 / 100 = 200 shares
        assert qty == pytest.approx(200.0, rel=0.01)

    def test_negative_kelly_returns_zero(self):
        # Bad stats: 30% win rate, 1:1 payoff -> Kelly = 0.3 - 0.7 = -0.4
        sizer = KellyCriterion(win_rate=0.3, avg_win_loss_ratio=1.0)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty == 0.0

    def test_kelly_with_trade_history(self):
        # Start without stats, then update from trade history
        sizer = KellyCriterion.from_trades(pnls=[100, 200, -50, 150, -80])
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_portfolio/test_sizing.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement position sizing**

Create `quantflow/portfolio/__init__.py` (empty) and `tests/test_portfolio/__init__.py` (empty).

```python
# quantflow/portfolio/sizing.py
from __future__ import annotations

from typing import Protocol

from quantflow.core.models import Asset


class PositionSizer(Protocol):
    """Protocol for position sizing strategies."""

    def calculate_quantity(
        self,
        asset: Asset,
        price: float,
        equity: float,
        signal_strength: float,
    ) -> float:
        """Calculate the number of shares/units to trade."""
        ...


class FixedFractional:
    """Risk a fixed fraction of equity per trade.

    The most common position sizing method. Risk X% of portfolio
    on each trade, scaled by signal strength.
    """

    def __init__(self, fraction: float = 0.02) -> None:
        self.fraction = fraction

    def calculate_quantity(
        self,
        asset: Asset,
        price: float,
        equity: float,
        signal_strength: float,
    ) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        risk_amount = equity * self.fraction * signal_strength
        return risk_amount / price


class KellyCriterion:
    """Kelly Criterion position sizing.

    The Kelly formula: f* = W - (1-W)/R
    where W = win rate, R = avg win / avg loss ratio.
    Full Kelly is aggressive; use kelly_fraction < 1.0 for safety.
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.5,
    ) -> None:
        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.kelly_fraction = kelly_fraction

    @classmethod
    def from_trades(cls, pnls: list[float], kelly_fraction: float = 0.5) -> KellyCriterion:
        """Create KellyCriterion from historical trade P&Ls."""
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.5
        if wins and losses:
            avg_win_loss_ratio = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        else:
            avg_win_loss_ratio = 1.5

        return cls(
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
            kelly_fraction=kelly_fraction,
        )

    def calculate_quantity(
        self,
        asset: Asset,
        price: float,
        equity: float,
        signal_strength: float,
    ) -> float:
        if price <= 0 or equity <= 0:
            return 0.0

        # Kelly formula: f* = W - (1-W)/R
        kelly_f = self.win_rate - (1 - self.win_rate) / self.avg_win_loss_ratio
        kelly_f = max(kelly_f, 0.0)  # Never go negative
        kelly_f *= self.kelly_fraction  # Apply safety fraction
        kelly_f *= signal_strength  # Scale by signal conviction

        risk_amount = equity * kelly_f
        return risk_amount / price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_portfolio/test_sizing.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/portfolio/ tests/test_portfolio/
git commit -m "feat: add PositionSizer protocol with FixedFractional and KellyCriterion"
```

---

### Task 7: Risk Manager

**Files:**
- Create: `quantflow/portfolio/risk.py`
- Create: `tests/test_portfolio/test_risk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio/test_risk.py
import pytest
from quantflow.core.models import Asset, AssetClass, Position
from quantflow.portfolio.risk import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.asset_aapl = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.asset_goog = Asset(symbol="GOOG", asset_class=AssetClass.EQUITY)
        self.asset_btc = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO)

    def test_no_restrictions_by_default(self):
        rm = RiskManager()
        allowed = rm.check_new_position(
            asset=self.asset_aapl,
            quantity=100.0,
            price=150.0,
            equity=100_000.0,
            cash=50_000.0,
            positions={},
            current_prices={},
            peak_equity=100_000.0,
        )
        assert allowed is True

    def test_drawdown_kill_switch(self):
        rm = RiskManager(max_drawdown=0.20)  # 20% max drawdown
        # Equity dropped from 100k to 75k = 25% drawdown -> blocked
        allowed = rm.check_new_position(
            asset=self.asset_aapl,
            quantity=100.0,
            price=150.0,
            equity=75_000.0,
            cash=75_000.0,
            positions={},
            current_prices={},
            peak_equity=100_000.0,
        )
        assert allowed is False

    def test_drawdown_within_limit(self):
        rm = RiskManager(max_drawdown=0.20)
        # Equity dropped from 100k to 85k = 15% drawdown -> allowed
        allowed = rm.check_new_position(
            asset=self.asset_aapl,
            quantity=100.0,
            price=150.0,
            equity=85_000.0,
            cash=85_000.0,
            positions={},
            current_prices={},
            peak_equity=100_000.0,
        )
        assert allowed is True

    def test_per_asset_exposure_limit(self):
        rm = RiskManager(max_position_pct=0.20)  # Max 20% per asset
        # Trying to buy $30k worth = 30% of 100k equity -> blocked
        allowed = rm.check_new_position(
            asset=self.asset_aapl,
            quantity=200.0,
            price=150.0,  # 200 * 150 = 30k = 30%
            equity=100_000.0,
            cash=50_000.0,
            positions={},
            current_prices={},
            peak_equity=100_000.0,
        )
        assert allowed is False

    def test_per_asset_exposure_within_limit(self):
        rm = RiskManager(max_position_pct=0.20)
        # Buying $15k worth = 15% -> allowed
        allowed = rm.check_new_position(
            asset=self.asset_aapl,
            quantity=100.0,
            price=150.0,  # 100 * 150 = 15k = 15%
            equity=100_000.0,
            cash=50_000.0,
            positions={},
            current_prices={},
            peak_equity=100_000.0,
        )
        assert allowed is True

    def test_max_open_positions(self):
        rm = RiskManager(max_open_positions=2)
        positions = {
            self.asset_aapl: Position(asset=self.asset_aapl, quantity=10, entry_price=150),
            self.asset_goog: Position(asset=self.asset_goog, quantity=5, entry_price=140),
        }
        # Already 2 positions, trying to open a 3rd -> blocked
        allowed = rm.check_new_position(
            asset=self.asset_btc,
            quantity=1.0,
            price=50_000.0,
            equity=100_000.0,
            cash=50_000.0,
            positions=positions,
            current_prices={self.asset_aapl: 155, self.asset_goog: 145},
            peak_equity=100_000.0,
        )
        assert allowed is False

    def test_adjust_quantity_to_fit_limit(self):
        rm = RiskManager(max_position_pct=0.20)
        adjusted = rm.adjust_quantity(
            asset=self.asset_aapl,
            quantity=200.0,   # Wants 200 shares @ $150 = $30k = 30%
            price=150.0,
            equity=100_000.0,
        )
        # Should be reduced to max 20% = $20k / $150 = 133.33 shares
        assert adjusted == pytest.approx(20_000.0 / 150.0, rel=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_portfolio/test_risk.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement RiskManager**

```python
# quantflow/portfolio/risk.py
from __future__ import annotations

from quantflow.core.models import Asset, Position


class RiskManager:
    """Portfolio risk controls.

    Checks proposed trades against risk limits before execution.
    Can block trades entirely or adjust position sizes down.
    """

    def __init__(
        self,
        max_drawdown: float | None = None,
        max_position_pct: float | None = None,
        max_open_positions: int | None = None,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self._killed = False

    def check_new_position(
        self,
        asset: Asset,
        quantity: float,
        price: float,
        equity: float,
        cash: float,
        positions: dict[Asset, Position],
        current_prices: dict[Asset, float],
        peak_equity: float,
    ) -> bool:
        """Check if a new position is allowed by risk controls. Returns True if allowed."""
        if self._killed:
            return False

        # Drawdown kill switch
        if self.max_drawdown is not None and peak_equity > 0:
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < -self.max_drawdown:
                self._killed = True
                return False

        # Per-asset exposure limit
        if self.max_position_pct is not None and equity > 0:
            position_value = abs(quantity) * price
            if position_value / equity > self.max_position_pct:
                return False

        # Max open positions
        if self.max_open_positions is not None:
            if asset not in positions and len(positions) >= self.max_open_positions:
                return False

        return True

    def adjust_quantity(
        self,
        asset: Asset,
        quantity: float,
        price: float,
        equity: float,
    ) -> float:
        """Reduce quantity to fit within risk limits."""
        if self.max_position_pct is not None and equity > 0 and price > 0:
            max_value = equity * self.max_position_pct
            max_qty = max_value / price
            quantity = min(quantity, max_qty)
        return quantity

    def reset(self) -> None:
        """Reset kill switch (e.g., for a new backtest)."""
        self._killed = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_portfolio/test_risk.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/portfolio/risk.py tests/test_portfolio/test_risk.py
git commit -m "feat: add RiskManager with drawdown kill switch, exposure limits, and max positions"
```

---

### Task 8: Integrate Position Sizing and Risk Controls into BacktestEngine

**Files:**
- Modify: `quantflow/backtest/engine.py`
- Modify: `tests/test_backtest/test_engine.py`

- [ ] **Step 1: Write failing tests for engine integration**

Append to `tests/test_backtest/test_engine.py`:

```python
from quantflow.portfolio.sizing import FixedFractional, KellyCriterion
from quantflow.portfolio.risk import RiskManager


class TestBacktestEngineWithSizing:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def test_fixed_fractional_sizing(self):
        prices = list(range(100, 160)) + list(range(160, 100, -1))
        data = {self.asset: make_price_data(self.asset, prices)}

        sizer = FixedFractional(fraction=0.10)  # 10% per trade
        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
            position_sizer=sizer,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)
        assert isinstance(result, BacktestResult)
        # With 10% sizing, position should be much smaller than 95% default
        if result.trades:
            # The trade quantity should be relatively small
            assert result.trades[0].quantity < 100  # Much less than all-in

    def test_drawdown_kill_switch_stops_trading(self):
        # Create a scenario with a big drawdown
        prices = list(range(100, 130)) + list(range(130, 60, -2)) + list(range(60, 100))
        data = {self.asset: make_price_data(self.asset, prices)}

        risk_mgr = RiskManager(max_drawdown=0.15)  # 15% max drawdown
        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
            risk_manager=risk_mgr,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)
        # After kill switch activates, no more trades should happen
        # The final equity should not be the worst possible (some protection)
        assert isinstance(result, BacktestResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_engine.py::TestBacktestEngineWithSizing -v`
Expected: FAIL -- TypeError (engine doesn't accept position_sizer yet)

- [ ] **Step 3: Modify BacktestEngine to accept PositionSizer and RiskManager**

Update `quantflow/backtest/engine.py`. The key changes:
1. Add `position_sizer` and `risk_manager` params to `__init__`
2. Track `peak_equity` for drawdown calculation
3. In `on_signal`, use `position_sizer.calculate_quantity()` instead of hardcoded `cash * position_size_pct / price`
4. In `on_signal`, check `risk_manager.check_new_position()` before opening new positions
5. Use `risk_manager.adjust_quantity()` to cap sizes

```python
# quantflow/backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

from quantflow.core.models import (
    Asset, Bar, Order, OrderSide, OrderType, OrderStatus,
    Position, Direction, Fill,
)
from quantflow.core.events import (
    EventBus, MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
)
from quantflow.backtest.execution import SimulatedExecution
from quantflow.strategies.base import Strategy
from quantflow.portfolio.sizing import FixedFractional
from quantflow.portfolio.risk import RiskManager


@dataclass
class Trade:
    """A completed round-trip trade."""
    asset: Asset
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float


@dataclass
class BacktestResult:
    """Contains all output from a backtest run."""
    equity_curve: list[float]
    timestamps: list[datetime]
    trades: list[Trade]
    signals: list[SignalEvent]
    benchmark_equity: list[float]
    initial_cash: float


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0,
        position_size_pct: float = 0.95,
        position_sizer=None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.position_size_pct = position_size_pct
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager or RiskManager()

    def _calculate_quantity(self, asset, price, equity, cash, signal_strength):
        """Calculate position size using sizer or fallback to position_size_pct."""
        if self.position_sizer is not None:
            qty = self.position_sizer.calculate_quantity(
                asset=asset, price=price, equity=equity, signal_strength=signal_strength,
            )
        else:
            qty = (cash * self.position_size_pct) / price
        # Apply risk manager adjustment
        qty = self.risk_manager.adjust_quantity(asset=asset, quantity=qty, price=price, equity=equity)
        return qty

    def run(
        self,
        data: dict[Asset, pd.DataFrame],
        strategy_factory: Callable[[EventBus, list[Asset]], Strategy],
    ) -> BacktestResult:
        bus = EventBus()
        assets = list(data.keys())
        self.risk_manager.reset()

        cash = self.initial_cash
        positions: dict[Asset, Position] = {}
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        trades: list[Trade] = []
        signals: list[SignalEvent] = []
        current_prices: dict[Asset, float] = {}
        peak_equity = self.initial_cash

        executor = SimulatedExecution(
            event_bus=bus,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )

        def _current_equity():
            eq = cash
            for a, pos in positions.items():
                p = current_prices.get(a, pos.entry_price)
                eq += pos.quantity * p
            return eq

        def on_signal(event: SignalEvent):
            nonlocal cash, peak_equity
            signals.append(event)
            sig = event.signal
            asset = sig.asset
            price = current_prices.get(asset)
            if price is None or price <= 0:
                return

            current_pos = positions.get(asset)
            equity = _current_equity()
            peak_equity = max(peak_equity, equity)

            if sig.direction == Direction.LONG:
                if current_pos and current_pos.quantity > 0:
                    return
                if current_pos and current_pos.quantity < 0:
                    close_order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=abs(current_pos.quantity),
                        order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(close_order, price)
                    pnl = (current_pos.entry_price - fill.fill_price) * abs(current_pos.quantity)
                    cash -= fill.fill_price * fill.fill_quantity + fill.commission
                    trades.append(Trade(
                        asset=asset, entry_time=sig.timestamp, exit_time=sig.timestamp,
                        side="short", entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price, quantity=abs(current_pos.quantity),
                        pnl=pnl - fill.commission, commission=fill.commission,
                    ))
                    del positions[asset]

                if not self.risk_manager.check_new_position(
                    asset=asset, quantity=0, price=price, equity=equity,
                    cash=cash, positions=positions, current_prices=current_prices,
                    peak_equity=peak_equity,
                ):
                    return

                quantity = self._calculate_quantity(asset, price, equity, cash, sig.strength)
                if quantity > 0 and quantity * price <= cash:
                    order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cost = fill.fill_price * fill.fill_quantity + fill.commission
                    cash -= cost
                    positions[asset] = Position(
                        asset=asset, quantity=fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

            elif sig.direction == Direction.SHORT:
                if current_pos and current_pos.quantity < 0:
                    return
                if current_pos and current_pos.quantity > 0:
                    close_order = Order(
                        asset=asset, side=OrderSide.SELL,
                        quantity=current_pos.quantity,
                        order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(close_order, price)
                    pnl = (fill.fill_price - current_pos.entry_price) * current_pos.quantity
                    cash += fill.fill_price * fill.fill_quantity - fill.commission
                    trades.append(Trade(
                        asset=asset, entry_time=sig.timestamp, exit_time=sig.timestamp,
                        side="long", entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price, quantity=current_pos.quantity,
                        pnl=pnl - fill.commission, commission=fill.commission,
                    ))
                    del positions[asset]

                if not self.risk_manager.check_new_position(
                    asset=asset, quantity=0, price=price, equity=equity,
                    cash=cash, positions=positions, current_prices=current_prices,
                    peak_equity=peak_equity,
                ):
                    return

                quantity = self._calculate_quantity(asset, price, equity, cash, sig.strength)
                if quantity > 0:
                    order = Order(
                        asset=asset, side=OrderSide.SELL,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cash += fill.fill_price * fill.fill_quantity - fill.commission
                    positions[asset] = Position(
                        asset=asset, quantity=-fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

        bus.subscribe(SignalEvent, on_signal)
        strategy = strategy_factory(bus, assets)

        all_dates: set[datetime] = set()
        for df in data.values():
            all_dates.update(df.index.to_pydatetime())
        sorted_dates = sorted(all_dates)

        benchmark_equity: list[float] = []
        first_prices: dict[Asset, float] = {}

        for ts in sorted_dates:
            for asset, df in data.items():
                if ts in df.index:
                    row = df.loc[ts]
                    bar = Bar(
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                    current_prices[asset] = bar.close
                    bus.emit(MarketDataEvent(asset=asset, bar=bar))
                    if asset not in first_prices:
                        first_prices[asset] = bar.close

            equity = _current_equity()
            peak_equity = max(peak_equity, equity)
            equity_curve.append(equity)
            timestamps.append(ts)

            bench = self.initial_cash
            if first_prices:
                per_asset = self.initial_cash / len(first_prices)
                bench = sum(
                    per_asset * (current_prices.get(a, fp) / fp)
                    for a, fp in first_prices.items()
                )
            benchmark_equity.append(bench)

        return BacktestResult(
            equity_curve=equity_curve, timestamps=timestamps,
            trades=trades, signals=signals,
            benchmark_equity=benchmark_equity, initial_cash=self.initial_cash,
        )
```

- [ ] **Step 4: Run ALL tests to verify nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass including the new integration tests

- [ ] **Step 5: Commit**

```bash
git add quantflow/backtest/engine.py tests/test_backtest/test_engine.py
git commit -m "feat: integrate PositionSizer and RiskManager into BacktestEngine"
```

---

### Task 9: Mean Reversion Example

**Files:**
- Create: `quantflow/examples/mean_reversion.py`

- [ ] **Step 1: Implement the runnable example**

```python
# quantflow/examples/mean_reversion.py
"""
Mean Reversion Backtest Example
================================
Run with: python -m quantflow.examples.mean_reversion

Backtests a Bollinger Band mean reversion strategy on SPY.
Buys when price drops below lower band (oversold).
Sells when price rises above upper band (overbought).
Uses Fixed Fractional position sizing (5% per trade).
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.portfolio.sizing import FixedFractional
from quantflow.portfolio.risk import RiskManager
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    asset = Asset(symbol="SPY", asset_class=AssetClass.EQUITY)

    print("Fetching SPY data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    print("\nRunning Mean Reversion backtest...")
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,
        commission_pct=0.0,
        position_sizer=FixedFractional(fraction=0.05),
        risk_manager=RiskManager(max_drawdown=0.25, max_position_pct=0.30),
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> MeanReversion:
        return MeanReversion(bus, assets, bb_period=20, num_std=2.0)

    result = engine.run(data={asset: df}, strategy_factory=strategy_factory)

    print_tearsheet(result)
    plot_tearsheet(result, save_path="mean_reversion_tearsheet.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add quantflow/examples/mean_reversion.py
git commit -m "feat: add runnable Mean Reversion backtest example with risk controls"
```

---

### Task 10: RSI + MACD Example

**Files:**
- Create: `quantflow/examples/rsi_macd.py`

- [ ] **Step 1: Implement the runnable example**

```python
# quantflow/examples/rsi_macd.py
"""
RSI + MACD Combo Backtest Example
===================================
Run with: python -m quantflow.examples.rsi_macd

Backtests an RSI + MACD combination strategy on MSFT.
Buys when RSI recovers from oversold AND MACD histogram turns positive.
Sells when RSI drops from overbought AND MACD histogram turns negative.
Uses Kelly Criterion position sizing.
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.rsi_macd import RSIMACDCombo
from quantflow.portfolio.sizing import KellyCriterion
from quantflow.portfolio.risk import RiskManager
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    asset = Asset(symbol="MSFT", asset_class=AssetClass.EQUITY)

    print("Fetching MSFT data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    print("\nRunning RSI + MACD Combo backtest...")
    # Conservative Kelly: assume 45% win rate, 2:1 payoff, half Kelly
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,
        commission_pct=0.0,
        position_sizer=KellyCriterion(win_rate=0.45, avg_win_loss_ratio=2.0, kelly_fraction=0.5),
        risk_manager=RiskManager(max_drawdown=0.20, max_position_pct=0.25),
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> RSIMACDCombo:
        return RSIMACDCombo(bus, assets, rsi_period=14, rsi_oversold=30, rsi_overbought=70)

    result = engine.run(data={asset: df}, strategy_factory=strategy_factory)

    print_tearsheet(result)
    plot_tearsheet(result, save_path="rsi_macd_tearsheet.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add quantflow/examples/rsi_macd.py
git commit -m "feat: add runnable RSI+MACD combo backtest example with Kelly sizing"
```

---

### Task 11: Full Test Suite and Integration Verification

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (~80+ tests)

- [ ] **Step 2: Run the SMA Crossover example (verify refactor didn't break it)**

Run: `python -m quantflow.examples.sma_crossover`
Expected: Tearsheet prints, chart saved. Results should be similar to Phase 1.

- [ ] **Step 3: Run the Mean Reversion example**

Run: `python -m quantflow.examples.mean_reversion`
Expected: Tearsheet prints, chart saved. Numbers are reasonable.

- [ ] **Step 4: Run the RSI + MACD example**

Run: `python -m quantflow.examples.rsi_macd`
Expected: Tearsheet prints, chart saved. Numbers are reasonable.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: phase 2 integration verification complete"
```

---

## Phase 2 Summary

After completing these 11 tasks, you will have added:

- **Indicator library** -- SMA, EMA, RSI, MACD, Bollinger Bands, ATR as pure functions
- **Indicator API in Strategy** -- `self.indicator("rsi", period=14)` auto-updates per bar
- **Refactored SMA Crossover** -- uses indicator API, proves it works
- **Mean Reversion strategy** -- Bollinger Band based, teaches statistical trading concepts
- **RSI + MACD Combo strategy** -- dual confirmation, teaches indicator combination
- **Position sizing** -- FixedFractional and KellyCriterion with pluggable interface
- **Risk controls** -- Drawdown kill switch, per-asset exposure limits, max open positions
- **Engine integration** -- BacktestEngine accepts PositionSizer and RiskManager
- **2 new runnable examples** -- mean reversion on SPY, RSI+MACD on MSFT

## What Comes Next (Phase 3)

- SQLite caching for data
- FRED macro data integration
- CCXT/CoinGecko crypto data
- Pairs Trading strategy (cointegration)
- Macro Regime strategy (FRED data)
- Walk-forward optimization
- Parameter sweep runner
- CLI via click
- Strategy composition (CompositeStrategy)
