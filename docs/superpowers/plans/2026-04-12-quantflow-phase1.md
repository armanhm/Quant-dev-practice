# QuantFlow Phase 1: End-to-End Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the thinnest possible vertical slice -- from data fetch to equity curve chart -- so you can run `python -m quantflow.examples.sma_crossover` and see an SMA crossover backtest on AAPL with performance metrics and an equity curve plot.

**Architecture:** Four modules built bottom-up: Core domain models + event bus, Data Engine (yfinance only, in-memory), Strategy framework + SMA Crossover, Event-driven Backtester + basic analytics. No CLI, no dashboard, no caching -- just a working backtest you can run and see.

**Tech Stack:** Python 3.11+, pandas, numpy, yfinance, matplotlib, pytest

---

## File Structure

```
quantflow/
    __init__.py
    core/
        __init__.py
        models.py          -- Asset, OHLCV bar, Signal, Order, Fill, Position, Portfolio dataclasses
        events.py          -- Event classes and EventBus
        interfaces.py      -- DataFetcher Protocol, Strategy ABC, ExecutionModel ABC
    data/
        __init__.py
        yahoo_fetcher.py   -- yfinance DataFetcher implementation
    strategies/
        __init__.py
        base.py            -- Strategy abstract base class
        sma_crossover.py   -- SMA Crossover concrete strategy
    backtest/
        __init__.py
        engine.py          -- Main backtest loop
        execution.py       -- Simulated execution (slippage + commission)
    analytics/
        __init__.py
        metrics.py         -- Sharpe, drawdown, CAGR, win rate, etc.
        tearsheet.py       -- Console + matplotlib tearsheet
    examples/
        __init__.py
        sma_crossover.py   -- Runnable example: python -m quantflow.examples.sma_crossover
tests/
    __init__.py
    test_core/
        __init__.py
        test_models.py
        test_events.py
    test_data/
        __init__.py
        test_yahoo_fetcher.py
    test_strategies/
        __init__.py
        test_sma_crossover.py
    test_backtest/
        __init__.py
        test_engine.py
        test_execution.py
    test_analytics/
        __init__.py
        test_metrics.py
pyproject.toml
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `quantflow/__init__.py`
- Create: all `__init__.py` files for packages
- Create: `tests/__init__.py` and all test package `__init__.py` files

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "quantflow"
version = "0.1.0"
description = "Quantitative research and backtesting platform"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "yfinance>=0.2.18",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create all package __init__.py files**

Create these empty files:
```
quantflow/__init__.py
quantflow/core/__init__.py
quantflow/data/__init__.py
quantflow/strategies/__init__.py
quantflow/backtest/__init__.py
quantflow/analytics/__init__.py
quantflow/examples/__init__.py
tests/__init__.py
tests/test_core/__init__.py
tests/test_data/__init__.py
tests/test_strategies/__init__.py
tests/test_backtest/__init__.py
tests/test_analytics/__init__.py
```

- [ ] **Step 3: Install the project in development mode**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed quantflow and dev dependencies

- [ ] **Step 4: Verify pytest runs**

Run: `pytest -v`
Expected: "no tests ran" (0 collected), exit code 5 (no tests). No import errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml quantflow/ tests/
git commit -m "feat: scaffold quantflow project structure with pyproject.toml"
```

---

### Task 2: Core Domain Models

**Files:**
- Create: `quantflow/core/models.py`
- Create: `tests/test_core/test_models.py`

- [ ] **Step 1: Write failing tests for domain models**

```python
# tests/test_core/test_models.py
import pytest
from datetime import datetime, timezone
from quantflow.core.models import (
    Asset,
    AssetClass,
    Bar,
    Signal,
    Direction,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Fill,
    Position,
)


class TestAsset:
    def test_create_equity(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert asset.symbol == "AAPL"
        assert asset.asset_class == AssetClass.EQUITY
        assert asset.exchange is None

    def test_create_crypto(self):
        asset = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO, exchange="binance")
        assert asset.exchange == "binance"

    def test_asset_equality(self):
        a1 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        a2 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert a1 == a2

    def test_asset_hash(self):
        a1 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        a2 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert hash(a1) == hash(a2)
        assert len({a1, a2}) == 1


class TestBar:
    def test_create_bar(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bar = Bar(
            timestamp=ts,
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000.0,
        )
        assert bar.close == 153.0
        assert bar.timestamp == ts


class TestSignal:
    def test_create_long_signal(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        signal = Signal(
            timestamp=ts,
            asset=asset,
            direction=Direction.LONG,
            strength=0.8,
        )
        assert signal.direction == Direction.LONG
        assert signal.strength == 0.8

    def test_signal_strength_clamped(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            Signal(timestamp=ts, asset=asset, direction=Direction.LONG, strength=1.5)


class TestOrder:
    def test_create_market_order(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        order = Order(
            asset=asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 100.0


class TestFill:
    def test_create_fill(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        order = Order(asset=asset, side=OrderSide.BUY, quantity=100.0, order_type=OrderType.MARKET)
        fill = Fill(
            order=order,
            fill_price=150.0,
            fill_quantity=100.0,
            commission=0.0,
            slippage=0.075,
        )
        assert fill.fill_price == 150.0


class TestPosition:
    def test_create_position(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        assert pos.quantity == 100.0

    def test_unrealized_pnl(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        pnl = pos.unrealized_pnl(current_price=160.0)
        assert pnl == 1000.0  # (160 - 150) * 100

    def test_unrealized_pnl_short(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=-50.0, entry_price=150.0)
        pnl = pos.unrealized_pnl(current_price=140.0)
        assert pnl == 500.0  # (150 - 140) * 50, short profits when price drops

    def test_market_value(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        assert pos.market_value(current_price=160.0) == 16000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_models.py -v`
Expected: FAIL -- ModuleNotFoundError (models.py doesn't exist yet)

- [ ] **Step 3: Implement domain models**

```python
# quantflow/core/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AssetClass(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    MACRO = "macro"
    OPTION = "option"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_class: AssetClass
    exchange: str | None = None


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    asset: Asset
    direction: Direction
    strength: float

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between 0 and 1, got {self.strength}")


@dataclass
class Order:
    asset: Asset
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING


@dataclass(frozen=True)
class Fill:
    order: Order
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float


@dataclass
class Position:
    asset: Asset
    quantity: float
    entry_price: float

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity

    def market_value(self, current_price: float) -> float:
        return abs(self.quantity) * current_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_models.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/core/models.py tests/test_core/test_models.py
git commit -m "feat: add core domain models (Asset, Bar, Signal, Order, Fill, Position)"
```

---

### Task 3: Event Bus

**Files:**
- Create: `quantflow/core/events.py`
- Create: `tests/test_core/test_events.py`

- [ ] **Step 1: Write failing tests for events and event bus**

```python
# tests/test_core/test_events.py
from datetime import datetime, timezone
from quantflow.core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventBus,
)
from quantflow.core.models import (
    Asset,
    AssetClass,
    Bar,
    Signal,
    Direction,
    Order,
    OrderSide,
    OrderType,
    Fill,
)


class TestEvents:
    def test_market_data_event(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        event = MarketDataEvent(asset=asset, bar=bar)
        assert event.asset == asset
        assert event.bar == bar

    def test_signal_event(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            asset=asset, direction=Direction.LONG, strength=0.8,
        )
        event = SignalEvent(signal=signal)
        assert event.signal.direction == Direction.LONG


class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []

        def handler(event: MarketDataEvent):
            received.append(event)

        bus.subscribe(MarketDataEvent, handler)

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        event = MarketDataEvent(asset=asset, bar=bar)
        bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self):
        bus = EventBus()
        received_a = []
        received_b = []

        bus.subscribe(MarketDataEvent, lambda e: received_a.append(e))
        bus.subscribe(MarketDataEvent, lambda e: received_b.append(e))

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        bus.emit(MarketDataEvent(asset=asset, bar=bar))

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_no_crosstalk_between_event_types(self):
        bus = EventBus()
        received = []

        bus.subscribe(SignalEvent, lambda e: received.append(e))

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        bus.emit(MarketDataEvent(asset=asset, bar=bar))

        assert len(received) == 0

    def test_emit_with_no_subscribers(self):
        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        # Should not raise
        bus.emit(MarketDataEvent(asset=asset, bar=bar))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_events.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement events and event bus**

```python
# quantflow/core/events.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from quantflow.core.models import Asset, Bar, Signal, Order, Fill


@dataclass
class Event:
    """Base class for all events."""
    pass


@dataclass
class MarketDataEvent(Event):
    asset: Asset
    bar: Bar


@dataclass
class SignalEvent(Event):
    signal: Signal


@dataclass
class OrderEvent(Event):
    order: Order


@dataclass
class FillEvent(Event):
    fill: Fill


class EventBus:
    """Simple synchronous event bus. Handlers are called in registration order."""

    def __init__(self):
        self._handlers: dict[type[Event], list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type[Event], handler: Callable[[Any], None]) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers[type(event)]:
            handler(event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_events.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/core/events.py tests/test_core/test_events.py
git commit -m "feat: add event system with MarketData, Signal, Order, Fill events and EventBus"
```

---

### Task 4: Data Fetcher Interface and Yahoo Finance Implementation

**Files:**
- Create: `quantflow/core/interfaces.py`
- Create: `quantflow/data/yahoo_fetcher.py`
- Create: `tests/test_data/test_yahoo_fetcher.py`

- [ ] **Step 1: Write the DataFetcher protocol and tests**

```python
# quantflow/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import pandas as pd

from quantflow.core.models import Asset, AssetClass


class DataFetcher(Protocol):
    """Protocol for all data source adapters."""

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data. Returns DataFrame with DatetimeIndex (UTC)
        and columns: [open, high, low, close, volume]."""
        ...

    def supported_asset_classes(self) -> list[AssetClass]:
        ...
```

- [ ] **Step 2: Write failing tests for YahooFetcher**

```python
# tests/test_data/test_yahoo_fetcher.py
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
        # yfinance sometimes returns capitalized columns
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_data/test_yahoo_fetcher.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 4: Implement YahooFetcher**

```python
# quantflow/data/yahoo_fetcher.py
from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from quantflow.core.models import Asset, AssetClass

TIMEFRAME_MAP = {
    "1d": "1d",
    "1h": "1h",
    "5m": "5m",
    "1m": "1m",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YahooFetcher:
    """Fetches OHLCV data from Yahoo Finance via yfinance."""

    def fetch_ohlcv(
        self,
        asset: Asset,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        yf_interval = TIMEFRAME_MAP.get(timeframe, "1d")

        df = yf.download(
            asset.symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            return df

        # Normalize column names to lowercase
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]

        # Keep only OHLCV columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in expected_cols if c in df.columns]]

        # Ensure UTC timezone
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")

        return df

    def supported_asset_classes(self) -> list[AssetClass]:
        return [
            AssetClass.EQUITY,
            AssetClass.CRYPTO,
            AssetClass.FOREX,
            AssetClass.COMMODITY,
            AssetClass.INDEX,
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_yahoo_fetcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add quantflow/core/interfaces.py quantflow/data/yahoo_fetcher.py tests/test_data/test_yahoo_fetcher.py
git commit -m "feat: add DataFetcher protocol and YahooFetcher implementation"
```

---

### Task 5: Strategy Base Class

**Files:**
- Create: `quantflow/strategies/base.py`
- Create: `tests/test_strategies/test_sma_crossover.py` (we write the test file now, targeting both base and SMA)

- [ ] **Step 1: Write failing tests for the Strategy base class**

```python
# tests/test_strategies/test_sma_crossover.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.base import Strategy


class DummyStrategy(Strategy):
    """A minimal strategy for testing the base class."""

    def init(self) -> None:
        self.call_count = 0

    def next(self, event: MarketDataEvent) -> None:
        self.call_count += 1
        if event.bar.close > 150.0:
            self.signal(direction=Direction.LONG, strength=0.5)


class TestStrategyBase:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.strategy = DummyStrategy(
            event_bus=self.bus,
            assets=[self.asset],
        )

    def test_strategy_init_called(self):
        assert self.strategy.call_count == 0

    def test_strategy_receives_market_data(self):
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        event = MarketDataEvent(asset=self.asset, bar=bar)
        self.bus.emit(event)

        assert self.strategy.call_count == 1

    def test_strategy_emits_signal(self):
        signals_received = []
        self.bus.subscribe(SignalEvent, lambda e: signals_received.append(e))

        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

        assert len(signals_received) == 1
        assert signals_received[0].signal.direction == Direction.LONG

    def test_strategy_no_signal_below_threshold(self):
        signals_received = []
        self.bus.subscribe(SignalEvent, lambda e: signals_received.append(e))

        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=148.0, high=150.0, low=147.0, close=149.0, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

        assert len(signals_received) == 0

    def test_strategy_tracks_bar_history(self):
        for i in range(3):
            bar = Bar(
                timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                open=150.0 + i, high=155.0 + i, low=149.0 + i,
                close=153.0 + i, volume=1e6,
            )
            self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

        assert len(self.strategy.bars[self.asset]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_sma_crossover.py::TestStrategyBase -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement Strategy base class**

```python
# quantflow/strategies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from quantflow.core.models import Asset, Bar, Signal, Direction
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement init() for setup and next() for per-bar logic.
    The base class handles event subscription, bar history, and signal emission.
    """

    def __init__(self, event_bus: EventBus, assets: list[Asset]) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.bars: dict[Asset, list[Bar]] = defaultdict(list)
        self._current_event: MarketDataEvent | None = None

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
        self._current_event = event
        self.next(event)
        self._current_event = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies/test_sma_crossover.py::TestStrategyBase -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/base.py tests/test_strategies/test_sma_crossover.py
git commit -m "feat: add Strategy abstract base class with bar history and signal emission"
```

---

### Task 6: SMA Crossover Strategy

**Files:**
- Create: `quantflow/strategies/sma_crossover.py`
- Modify: `tests/test_strategies/test_sma_crossover.py` (add SMA tests)

- [ ] **Step 1: Add failing tests for SMA Crossover to the existing test file**

Append to `tests/test_strategies/test_sma_crossover.py`:

```python
from quantflow.strategies.sma_crossover import SMACrossover


class TestSMACrossover:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, close: float, day: int):
        bar = Bar(
            timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
            open=close - 1, high=close + 1, low=close - 2,
            close=close, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_no_signal_before_slow_period(self):
        strategy = SMACrossover(
            event_bus=self.bus,
            assets=[self.asset],
            fast_period=3,
            slow_period=5,
        )
        # Emit 4 bars (less than slow_period=5)
        for i in range(4):
            self._emit_bar(close=150.0 + i, day=i + 1)

        assert len(self.signals) == 0

    def test_long_signal_on_golden_cross(self):
        strategy = SMACrossover(
            event_bus=self.bus,
            assets=[self.asset],
            fast_period=3,
            slow_period=5,
        )
        # Create a rising sequence where fast SMA crosses above slow SMA
        prices = [100, 101, 102, 103, 110, 120, 130]
        for i, price in enumerate(prices):
            self._emit_bar(close=price, day=i + 1)

        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(long_signals) > 0

    def test_short_signal_on_death_cross(self):
        strategy = SMACrossover(
            event_bus=self.bus,
            assets=[self.asset],
            fast_period=3,
            slow_period=5,
        )
        # Rising then falling: should eventually get a short signal
        prices = [100, 110, 120, 130, 140, 130, 120, 110, 100, 90]
        for i, price in enumerate(prices):
            self._emit_bar(close=price, day=i + 1)

        short_signals = [s for s in self.signals if s.signal.direction == Direction.SHORT]
        assert len(short_signals) > 0

    def test_no_duplicate_signals(self):
        strategy = SMACrossover(
            event_bus=self.bus,
            assets=[self.asset],
            fast_period=3,
            slow_period=5,
        )
        # Steady uptrend -- should signal long once, not every bar
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        for i, price in enumerate(prices):
            self._emit_bar(close=price, day=i + 1)

        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        # Should only signal on the crossover, not on every bar
        assert len(long_signals) <= 2  # At most a couple of signals, not 5+
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_sma_crossover.py::TestSMACrossover -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement SMA Crossover strategy**

```python
# quantflow/strategies/sma_crossover.py
from __future__ import annotations

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


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
        # Must be set before super().__init__ calls init()
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        bars = self.bars[event.asset]
        if len(bars) < self.slow_period:
            return

        closes = [b.close for b in bars]
        fast_sma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_sma = sum(closes[-self.slow_period:]) / self.slow_period

        if fast_sma > slow_sma:
            new_direction = Direction.LONG
        elif fast_sma < slow_sma:
            new_direction = Direction.SHORT
        else:
            return

        prev = self._prev_position[event.asset]
        if new_direction != prev:
            self.signal(direction=new_direction, strength=1.0)
            self._prev_position[event.asset] = new_direction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies/test_sma_crossover.py -v`
Expected: All 9 tests PASS (5 base + 4 SMA)

- [ ] **Step 5: Commit**

```bash
git add quantflow/strategies/sma_crossover.py tests/test_strategies/test_sma_crossover.py
git commit -m "feat: add SMA Crossover strategy with golden/death cross detection"
```

---

### Task 7: Execution Simulator

**Files:**
- Create: `quantflow/backtest/execution.py`
- Create: `tests/test_backtest/test_execution.py`

- [ ] **Step 1: Write failing tests for execution simulator**

```python
# tests/test_backtest/test_execution.py
import pytest
from quantflow.core.models import (
    Asset, AssetClass, Order, OrderSide, OrderType, OrderStatus,
)
from quantflow.core.events import EventBus, OrderEvent, FillEvent
from quantflow.backtest.execution import SimulatedExecution


class TestSimulatedExecution:
    def setup_method(self):
        self.bus = EventBus()
        self.fills = []
        self.bus.subscribe(FillEvent, lambda e: self.fills.append(e))
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def test_market_order_fills_at_price_plus_slippage(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.001,  # 0.1%
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        assert len(self.fills) == 1
        fill = self.fills[0].fill
        assert fill.fill_quantity == 100.0
        # Buy slippage should increase price
        assert fill.fill_price == pytest.approx(150.0 * 1.001)
        assert fill.commission == 0.0
        assert fill.order.status == OrderStatus.FILLED

    def test_sell_order_slippage_decreases_price(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.001,
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.SELL,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        fill = self.fills[0].fill
        # Sell slippage should decrease price
        assert fill.fill_price == pytest.approx(150.0 * 0.999)

    def test_commission_calculated(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.0,
            commission_pct=0.001,  # 0.1%
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        fill = self.fills[0].fill
        # Commission = 0.1% of (100 * 150) = 15.0
        assert fill.commission == pytest.approx(15.0)

    def test_zero_slippage_and_commission(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=50.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=200.0)

        fill = self.fills[0].fill
        assert fill.fill_price == 200.0
        assert fill.commission == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_execution.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement SimulatedExecution**

```python
# quantflow/backtest/execution.py
from __future__ import annotations

from quantflow.core.models import Order, OrderSide, OrderStatus, Fill
from quantflow.core.events import EventBus, FillEvent


class SimulatedExecution:
    """Simulates order execution with configurable slippage and commission."""

    def __init__(
        self,
        event_bus: EventBus,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0,
    ) -> None:
        self.event_bus = event_bus
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

    def execute(self, order: Order, current_price: float) -> Fill:
        # Apply slippage: buys pay more, sells receive less
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1.0 + self.slippage_pct)
        else:
            fill_price = current_price * (1.0 - self.slippage_pct)

        slippage = abs(fill_price - current_price) * order.quantity

        # Commission based on notional value
        notional = order.quantity * fill_price
        commission = notional * self.commission_pct

        order.status = OrderStatus.FILLED

        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage,
        )

        self.event_bus.emit(FillEvent(fill=fill))
        return fill
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_execution.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/backtest/execution.py tests/test_backtest/test_execution.py
git commit -m "feat: add SimulatedExecution with slippage and commission models"
```

---

### Task 8: Backtest Engine

**Files:**
- Create: `quantflow/backtest/engine.py`
- Create: `tests/test_backtest/test_engine.py`

- [ ] **Step 1: Write failing tests for the backtest engine**

```python
# tests/test_backtest/test_engine.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Direction
from quantflow.core.events import EventBus
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.engine import BacktestEngine, BacktestResult


def make_price_data(asset: Asset, prices: list[float], start_year: int = 2024) -> pd.DataFrame:
    """Helper to create a DataFrame of OHLCV data from a list of close prices."""
    dates = pd.date_range(
        start=f"{start_year}-01-01", periods=len(prices), freq="B", tz="UTC"
    )
    df = pd.DataFrame(
        {
            "open": [p - 1 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )
    return df


class TestBacktestEngine:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def test_backtest_runs_without_error(self):
        prices = list(range(100, 200))  # Steady uptrend, 100 bars
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=20)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_backtest_result_has_trades(self):
        # Uptrend then downtrend to generate at least one trade
        prices = list(range(100, 140)) + list(range(140, 100, -1))
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(
            initial_cash=100_000.0,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert len(result.trades) > 0

    def test_equity_curve_starts_at_initial_cash(self):
        prices = list(range(100, 150))
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(initial_cash=50_000.0)

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=5, slow_period=10)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        assert result.equity_curve[0] == pytest.approx(50_000.0)

    def test_buy_and_hold_benchmark(self):
        prices = [100.0] * 10 + [200.0] * 10  # Doubles in price
        data = {self.asset: make_price_data(self.asset, prices)}

        engine = BacktestEngine(initial_cash=100_000.0)

        def strategy_factory(bus, assets):
            return SMACrossover(bus, assets, fast_period=3, slow_period=5)

        result = engine.run(data=data, strategy_factory=strategy_factory)

        # Benchmark should reflect the price change
        assert result.benchmark_equity[-1] > result.benchmark_equity[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_engine.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement BacktestEngine**

```python
# quantflow/backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class Trade:
    """A completed round-trip trade."""
    asset: Asset
    entry_time: datetime
    exit_time: datetime
    side: str  # "long" or "short"
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
    """Event-driven backtesting engine.

    Feeds historical bars one at a time through the event bus.
    Strategies emit signals, which become orders, which get filled.
    Tracks portfolio state bar-by-bar.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0,
        position_size_pct: float = 0.95,
    ) -> None:
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.position_size_pct = position_size_pct

    def run(
        self,
        data: dict[Asset, pd.DataFrame],
        strategy_factory: Callable[[EventBus, list[Asset]], Strategy],
    ) -> BacktestResult:
        bus = EventBus()
        assets = list(data.keys())

        # State
        cash = self.initial_cash
        positions: dict[Asset, Position] = {}
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        trades: list[Trade] = []
        signals: list[SignalEvent] = []
        current_prices: dict[Asset, float] = {}

        # Execution
        executor = SimulatedExecution(
            event_bus=bus,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )

        # Signal handler: convert signals to orders and fills
        def on_signal(event: SignalEvent):
            nonlocal cash
            signals.append(event)
            sig = event.signal
            asset = sig.asset
            price = current_prices.get(asset)
            if price is None or price <= 0:
                return

            current_pos = positions.get(asset)

            if sig.direction == Direction.LONG:
                if current_pos and current_pos.quantity > 0:
                    return  # Already long
                # Close short if exists
                if current_pos and current_pos.quantity < 0:
                    close_order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=abs(current_pos.quantity),
                        order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(close_order, price)
                    pnl = (current_pos.entry_price - fill.fill_price) * abs(current_pos.quantity)
                    cash += abs(current_pos.quantity) * fill.fill_price - fill.commission
                    trades.append(Trade(
                        asset=asset,
                        entry_time=current_pos.entry_price,  # Will improve later
                        exit_time=sig.timestamp,
                        side="short",
                        entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price,
                        quantity=abs(current_pos.quantity),
                        pnl=pnl - fill.commission,
                        commission=fill.commission,
                    ))
                    del positions[asset]

                # Open long
                available = cash * self.position_size_pct
                quantity = available / price
                if quantity > 0:
                    order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cost = fill.fill_price * fill.fill_quantity + fill.commission
                    cash -= cost
                    positions[asset] = Position(
                        asset=asset,
                        quantity=fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

            elif sig.direction == Direction.SHORT:
                if current_pos and current_pos.quantity < 0:
                    return  # Already short
                # Close long if exists
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
                        asset=asset,
                        entry_time=sig.timestamp,
                        exit_time=sig.timestamp,
                        side="long",
                        entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price,
                        quantity=current_pos.quantity,
                        pnl=pnl - fill.commission,
                        commission=fill.commission,
                    ))
                    del positions[asset]

                # Open short
                available = cash * self.position_size_pct
                quantity = available / price
                if quantity > 0:
                    order = Order(
                        asset=asset, side=OrderSide.SELL,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cash += fill.fill_price * fill.fill_quantity - fill.commission
                    positions[asset] = Position(
                        asset=asset,
                        quantity=-fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

        bus.subscribe(SignalEvent, on_signal)

        # Create strategy (this subscribes to MarketDataEvent)
        strategy = strategy_factory(bus, assets)

        # Build unified timeline from all assets
        all_dates: set[datetime] = set()
        for df in data.values():
            all_dates.update(df.index.to_pydatetime())
        sorted_dates = sorted(all_dates)

        # Benchmark: buy-and-hold from first bar
        benchmark_equity: list[float] = []
        first_prices: dict[Asset, float] = {}

        # Main loop: emit bars in chronological order
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

            # Calculate equity
            equity = cash
            for asset, pos in positions.items():
                price = current_prices.get(asset, pos.entry_price)
                if pos.quantity > 0:
                    equity += pos.quantity * price
                else:
                    # Short position: we received cash when we opened, now owe shares
                    equity += pos.quantity * price  # quantity is negative

            equity_curve.append(equity)
            timestamps.append(ts)

            # Benchmark
            bench = self.initial_cash
            if first_prices:
                # Equal-weight buy-and-hold
                per_asset = self.initial_cash / len(first_prices)
                bench = sum(
                    per_asset * (current_prices.get(a, fp) / fp)
                    for a, fp in first_prices.items()
                )
            benchmark_equity.append(bench)

        return BacktestResult(
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            signals=signals,
            benchmark_equity=benchmark_equity,
            initial_cash=self.initial_cash,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_engine.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/backtest/engine.py tests/test_backtest/test_engine.py
git commit -m "feat: add event-driven BacktestEngine with portfolio tracking and benchmark"
```

---

### Task 9: Performance Metrics

**Files:**
- Create: `quantflow/analytics/metrics.py`
- Create: `tests/test_analytics/test_metrics.py`

- [ ] **Step 1: Write failing tests for metrics**

```python
# tests/test_analytics/test_metrics.py
import pytest
import numpy as np
from quantflow.analytics.metrics import (
    total_return,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    profit_factor,
    avg_win_loss_ratio,
)


class TestReturnMetrics:
    def test_total_return(self):
        equity = [100_000, 110_000, 120_000, 115_000, 130_000]
        assert total_return(equity) == pytest.approx(0.3)  # 30%

    def test_total_return_negative(self):
        equity = [100_000, 90_000, 80_000]
        assert total_return(equity) == pytest.approx(-0.2)

    def test_total_return_single_point(self):
        assert total_return([100_000]) == 0.0

    def test_cagr(self):
        # $100k -> $200k over 3 years
        equity = [100_000] + [0] * 755 + [200_000]  # ~3 years of daily bars
        result = cagr(equity, periods_per_year=252)
        assert 0.20 < result < 0.30  # roughly 26% CAGR


class TestRiskMetrics:
    def test_max_drawdown(self):
        equity = [100, 110, 105, 120, 100, 130]
        dd = max_drawdown(equity)
        # Peak was 120, trough was 100, drawdown = -20/120 = -16.67%
        assert dd == pytest.approx(-20.0 / 120.0, abs=0.001)

    def test_max_drawdown_no_drawdown(self):
        equity = [100, 110, 120, 130]
        assert max_drawdown(equity) == 0.0

    def test_sharpe_ratio(self):
        # Consistent positive returns
        equity = [100_000 + i * 100 for i in range(252)]
        sr = sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=252)
        assert sr > 0  # Positive and high since returns are very consistent

    def test_sharpe_ratio_flat(self):
        equity = [100_000] * 100
        sr = sharpe_ratio(equity)
        assert sr == 0.0

    def test_sortino_ratio(self):
        equity = [100_000 + i * 100 for i in range(252)]
        sr = sortino_ratio(equity, risk_free_rate=0.0, periods_per_year=252)
        assert sr > 0


class TestTradeMetrics:
    def test_win_rate(self):
        pnls = [100, -50, 200, -30, 150]
        assert win_rate(pnls) == pytest.approx(0.6)  # 3 wins out of 5

    def test_win_rate_all_wins(self):
        assert win_rate([100, 200, 50]) == pytest.approx(1.0)

    def test_win_rate_no_trades(self):
        assert win_rate([]) == 0.0

    def test_profit_factor(self):
        pnls = [100, -50, 200, -30]
        # Gross profit = 300, gross loss = 80
        assert profit_factor(pnls) == pytest.approx(300.0 / 80.0)

    def test_profit_factor_no_losses(self):
        assert profit_factor([100, 200]) == float("inf")

    def test_avg_win_loss_ratio(self):
        pnls = [100, -50, 200, -30]
        # Avg win = 150, avg loss = 40
        assert avg_win_loss_ratio(pnls) == pytest.approx(150.0 / 40.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analytics/test_metrics.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement metrics**

```python
# quantflow/analytics/metrics.py
from __future__ import annotations

import numpy as np


def total_return(equity_curve: list[float]) -> float:
    """Total return as a fraction (0.3 = 30%)."""
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]


def cagr(equity_curve: list[float], periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    total = equity_curve[-1] / equity_curve[0]
    n_years = (len(equity_curve) - 1) / periods_per_year
    if n_years <= 0:
        return 0.0
    if total <= 0:
        return -1.0
    return total ** (1.0 / n_years) - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a negative fraction (-0.2 = 20% drawdown). Returns 0.0 if no drawdown."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    worst_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        if dd < worst_dd:
            worst_dd = dd
    return worst_dd


def sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    if len(equity_curve) < 2:
        return 0.0
    prices = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]
    if len(returns) == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    if len(equity_curve) < 2:
        return 0.0
    prices = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]
    if len(returns) == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        downside_std = 0.0
    else:
        downside_std = float(np.std(downside, ddof=1))

    if downside_std == 0:
        return float(np.mean(excess) * np.sqrt(periods_per_year)) if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def win_rate(pnls: list[float]) -> float:
    """Fraction of trades that were profitable."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_win_loss_ratio(pnls: list[float]) -> float:
    """Average winning trade / average losing trade."""
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    if not wins or not losses:
        return 0.0
    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analytics/test_metrics.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add quantflow/analytics/metrics.py tests/test_analytics/test_metrics.py
git commit -m "feat: add performance metrics (Sharpe, Sortino, drawdown, CAGR, win rate, etc.)"
```

---

### Task 10: Tearsheet (Console + Chart)

**Files:**
- Create: `quantflow/analytics/tearsheet.py`

- [ ] **Step 1: Implement the tearsheet**

```python
# quantflow/analytics/tearsheet.py
from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from quantflow.backtest.engine import BacktestResult
from quantflow.analytics.metrics import (
    total_return,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    profit_factor,
    avg_win_loss_ratio,
)


def print_tearsheet(result: BacktestResult) -> dict[str, float]:
    """Print performance summary to console and return metrics dict."""
    pnls = [t.pnl for t in result.trades]

    metrics = {
        "Total Return": total_return(result.equity_curve),
        "CAGR": cagr(result.equity_curve),
        "Max Drawdown": max_drawdown(result.equity_curve),
        "Sharpe Ratio": sharpe_ratio(result.equity_curve),
        "Sortino Ratio": sortino_ratio(result.equity_curve),
        "Win Rate": win_rate(pnls),
        "Profit Factor": profit_factor(pnls),
        "Avg Win/Loss": avg_win_loss_ratio(pnls),
        "Total Trades": len(result.trades),
        "Final Equity": result.equity_curve[-1] if result.equity_curve else 0.0,
    }

    print("\n" + "=" * 50)
    print("       QUANTFLOW BACKTEST TEARSHEET")
    print("=" * 50)
    print(f"  Initial Capital:   ${result.initial_cash:>14,.2f}")
    print(f"  Final Equity:      ${metrics['Final Equity']:>14,.2f}")
    print("-" * 50)
    print(f"  Total Return:      {metrics['Total Return']:>14.2%}")
    print(f"  CAGR:              {metrics['CAGR']:>14.2%}")
    print(f"  Max Drawdown:      {metrics['Max Drawdown']:>14.2%}")
    print("-" * 50)
    print(f"  Sharpe Ratio:      {metrics['Sharpe Ratio']:>14.2f}")
    print(f"  Sortino Ratio:     {metrics['Sortino Ratio']:>14.2f}")
    print("-" * 50)
    print(f"  Total Trades:      {metrics['Total Trades']:>14d}")
    print(f"  Win Rate:          {metrics['Win Rate']:>14.2%}")
    print(f"  Profit Factor:     {metrics['Profit Factor']:>14.2f}")
    print(f"  Avg Win/Loss:      {metrics['Avg Win/Loss']:>14.2f}")
    print("=" * 50)

    return metrics


def plot_tearsheet(result: BacktestResult, save_path: str | None = None) -> None:
    """Plot equity curve, drawdown, and trade markers."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    dates = result.timestamps
    equity = np.array(result.equity_curve)
    benchmark = np.array(result.benchmark_equity)

    # --- Equity Curve ---
    ax1 = axes[0]
    ax1.plot(dates, equity, label="Strategy", color="#2196F3", linewidth=1.5)
    ax1.plot(dates, benchmark, label="Buy & Hold", color="#9E9E9E", linewidth=1, linestyle="--")
    ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Drawdown ---
    ax2 = axes[1]
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    ax2.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.3)
    ax2.plot(dates, drawdown, color="#F44336", linewidth=1)
    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {save_path}")
    else:
        plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add quantflow/analytics/tearsheet.py
git commit -m "feat: add tearsheet with console metrics and matplotlib equity/drawdown charts"
```

---

### Task 11: Runnable Example

**Files:**
- Create: `quantflow/examples/sma_crossover.py`

- [ ] **Step 1: Implement the runnable example**

```python
# quantflow/examples/sma_crossover.py
"""
SMA Crossover Backtest Example
===============================
Run with: python -m quantflow.examples.sma_crossover

Backtests a Simple Moving Average crossover strategy on AAPL.
Fast SMA (10) crossing above Slow SMA (50) = buy signal.
Fast SMA (10) crossing below Slow SMA (50) = sell signal.
"""
from __future__ import annotations

from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    # 1. Define what we're trading
    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    # 2. Fetch historical data
    print("Fetching AAPL data from Yahoo Finance...")
    fetcher = YahooFetcher()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df = fetcher.fetch_ohlcv(asset, start, end)
    print(f"Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # 3. Run the backtest
    print("\nRunning SMA Crossover backtest...")
    engine = BacktestEngine(
        initial_cash=100_000.0,
        slippage_pct=0.0005,   # 0.05%
        commission_pct=0.0,     # Commission-free (modern brokers)
    )

    def strategy_factory(bus: EventBus, assets: list[Asset]) -> SMACrossover:
        return SMACrossover(bus, assets, fast_period=10, slow_period=50)

    result = engine.run(
        data={asset: df},
        strategy_factory=strategy_factory,
    )

    # 4. Print results
    print_tearsheet(result)

    # 5. Plot charts
    plot_tearsheet(result, save_path="sma_crossover_tearsheet.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run: `python -m quantflow.examples.sma_crossover`
Expected: Console tearsheet with metrics + `sma_crossover_tearsheet.png` saved to disk. No errors.

- [ ] **Step 3: Commit**

```bash
git add quantflow/examples/sma_crossover.py
git commit -m "feat: add runnable SMA crossover backtest example"
```

---

### Task 12: Full Test Suite and Cleanup

**Files:**
- Modify: `main.py` (remove PyCharm boilerplate)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (approximately 30 tests across all test files)

- [ ] **Step 2: Replace main.py with a useful entry point**

```python
# main.py
"""QuantFlow - Quantitative Research & Backtesting Platform.

Run the example:
    python -m quantflow.examples.sma_crossover
"""
```

- [ ] **Step 3: Add .gitignore**

Create `.gitignore`:
```
__pycache__/
*.pyc
.idea/
*.egg-info/
dist/
build/
.env
*.png
*.db
.pytest_cache/
```

- [ ] **Step 4: Run full test suite one more time**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Final commit**

```bash
git add main.py .gitignore
git commit -m "chore: clean up main.py, add .gitignore"
```

---

## Phase 1 Summary

After completing these 12 tasks, you will have:

- **Core domain models** -- Asset, Bar, Signal, Order, Fill, Position with proper enums
- **Event bus** -- synchronous pub/sub decoupling all modules
- **Yahoo Finance data fetcher** -- fetches OHLCV for equities, crypto, forex, commodities
- **Strategy framework** -- abstract base class with bar history and signal emission
- **SMA Crossover strategy** -- golden cross / death cross with duplicate signal prevention
- **Execution simulator** -- slippage and commission modeling
- **Backtest engine** -- event-driven bar-by-bar simulation with portfolio tracking
- **Performance metrics** -- Sharpe, Sortino, max drawdown, CAGR, win rate, profit factor
- **Tearsheet** -- console output + matplotlib equity curve and drawdown chart
- **Runnable example** -- `python -m quantflow.examples.sma_crossover`
- **~30 unit tests** covering all modules

## What Comes Next (Phase 2)

- SQLite caching for data
- Indicator library (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- More strategies (Mean Reversion, RSI+MACD, Pairs Trading)
- Position sizing models (Fixed Fractional, Kelly Criterion)
- Risk controls (drawdown kill switch, exposure limits)
- FRED macro data integration
- CCXT/CoinGecko crypto data
- CLI via click
