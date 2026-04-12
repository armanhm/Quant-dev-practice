import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from quantflow.data.indicators import sma as sma_func

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.base import Strategy
from quantflow.strategies.sma_crossover import SMACrossover


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
        assert len(long_signals) <= 2


class IndicatorStrategy(Strategy):
    """Strategy that uses the indicator API for testing."""
    def init(self) -> None:
        self.sma_fast = self.indicator("sma", period=3)
        self.sma_slow = self.indicator("sma", period=5)
        self.rsi_val = self.indicator("rsi", period=5)
    def next(self, event: MarketDataEvent) -> None:
        pass


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
        assert not math.isnan(strategy.sma_fast[self.asset][-1])
        assert strategy.sma_fast[self.asset][-1] == pytest.approx(108.0)

    def test_indicator_latest_shortcut(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        prices = [100, 102, 104, 106, 108, 110]
        for i, p in enumerate(prices):
            self._emit_bar(close=p, day=i + 1)
        assert strategy.sma_fast.latest(self.asset) == pytest.approx(108.0)

    def test_indicator_not_enough_data_returns_nan(self):
        strategy = IndicatorStrategy(event_bus=self.bus, assets=[self.asset])
        self._emit_bar(close=100, day=1)
        assert math.isnan(strategy.sma_fast.latest(self.asset))
