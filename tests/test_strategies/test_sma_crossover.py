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
