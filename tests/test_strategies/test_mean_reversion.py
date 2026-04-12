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
            event_bus=self.bus, assets=[self.asset], bb_period=10,
        )
        for i in range(8):
            self._emit_bar(close=100.0, day=i + 1)
        assert len(self.signals) == 0

    def test_long_signal_below_lower_band(self):
        strategy = MeanReversion(
            event_bus=self.bus, assets=[self.asset], bb_period=10, num_std=2.0,
        )
        for i in range(15):
            self._emit_bar(close=100.0, day=i + 1)
        self._emit_bar(close=80.0, day=16)
        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(long_signals) > 0

    def test_short_signal_above_upper_band(self):
        strategy = MeanReversion(
            event_bus=self.bus, assets=[self.asset], bb_period=10, num_std=2.0,
        )
        for i in range(15):
            self._emit_bar(close=100.0, day=i + 1)
        self._emit_bar(close=120.0, day=16)
        short_signals = [s for s in self.signals if s.signal.direction == Direction.SHORT]
        assert len(short_signals) > 0

    def test_flat_signal_at_mean(self):
        strategy = MeanReversion(
            event_bus=self.bus, assets=[self.asset], bb_period=10, num_std=2.0,
        )
        for i in range(20):
            self._emit_bar(close=100.0, day=i + 1)
        assert len(self.signals) == 0
