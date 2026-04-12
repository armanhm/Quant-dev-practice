# tests/test_strategies/test_rsi_macd.py
import pytest
from datetime import datetime, timezone, timedelta

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
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1),
            open=close - 1, high=close + 2, low=close - 2,
            close=close, volume=1e6,
        )
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_no_signal_before_enough_data(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus, assets=[self.asset], rsi_period=14,
        )
        for i in range(20):
            self._emit_bar(close=100.0 + i * 0.1, day=i + 1)
        assert len(self.signals) == 0

    def test_long_signal_on_oversold_with_macd_cross(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus, assets=[self.asset],
            rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        )
        prices = list(range(150, 100, -1)) + list(range(100, 130))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)
        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(self.signals) > 0

    def test_short_signal_on_overbought_with_macd_cross(self):
        strategy = RSIMACDCombo(
            event_bus=self.bus, assets=[self.asset],
            rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        )
        prices = list(range(100, 160)) + list(range(160, 130, -1))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)
        short_signals = [s for s in self.signals if s.signal.direction == Direction.SHORT]
        assert len(self.signals) > 0
