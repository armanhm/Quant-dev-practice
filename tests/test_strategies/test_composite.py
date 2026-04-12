# tests/test_strategies/test_composite.py
import pytest
from datetime import datetime, timezone, timedelta
from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.strategies.composite import CompositeStrategy


class TestCompositeStrategy:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, close: float, day: int):
        bar = Bar(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1),
                  open=close - 1, high=close + 2, low=close - 2,
                  close=close, volume=1e6)
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_composite_creates_without_error(self):
        composite = CompositeStrategy(
            event_bus=self.bus, assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.6),
                (lambda bus, assets: MeanReversion(bus, assets, bb_period=5, num_std=2.0), 0.4),
            ],
        )
        assert composite is not None

    def test_composite_emits_merged_signals(self):
        composite = CompositeStrategy(
            event_bus=self.bus, assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.6),
                (lambda bus, assets: MeanReversion(bus, assets, bb_period=5, num_std=2.0), 0.4),
            ],
            min_strength=0.1,
        )
        prices = list(range(100, 130)) + list(range(130, 100, -1))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)
        assert len(self.signals) > 0

    def test_composite_respects_min_strength(self):
        composite = CompositeStrategy(
            event_bus=self.bus, assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.5),
            ],
            min_strength=0.99,
        )
        prices = list(range(100, 120))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)
        assert len(self.signals) == 0
