# tests/test_strategies/test_pairs_trading.py
import pytest
from datetime import datetime, timezone
from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.pairs_trading import PairsTrading


class TestPairsTrading:
    def setup_method(self):
        self.bus = EventBus()
        self.asset_a = Asset(symbol="KO", asset_class=AssetClass.EQUITY)
        self.asset_b = Asset(symbol="PEP", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bars(self, close_a: float, close_b: float, day: int):
        ts = datetime(2024, 1, day, tzinfo=timezone.utc)
        for asset, close in [(self.asset_a, close_a), (self.asset_b, close_b)]:
            bar = Bar(timestamp=ts, open=close - 1, high=close + 1,
                      low=close - 1, close=close, volume=1e6)
            self.bus.emit(MarketDataEvent(asset=asset, bar=bar))

    def test_no_signal_before_lookback(self):
        strategy = PairsTrading(event_bus=self.bus, assets=[self.asset_a, self.asset_b], lookback_period=20)
        for i in range(15):
            self._emit_bars(close_a=50.0 + i * 0.1, close_b=55.0 + i * 0.1, day=i + 1)
        assert len(self.signals) == 0

    def test_signal_on_spread_divergence(self):
        strategy = PairsTrading(event_bus=self.bus, assets=[self.asset_a, self.asset_b], lookback_period=20, entry_z=1.5)
        for i in range(25):
            self._emit_bars(close_a=50.0 + i * 0.1, close_b=55.0 + i * 0.1, day=i + 1)
        for i in range(5):
            self._emit_bars(close_a=60.0 + i * 3.0, close_b=57.5, day=26 + i)
        assert len(self.signals) > 0

    def test_requires_exactly_two_assets(self):
        with pytest.raises(ValueError, match="exactly 2"):
            PairsTrading(event_bus=self.bus, assets=[self.asset_a], lookback_period=20)
