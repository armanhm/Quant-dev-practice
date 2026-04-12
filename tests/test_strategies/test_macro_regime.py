# tests/test_strategies/test_macro_regime.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.macro_regime import MacroRegime, Regime


class TestMacroRegime:
    def setup_method(self):
        self.bus = EventBus()
        self.equity = Asset(symbol="SPY", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, asset, close, day):
        bar = Bar(timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
                  open=close - 1, high=close + 1, low=close - 1, close=close, volume=1e6)
        self.bus.emit(MarketDataEvent(asset=asset, bar=bar))

    def test_detect_growth_regime(self):
        assert MacroRegime.detect_regime(yield_spread=1.5, vix=15.0, cpi_yoy=2.0) == Regime.GROWTH

    def test_detect_recession_regime(self):
        assert MacroRegime.detect_regime(yield_spread=-0.5, vix=18.0, cpi_yoy=2.0) == Regime.RECESSION

    def test_detect_high_vol_regime(self):
        assert MacroRegime.detect_regime(yield_spread=1.0, vix=30.0, cpi_yoy=2.0) == Regime.HIGH_VOLATILITY

    def test_detect_inflation_regime(self):
        assert MacroRegime.detect_regime(yield_spread=1.0, vix=18.0, cpi_yoy=5.0) == Regime.INFLATION

    def test_strategy_with_macro_data(self):
        macro_data = {
            "T10Y2Y": pd.DataFrame({"value": [1.5, 1.5, 1.5]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")),
            "VIXCLS": pd.DataFrame({"value": [15.0, 15.0, 15.0]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")),
            "CPIAUCSL_PC1": pd.DataFrame({"value": [2.0, 2.0, 2.0]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")),
        }
        strategy = MacroRegime(event_bus=self.bus, assets=[self.equity], macro_data=macro_data)
        for i in range(30):
            self._emit_bar(self.equity, close=450.0 + i, day=i + 1)
        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(long_signals) > 0
