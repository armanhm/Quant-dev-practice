"""Macro Regime strategy: shifts allocation based on economic indicators."""
from __future__ import annotations
import math
from enum import Enum
import pandas as pd
from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class Regime(Enum):
    GROWTH = "growth"
    RECESSION = "recession"
    HIGH_VOLATILITY = "high_volatility"
    INFLATION = "inflation"


class MacroRegime(Strategy):
    """Shifts asset allocation based on detected economic regime using FRED data."""

    def __init__(self, event_bus: EventBus, assets: list[Asset],
                 macro_data: dict[str, pd.DataFrame] | None = None, min_bars: int = 20) -> None:
        self.macro_data = macro_data or {}
        self.min_bars = min_bars
        self._current_regime: Regime | None = None
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    @staticmethod
    def detect_regime(yield_spread: float, vix: float, cpi_yoy: float) -> Regime:
        if vix > 25:
            return Regime.HIGH_VOLATILITY
        if yield_spread < 0:
            return Regime.RECESSION
        if cpi_yoy > 4.0:
            return Regime.INFLATION
        return Regime.GROWTH

    def _get_macro_value(self, indicator: str, as_of: pd.Timestamp) -> float | None:
        df = self.macro_data.get(indicator)
        if df is None or df.empty:
            return None
        mask = df.index <= as_of
        if not mask.any():
            return None
        return float(df.loc[mask].iloc[-1]["value"])

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        if len(self.bars[asset]) < self.min_bars:
            return
        ts = pd.Timestamp(event.bar.timestamp)
        yield_spread = self._get_macro_value("T10Y2Y", ts)
        vix = self._get_macro_value("VIXCLS", ts)
        cpi_yoy = self._get_macro_value("CPIAUCSL_PC1", ts)
        if yield_spread is None or vix is None or cpi_yoy is None:
            return
        regime = self.detect_regime(yield_spread, vix, cpi_yoy)
        self._current_regime = regime
        prev = self._prev_position.get(asset, Direction.FLAT)
        if regime == Regime.GROWTH:
            new_dir, strength = Direction.LONG, 0.8
        elif regime == Regime.RECESSION:
            new_dir, strength = Direction.SHORT, 0.6
        elif regime == Regime.HIGH_VOLATILITY:
            new_dir, strength = Direction.SHORT, 0.4
        elif regime == Regime.INFLATION:
            if asset.asset_class.value == "commodity":
                new_dir, strength = Direction.LONG, 0.7
            else:
                new_dir, strength = Direction.SHORT, 0.3
        else:
            return
        if new_dir != prev:
            self.signal(direction=new_dir, strength=strength)
            self._prev_position[asset] = new_dir
