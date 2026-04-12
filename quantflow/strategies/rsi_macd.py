# quantflow/strategies/rsi_macd.py
from __future__ import annotations

import math

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class RSIMACDCombo(Strategy):
    """RSI + MACD Combination strategy.

    Uses RSI to identify overbought/oversold conditions and MACD
    for trend confirmation.
    - Long: RSI recovers from oversold (<30) AND MACD histogram turns positive
    - Short: RSI drops from overbought (>70) AND MACD histogram turns negative
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
        if len(bars) < 35:
            return

        from quantflow.data.indicators import macd
        closes = [b.close for b in bars]
        macd_line, signal_line, histogram = macd(closes)

        if not histogram or math.isnan(histogram[-1]):
            return

        hist = histogram[-1]

        if rsi_val < self.rsi_oversold:
            self._was_oversold[asset] = True
            self._was_overbought[asset] = False
        elif rsi_val > self.rsi_overbought:
            self._was_overbought[asset] = True
            self._was_oversold[asset] = False

        prev = self._prev_position[asset]

        if self._was_oversold[asset] and rsi_val > self.rsi_oversold and hist > 0:
            if prev != Direction.LONG:
                self.signal(direction=Direction.LONG, strength=0.7)
                self._prev_position[asset] = Direction.LONG
                self._was_oversold[asset] = False

        elif self._was_overbought[asset] and rsi_val < self.rsi_overbought and hist < 0:
            if prev != Direction.SHORT:
                self.signal(direction=Direction.SHORT, strength=0.7)
                self._prev_position[asset] = Direction.SHORT
                self._was_overbought[asset] = False
