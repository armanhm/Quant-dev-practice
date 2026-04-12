# quantflow/strategies/mean_reversion.py
from __future__ import annotations

import math

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bollinger Band Mean Reversion strategy.

    Buys when price drops below the lower Bollinger Band (oversold).
    Sells when price rises above the upper Bollinger Band (overbought).
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        bb_period: int = 20,
        num_std: float = 2.0,
    ) -> None:
        self.bb_period = bb_period
        self.num_std = num_std
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        self._bb = self.indicator("bollinger_bands", period=self.bb_period, num_std=self.num_std)
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        price = event.bar.close

        raw = self._bb.raw(asset)
        if raw is None:
            return
        upper, middle, lower = raw

        if not upper or math.isnan(upper[-1]):
            return

        current_upper = upper[-1]
        current_middle = middle[-1]
        current_lower = lower[-1]

        prev = self._prev_position[asset]

        if price < current_lower:
            if prev != Direction.LONG:
                self.signal(direction=Direction.LONG, strength=0.8)
                self._prev_position[asset] = Direction.LONG
        elif price > current_upper:
            if prev != Direction.SHORT:
                self.signal(direction=Direction.SHORT, strength=0.8)
                self._prev_position[asset] = Direction.SHORT
