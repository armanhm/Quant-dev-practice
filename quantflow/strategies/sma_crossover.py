# quantflow/strategies/sma_crossover.py
from __future__ import annotations

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy, IndicatorBuffer


class SMACrossover(Strategy):
    """Simple Moving Average Crossover strategy.

    Goes long when the fast SMA crosses above the slow SMA (golden cross).
    Goes short when the fast SMA crosses below the slow SMA (death cross).
    Only emits a signal on the actual crossover, not on every bar.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        fast_period: int = 10,
        slow_period: int = 50,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        self._sma_fast = self.indicator("sma", period=self.fast_period)
        self._sma_slow = self.indicator("sma", period=self.slow_period)
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        fast = self._sma_fast.latest(asset)
        slow = self._sma_slow.latest(asset)

        if fast != fast or slow != slow:  # NaN check
            return

        if fast > slow:
            new_direction = Direction.LONG
        elif fast < slow:
            new_direction = Direction.SHORT
        else:
            return

        prev = self._prev_position[asset]
        if new_direction != prev:
            self.signal(direction=new_direction, strength=1.0)
            self._prev_position[asset] = new_direction
