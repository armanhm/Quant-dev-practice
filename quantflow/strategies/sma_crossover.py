from __future__ import annotations

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


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
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    def next(self, event: MarketDataEvent) -> None:
        bars = self.bars[event.asset]
        if len(bars) < self.slow_period:
            return

        closes = [b.close for b in bars]
        fast_sma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_sma = sum(closes[-self.slow_period:]) / self.slow_period

        if fast_sma > slow_sma:
            new_direction = Direction.LONG
        elif fast_sma < slow_sma:
            new_direction = Direction.SHORT
        else:
            return

        prev = self._prev_position[event.asset]
        if new_direction != prev:
            self.signal(direction=new_direction, strength=1.0)
            self._prev_position[event.asset] = new_direction
