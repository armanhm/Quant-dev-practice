from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from quantflow.core.models import Asset, Bar, Signal, Direction
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement init() for setup and next() for per-bar logic.
    The base class handles event subscription, bar history, and signal emission.
    """

    def __init__(self, event_bus: EventBus, assets: list[Asset]) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.bars: dict[Asset, list[Bar]] = defaultdict(list)
        self._current_event: MarketDataEvent | None = None

        self.event_bus.subscribe(MarketDataEvent, self._on_market_data)
        self.init()

    @abstractmethod
    def init(self) -> None:
        """Called once at construction. Set up indicators, state, etc."""
        ...

    @abstractmethod
    def next(self, event: MarketDataEvent) -> None:
        """Called on each new bar. Implement trading logic here."""
        ...

    def signal(self, direction: Direction, strength: float) -> None:
        """Emit a trading signal from within next()."""
        if self._current_event is None:
            raise RuntimeError("signal() can only be called from within next()")

        sig = Signal(
            timestamp=self._current_event.bar.timestamp,
            asset=self._current_event.asset,
            direction=direction,
            strength=strength,
        )
        self.event_bus.emit(SignalEvent(signal=sig))

    def _on_market_data(self, event: MarketDataEvent) -> None:
        if event.asset not in self.assets:
            return
        self.bars[event.asset].append(event.bar)
        self._current_event = event
        self.next(event)
        self._current_event = None
