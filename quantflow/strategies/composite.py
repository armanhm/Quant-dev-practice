"""CompositeStrategy: weighted ensemble of multiple strategies."""
from __future__ import annotations
from collections import defaultdict
from typing import Callable
from quantflow.core.models import Asset, Direction, Signal
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent


class CompositeStrategy:
    """Combines multiple strategies with weighted signal aggregation.

    Each sub-strategy runs on its own internal event bus. Their signals are
    intercepted, weighted, and merged. The merged signal is emitted on the
    main event bus only if the weighted net score meets min_strength.
    """

    def __init__(self, event_bus: EventBus, assets: list[Asset],
                 components: list[tuple[Callable, float]], min_strength: float = 0.2) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.min_strength = min_strength
        # Each entry: (direction, strength, weight)
        self._pending_signals: dict[Asset, list[tuple[Direction, float, float]]] = defaultdict(list)
        self._prev_direction: dict[Asset, Direction] = {a: Direction.FLAT for a in assets}

        # One internal bus per sub-strategy so we can attribute correct weight
        self._internal_buses: list[EventBus] = []
        self._strategies = []
        for factory, weight in components:
            bus = EventBus()
            bus.subscribe(
                SignalEvent,
                lambda e, w=weight: self._on_sub_signal(e, w),
            )
            self._internal_buses.append(bus)
            self._strategies.append(factory(bus, assets))

        self.event_bus.subscribe(MarketDataEvent, self._on_market_data)

    def _on_market_data(self, event: MarketDataEvent) -> None:
        if event.asset not in self.assets:
            return
        self._pending_signals[event.asset] = []
        for bus in self._internal_buses:
            bus.emit(event)
        self._merge_and_emit(event.asset, event)

    def _on_sub_signal(self, event: SignalEvent, weight: float) -> None:
        sig = event.signal
        self._pending_signals[sig.asset].append((sig.direction, sig.strength, weight))

    def _merge_and_emit(self, asset: Asset, market_event: MarketDataEvent) -> None:
        pending = self._pending_signals.get(asset, [])
        if not pending:
            return

        net_score = 0.0
        for direction, strength, weight in pending:
            vote = 1.0 if direction == Direction.LONG else -1.0
            net_score += vote * strength * weight

        abs_score = abs(net_score)
        if abs_score < self.min_strength:
            return

        new_dir = Direction.LONG if net_score > 0 else Direction.SHORT
        strength = min(abs_score, 1.0)

        prev = self._prev_direction.get(asset, Direction.FLAT)
        if new_dir != prev:
            sig = Signal(timestamp=market_event.bar.timestamp, asset=asset,
                         direction=new_dir, strength=strength)
            self.event_bus.emit(SignalEvent(signal=sig))
            self._prev_direction[asset] = new_dir
