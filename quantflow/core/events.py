from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from quantflow.core.models import Asset, Bar, Signal, Order, Fill


@dataclass
class Event:
    """Base class for all events."""
    pass


@dataclass
class MarketDataEvent(Event):
    asset: Asset
    bar: Bar


@dataclass
class SignalEvent(Event):
    signal: Signal


@dataclass
class OrderEvent(Event):
    order: Order


@dataclass
class FillEvent(Event):
    fill: Fill


class EventBus:
    """Simple synchronous event bus. Handlers are called in registration order."""

    def __init__(self):
        self._handlers: dict[type[Event], list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type[Event], handler: Callable[[Any], None]) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers[type(event)]:
            handler(event)
