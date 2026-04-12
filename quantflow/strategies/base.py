# quantflow/strategies/base.py
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from quantflow.core.models import Asset, Bar, Signal, Direction
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent
from quantflow.data import indicators as ind


_INDICATOR_REGISTRY: dict[str, tuple[Callable, str]] = {
    "sma": (ind.sma, "close"),
    "ema": (ind.ema, "close"),
    "rsi": (ind.rsi, "close"),
    "bollinger_bands": (ind.bollinger_bands, "close"),
    "atr": (ind.atr, "hlc"),
}


class IndicatorBuffer:
    """Stores computed indicator values per asset. Updated by the Strategy on each bar."""

    def __init__(self, name: str, func: Callable, input_type: str, params: dict):
        self.name = name
        self.func = func
        self.input_type = input_type
        self.params = params
        self._values: dict[Asset, list[float]] = defaultdict(list)

    def update(self, asset: Asset, bars: list[Bar]) -> None:
        if not bars:
            return
        if self.input_type == "close":
            closes = [b.close for b in bars]
            result = self.func(closes, **self.params)
        elif self.input_type == "hlc":
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            closes = [b.close for b in bars]
            result = self.func(highs, lows, closes, **self.params)
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

        if isinstance(result, tuple):
            self._values[asset] = list(result[0])
            if not hasattr(self, "_raw"):
                self._raw: dict[Asset, tuple] = {}
            self._raw[asset] = result
        else:
            self._values[asset] = result

    def __getitem__(self, asset: Asset) -> list[float]:
        return self._values[asset]

    def latest(self, asset: Asset) -> float:
        values = self._values.get(asset, [])
        if not values:
            return float("nan")
        return values[-1]

    def raw(self, asset: Asset) -> tuple | None:
        raw = getattr(self, "_raw", {})
        return raw.get(asset)


class Strategy(ABC):
    def __init__(self, event_bus: EventBus, assets: list[Asset]) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.bars: dict[Asset, list[Bar]] = defaultdict(list)
        self._current_event: MarketDataEvent | None = None
        self._indicators: list[IndicatorBuffer] = []

        self.event_bus.subscribe(MarketDataEvent, self._on_market_data)
        self.init()

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def next(self, event: MarketDataEvent) -> None:
        ...

    def indicator(self, name: str, **params) -> IndicatorBuffer:
        if name not in _INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator: {name}. Available: {list(_INDICATOR_REGISTRY.keys())}")
        func, input_type = _INDICATOR_REGISTRY[name]
        buf = IndicatorBuffer(name=name, func=func, input_type=input_type, params=params)
        self._indicators.append(buf)
        return buf

    def signal(self, direction: Direction, strength: float) -> None:
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
        for buf in self._indicators:
            buf.update(event.asset, self.bars[event.asset])
        self._current_event = event
        self.next(event)
        self._current_event = None
