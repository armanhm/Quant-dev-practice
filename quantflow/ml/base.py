"""Base class for ML-powered trading strategies."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy
from quantflow.ml.features import build_features


class MLStrategy(Strategy):
    """Base class for ML strategies.

    Subclasses implement:
    - load_model(): load a pre-trained model
    - predict(features): return prediction from model
    - prediction_to_signal(prediction): convert prediction to direction + strength
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        model_path: str | None = None,
        lookback: int = 20,
        min_bars: int = 50,
    ) -> None:
        self.model_path = model_path
        self.lookback = lookback
        self.min_bars = min_bars
        self.model = None
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        if self.model_path:
            self.model = self.load_model()
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    @abstractmethod
    def load_model(self):
        """Load the pre-trained model from self.model_path."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray):
        """Run prediction on a single feature vector."""
        ...

    @abstractmethod
    def prediction_to_signal(self, prediction) -> tuple[Direction, float]:
        """Convert model prediction to (direction, strength)."""
        ...

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        bars = self.bars[asset]
        if len(bars) < self.min_bars:
            return
        if self.model is None:
            return

        X, _ = build_features(bars, lookback=self.lookback, horizon=1)
        if X.shape[0] == 0:
            return

        latest_features = X[-1:].reshape(1, -1)
        prediction = self.predict(latest_features)
        direction, strength = self.prediction_to_signal(prediction)

        if direction == Direction.FLAT:
            return

        prev = self._prev_position.get(asset, Direction.FLAT)
        if direction != prev:
            self.signal(direction=direction, strength=strength)
            self._prev_position[asset] = direction
