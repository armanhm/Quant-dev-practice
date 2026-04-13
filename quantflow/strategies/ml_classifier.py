"""Random Forest / XGBoost classifier strategy."""
from __future__ import annotations
import numpy as np
import joblib
from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus
from quantflow.ml.base import MLStrategy


class MLClassifier(MLStrategy):
    """ML strategy using a trained sklearn classifier. Predicts: 1 (long), -1 (short), 0 (flat)."""

    def __init__(self, event_bus: EventBus, assets: list[Asset],
                 model_path: str | None = None, lookback: int = 20, min_bars: int = 50) -> None:
        super().__init__(event_bus, assets, model_path, lookback, min_bars)

    def load_model(self):
        return joblib.load(self.model_path)

    def predict(self, features: np.ndarray):
        prediction = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)[0]
            confidence = max(probas)
        else:
            confidence = 0.7
        return prediction, confidence

    def prediction_to_signal(self, prediction) -> tuple[Direction, float]:
        pred_class, confidence = prediction
        if pred_class == 1:
            return Direction.LONG, min(confidence, 1.0)
        elif pred_class == -1:
            return Direction.SHORT, min(confidence, 1.0)
        return Direction.FLAT, 0.0
