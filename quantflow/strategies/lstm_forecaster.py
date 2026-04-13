"""LSTM-based price forecasting strategy using PyTorch."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus
from quantflow.ml.base import MLStrategy
from quantflow.ml.features import build_features


class LSTMModel(nn.Module):
    """Simple LSTM for sequence classification (up/flat/down)."""
    def __init__(self, input_size: int = 11, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


class LSTMForecaster(MLStrategy):
    """LSTM-powered trading strategy.

    Uses a trained LSTM model to predict price direction from a sequence
    of feature vectors. The model outputs probabilities for 3 classes:
    down (-1), flat (0), up (1).
    """

    def __init__(self, event_bus: EventBus, assets: list[Asset],
                 model_path: str | None = None, lookback: int = 20,
                 min_bars: int = 50, seq_length: int = 20,
                 hidden_size: int = 64, num_layers: int = 2) -> None:
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(event_bus, assets, model_path, lookback, min_bars)

    def load_model(self):
        model = LSTMModel(input_size=11, hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict(self, features: np.ndarray):
        asset = self.assets[0]
        bars = self.bars[asset]
        X_all, _ = build_features(bars, lookback=self.lookback, horizon=1)
        if X_all.shape[0] < self.seq_length:
            return None
        seq = X_all[-self.seq_length:]
        tensor = torch.FloatTensor(seq).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        direction_map = {0: -1, 1: 0, 2: 1}
        return direction_map[pred_class], confidence

    def prediction_to_signal(self, prediction) -> tuple[Direction, float]:
        if prediction is None:
            return Direction.FLAT, 0.0
        pred_class, confidence = prediction
        if pred_class == 1:
            return Direction.LONG, min(confidence, 1.0)
        elif pred_class == -1:
            return Direction.SHORT, min(confidence, 1.0)
        return Direction.FLAT, 0.0
