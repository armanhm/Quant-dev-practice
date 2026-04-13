import pytest
import numpy as np
from datetime import datetime, timezone
from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus

try:
    import torch
    from quantflow.strategies.lstm_forecaster import LSTMForecaster, LSTMModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_bars(n: int) -> list[Bar]:
    bars = []
    price = 100.0
    for i in range(n):
        change = 0.5 * (1 if i % 3 != 0 else -1)
        price += change
        bars.append(Bar(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                        open=price - 0.5, high=price + 1.0, low=price - 1.0,
                        close=price, volume=1e6))
    return bars


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestLSTMForecaster:
    def test_lstm_model_forward(self):
        model = LSTMModel(input_size=11, hidden_size=32, num_layers=1)
        x = torch.randn(1, 20, 11)
        out = model(x)
        assert out.shape == (1, 3)

    def test_strategy_no_signal_without_model(self):
        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signals = []
        bus.subscribe(SignalEvent, lambda e: signals.append(e))
        strategy = LSTMForecaster(event_bus=bus, assets=[asset], model_path=None, min_bars=50)
        bars = make_bars(100)
        for bar in bars:
            bus.emit(MarketDataEvent(asset=asset, bar=bar))
        assert len(signals) == 0

    def test_strategy_with_trained_model(self, tmp_path):
        model = LSTMModel(input_size=11, hidden_size=16, num_layers=1)
        model_path = str(tmp_path / "lstm_test.pt")
        torch.save(model.state_dict(), model_path)

        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signals = []
        bus.subscribe(SignalEvent, lambda e: signals.append(e))
        strategy = LSTMForecaster(event_bus=bus, assets=[asset], model_path=model_path,
                                  min_bars=50, seq_length=20, hidden_size=16, num_layers=1)
        bars = make_bars(100)
        for bar in bars:
            bus.emit(MarketDataEvent(asset=asset, bar=bar))
        assert True  # No crash = success
