import pytest
import numpy as np
from datetime import datetime, timezone
from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.ml_classifier import MLClassifier
from quantflow.ml.features import build_features
from quantflow.ml.registry import save_model


def make_bars(n: int, trend: float = 0.5) -> list[Bar]:
    bars = []
    price = 100.0
    for i in range(n):
        change = trend * (1 if i % 3 != 0 else -1) + np.random.normal(0, 0.2)
        price = max(price + change, 1.0)
        bars.append(Bar(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                        open=price - 0.5, high=price + 1.0, low=price - 1.0,
                        close=price, volume=1e6))
    return bars


class TestMLClassifier:
    def test_train_and_predict(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        np.random.seed(42)
        bars = make_bars(300)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert X.shape[0] > 0

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        model_path = save_model(model, "rf_test", "1", model_dir=str(tmp_path))

        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signals = []
        bus.subscribe(SignalEvent, lambda e: signals.append(e))

        strategy = MLClassifier(event_bus=bus, assets=[asset],
                                model_path=str(model_path), min_bars=50)
        test_bars = make_bars(100)
        for bar in test_bars:
            bus.emit(MarketDataEvent(asset=asset, bar=bar))
        # Should run without error
        assert True

    def test_no_signal_without_model(self):
        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signals = []
        bus.subscribe(SignalEvent, lambda e: signals.append(e))

        strategy = MLClassifier(event_bus=bus, assets=[asset], model_path=None, min_bars=50)
        bars = make_bars(100)
        for bar in bars:
            bus.emit(MarketDataEvent(asset=asset, bar=bar))
        assert len(signals) == 0
