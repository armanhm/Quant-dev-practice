# tests/test_ml/test_features.py
import pytest
import numpy as np
from datetime import datetime, timezone
from quantflow.core.models import Bar
from quantflow.ml.features import build_features, time_series_split


def make_bars(n: int) -> list[Bar]:
    bars = []
    price = 100.0
    for i in range(n):
        change = 0.5 * (1 if i % 3 != 0 else -1)
        price += change
        bars.append(Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=price - 0.5, high=price + 1.0, low=price - 1.0,
            close=price, volume=1e6 * (1 + i % 5 * 0.1),
        ))
    return bars


class TestBuildFeatures:
    def test_returns_numpy_arrays(self):
        bars = make_bars(100)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_correct_shape(self):
        bars = make_bars(100)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

    def test_no_nan_in_output(self):
        bars = make_bars(200)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_labels_are_valid(self):
        bars = make_bars(200)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert set(np.unique(y)).issubset({-1, 0, 1})

    def test_not_enough_data(self):
        bars = make_bars(10)
        X, y = build_features(bars, lookback=20, horizon=5)
        assert X.shape[0] == 0


class TestTimeSeriesSplit:
    def test_split_preserves_order(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio=0.7)
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert X_train[-1, 0] < X_test[0, 0]

    def test_no_shuffle(self):
        X = np.arange(50).reshape(50, 1)
        y = np.arange(50)
        X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio=0.8)
        assert np.array_equal(y_train, np.arange(40))
        assert np.array_equal(y_test, np.arange(40, 50))
