# tests/test_ml/test_registry.py
import pytest
import numpy as np
from pathlib import Path
from quantflow.ml.registry import save_model, load_model, list_models


class TestModelRegistry:
    def test_save_and_load_sklearn(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.choice([0, 1], 50)
        model.fit(X, y)

        save_model(model, name="test_rf", version="1", model_dir=str(tmp_path))
        loaded = load_model(name="test_rf", version="1", model_dir=str(tmp_path))
        assert np.array_equal(model.predict(X), loaded.predict(X))

    def test_list_models(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5)
        model.fit(np.random.rand(20, 3), np.random.choice([0, 1], 20))

        save_model(model, name="model_a", version="1", model_dir=str(tmp_path))
        save_model(model, name="model_b", version="2", model_dir=str(tmp_path))
        models = list_models(model_dir=str(tmp_path))
        assert len(models) == 2

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(name="nope", version="1", model_dir=str(tmp_path))
