"""Model registry: save, load, and list trained ML models."""
from __future__ import annotations

from pathlib import Path

import joblib

_DEFAULT_MODEL_DIR = Path.home() / ".quantflow" / "models"


def save_model(model, name: str, version: str, model_dir: str | None = None) -> Path:
    base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_v{version}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, version: str, model_dir: str | None = None):
    base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    path = base / f"{name}_v{version}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def list_models(model_dir: str | None = None) -> list[dict]:
    base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    if not base.exists():
        return []
    models = []
    for p in sorted(base.glob("*.joblib")):
        parts = p.stem.rsplit("_v", 1)
        name = parts[0] if len(parts) == 2 else p.stem
        version = parts[1] if len(parts) == 2 else "unknown"
        models.append({"name": name, "version": version, "path": str(p), "size_mb": p.stat().st_size / 1e6})
    return models
