# QuantFlow Phase 3B Part 1: ML/DL Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add machine learning infrastructure (feature pipeline, model registry) and two ML strategies (Random Forest classifier, LSTM forecaster) that plug into the existing backtest engine via the Strategy base class.

**Architecture:** Feature pipeline transforms OHLCV + indicators into feature matrices. Model registry saves/loads trained models. MLStrategy base class extends Strategy with model loading. Training is separate from backtesting (no lookahead bias). PyTorch/sklearn/xgboost are optional dependencies.

**Tech Stack:** Python 3.11+, scikit-learn, XGBoost, PyTorch, joblib, pytest

---

## Tasks

### Task 1: ML Dependencies + Package Setup
- Add `[project.optional-dependencies] ml = ["scikit-learn>=1.3", "xgboost>=2.0", "torch>=2.0", "joblib>=1.3"]` to pyproject.toml
- Create `quantflow/ml/__init__.py` and `tests/test_ml/__init__.py` (empty)
- Install: `pip install -e ".[dev,ml]"`
- Commit: `chore: add optional ML dependencies`

### Task 2: Feature Pipeline
- Create `quantflow/ml/features.py` with `build_features(bars, lookback, horizon) -> (X, y)` and `time_series_split(X, y, train_ratio)`
- Create `tests/test_ml/test_features.py` with tests for shape, NaN-free, valid labels, time-series ordering
- Features: returns (1d/5d/10d/20d), RSI, MACD histogram, Bollinger %B, ATR ratio, volume ratio, volatility (10d/20d)
- Labels: 1 if forward return > 0.1%, -1 if < -0.1%, else 0
- Commit: `feat: add ML feature pipeline`

### Task 3: Model Registry
- Create `quantflow/ml/registry.py` with `save_model()`, `load_model()`, `list_models()`
- Uses joblib for sklearn models, saves to `~/.quantflow/models/`
- Create `tests/test_ml/test_registry.py`
- Commit: `feat: add model registry for saving/loading trained models`

### Task 4: MLStrategy Base Class
- Create `quantflow/ml/base.py` extending Strategy with `load_model()`, `predict()`, `prediction_to_signal()` abstract methods
- Commit: `feat: add MLStrategy base class`

### Task 5: Random Forest Classifier Strategy
- Create `quantflow/strategies/ml_classifier.py` (MLClassifier extending MLStrategy)
- Create `tests/test_strategies/test_ml_classifier.py`
- Create `quantflow/examples/ml_classifier.py` (train RF on AAPL, backtest on test period)
- Commit: `feat: add Random Forest ML classifier strategy`

### Task 6: LSTM Price Forecaster Strategy
- Create `quantflow/strategies/lstm_forecaster.py` (LSTMForecaster + LSTMModel)
- Create `tests/test_strategies/test_lstm_forecaster.py` (skip if no torch)
- Commit: `feat: add LSTM price forecaster strategy`

### Task 7: Registry + README Update
- Register ml_classifier and lstm_forecaster in strategy registry
- Update README.md roadmap
- Run full test suite
- Push to GitHub
- Commit: `chore: register ML strategies and update README`
