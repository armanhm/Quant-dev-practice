# QuantFlow Phase 3B: ML/DL Strategies, LLM Assistant, Dashboard & Paper Trading

**Date:** 2026-04-12
**Status:** Design approved

## Overview

Phase 3B adds machine learning and deep learning strategies, an LLM-powered research assistant, a Streamlit dashboard, and paper trading capability. Split into two parts for manageable execution.

---

## Part 1: ML/DL Strategies

### Feature Pipeline (`quantflow/ml/features.py`)

Transforms OHLCV data + technical indicators into feature matrices for ML models. Enforces time-series integrity -- no lookahead bias.

**API:**
```python
X, y = build_features(bars, lookback=20, prediction_horizon=5)
# X: (n_samples, n_features) -- features from indicators + price patterns
# y: (n_samples,) -- 1 (price went up), 0 (flat), -1 (price went down)
```

**Features generated:**
- Returns: 1-day, 5-day, 10-day, 20-day returns
- Indicators: RSI, MACD histogram, Bollinger Band %B, ATR ratio
- Volume: volume ratio (current / 20-day average)
- Volatility: rolling 10-day and 20-day standard deviation of returns

**Time-series train/test split:** `time_series_split(X, y, train_ratio=0.7)` -- never shuffles, always splits chronologically.

### Model Registry (`quantflow/ml/registry.py`)

Save/load trained models with versioning.

- `save_model(model, name, version, path)` -- saves to `~/.quantflow/models/{name}_v{version}.joblib` (sklearn) or `.pt` (PyTorch)
- `load_model(name, version, path)` -- loads and returns the model
- `list_models(path)` -- lists all saved models

### ML Strategy Base Class (`quantflow/ml/base.py`)

Extends `Strategy` with ML lifecycle:
```python
class MLStrategy(Strategy):
    def __init__(self, event_bus, assets, model_path):
        self.model_path = model_path
        super().__init__(event_bus, assets)

    def init(self):
        self.model = load_model_from_path(self.model_path)

    def next(self, event):
        features = self.extract_features(event.asset)
        if features is not None:
            prediction = self.model.predict(features)
            # Convert prediction to signal
```

Training is separate (`quantflow train` CLI command). No training during backtests.

### Feature-Based Classifier (`quantflow/strategies/ml_classifier.py`)

- Uses Random Forest or XGBoost
- Features: returns, RSI, MACD, BB%B, ATR, volume ratio
- Target: 1 if 5-day forward return > 0, else -1
- Walk-forward training: train on historical window, predict next period

### LSTM Price Forecaster (`quantflow/strategies/lstm_forecaster.py`)

- PyTorch LSTM model
- Input: sequence of 20 bars of features
- Output: predicted next-day return direction
- Teaches: sequence modeling, data normalization, PyTorch training loop

---

## Part 2: LLM Assistant + Dashboard + Paper Trading

### LLM Assistant

**Provider adapter pattern:**
- `quantflow/assistant/provider.py` -- `LLMProvider` protocol with `chat(messages, tools)` method
- `quantflow/assistant/claude_provider.py` -- Claude API via Anthropic SDK
- Tool definitions that let the LLM call platform functions (run_backtest, fetch_data, etc.)
- `quantflow chat` CLI command with conversation memory

### Streamlit Dashboard

- `quantflow/dashboard/app.py` -- main Streamlit app
- Pages: Data Explorer, Strategy Lab, Results Viewer, Chat panel
- Run with: `quantflow dashboard` or `streamlit run quantflow/dashboard/app.py`

### Paper Trading

- `quantflow/live/paper_trader.py` -- receives real-time bars from Alpaca API
- Translates live bars into `MarketDataEvent` and feeds them through the existing event bus
- Same strategies, same risk controls -- just a different data source

### Docker

- `Dockerfile` for containerizing the entire platform
- `docker-compose.yml` for dashboard + paper trading services

---

## New Dependencies

| Library | Purpose | Install Group |
|---------|---------|--------------|
| scikit-learn | Random Forest, feature engineering | `[ml]` optional |
| xgboost | Gradient boosting classifier | `[ml]` optional |
| torch | LSTM deep learning | `[ml]` optional |
| anthropic | Claude API SDK | `[llm]` optional |
| streamlit | Dashboard | `[dashboard]` optional |
| alpaca-py | Paper trading | `[live]` optional |
