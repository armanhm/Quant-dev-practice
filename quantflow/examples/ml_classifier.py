"""
ML Classifier Backtest Example
================================
Run with: python -m quantflow.examples.ml_classifier
Trains a Random Forest on AAPL, then backtests on the test period.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from quantflow.core.models import Asset, AssetClass, Bar
from quantflow.core.events import EventBus
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.ml_classifier import MLClassifier
from quantflow.ml.features import build_features, time_series_split
from quantflow.ml.registry import save_model
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


def main():
    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
    print("Fetching AAPL data...")
    fetcher = YahooFetcher()
    df = fetcher.fetch_ohlcv(asset,
        datetime(2018, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc))
    print(f"Got {len(df)} bars")

    bars = [Bar(timestamp=ts, open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]), volume=float(r["volume"]))
            for ts, r in df.iterrows()]

    print("\nBuilding features...")
    X, y = build_features(bars, lookback=20, horizon=5)
    X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio=0.7)
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print(f"Train acc: {model.score(X_train, y_train):.2%}, Test acc: {model.score(X_test, y_test):.2%}")

    model_path = save_model(model, "rf_aapl", "1")
    print(f"Model saved to {model_path}")

    split_idx = int(len(df) * 0.7)
    test_df = df.iloc[split_idx:]
    print(f"\nBacktesting on {len(test_df)} bars...")

    engine = BacktestEngine(initial_cash=100_000.0, slippage_pct=0.0005)
    def factory(bus, assets):
        return MLClassifier(bus, assets, model_path=str(model_path), min_bars=50)
    result = engine.run(data={asset: test_df}, strategy_factory=factory)
    print_tearsheet(result)
    plot_tearsheet(result, save_path="ml_classifier_tearsheet.png")


if __name__ == "__main__":
    main()
