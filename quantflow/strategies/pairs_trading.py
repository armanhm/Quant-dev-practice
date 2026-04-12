"""Pairs Trading strategy using spread z-score."""
from __future__ import annotations
import numpy as np
from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class PairsTrading(Strategy):
    """Statistical arbitrage pairs trading. Trades the spread between two correlated assets."""

    def __init__(self, event_bus: EventBus, assets: list[Asset], lookback_period: int = 60,
                 entry_z: float = 2.0, exit_z: float = 0.0) -> None:
        if len(assets) != 2:
            raise ValueError(f"PairsTrading requires exactly 2 assets, got {len(assets)}")
        self.lookback_period = lookback_period
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._position_state: str = "flat"
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        pass

    def next(self, event: MarketDataEvent) -> None:
        asset_a, asset_b = self.assets
        bars_a = self.bars[asset_a]
        bars_b = self.bars[asset_b]
        min_bars = min(len(bars_a), len(bars_b))
        if min_bars < self.lookback_period:
            return

        closes_a = np.array([b.close for b in bars_a[-self.lookback_period:]])
        closes_b = np.array([b.close for b in bars_b[-self.lookback_period:]])

        mean_a, mean_b = np.mean(closes_a), np.mean(closes_b)
        cov = np.sum((closes_a - mean_a) * (closes_b - mean_b))
        var_b = np.sum((closes_b - mean_b) ** 2)
        if var_b == 0:
            return
        hedge_ratio = cov / var_b

        spread = closes_a - hedge_ratio * closes_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        if spread_std == 0:
            return
        current_z = (spread[-1] - spread_mean) / spread_std

        if self._position_state == "flat":
            if current_z > self.entry_z:
                self._position_state = "short_spread"
                if event.asset == asset_a:
                    self.signal(direction=Direction.SHORT, strength=min(abs(current_z) / 3.0, 1.0))
            elif current_z < -self.entry_z:
                self._position_state = "long_spread"
                if event.asset == asset_a:
                    self.signal(direction=Direction.LONG, strength=min(abs(current_z) / 3.0, 1.0))
        elif self._position_state == "long_spread":
            if current_z >= self.exit_z:
                self._position_state = "flat"
                if event.asset == asset_a:
                    self.signal(direction=Direction.SHORT, strength=0.5)
        elif self._position_state == "short_spread":
            if current_z <= self.exit_z:
                self._position_state = "flat"
                if event.asset == asset_a:
                    self.signal(direction=Direction.LONG, strength=0.5)
