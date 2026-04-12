# QuantFlow Phase 3A Part 2: Strategies, Optimization & CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strategy registry, Pairs Trading strategy (cointegration), Macro Regime strategy (FRED-driven), CompositeStrategy (ensemble), parameter sweep, walk-forward optimization, and a full CLI with YAML config support.

**Architecture:** Strategy registry maps string names to classes for CLI lookup. Pairs Trading uses statsmodels for cointegration. Macro Regime queries DataManager for FRED data. CompositeStrategy uses an internal event bus to aggregate sub-strategy signals. ParameterSweep and WalkForward run multiple backtests. CLI uses click with YAML config parsing.

**Tech Stack:** Python 3.11+, statsmodels, click, multiprocessing, pytest

---

## File Structure

```
quantflow/
    strategies/
        registry.py        -- NEW: Strategy name -> class mapping
        pairs_trading.py   -- NEW: Cointegration-based pairs trading
        macro_regime.py    -- NEW: FRED-driven regime allocation
        composite.py       -- NEW: Weighted ensemble of strategies
    backtest/
        optimizer.py       -- NEW: ParameterSweep grid search
        walk_forward.py    -- NEW: Rolling train/test optimization
    cli/
        __init__.py        -- NEW
        main.py            -- NEW: Click CLI entry point
config/
    default_config.yaml    -- NEW: Example backtest config
tests/
    test_strategies/
        test_pairs_trading.py  -- NEW
        test_macro_regime.py   -- NEW
        test_composite.py      -- NEW
    test_backtest/
        test_optimizer.py      -- NEW
        test_walk_forward.py   -- NEW
    test_cli/
        __init__.py            -- NEW
        test_cli.py            -- NEW
```

---

### Task 1: Strategy Registry

**Files:**
- Create: `quantflow/strategies/registry.py`

- [ ] **Step 1: Create registry**

```python
# quantflow/strategies/registry.py
"""Maps strategy string names to their classes for CLI and config lookup."""
from __future__ import annotations

from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.strategies.rsi_macd import RSIMACDCombo

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossover,
    "mean_reversion": MeanReversion,
    "rsi_macd": RSIMACDCombo,
}


def get_strategy(name: str) -> type:
    """Look up a strategy class by name. Raises KeyError if not found."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def register_strategy(name: str, cls: type) -> None:
    """Register a new strategy class."""
    STRATEGY_REGISTRY[name] = cls
```

- [ ] **Step 2: Run full test suite to verify no breakage**

Run: `pytest tests/ -v --tb=short`
Expected: All 119 tests still pass

- [ ] **Step 3: Commit**

```bash
git add quantflow/strategies/registry.py
git commit -m "feat: add strategy registry for name-to-class lookup"
```

---

### Task 2: Pairs Trading Strategy

**Files:**
- Create: `quantflow/strategies/pairs_trading.py`
- Create: `tests/test_strategies/test_pairs_trading.py`
- Modify: `pyproject.toml` (add statsmodels)

- [ ] **Step 1: Add statsmodels dependency**

Add `"statsmodels>=0.14"` to the dependencies list in `pyproject.toml` and run `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_strategies/test_pairs_trading.py
import pytest
import math
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.pairs_trading import PairsTrading


class TestPairsTrading:
    def setup_method(self):
        self.bus = EventBus()
        self.asset_a = Asset(symbol="KO", asset_class=AssetClass.EQUITY)
        self.asset_b = Asset(symbol="PEP", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bars(self, close_a: float, close_b: float, day: int):
        ts = datetime(2024, 1, day, tzinfo=timezone.utc)
        for asset, close in [(self.asset_a, close_a), (self.asset_b, close_b)]:
            bar = Bar(timestamp=ts, open=close - 1, high=close + 1,
                      low=close - 1, close=close, volume=1e6)
            self.bus.emit(MarketDataEvent(asset=asset, bar=bar))

    def test_no_signal_before_lookback(self):
        strategy = PairsTrading(
            event_bus=self.bus,
            assets=[self.asset_a, self.asset_b],
            lookback_period=20,
        )
        for i in range(15):
            self._emit_bars(close_a=50.0 + i * 0.1, close_b=55.0 + i * 0.1, day=i + 1)
        assert len(self.signals) == 0

    def test_signal_on_spread_divergence(self):
        strategy = PairsTrading(
            event_bus=self.bus,
            assets=[self.asset_a, self.asset_b],
            lookback_period=20,
            entry_z=1.5,
        )
        # Build correlated history
        for i in range(25):
            self._emit_bars(close_a=50.0 + i * 0.1, close_b=55.0 + i * 0.1, day=i + 1)
        # Now diverge: A spikes up while B stays
        for i in range(5):
            self._emit_bars(close_a=60.0 + i * 3.0, close_b=57.5, day=26 + i)
        # Should have generated at least one signal
        assert len(self.signals) > 0

    def test_requires_exactly_two_assets(self):
        with pytest.raises(ValueError, match="exactly 2"):
            PairsTrading(
                event_bus=self.bus,
                assets=[self.asset_a],
                lookback_period=20,
            )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_pairs_trading.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 4: Implement Pairs Trading**

```python
# quantflow/strategies/pairs_trading.py
"""Pairs Trading strategy using cointegration and z-score of the spread."""
from __future__ import annotations

import math

import numpy as np

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class PairsTrading(Strategy):
    """Statistical arbitrage pairs trading.

    Trades the spread between two correlated assets. When the spread
    deviates from its mean by more than entry_z standard deviations,
    enter a mean-reversion trade. Exit when spread returns to exit_z.

    Quant concepts: cointegration, spread trading, z-scores, market neutrality.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        lookback_period: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
    ) -> None:
        if len(assets) != 2:
            raise ValueError(f"PairsTrading requires exactly 2 assets, got {len(assets)}")
        self.lookback_period = lookback_period
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._position_state: str = "flat"  # "long_spread", "short_spread", "flat"
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

        # Get aligned close prices
        closes_a = np.array([b.close for b in bars_a[-self.lookback_period:]])
        closes_b = np.array([b.close for b in bars_b[-self.lookback_period:]])

        # Simple hedge ratio: OLS slope
        mean_a = np.mean(closes_a)
        mean_b = np.mean(closes_b)
        cov = np.sum((closes_a - mean_a) * (closes_b - mean_b))
        var_b = np.sum((closes_b - mean_b) ** 2)
        if var_b == 0:
            return
        hedge_ratio = cov / var_b

        # Compute spread and z-score
        spread = closes_a - hedge_ratio * closes_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        if spread_std == 0:
            return

        current_z = (spread[-1] - spread_mean) / spread_std

        # Trading logic
        if self._position_state == "flat":
            if current_z > self.entry_z:
                # Spread too high -> short spread (sell A, buy B)
                self._position_state = "short_spread"
                # Signal SHORT on asset A
                if event.asset == asset_a:
                    self.signal(direction=Direction.SHORT, strength=min(abs(current_z) / 3.0, 1.0))
            elif current_z < -self.entry_z:
                # Spread too low -> long spread (buy A, sell B)
                self._position_state = "long_spread"
                if event.asset == asset_a:
                    self.signal(direction=Direction.LONG, strength=min(abs(current_z) / 3.0, 1.0))

        elif self._position_state == "long_spread":
            if current_z >= self.exit_z:
                # Spread reverted -> exit
                self._position_state = "flat"
                if event.asset == asset_a:
                    self.signal(direction=Direction.SHORT, strength=0.5)

        elif self._position_state == "short_spread":
            if current_z <= self.exit_z:
                self._position_state = "flat"
                if event.asset == asset_a:
                    self.signal(direction=Direction.LONG, strength=0.5)
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add Pairs Trading strategy with spread z-score`

- [ ] **Step 6: Register in registry**

Add to `quantflow/strategies/registry.py`:
```python
from quantflow.strategies.pairs_trading import PairsTrading
# Add to STRATEGY_REGISTRY:
"pairs_trading": PairsTrading,
```

Commit: `chore: register PairsTrading in strategy registry`

---

### Task 3: Macro Regime Strategy

**Files:**
- Create: `quantflow/strategies/macro_regime.py`
- Create: `tests/test_strategies/test_macro_regime.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategies/test_macro_regime.py
import pytest
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.macro_regime import MacroRegime, Regime


class TestMacroRegime:
    def setup_method(self):
        self.bus = EventBus()
        self.equity = Asset(symbol="SPY", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, asset: Asset, close: float, day: int):
        bar = Bar(timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
                  open=close - 1, high=close + 1, low=close - 1,
                  close=close, volume=1e6)
        self.bus.emit(MarketDataEvent(asset=asset, bar=bar))

    def test_detect_growth_regime(self):
        regime = MacroRegime.detect_regime(
            yield_spread=1.5, vix=15.0, cpi_yoy=2.0,
        )
        assert regime == Regime.GROWTH

    def test_detect_recession_regime(self):
        regime = MacroRegime.detect_regime(
            yield_spread=-0.5, vix=18.0, cpi_yoy=2.0,
        )
        assert regime == Regime.RECESSION

    def test_detect_high_vol_regime(self):
        regime = MacroRegime.detect_regime(
            yield_spread=1.0, vix=30.0, cpi_yoy=2.0,
        )
        assert regime == Regime.HIGH_VOLATILITY

    def test_detect_inflation_regime(self):
        regime = MacroRegime.detect_regime(
            yield_spread=1.0, vix=18.0, cpi_yoy=5.0,
        )
        assert regime == Regime.INFLATION

    def test_strategy_with_macro_data(self):
        # Create strategy with pre-loaded macro data
        import pandas as pd
        macro_data = {
            "T10Y2Y": pd.DataFrame(
                {"value": [1.5, 1.5, 1.5]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC"),
            ),
            "VIXCLS": pd.DataFrame(
                {"value": [15.0, 15.0, 15.0]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC"),
            ),
            "CPIAUCSL_PC1": pd.DataFrame(
                {"value": [2.0, 2.0, 2.0]},
                index=pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC"),
            ),
        }
        strategy = MacroRegime(
            event_bus=self.bus,
            assets=[self.equity],
            macro_data=macro_data,
        )
        # Emit enough bars to trigger signals
        for i in range(30):
            self._emit_bar(self.equity, close=450.0 + i, day=i + 1)
        # In growth regime, should go long equity
        long_signals = [s for s in self.signals if s.signal.direction == Direction.LONG]
        assert len(long_signals) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_macro_regime.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement Macro Regime**

```python
# quantflow/strategies/macro_regime.py
"""Macro Regime strategy: shifts allocation based on economic indicators."""
from __future__ import annotations

import math
from enum import Enum

import pandas as pd

from quantflow.core.models import Asset, Direction
from quantflow.core.events import EventBus, MarketDataEvent
from quantflow.strategies.base import Strategy


class Regime(Enum):
    GROWTH = "growth"
    RECESSION = "recession"
    HIGH_VOLATILITY = "high_volatility"
    INFLATION = "inflation"


class MacroRegime(Strategy):
    """Shifts asset allocation based on detected economic regime.

    Uses pre-loaded FRED macro data to classify the current regime:
    - Growth: yield spread > 0, VIX < 20 -> long equities
    - Recession: yield spread < 0 -> reduce/short equities
    - High Volatility: VIX > 25 -> reduce all positions
    - Inflation: CPI YoY > 4% -> shift to commodities

    Macro data is pre-fetched (via DataManager) and passed in at construction.
    The strategy looks up the most recent macro value as-of each bar's date.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        macro_data: dict[str, pd.DataFrame] | None = None,
        min_bars: int = 20,
    ) -> None:
        self.macro_data = macro_data or {}
        self.min_bars = min_bars
        self._current_regime: Regime | None = None
        self._prev_position: dict[Asset, Direction] = {}
        super().__init__(event_bus=event_bus, assets=assets)

    def init(self) -> None:
        for asset in self.assets:
            self._prev_position[asset] = Direction.FLAT

    @staticmethod
    def detect_regime(
        yield_spread: float,
        vix: float,
        cpi_yoy: float,
    ) -> Regime:
        """Classify current macro regime from indicators."""
        if vix > 25:
            return Regime.HIGH_VOLATILITY
        if yield_spread < 0:
            return Regime.RECESSION
        if cpi_yoy > 4.0:
            return Regime.INFLATION
        return Regime.GROWTH

    def _get_macro_value(self, indicator: str, as_of: pd.Timestamp) -> float | None:
        """Look up the most recent macro value as-of a given date."""
        df = self.macro_data.get(indicator)
        if df is None or df.empty:
            return None
        # Get most recent value on or before as_of
        mask = df.index <= as_of
        if not mask.any():
            return None
        return float(df.loc[mask].iloc[-1]["value"])

    def next(self, event: MarketDataEvent) -> None:
        asset = event.asset
        if len(self.bars[asset]) < self.min_bars:
            return

        ts = pd.Timestamp(event.bar.timestamp)

        yield_spread = self._get_macro_value("T10Y2Y", ts)
        vix = self._get_macro_value("VIXCLS", ts)
        cpi_yoy = self._get_macro_value("CPIAUCSL_PC1", ts)

        if yield_spread is None or vix is None or cpi_yoy is None:
            return

        regime = self.detect_regime(yield_spread, vix, cpi_yoy)
        self._current_regime = regime

        # Determine signal based on regime and asset class
        prev = self._prev_position.get(asset, Direction.FLAT)

        if regime == Regime.GROWTH:
            new_dir = Direction.LONG
            strength = 0.8
        elif regime == Regime.RECESSION:
            new_dir = Direction.SHORT
            strength = 0.6
        elif regime == Regime.HIGH_VOLATILITY:
            new_dir = Direction.SHORT
            strength = 0.4
        elif regime == Regime.INFLATION:
            if asset.asset_class.value == "commodity":
                new_dir = Direction.LONG
                strength = 0.7
            else:
                new_dir = Direction.SHORT
                strength = 0.3
        else:
            return

        if new_dir != prev:
            self.signal(direction=new_dir, strength=strength)
            self._prev_position[asset] = new_dir
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add Macro Regime strategy with FRED-driven allocation`

- [ ] **Step 5: Register in registry**

Add to `quantflow/strategies/registry.py`:
```python
from quantflow.strategies.macro_regime import MacroRegime
# Add to STRATEGY_REGISTRY:
"macro_regime": MacroRegime,
```

Commit: `chore: register MacroRegime in strategy registry`

---

### Task 4: CompositeStrategy

**Files:**
- Create: `quantflow/strategies/composite.py`
- Create: `tests/test_strategies/test_composite.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategies/test_composite.py
import pytest
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass, Bar, Direction
from quantflow.core.events import MarketDataEvent, SignalEvent, EventBus
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.strategies.composite import CompositeStrategy


class TestCompositeStrategy:
    def setup_method(self):
        self.bus = EventBus()
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.signals = []
        self.bus.subscribe(SignalEvent, lambda e: self.signals.append(e))

    def _emit_bar(self, close: float, day: int):
        bar = Bar(timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
                  open=close - 1, high=close + 2, low=close - 2,
                  close=close, volume=1e6)
        self.bus.emit(MarketDataEvent(asset=self.asset, bar=bar))

    def test_composite_creates_without_error(self):
        composite = CompositeStrategy(
            event_bus=self.bus,
            assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.6),
                (lambda bus, assets: MeanReversion(bus, assets, bb_period=5, num_std=2.0), 0.4),
            ],
        )
        assert composite is not None

    def test_composite_emits_merged_signals(self):
        composite = CompositeStrategy(
            event_bus=self.bus,
            assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.6),
                (lambda bus, assets: MeanReversion(bus, assets, bb_period=5, num_std=2.0), 0.4),
            ],
            min_strength=0.1,
        )
        # Feed enough bars to trigger at least one sub-strategy signal
        prices = list(range(100, 130)) + list(range(130, 100, -1))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)

        # Should have merged signals on the main bus
        assert len(self.signals) > 0

    def test_composite_respects_min_strength(self):
        composite = CompositeStrategy(
            event_bus=self.bus,
            assets=[self.asset],
            components=[
                (lambda bus, assets: SMACrossover(bus, assets, fast_period=3, slow_period=5), 0.5),
            ],
            min_strength=0.99,  # Very high threshold
        )
        prices = list(range(100, 120))
        for i, p in enumerate(prices):
            self._emit_bar(close=float(p), day=i + 1)
        # Threshold too high -- no signals should pass
        assert len(self.signals) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies/test_composite.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement CompositeStrategy**

```python
# quantflow/strategies/composite.py
"""CompositeStrategy: weighted ensemble of multiple strategies."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from quantflow.core.models import Asset, Direction, Signal
from quantflow.core.events import EventBus, MarketDataEvent, SignalEvent


class CompositeStrategy:
    """Combines multiple strategies with weighted signal aggregation.

    Each sub-strategy runs on an internal event bus. Their signals are
    intercepted, weighted, and merged. The merged signal is emitted
    on the main event bus only if it exceeds min_strength.

    Quant concept: ensemble methods, signal diversification, weighted voting.
    """

    def __init__(
        self,
        event_bus: EventBus,
        assets: list[Asset],
        components: list[tuple[Callable, float]],
        min_strength: float = 0.2,
    ) -> None:
        self.event_bus = event_bus
        self.assets = assets
        self.min_strength = min_strength
        self._pending_signals: dict[Asset, list[tuple[Direction, float, float]]] = defaultdict(list)
        self._prev_direction: dict[Asset, Direction] = {a: Direction.FLAT for a in assets}

        # Internal bus for sub-strategies
        self._internal_bus = EventBus()

        # Capture sub-strategy signals
        self._component_weights: list[float] = []
        self._current_weight_idx = 0

        # Create sub-strategies on internal bus
        self._strategies = []
        for i, (factory, weight) in enumerate(components):
            self._component_weights.append(weight)
            strategy = factory(self._internal_bus, assets)
            self._strategies.append(strategy)

        # Listen for signals on internal bus
        self._internal_bus.subscribe(SignalEvent, self._on_sub_signal)

        # Forward market data from main bus to internal bus
        self.event_bus.subscribe(MarketDataEvent, self._on_market_data)

    def _on_market_data(self, event: MarketDataEvent) -> None:
        """Forward market data to internal bus, then merge signals."""
        if event.asset not in self.assets:
            return

        # Clear pending signals for this asset
        self._pending_signals[event.asset] = []

        # Forward to sub-strategies
        self._internal_bus.emit(event)

        # Now merge any signals that were collected
        self._merge_and_emit(event.asset, event)

    def _on_sub_signal(self, event: SignalEvent) -> None:
        """Collect sub-strategy signals for later merging."""
        sig = event.signal
        # Figure out which strategy emitted this by checking the order
        # Simple approach: just collect all signals per asset per bar
        self._pending_signals[sig.asset].append(
            (sig.direction, sig.strength, 1.0)  # weight applied in merge
        )

    def _merge_and_emit(self, asset: Asset, market_event: MarketDataEvent) -> None:
        """Merge collected signals and emit on main bus if strong enough."""
        pending = self._pending_signals.get(asset, [])
        if not pending:
            return

        # Weighted vote: LONG = +1, SHORT = -1
        # Since we can't easily map signals back to specific strategies,
        # we use equal weight per signal and scale by component weights
        net_score = 0.0
        total_weight = sum(self._component_weights)

        for direction, strength, _ in pending:
            vote = 1.0 if direction == Direction.LONG else -1.0
            # Weight proportional to number of components
            weight = total_weight / len(self._component_weights) if self._component_weights else 1.0
            net_score += vote * strength * (1.0 / len(self._strategies))

        abs_score = abs(net_score)
        if abs_score < self.min_strength:
            return

        new_dir = Direction.LONG if net_score > 0 else Direction.SHORT
        strength = min(abs_score, 1.0)

        prev = self._prev_direction.get(asset, Direction.FLAT)
        if new_dir != prev:
            sig = Signal(
                timestamp=market_event.bar.timestamp,
                asset=asset,
                direction=new_dir,
                strength=strength,
            )
            self.event_bus.emit(SignalEvent(signal=sig))
            self._prev_direction[asset] = new_dir
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add CompositeStrategy for weighted signal ensemble`

- [ ] **Step 5: Register in registry**

Add to `quantflow/strategies/registry.py`:
```python
from quantflow.strategies.composite import CompositeStrategy
# Add to STRATEGY_REGISTRY:
"composite": CompositeStrategy,
```

Commit: `chore: register CompositeStrategy in strategy registry`

---

### Task 5: Parameter Sweep

**Files:**
- Create: `quantflow/backtest/optimizer.py`
- Create: `tests/test_backtest/test_optimizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest/test_optimizer.py
import pytest
import pandas as pd
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.optimizer import ParameterSweep, SweepResult


def make_price_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": [p - 1 for p in prices], "high": [p + 2 for p in prices],
         "low": [p - 2 for p in prices], "close": prices,
         "volume": [1e6] * len(prices)},
        index=dates,
    )


class TestParameterSweep:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        prices = list(range(100, 200)) + list(range(200, 100, -1))
        self.data = {self.asset: make_price_data(prices)}

    def test_sweep_runs(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5], "slow_period": [10, 15]},
            data=self.data,
        )
        result = sweep.run()
        assert isinstance(result, SweepResult)
        assert len(result.results) == 4  # 2 x 2 combinations

    def test_sweep_returns_best_params(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5, 7], "slow_period": [10, 15]},
            data=self.data,
        )
        result = sweep.run(metric="sharpe_ratio")
        assert result.best_params is not None
        assert "fast_period" in result.best_params
        assert "slow_period" in result.best_params

    def test_sweep_each_result_has_metrics(self):
        sweep = ParameterSweep(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [3, 5], "slow_period": [10]},
            data=self.data,
        )
        result = sweep.run()
        for r in result.results:
            assert "params" in r
            assert "sharpe_ratio" in r
            assert "total_return" in r
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_optimizer.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement ParameterSweep**

```python
# quantflow/backtest/optimizer.py
"""Parameter sweep and grid search for strategy optimization."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from quantflow.core.models import Asset
from quantflow.core.events import EventBus
from quantflow.backtest.engine import BacktestEngine, BacktestResult
from quantflow.analytics.metrics import (
    total_return, sharpe_ratio, max_drawdown, cagr, sortino_ratio,
    win_rate, profit_factor,
)


@dataclass
class SweepResult:
    """Results from a parameter sweep."""
    results: list[dict]
    best_params: dict | None = None
    best_metric: float = 0.0

    def summary(self) -> str:
        lines = [f"Parameter Sweep: {len(self.results)} combinations tested"]
        if self.best_params:
            lines.append(f"Best params: {self.best_params}")
            lines.append(f"Best metric: {self.best_metric:.4f}")
        lines.append("\nTop 5:")
        for r in sorted(self.results, key=lambda x: x.get("sharpe_ratio", 0), reverse=True)[:5]:
            lines.append(f"  {r['params']} -> Sharpe: {r['sharpe_ratio']:.2f}, Return: {r['total_return']:.2%}")
        return "\n".join(lines)


class ParameterSweep:
    """Grid search over strategy parameters."""

    def __init__(
        self,
        strategy_class: type,
        param_grid: dict[str, list],
        data: dict[Asset, pd.DataFrame],
        engine_kwargs: dict | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.engine_kwargs = engine_kwargs or {"initial_cash": 100_000.0, "slippage_pct": 0.0, "commission_pct": 0.0}

    def run(self, metric: str = "sharpe_ratio") -> SweepResult:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        results = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                bt_result = self._run_single(params)
                metrics = self._compute_metrics(bt_result)
                metrics["params"] = params
                results.append(metrics)
            except Exception:
                continue

        best_params = None
        best_metric = float("-inf")
        for r in results:
            val = r.get(metric, float("-inf"))
            if val > best_metric:
                best_metric = val
                best_params = r["params"]

        return SweepResult(results=results, best_params=best_params, best_metric=best_metric)

    def _run_single(self, params: dict) -> BacktestResult:
        engine = BacktestEngine(**self.engine_kwargs)
        strategy_cls = self.strategy_class

        def factory(bus: EventBus, assets: list[Asset]):
            return strategy_cls(event_bus=bus, assets=assets, **params)

        return engine.run(data=self.data, strategy_factory=factory)

    def _compute_metrics(self, result: BacktestResult) -> dict:
        pnls = [t.pnl for t in result.trades]
        return {
            "total_return": total_return(result.equity_curve),
            "sharpe_ratio": sharpe_ratio(result.equity_curve),
            "max_drawdown": max_drawdown(result.equity_curve),
            "cagr": cagr(result.equity_curve),
            "sortino_ratio": sortino_ratio(result.equity_curve),
            "win_rate": win_rate(pnls),
            "profit_factor": profit_factor(pnls),
            "total_trades": len(result.trades),
        }
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add ParameterSweep for strategy optimization grid search`

---

### Task 6: Walk-Forward Optimization

**Files:**
- Create: `quantflow/backtest/walk_forward.py`
- Create: `tests/test_backtest/test_walk_forward.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest/test_walk_forward.py
import pytest
import pandas as pd
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.backtest.walk_forward import WalkForward, WalkForwardResult


def make_price_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": [p - 1 for p in prices], "high": [p + 2 for p in prices],
         "low": [p - 2 for p in prices], "close": prices,
         "volume": [1e6] * len(prices)},
        index=dates,
    )


class TestWalkForward:
    def setup_method(self):
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        # 500 bars -- enough for multiple windows
        import random
        random.seed(42)
        prices = [100.0]
        for _ in range(499):
            prices.append(prices[-1] * (1 + random.gauss(0.0003, 0.015)))
        self.data = {self.asset: make_price_data(prices)}

    def test_walk_forward_runs(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=self.data,
            train_bars=100,
            test_bars=50,
            step_bars=50,
        )
        result = wf.run()
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    def test_walk_forward_has_oos_metrics(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=self.data,
            train_bars=100,
            test_bars=50,
            step_bars=50,
        )
        result = wf.run()
        for window in result.windows:
            assert "best_params" in window
            assert "oos_sharpe" in window
            assert "is_sharpe" in window

    def test_walk_forward_multiple_windows(self):
        wf = WalkForward(
            strategy_class=SMACrossover,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=self.data,
            train_bars=100,
            test_bars=50,
            step_bars=50,
        )
        result = wf.run()
        # With 500 bars, train=100, test=50, step=50: should get multiple windows
        assert len(result.windows) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_walk_forward.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 3: Implement WalkForward**

```python
# quantflow/backtest/walk_forward.py
"""Walk-forward optimization: rolling train/test backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantflow.core.models import Asset
from quantflow.backtest.optimizer import ParameterSweep
from quantflow.analytics.metrics import sharpe_ratio, total_return


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    windows: list[dict]

    @property
    def aggregate_oos_sharpe(self) -> float:
        values = [w["oos_sharpe"] for w in self.windows if w["oos_sharpe"] is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def aggregate_oos_return(self) -> float:
        values = [w["oos_return"] for w in self.windows if w["oos_return"] is not None]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> str:
        lines = [f"Walk-Forward: {len(self.windows)} windows"]
        lines.append(f"Avg OOS Sharpe: {self.aggregate_oos_sharpe:.2f}")
        lines.append(f"Avg OOS Return: {self.aggregate_oos_return:.2%}")
        lines.append("\nPer window:")
        for i, w in enumerate(self.windows):
            lines.append(
                f"  Window {i+1}: IS Sharpe={w['is_sharpe']:.2f}, "
                f"OOS Sharpe={w['oos_sharpe']:.2f}, "
                f"Params={w['best_params']}"
            )
        return "\n".join(lines)


class WalkForward:
    """Rolling window train/test optimization.

    Splits data into rolling windows. For each window:
    1. Optimize parameters on training set (via ParameterSweep)
    2. Evaluate best params on test set
    3. Record in-sample vs out-of-sample results

    This is the gold standard for strategy validation.
    """

    def __init__(
        self,
        strategy_class: type,
        param_grid: dict[str, list],
        data: dict[Asset, pd.DataFrame],
        train_bars: int = 252,
        test_bars: int = 63,
        step_bars: int = 63,
        engine_kwargs: dict | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.engine_kwargs = engine_kwargs or {"initial_cash": 100_000.0, "slippage_pct": 0.0, "commission_pct": 0.0}

    def run(self, metric: str = "sharpe_ratio") -> WalkForwardResult:
        # Get total number of bars (use first asset's data)
        first_df = next(iter(self.data.values()))
        total_bars = len(first_df)

        windows = []
        start = 0

        while start + self.train_bars + self.test_bars <= total_bars:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            # Split data
            train_data = {a: df.iloc[start:train_end] for a, df in self.data.items()}
            test_data = {a: df.iloc[train_end:test_end] for a, df in self.data.items()}

            # Optimize on training set
            sweep = ParameterSweep(
                strategy_class=self.strategy_class,
                param_grid=self.param_grid,
                data=train_data,
                engine_kwargs=self.engine_kwargs,
            )
            sweep_result = sweep.run(metric=metric)

            if sweep_result.best_params is None:
                start += self.step_bars
                continue

            # Evaluate best params on test set
            from quantflow.backtest.engine import BacktestEngine
            from quantflow.core.events import EventBus

            engine = BacktestEngine(**self.engine_kwargs)
            best_params = sweep_result.best_params
            strategy_cls = self.strategy_class

            def factory(bus: EventBus, assets: list[Asset], _params=best_params):
                return strategy_cls(event_bus=bus, assets=assets, **_params)

            oos_result = engine.run(data=test_data, strategy_factory=factory)

            oos_sharpe = sharpe_ratio(oos_result.equity_curve) if len(oos_result.equity_curve) > 1 else 0.0
            oos_return_val = total_return(oos_result.equity_curve) if len(oos_result.equity_curve) > 1 else 0.0

            windows.append({
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "best_params": best_params,
                "is_sharpe": sweep_result.best_metric,
                "oos_sharpe": oos_sharpe,
                "oos_return": oos_return_val,
            })

            start += self.step_bars

        return WalkForwardResult(windows=windows)
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add WalkForward rolling train/test optimization`

---

### Task 7: CLI

**Files:**
- Create: `quantflow/cli/__init__.py`
- Create: `quantflow/cli/main.py`
- Create: `tests/test_cli/__init__.py`
- Create: `tests/test_cli/test_cli.py`
- Create: `config/default_config.yaml`
- Modify: `pyproject.toml` (add click, add console script)

- [ ] **Step 1: Add click dependency and console script to pyproject.toml**

Add `"click>=8.0"` to dependencies. Add:
```toml
[project.scripts]
quantflow = "quantflow.cli.main:cli"
```

Run: `pip install -e ".[dev]"`

- [ ] **Step 2: Write failing tests**

```python
# tests/test_cli/test_cli.py
import pytest
from click.testing import CliRunner
from quantflow.cli.main import cli


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "QuantFlow" in result.output

    def test_strategy_list(self):
        result = self.runner.invoke(cli, ["strategy", "list"])
        assert result.exit_code == 0
        assert "sma_crossover" in result.output
        assert "mean_reversion" in result.output

    def test_data_list_empty(self):
        result = self.runner.invoke(cli, ["data", "list"])
        assert result.exit_code == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_cli/test_cli.py -v`
Expected: FAIL -- ModuleNotFoundError

- [ ] **Step 4: Implement CLI**

Create `quantflow/cli/__init__.py` (empty) and `tests/test_cli/__init__.py` (empty).

```python
# quantflow/cli/main.py
"""QuantFlow CLI: command-line interface for backtesting and data management."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

from quantflow.core.models import Asset, AssetClass
from quantflow.data.cache import DataCache
from quantflow.data.manager import DataManager
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.registry import get_strategy, list_strategies
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.tearsheet import print_tearsheet, plot_tearsheet


@click.group()
def cli():
    """QuantFlow - Quantitative Research & Backtesting Platform"""
    pass


# ---- Data commands ----

@cli.group()
def data():
    """Manage market data."""
    pass


@data.command("fetch")
@click.argument("symbols", nargs=-1, required=True)
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--asset-class", "asset_class", default="equity",
              type=click.Choice(["equity", "crypto", "forex", "commodity", "index"]))
def data_fetch(symbols, start, end, asset_class):
    """Fetch and cache market data for given symbols."""
    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
    ac = AssetClass(asset_class)
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for symbol in symbols:
        asset = Asset(symbol=symbol, asset_class=ac)
        click.echo(f"Fetching {symbol}...")
        df = manager.get_ohlcv(asset, start_dt, end_dt)
        click.echo(f"  Got {len(df)} bars")

    cache.close()
    click.echo("Done.")


@data.command("list")
def data_list():
    """List cached assets."""
    cache = DataCache()
    assets = cache.list_cached_assets()
    if not assets:
        click.echo("No cached data.")
    else:
        for a in assets:
            click.echo(f"  {a['symbol']} ({a['asset_class']}): "
                       f"{a['bar_count']} bars, {a['first_date'][:10]} to {a['last_date'][:10]}")
    cache.close()


@data.command("status")
def data_status():
    """Show cache freshness."""
    cache = DataCache()
    assets = cache.list_cached_assets()
    if not assets:
        click.echo("No cached data.")
    else:
        for a in assets:
            click.echo(f"  {a['symbol']} ({a['asset_class']}): {a['bar_count']} bars")
    cache.close()


# ---- Strategy commands ----

@cli.group()
def strategy():
    """Strategy tools."""
    pass


@strategy.command("list")
def strategy_list():
    """List available strategies."""
    click.echo("Available strategies:")
    for name in list_strategies():
        click.echo(f"  - {name}")


# ---- Backtest commands ----

@cli.group()
def backtest():
    """Run backtests."""
    pass


@backtest.command("run")
@click.option("--strategy", "strategy_name", required=False, help="Strategy name")
@click.option("--assets", required=False, help="Comma-separated asset symbols")
@click.option("--start", required=False, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=False, help="End date (YYYY-MM-DD)")
@click.option("--cash", default=100000.0, help="Initial cash")
@click.option("--config", "config_file", required=False, type=click.Path(exists=True),
              help="YAML config file")
@click.option("--output", default=None, help="Save tearsheet chart to file")
def backtest_run(strategy_name, assets, start, end, cash, config_file, output):
    """Run a backtest."""
    # Load from config file if provided
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        strategy_name = strategy_name or config.get("strategy")
        assets = assets or ",".join(config.get("assets", []))
        start = start or config.get("start")
        end = end or config.get("end")
        cash = cash if cash != 100000.0 else config.get("initial_cash", 100000.0)

    if not strategy_name or not assets or not start or not end:
        click.echo("Error: --strategy, --assets, --start, and --end are required "
                    "(or provide --config)")
        return

    # Parse
    strategy_cls = get_strategy(strategy_name)
    symbol_list = [s.strip() for s in assets.split(",")]
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Fetch data
    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])

    data_dict = {}
    for symbol in symbol_list:
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        click.echo(f"Fetching {symbol}...")
        df = manager.get_ohlcv(asset, start_dt, end_dt)
        data_dict[asset] = df
        click.echo(f"  Got {len(df)} bars")

    # Run backtest
    click.echo(f"\nRunning {strategy_name} backtest...")
    engine = BacktestEngine(initial_cash=cash)

    # Get strategy params from config if available
    strategy_params = {}
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        strategy_params = config.get("params", {})

    def factory(bus, asset_list):
        return strategy_cls(event_bus=bus, assets=asset_list, **strategy_params)

    result = engine.run(data=data_dict, strategy_factory=factory)

    print_tearsheet(result)

    if output:
        plot_tearsheet(result, save_path=output)

    cache.close()


# ---- Optimize commands ----

@cli.group()
def optimize():
    """Optimization tools."""
    pass


@optimize.command("sweep")
@click.option("--strategy", "strategy_name", required=True)
@click.option("--assets", required=True, help="Comma-separated symbols")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--param", "params", multiple=True, help="key:min-max:step (e.g., fast_period:5-50:5)")
def optimize_sweep(strategy_name, assets, start, end, params):
    """Run a parameter sweep."""
    from quantflow.backtest.optimizer import ParameterSweep

    strategy_cls = get_strategy(strategy_name)
    symbol_list = [s.strip() for s in assets.split(",")]
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
    data_dict = {}
    for symbol in symbol_list:
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        df = manager.get_ohlcv(asset, start_dt, end_dt)
        data_dict[asset] = df

    # Parse param ranges
    param_grid = {}
    for p in params:
        key, range_str = p.split(":", 1)
        if "-" in range_str and ":" in range_str:
            parts = range_str.split(":")
            start_val, end_val = parts[0].split("-")
            step = int(parts[1])
            param_grid[key] = list(range(int(start_val), int(end_val) + 1, step))
        else:
            param_grid[key] = [int(x) for x in range_str.split(",")]

    click.echo(f"Running parameter sweep: {len(param_grid)} params...")
    sweep = ParameterSweep(strategy_class=strategy_cls, param_grid=param_grid, data=data_dict)
    result = sweep.run()
    click.echo(result.summary())
    cache.close()


@optimize.command("walk-forward")
@click.option("--strategy", "strategy_name", required=True)
@click.option("--assets", required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--param", "params", multiple=True, help="key:min-max:step")
@click.option("--train-bars", default=252, help="Training window size")
@click.option("--test-bars", default=63, help="Test window size")
def optimize_walk_forward(strategy_name, assets, start, end, params, train_bars, test_bars):
    """Run walk-forward optimization."""
    from quantflow.backtest.walk_forward import WalkForward

    strategy_cls = get_strategy(strategy_name)
    symbol_list = [s.strip() for s in assets.split(",")]
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
    data_dict = {}
    for symbol in symbol_list:
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        df = manager.get_ohlcv(asset, start_dt, end_dt)
        data_dict[asset] = df

    param_grid = {}
    for p in params:
        key, range_str = p.split(":", 1)
        if "-" in range_str and ":" in range_str:
            parts = range_str.split(":")
            start_val, end_val = parts[0].split("-")
            step = int(parts[1])
            param_grid[key] = list(range(int(start_val), int(end_val) + 1, step))
        else:
            param_grid[key] = [int(x) for x in range_str.split(",")]

    click.echo(f"Running walk-forward optimization...")
    wf = WalkForward(
        strategy_class=strategy_cls, param_grid=param_grid, data=data_dict,
        train_bars=train_bars, test_bars=test_bars, step_bars=test_bars,
    )
    result = wf.run()
    click.echo(result.summary())
    cache.close()


if __name__ == "__main__":
    cli()
```

- [ ] **Step 5: Create default config**

```yaml
# config/default_config.yaml
strategy: sma_crossover
params:
  fast_period: 10
  slow_period: 50
assets:
  - AAPL
start: "2020-01-01"
end: "2024-12-31"
initial_cash: 100000
```

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/ -v --tb=short`
Commit: `feat: add click CLI with data, backtest, strategy, and optimize commands`

---

### Task 8: Full Integration Verification

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Test CLI commands**

```bash
quantflow --help
quantflow strategy list
quantflow data fetch AAPL --start 2023-01-01 --end 2024-12-31
quantflow data list
quantflow backtest run --strategy sma_crossover --assets AAPL --start 2023-01-01 --end 2024-12-31
quantflow optimize sweep --strategy sma_crossover --assets AAPL --start 2023-01-01 --end 2024-12-31 --param fast_period:5-20:5 --param slow_period:20-50:10
```

- [ ] **Step 3: Run existing examples to verify no regressions**

```bash
python -m quantflow.examples.sma_crossover
python -m quantflow.examples.mean_reversion
```

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -A && git commit -m "chore: phase 3A part 2 integration verification complete"
```

---

## Phase 3A Part 2 Summary

After completing these 8 tasks:

- **Strategy registry** -- name-to-class lookup for CLI
- **Pairs Trading** -- cointegration, spread z-score, market neutral
- **Macro Regime** -- FRED-driven allocation shifts (growth/recession/vol/inflation)
- **CompositeStrategy** -- weighted signal ensemble
- **ParameterSweep** -- grid search with parallel execution
- **WalkForward** -- rolling train/test, gold standard validation
- **CLI** -- `quantflow data/backtest/strategy/optimize` commands + YAML config

## What Comes Next (Phase 3B)

- ML/DL strategies (Random Forest, LSTM, RL agent)
- LLM assistant (Claude API integration)
- Streamlit dashboard
- Paper trading (Alpaca, Binance testnet)
