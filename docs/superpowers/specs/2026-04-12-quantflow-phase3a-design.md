# QuantFlow Phase 3A: Infrastructure, Strategies, Optimization & CLI

**Date:** 2026-04-12
**Status:** Design approved, pending implementation plan

## Overview

Phase 3A transforms QuantFlow from a script-based backtesting tool into a complete quantitative research platform with persistent data caching, multiple data sources, advanced strategies (pairs trading, macro regime), rigorous optimization (walk-forward, parameter sweep), strategy composition, and a full CLI. After Phase 3A, the platform is usable entirely from the command line with YAML configuration.

**Goals:**
1. Stop re-fetching data -- SQLite cache makes backtests instant after first fetch
2. Multi-source data -- equities, crypto, forex, macro indicators from Yahoo, CCXT, FRED
3. Advanced strategies teaching new quant concepts -- cointegration, macro regimes, ensembles
4. Rigorous backtesting -- walk-forward optimization, deliberate overfitting exercise
5. Professional CLI interface with YAML config support

---

## Module 1: SQLite Data Cache

### DataCache (`quantflow/data/cache.py`)

SQLite-backed cache for all fetched market data. Stored at `~/.quantflow/cache.db` by default, configurable.

**Tables:**

```sql
market_data(
    asset_symbol TEXT,
    asset_class TEXT,
    timestamp TEXT,      -- ISO 8601 UTC
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT,         -- "yahoo", "ccxt", "fred"
    fetched_at TEXT,     -- when this row was cached
    PRIMARY KEY (asset_symbol, asset_class, timestamp)
)

macro_data(
    indicator TEXT,      -- e.g., "GDP", "CPI", "UNRATE"
    date TEXT,
    value REAL,
    source TEXT,
    fetched_at TEXT,
    PRIMARY KEY (indicator, date)
)
```

**DataCache API:**
- `get_ohlcv(asset, start, end) -> DataFrame | None` -- returns cached data or None
- `put_ohlcv(asset, df, source)` -- stores DataFrame rows
- `get_macro(indicator, start, end) -> DataFrame | None`
- `put_macro(indicator, df, source)`
- `is_fresh(asset, start, end, max_age_hours=24) -> bool`
- `list_cached_assets() -> list[dict]` -- for CLI `data list`
- `get_cache_status() -> list[dict]` -- for CLI `data status`

### DataManager (`quantflow/data/manager.py`)

Orchestrates fetchers and cache. The single entry point for all data access.

**Flow:**
```
DataManager.get_ohlcv(asset, start, end)
  -> check cache for date range
  -> if fully cached and fresh: return from cache
  -> if partially cached: fetch only missing dates, merge, cache
  -> if not cached: fetch from appropriate source, cache, return
```

**Source routing by asset class:**
- EQUITY, INDEX, COMMODITY, FOREX -> YahooFetcher
- CRYPTO -> CCXTFetcher (fallback: YahooFetcher)
- MACRO -> FREDFetcher

---

## Module 2: Additional Data Sources

### FREDFetcher (`quantflow/data/fred_fetcher.py`)

Fetches economic indicators from the Federal Reserve (FRED API). Free API key required, stored in `.env`.

**Key indicators:**
- GDP, CPI (inflation), UNRATE (unemployment), DFF (Fed funds rate)
- T10Y2Y (10Y-2Y yield spread -- recession predictor)
- VIXCLS (VIX -- volatility index)

**API:**
- `fetch_ohlcv(asset, start, end)` -- returns value as close column (open/high/low = close, volume = 0) for compatibility
- `fetch_series(indicator, start, end) -> DataFrame` -- raw series with DatetimeIndex and `value` column
- `supported_asset_classes() -> [AssetClass.MACRO]`

**Dependency:** `fredapi` library

### CCXTFetcher (`quantflow/data/ccxt_fetcher.py`)

Fetches crypto OHLCV from Binance public API via CCXT. No API key needed.

**Features:**
- Better granularity than yfinance: 1m, 5m, 15m, 1h, 4h, 1d
- Handles CCXT's pagination for long date ranges
- Normalizes to standard DataFrame format (DatetimeIndex UTC, lowercase columns)

**API:**
- `fetch_ohlcv(asset, start, end, timeframe)` -- standard DataFetcher protocol
- `supported_asset_classes() -> [AssetClass.CRYPTO]`

**Dependency:** `ccxt` library

---

## Module 3: Pairs Trading Strategy

### PairsTrading (`quantflow/strategies/pairs_trading.py`)

Market-neutral statistical arbitrage strategy. Trades the spread between two cointegrated assets.

**Setup (in `init()`):**
- Takes exactly 2 assets
- Computes cointegration using Engle-Granger method: OLS regression of asset A on asset B, ADF test on residuals
- Calculates hedge ratio from the regression coefficient

**Trading logic (in `next()`):**
- Computes rolling spread: `price_A - hedge_ratio * price_B`
- Calculates z-score of the spread using rolling mean and std (lookback window configurable)
- Long spread when z < -entry_z (default -2.0): buy A, sell B
- Short spread when z > entry_z (default 2.0): sell A, buy B
- Exit when z crosses 0 (spread reverted to mean)

**Parameters:** `lookback_period` (default 60), `entry_z` (default 2.0), `exit_z` (default 0.0)

**Dependency:** `statsmodels` for OLS and ADF test

---

## Module 4: Macro Regime Strategy

### MacroRegime (`quantflow/strategies/macro_regime.py`)

Shifts asset allocation based on detected economic regime using FRED data.

**Regime detection (simple rule-based):**

| Regime | Conditions | Allocation Bias |
|--------|-----------|----------------|
| Growth | Yield spread > 0, VIX < 20 | Long equities, long crypto |
| Recession | Yield spread < 0 or inverting | Reduce exposure, short equities |
| High Volatility | VIX > 25 | Reduce all positions, favor commodities |
| Inflation | CPI YoY > 4% | Favor commodities, reduce bond-sensitive assets |

**Design:**
- Takes a dict mapping asset roles to assets: `{"equity": SPY, "crypto": BTC-USD, "commodity": GLD}`
- Requires `DataManager` to fetch FRED macro data alongside price data
- On each bar, evaluates regime from latest macro indicators
- Emits signals with direction and strength per asset based on regime

**Depends on:** DataManager with FREDFetcher integrated

---

## Module 5: Walk-Forward Optimization & Parameter Sweep

### ParameterSweep (`quantflow/backtest/optimizer.py`)

Grid search over strategy parameters.

**API:**
```python
sweep = ParameterSweep(
    strategy_class=SMACrossover,
    param_grid={"fast_period": range(5, 50, 5), "slow_period": range(20, 100, 10)},
    data=data,
    engine_kwargs={"initial_cash": 100_000},
)
results = sweep.run(metric="sharpe_ratio")  # Returns SweepResult
```

**SweepResult:**
- `results: list[dict]` -- each dict has params + all metrics
- `best_params: dict`
- `best_metric: float`
- `show_overfit_danger(train_data, test_data)` -- runs best in-sample params on OOS data, prints comparison

**Parallelization:** `multiprocessing.Pool` for running parameter combos concurrently.

### WalkForward (`quantflow/backtest/walk_forward.py`)

Rolling train/test optimization -- the gold standard for strategy validation.

**API:**
```python
wf = WalkForward(
    strategy_class=SMACrossover,
    param_grid={"fast_period": range(5, 50, 5), "slow_period": range(20, 100, 10)},
    data=data,
    train_bars=252,    # 1 year training
    test_bars=63,      # 3 months testing
    step_bars=63,      # slide forward 3 months
)
result = wf.run(metric="sharpe_ratio")  # Returns WalkForwardResult
```

**WalkForwardResult:**
- `windows: list[dict]` -- per-window: train period, test period, best params, in-sample metrics, out-of-sample metrics
- `aggregate_oos_sharpe: float` -- average out-of-sample Sharpe across all windows
- `aggregate_oos_return: float`

---

## Module 6: Strategy Composition

### CompositeStrategy (`quantflow/strategies/composite.py`)

Combines multiple strategies into a weighted ensemble.

**Design:**
- Takes list of `(strategy_factory, weight)` tuples
- Creates sub-strategies on the same event bus
- Intercepts sub-strategy signals before they reach the engine
- Merges signals per asset: weighted sum of directions (+1 for LONG, -1 for SHORT)
- Final signal: direction from sign of weighted sum, strength from absolute value (capped at 1.0)
- Configurable minimum threshold (default 0.2) -- only emit if net strength exceeds threshold

**Implementation approach:**
- Uses a separate internal EventBus for sub-strategies
- Sub-strategies emit signals on the internal bus
- CompositeStrategy listens on internal bus, aggregates, then emits merged signals on the main bus

---

## Module 7: CLI

### Entry point (`quantflow/cli/main.py`)

Click-based CLI registered as `quantflow` console script.

**Commands:**

```
quantflow data fetch <symbols> --start --end     Fetch and cache market data
quantflow data list                               List cached assets
quantflow data status                             Show cache freshness

quantflow backtest run --strategy --assets --start --end [options]
quantflow backtest run --config backtest.yaml

quantflow strategy list                           List available strategies
quantflow strategy new <name>                     Scaffold a new strategy file

quantflow optimize sweep --strategy --assets --param key:min-max:step
quantflow optimize walk-forward --strategy --assets --train-bars --test-bars

quantflow report generate <result.json> --output tearsheet.html
```

### Strategy Registry (`quantflow/strategies/registry.py`)

Maps string names to strategy classes for CLI lookup:
```python
STRATEGY_REGISTRY = {
    "sma_crossover": SMACrossover,
    "mean_reversion": MeanReversion,
    "rsi_macd": RSIMACDCombo,
    "pairs_trading": PairsTrading,
    "macro_regime": MacroRegime,
}
```

### YAML Config (`config/default_config.yaml`)

```yaml
strategy: sma_crossover
params:
  fast_period: 10
  slow_period: 50
assets: [AAPL, MSFT]
start: 2020-01-01
end: 2024-12-31
initial_cash: 100000
position_sizer:
  type: fixed_fractional
  fraction: 0.05
risk_manager:
  max_drawdown: 0.20
  max_position_pct: 0.25
```

---

## New Dependencies

| Library | Purpose |
|---------|---------|
| `fredapi` | FRED macro data |
| `ccxt` | Crypto data via Binance |
| `statsmodels` | Cointegration test, OLS regression |
| `click` | CLI framework |
| `python-dotenv` | Load `.env` for API keys |

---

## Project Structure (new/modified files)

```
quantflow/
    data/
        cache.py           -- NEW: SQLite DataCache
        manager.py         -- NEW: DataManager (orchestrates fetchers + cache)
        fred_fetcher.py    -- NEW: FRED macro data fetcher
        ccxt_fetcher.py    -- NEW: CCXT crypto data fetcher
    strategies/
        pairs_trading.py   -- NEW: Pairs Trading (cointegration)
        macro_regime.py    -- NEW: Macro Regime (FRED-driven allocation)
        composite.py       -- NEW: CompositeStrategy (weighted ensemble)
        registry.py        -- NEW: Strategy name -> class mapping
    backtest/
        optimizer.py       -- NEW: ParameterSweep (grid search)
        walk_forward.py    -- NEW: WalkForward optimization
    cli/
        __init__.py        -- NEW
        main.py            -- NEW: Click CLI entry point
    examples/
        pairs_trading.py   -- NEW
        macro_regime.py    -- NEW
        composite.py       -- NEW
        walk_forward.py    -- NEW
config/
    default_config.yaml    -- NEW
    example.env            -- NEW
tests/
    test_data/
        test_cache.py      -- NEW
        test_manager.py    -- NEW
        test_fred_fetcher.py   -- NEW
        test_ccxt_fetcher.py   -- NEW
    test_strategies/
        test_pairs_trading.py  -- NEW
        test_macro_regime.py   -- NEW
        test_composite.py      -- NEW
    test_backtest/
        test_optimizer.py  -- NEW
        test_walk_forward.py   -- NEW
    test_cli/
        __init__.py        -- NEW
        test_cli.py        -- NEW
```
