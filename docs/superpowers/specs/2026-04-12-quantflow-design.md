# QuantFlow -- Quantitative Research & Backtesting Platform

**Date:** 2026-04-12
**Status:** Design approved, pending implementation plan

## Overview

QuantFlow is a modular, multi-asset quantitative research and backtesting platform built from scratch in Python. It covers the full quant workflow: data ingestion, strategy development (rule-based, statistical, and ML/DL), event-driven backtesting, portfolio & risk management, performance analytics, and an LLM-powered research assistant.

**Goals:**
1. Portfolio-worthy project demonstrating quant development depth
2. Hands-on learning of quantitative finance concepts
3. Actually useful tool for personal research and trading decisions
4. Architected for future extension to paper and live trading (Approach C)

**Target user:** Strong Python developer learning quant finance.

---

## Architecture

```
+-----------------------------------------------------+
|                    CLI / Dashboard                    |
|              (commands, visualization)                |
+----------+----------+-----------+-------------------+
| Analytics| Back-    | Strategy  |  Portfolio &       |
| & Risk   | tester   | Framework |  Risk Engine       |
+----------+----------+-----------+-------------------+
|               LLM Assistant (optional)               |
+-----------------------------------------------------+
|                  Data Engine                          |
|  (fetchers, normalization, storage, indicators)      |
+-----------------------------------------------------+
|                  Core / Domain                       |
|  (models, events, interfaces, config)                |
+-----------------------------------------------------+
```

**Bottom-up dependency flow:**
1. **Core** -- shared domain models, event bus, abstract interfaces, configuration
2. **Data Engine** -- fetch, normalize, cache, and serve market data + economic indicators
3. **Strategy Framework** -- base classes, indicator library, signal generation, ML/DL strategies
4. **Backtester** -- event-driven simulation engine, fills, slippage, commission modeling
5. **Portfolio & Risk** -- position sizing, risk metrics, risk controls, portfolio optimization
6. **Analytics** -- tearsheet generation, performance metrics, visualization
7. **LLM Assistant** -- AI-powered research, analysis, code generation, tutoring
8. **CLI / Dashboard** -- command-line interface + optional Streamlit web dashboard

**Key architectural decisions:**
- Event-driven core (prepares for future live trading)
- Each module is a Python package with explicit public interfaces
- Data layer uses an adapter pattern (swap sources by implementing one interface)
- All timestamps in UTC
- Prices as float64 for indicators/analytics, Decimal for order/fill math

---

## Module 1: Core Domain

### Domain Models

```
Asset          -- symbol, asset_class, exchange, metadata
OHLCV          -- timestamp, open, high, low, close, volume
Signal         -- timestamp, asset, direction (long/short/flat), strength (0-1), metadata
Order          -- asset, side (buy/sell), quantity, order_type (market/limit), status
Fill           -- order reference, fill_price, fill_quantity, commission, slippage
Position       -- asset, quantity, entry_price, current_price, unrealized_pnl
Portfolio      -- positions, cash, equity, history
```

### Asset Classes

| Enum Value | Description |
|-----------|-------------|
| `EQUITY` | Stocks |
| `CRYPTO` | Cryptocurrencies |
| `FOREX` | Currency pairs |
| `COMMODITY` | Gold, oil, etc. |
| `INDEX` | S&P 500, VIX, etc. |
| `MACRO` | GDP, CPI, interest rates |
| `OPTION` | Options chains |

### Event System

Synchronous event bus decoupling all modules. Same event flow works for live trading -- only the event source changes.

```
Events:
  MarketDataEvent   -- new bar/tick available
  SignalEvent        -- strategy generated a signal
  OrderEvent         -- order submitted
  FillEvent          -- order was filled
  PortfolioEvent     -- portfolio state updated
```

**Backtest event flow:**
MarketDataEvent -> Strategy.next() -> SignalEvent -> PortfolioManager -> OrderEvent -> ExecutionSimulator -> FillEvent -> Portfolio update

### Configuration

YAML-based config for data sources, strategy parameters, backtest settings. Environment variables for API keys (`.env` file, never committed).

---

## Module 2: Data Engine

### Data Source Adapters (all free tier)

| Source | Asset Classes | What It Provides |
|--------|--------------|-----------------|
| Yahoo Finance (`yfinance`) | Equities, Crypto, Forex, Commodities, Indices, Options | Daily/intraday OHLCV, fundamentals, options chains |
| Alpha Vantage (free key) | Equities, Forex, Crypto | Daily OHLCV, technical indicators, 25 req/day |
| CoinGecko (free) | Crypto | Prices, market cap, volume, metadata, no key needed |
| CCXT (Binance public) | Crypto | OHLCV with better granularity than yfinance |
| FRED (free key) | Macro | GDP, CPI, unemployment, interest rates, 120 req/min |

### Adapter Protocol

```python
class DataFetcher(Protocol):
    def fetch_ohlcv(self, asset, start, end, timeframe) -> DataFrame
    def fetch_fundamentals(self, asset) -> dict
    def supported_asset_classes(self) -> list[AssetClass]
```

Every source implements this protocol. A `DataManager` routes requests to the right fetcher based on asset class, with fallback chains (try Yahoo first, fall back to Alpha Vantage).

### Local Storage & Caching

- SQLite database for cached OHLCV data -- fetch once, query many times
- Automatic staleness detection -- re-fetch if data is older than configurable threshold
- Schema: `market_data(asset_id, timestamp, open, high, low, close, volume, source)`
- Macro data in separate table: `macro_data(date, indicator, value, source)`

### Indicator Library

Built on top of the data layer, computed lazily and cached:

- **Trend:** SMA, EMA, MACD, ADX
- **Momentum:** RSI, Stochastic, ROC, Williams %R
- **Volatility:** Bollinger Bands, ATR, historical volatility
- **Volume:** OBV, VWAP
- **Custom:** users can register their own indicator functions

### Data Normalization

- All timestamps converted to UTC
- Consistent DataFrame format: DatetimeIndex, columns = [open, high, low, close, volume]
- Missing data handling: forward-fill for gaps, NaN-aware indicator calculations

---

## Module 3: Strategy Framework

### Strategy Base Class

```python
class MyStrategy(Strategy):
    def init(self):
        self.sma_fast = self.indicator("sma", period=10)
        self.sma_slow = self.indicator("sma", period=50)

    def next(self, bar: MarketDataEvent):
        if self.sma_fast[-1] > self.sma_slow[-1]:
            self.signal(direction="long", strength=0.8)
        elif self.sma_fast[-1] < self.sma_slow[-1]:
            self.signal(direction="short", strength=0.8)
```

### Built-in Rule-Based Strategies

| Strategy | Category | What You Learn |
|----------|----------|---------------|
| SMA Crossover | Technical/Trend | Signal generation basics, moving averages |
| Mean Reversion | Statistical | Z-scores, Bollinger Band strategies |
| RSI + MACD Combo | Technical/Momentum | Combining indicators, signal filtering |
| Pairs Trading | Statistical Arbitrage | Cointegration, spread trading, market-neutral |
| Momentum Factor | Factor Investing | Cross-sectional ranking, factor construction |
| Macro Regime | Fundamental/Macro | FRED data, regime detection, allocation shifts |
| Multi-Asset Risk Parity | Portfolio | Correlation matrices, inverse-vol weighting |

Each built-in strategy is documented with the quant theory behind it.

### ML/DL Strategies

| Strategy | Technique | What You Learn |
|----------|-----------|---------------|
| Feature-Based Classifier | Random Forest / XGBoost | Feature engineering, time-series CV, up/down prediction |
| Regime Detection | Hidden Markov Models | Unsupervised learning, market regime identification |
| Price Forecasting | LSTM / GRU | Sequence modeling, deep learning on time-series |
| Sentiment Signal | NLP (transformer-based) | Alternative data, news/social sentiment as signal |
| RL Trading Agent | DQN / PPO | Trading as RL problem, reward shaping, action spaces |
| Factor Model | Linear Regression / PCA | Dimensionality reduction, Fama-French style models |

### ML Infrastructure

- **Feature Pipeline:** Transforms raw data into feature matrices. Enforces walk-forward validation -- no lookahead bias.
- **Model Registry:** Trained models versioned and stored locally (joblib for sklearn, checkpoints for PyTorch).
- **Train/Predict Separation:** Training happens offline. At backtest time, strategy loads pre-trained model and calls `predict()` per bar.
- **Overfitting Diagnostics:** In-sample vs. out-of-sample comparison, feature importance reporting.

### Strategy Composition

`CompositeStrategy` combines multiple strategies with configurable weights. Enables ensemble models (e.g., 40% momentum + 30% mean reversion + 30% macro regime).

### Strategy Parameters

Strategies declare parameters for optimization:
```python
param("fast_period", default=10, min=5, max=50)
```

---

## Module 4: Backtesting Engine

### Event-Driven Simulation Loop

```
for each bar in historical_data:
    1. DataEngine emits MarketDataEvent
    2. Strategy.next() processes bar -> emits SignalEvent(s)
    3. PortfolioManager receives signals -> applies position sizing -> emits OrderEvent(s)
    4. ExecutionSimulator fills orders -> emits FillEvent(s)
    5. Portfolio updates positions, equity curve, logs state
```

### Execution Simulation

- **Slippage models:** Fixed (e.g., 0.1%), volume-based, or custom
- **Commission models:** Per-trade flat fee, percentage-based, or per-share -- configurable per asset class
- **Fill logic:** Market orders fill at next bar's open. Limit orders fill if price crosses limit. Partial fills for large orders relative to volume.
- **No lookahead guarantee:** Strategy only sees data up to current bar. Enforced architecturally.

### Transaction Cost Defaults

| Asset Class | Commission | Slippage |
|------------|-----------|---------|
| Equities | $0.00 | 0.05% |
| Crypto | 0.1% maker/taker | 0.05% |
| Forex | Spread-based (1 pip) | 0.02% |
| Commodities | 0.1% | 0.1% |
| Options | $0.65/contract | 0.1% |

### Backtest Modes

- **Single strategy, single asset** -- simplest, great for learning
- **Single strategy, multi-asset** -- test across a universe
- **Multi-strategy, multi-asset** -- ensemble with capital allocation
- **Walk-forward optimization** -- rolling train/test windows, prevents overfitting
- **Parameter sweep** -- grid/random search over strategy params, parallelized
- **Deliberate overfitting exercise** -- intentionally overfit, then show why out-of-sample testing matters

### Backtest Output

A `BacktestResult` object containing:
- Full equity curve (daily portfolio value)
- Trade log (every entry/exit with P&L)
- Signal log (every signal generated)
- Position history over time
- Benchmark comparison (buy-and-hold of same assets)

---

## Module 5: Portfolio & Risk Engine

### Portfolio State

```
Portfolio (updated every bar):
  +-- Cash balance
  +-- Positions (per asset: quantity, entry, current price, unrealized P&L)
  +-- Total equity (cash + positions market value)
  +-- Equity curve history
  +-- Trade history (closed trades with realized P&L)
```

### Position Sizing Models

| Model | How It Works | What You Learn |
|-------|-------------|---------------|
| Fixed Fractional | Risk X% of equity per trade | Basic money management |
| Kelly Criterion | Optimal fraction from win rate and win/loss ratio | Information theory in finance, why full Kelly is dangerous |
| Volatility-Targeted | Size inversely proportional to ATR/volatility | Risk parity concepts |
| Equal Weight | Divide capital equally across signals | Simplest baseline |
| Signal-Weighted | Size proportional to signal strength (0-1) | Connecting conviction to allocation |

### Risk Controls

| Control | What It Does |
|---------|-------------|
| Max Drawdown Kill Switch | Halts trading if portfolio drops X% from peak |
| Per-Asset Exposure Limit | No single asset exceeds X% of portfolio |
| Per-Asset-Class Limit | No asset class exceeds X% of portfolio |
| Max Open Positions | Cap on total simultaneous positions |
| Daily Loss Limit | Stop trading for the day if losses exceed threshold |
| Correlation Guard | Warn/block if new position highly correlated with existing |

### Portfolio Optimization

- Mean-Variance (Markowitz) optimization
- Minimum variance portfolio
- Risk parity (equal risk contribution)
- Black-Litterman (combine market equilibrium with strategy views)

---

## Module 6: Analytics & Tearsheet

### Performance Metrics

**Return metrics:** Total return, CAGR, monthly/annual returns
**Risk metrics:** Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, VaR (95%, 99%), CVaR/Expected Shortfall
**Trade metrics:** Win rate, avg win/loss, profit factor, total trades, win/loss streaks
**Benchmark metrics:** Alpha, Beta, correlation to benchmark

### Visualizations

- Equity curve vs benchmark
- Drawdown chart
- Monthly returns heatmap
- Rolling Sharpe (6-month window)
- Trade distribution histogram
- Win/loss streak chart
- Exposure over time (per asset)
- Correlation matrix (multi-asset)
- Efficient frontier plot (portfolio optimization)

### Output Formats

- Console (rich terminal tables and charts via `rich`)
- HTML tearsheet (self-contained, shareable)
- JSON (machine-readable for further analysis)
- PDF export

---

## Module 7: LLM Assistant

### Capabilities

| Capability | What It Does |
|-----------|-------------|
| Strategy Advisor | Explain quant concepts, suggest strategy ideas |
| Backtest Analyst | Interpret results in plain English, flag issues |
| Code Generator | Generate strategy boilerplate from natural language |
| Research Summarizer | Summarize financial news, macro data |
| Debug Helper | Diagnose underperformance, suggest improvements |
| Learning Tutor | Explain math/theory behind any platform concept |
| Sentiment Analyzer | Score news/articles for bullish/bearish sentiment as a trading signal |

### Architecture

```
LLM Assistant
  +-- Chat Interface
  +-- Tool Functions (function calling)
  +-- LLM Provider Adapter (Claude API / OpenAI / Ollama)

Tools the LLM can call:
  - run_backtest(strategy, params)
  - fetch_data(asset, range)
  - analyze_portfolio(metrics)
  - plot_chart(data, type)
  - search_strategies(criteria)
  - explain_concept(topic)
```

### Key Decisions

- **Provider-agnostic:** Adapter pattern. Start with Claude API, add OpenAI or Ollama.
- **Tool-use:** LLM invokes platform functions via function calling.
- **Conversation memory:** Per-session chat history.
- **Cost-conscious:** Ollama for free local models. Claude/OpenAI for those with API keys.
- **Optional:** Platform works fully without LLM integration.

---

## Module 8: CLI & Dashboard

### CLI (`quantflow` command via `click`)

```bash
# Data
quantflow data fetch AAPL BTC-USD EURUSD --start 2020-01-01
quantflow data list
quantflow data status

# Backtesting
quantflow backtest run --strategy sma_crossover --assets AAPL --start 2020-01-01
quantflow backtest run --config my_backtest.yaml
quantflow backtest compare result_1.json result_2.json

# Strategy
quantflow strategy list
quantflow strategy new my_strategy
quantflow strategy optimize --strategy sma_crossover --param fast_period:5-50

# Portfolio
quantflow portfolio optimize --assets AAPL,MSFT,GOOG --method risk_parity

# LLM Assistant
quantflow chat
quantflow chat "Why did my momentum strategy underperform in Q1 2022?"

# Tearsheet
quantflow report generate result.json --output tearsheet.html
```

### Streamlit Dashboard (optional)

| Page | What It Shows |
|------|-------------|
| Data Explorer | Browse cached data, plot assets, overlay indicators |
| Strategy Lab | Configure and run backtests visually, parameter sliders |
| Results Viewer | Interactive tearsheet with zoomable charts |
| Comparison | Side-by-side strategy comparison |
| Portfolio | Portfolio optimization, efficient frontier |
| Chat | LLM assistant panel |

---

## Tech Stack

| Layer | Libraries |
|-------|----------|
| Core | Python 3.11+, dataclasses, abc, typing |
| Data | pandas, numpy, SQLite, yfinance, ccxt, fredapi |
| ML/DL | scikit-learn, XGBoost, PyTorch (optional) |
| Visualization | plotly, matplotlib, rich |
| Dashboard | Streamlit |
| LLM | Anthropic SDK, OpenAI SDK, Ollama (optional) |
| CLI | click |
| Testing | pytest, pytest-cov |
| Packaging | pyproject.toml, pip |

---

## Stretch Goals (Future -- Approach C)

- **Paper trading:** Alpaca API (stocks), Binance testnet (crypto) -- engine switches from historical to real-time websocket bars
- **Live trading:** Full broker integration with real money
- **Docker:** Containerize for portability
- **Real-time data:** Websocket feeds for live market data
- **Web API:** FastAPI layer exposing platform functionality
- **Alerting:** Notifications when strategies generate signals in live mode

---

## Project Structure

```
quantflow/
  core/             -- domain models, events, interfaces, config
  data/             -- data fetchers, storage, indicators, caching
  strategies/       -- base class, built-in strategies, ML strategies
  backtest/         -- engine, execution simulator, parameter optimization
  portfolio/        -- portfolio manager, position sizing, risk controls
  analytics/        -- metrics, tearsheet, visualization
  assistant/        -- LLM integration, tools, chat interface
  cli/              -- click CLI commands
  dashboard/        -- Streamlit app
tests/
  test_core/
  test_data/
  test_strategies/
  test_backtest/
  test_portfolio/
  test_analytics/
  test_assistant/
docs/
config/
  default_config.yaml
  example.env
pyproject.toml
README.md
```
