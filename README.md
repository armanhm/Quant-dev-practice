# QuantFlow

A modular, multi-asset quantitative research and backtesting platform built from scratch in Python. No black-box libraries -- every component is hand-built so you understand exactly what's happening under the hood.

## What It Does

QuantFlow lets you write trading strategies, backtest them against historical data, and analyze performance with professional-grade metrics. The event-driven architecture mirrors how real trading systems work, making it a natural stepping stone to live trading.

```
quantflow data fetch  -->  strategy generates signals  -->  backtest simulates execution  -->  tearsheet shows results
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/armanhm/Quant-dev-practice.git
cd Quant-dev-practice
pip install -e ".[dev]"

# Run a backtest
python -m quantflow.examples.sma_crossover
python -m quantflow.examples.mean_reversion
python -m quantflow.examples.rsi_macd
```

The SMA Crossover example fetches AAPL data, runs a moving average crossover strategy, prints a performance tearsheet, and saves an equity curve chart.

## Architecture

```
+-----------------------------------------------------+
|                  Analytics & Tearsheet                |
+----------+----------+-----------+-------------------+
| Backtest | Strategy  | Portfolio | Risk              |
| Engine   | Framework | Sizing    | Manager           |
+----------+----------+-----------+-------------------+
|                   Data Engine                        |
|          (Yahoo Finance, Indicators)                 |
+-----------------------------------------------------+
|                   Core / Domain                      |
|        (Models, Events, Interfaces)                  |
+-----------------------------------------------------+
```

**Event-driven core:** Market data flows through an event bus. Strategies subscribe to price bars, emit signals, which become orders, which get filled. The same event flow will work for live trading -- only the data source changes.

## Modules

### Core (`quantflow/core/`)
Domain models (`Asset`, `Bar`, `Signal`, `Order`, `Fill`, `Position`), event system (`EventBus` with `MarketDataEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`), and adapter interfaces.

### Data Engine (`quantflow/data/`)
- **Yahoo Finance fetcher** -- equities, crypto, forex, commodities, indices via `yfinance`
- **Indicator library** -- SMA, EMA, RSI, MACD, Bollinger Bands, ATR as pure functions
- Adapter pattern: swap data sources by implementing the `DataFetcher` protocol

### Strategy Framework (`quantflow/strategies/`)
Abstract `Strategy` base class with built-in indicator registration:

```python
class MyStrategy(Strategy):
    def init(self):
        self.rsi = self.indicator("rsi", period=14)
        self.bb = self.indicator("bollinger_bands", period=20, num_std=2.0)

    def next(self, event):
        if self.rsi.latest(event.asset) < 30:
            self.signal(direction=Direction.LONG, strength=0.8)
```

**Built-in strategies:**
| Strategy | Type | Description |
|----------|------|-------------|
| SMA Crossover | Trend | Golden cross / death cross with fast & slow moving averages |
| Mean Reversion | Statistical | Bollinger Band reversion -- buy oversold, sell overbought |
| RSI + MACD Combo | Momentum | Dual confirmation -- RSI for extremes, MACD for trend |

### Backtest Engine (`quantflow/backtest/`)
Event-driven bar-by-bar simulation with:
- Configurable slippage and commission models
- No lookahead bias (enforced architecturally)
- Pluggable position sizing and risk management
- Buy-and-hold benchmark comparison

### Portfolio & Risk (`quantflow/portfolio/`)
**Position sizing:**
- `FixedFractional` -- risk X% of equity per trade
- `KellyCriterion` -- optimal sizing from information theory

**Risk controls:**
- Max drawdown kill switch -- halts trading if portfolio drops too far
- Per-asset exposure limits -- no single position exceeds X% of portfolio
- Max open positions cap

### Analytics (`quantflow/analytics/`)
Performance tearsheet with:
- Return metrics: total return, CAGR
- Risk metrics: Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- Trade metrics: win rate, profit factor, avg win/loss ratio
- Equity curve and drawdown charts via matplotlib

## Example Output

```
==================================================
       QUANTFLOW BACKTEST TEARSHEET
==================================================
  Initial Capital:   $    100,000.00
  Final Equity:      $    144,916.74
--------------------------------------------------
  Total Return:              44.92%
  CAGR:                       7.73%
  Max Drawdown:             -46.33%
--------------------------------------------------
  Sharpe Ratio:                0.41
  Sortino Ratio:               0.61
--------------------------------------------------
  Total Trades:                  35
  Win Rate:                  34.29%
  Profit Factor:               1.29
  Avg Win/Loss:                2.47
==================================================
```

## Project Structure

```
quantflow/
    core/           -- domain models, events, interfaces
    data/           -- data fetchers, technical indicators
    strategies/     -- strategy base class + built-in strategies
    backtest/       -- event-driven engine, execution simulator
    portfolio/      -- position sizing, risk controls
    analytics/      -- performance metrics, tearsheet, charts
    examples/       -- runnable backtest examples
tests/              -- 99 unit tests covering all modules
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- Python 3.11+
- pandas, numpy -- data manipulation
- yfinance -- free market data
- matplotlib -- charting
- pytest -- testing

## Roadmap

- [ ] SQLite caching for market data
- [ ] FRED macro data integration
- [ ] CCXT / CoinGecko crypto data
- [ ] Pairs Trading strategy (cointegration)
- [ ] Macro Regime strategy
- [ ] Walk-forward optimization
- [ ] Parameter sweep runner
- [ ] CLI via click
- [ ] ML/DL strategies (Random Forest, LSTM, RL)
- [ ] LLM-powered research assistant
- [ ] Streamlit dashboard
- [ ] Paper trading via Alpaca API
- [ ] Docker containerization

## License

MIT
