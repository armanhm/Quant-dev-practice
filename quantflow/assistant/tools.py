"""Tool definitions for LLM function calling."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from quantflow.core.models import Asset, AssetClass
from quantflow.data.cache import DataCache
from quantflow.data.manager import DataManager
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.registry import get_strategy, list_strategies
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.metrics import (
    total_return, sharpe_ratio, max_drawdown, cagr, sortino_ratio, win_rate,
)


TOOL_DEFINITIONS = [
    {
        "name": "list_strategies",
        "description": "List all available trading strategies in QuantFlow",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "fetch_data",
        "description": "Fetch OHLCV market data for an asset. Returns summary stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Asset symbol (e.g., AAPL, BTC-USD)"},
                "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end": {"type": "string", "description": "End date YYYY-MM-DD"},
                "asset_class": {"type": "string", "enum": ["equity", "crypto", "forex", "commodity", "index"], "description": "Asset class"},
            },
            "required": ["symbol", "start", "end"],
        },
    },
    {
        "name": "run_backtest",
        "description": "Run a backtest with a given strategy and asset. Returns performance metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "description": "Strategy name (e.g., sma_crossover)"},
                "symbol": {"type": "string", "description": "Asset symbol"},
                "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end": {"type": "string", "description": "End date YYYY-MM-DD"},
                "params": {"type": "object", "description": "Strategy parameters as key-value pairs"},
            },
            "required": ["strategy", "symbol", "start", "end"],
        },
    },
    {
        "name": "explain_concept",
        "description": "Explain a quantitative finance concept. Use this when the user asks about quant topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The concept to explain"},
            },
            "required": ["topic"],
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "list_strategies":
        strategies = list_strategies()
        return f"Available strategies: {', '.join(strategies)}"

    elif name == "fetch_data":
        cache = DataCache()
        manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
        ac = AssetClass(args.get("asset_class", "equity"))
        asset = Asset(symbol=args["symbol"], asset_class=ac)
        start = datetime.strptime(args["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        df = manager.get_ohlcv(asset, start, end)
        cache.close()
        if df.empty:
            return f"No data found for {args['symbol']}"
        return (f"Fetched {len(df)} bars for {args['symbol']} "
                f"from {df.index[0].date()} to {df.index[-1].date()}. "
                f"Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}, "
                f"Avg volume: {df['volume'].mean():,.0f}")

    elif name == "run_backtest":
        cache = DataCache()
        manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
        asset = Asset(symbol=args["symbol"], asset_class=AssetClass.EQUITY)
        start = datetime.strptime(args["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        df = manager.get_ohlcv(asset, start, end)

        strategy_cls = get_strategy(args["strategy"])
        params = args.get("params", {})
        engine = BacktestEngine(initial_cash=100_000.0, slippage_pct=0.0005)

        from quantflow.core.events import EventBus
        def factory(bus: EventBus, assets: list[Asset]):
            return strategy_cls(event_bus=bus, assets=assets, **params)

        result = engine.run(data={asset: df}, strategy_factory=factory)
        cache.close()

        pnls = [t.pnl for t in result.trades]
        return (f"Backtest results for {args['strategy']} on {args['symbol']}:\n"
                f"  Total Return: {total_return(result.equity_curve):.2%}\n"
                f"  Sharpe Ratio: {sharpe_ratio(result.equity_curve):.2f}\n"
                f"  Max Drawdown: {max_drawdown(result.equity_curve):.2%}\n"
                f"  CAGR: {cagr(result.equity_curve):.2%}\n"
                f"  Total Trades: {len(result.trades)}\n"
                f"  Win Rate: {win_rate(pnls):.2%}")

    elif name == "explain_concept":
        return f"[The LLM will explain '{args['topic']}' using its own knowledge]"

    return f"Unknown tool: {name}"
