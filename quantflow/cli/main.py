"""QuantFlow CLI: command-line interface for backtesting and data management."""
from __future__ import annotations

from datetime import datetime, timezone

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
    """Show cache status."""
    cache = DataCache()
    assets = cache.list_cached_assets()
    if not assets:
        click.echo("No cached data.")
    else:
        for a in assets:
            click.echo(f"  {a['symbol']} ({a['asset_class']}): {a['bar_count']} bars")
    cache.close()


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


@cli.group()
def backtest():
    """Run backtests."""
    pass


@backtest.command("run")
@click.option("--strategy", "strategy_name", required=False)
@click.option("--assets", required=False, help="Comma-separated symbols")
@click.option("--start", required=False)
@click.option("--end", required=False)
@click.option("--cash", default=100000.0)
@click.option("--config", "config_file", required=False, type=click.Path(exists=True))
@click.option("--output", default=None)
def backtest_run(strategy_name, assets, start, end, cash, config_file, output):
    """Run a backtest."""
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        strategy_name = strategy_name or config.get("strategy")
        assets = assets or ",".join(config.get("assets", []))
        start = start or config.get("start")
        end = end or config.get("end")
        cash = cash if cash != 100000.0 else config.get("initial_cash", 100000.0)

    if not strategy_name or not assets or not start or not end:
        click.echo("Error: --strategy, --assets, --start, and --end required (or --config)")
        return

    strategy_cls = get_strategy(strategy_name)
    symbol_list = [s.strip() for s in assets.split(",")]
    start_dt = datetime.strptime(str(start), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(str(end), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    cache = DataCache()
    manager = DataManager(cache=cache, fetchers=[YahooFetcher()])
    data_dict = {}
    for symbol in symbol_list:
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        click.echo(f"Fetching {symbol}...")
        df = manager.get_ohlcv(asset, start_dt, end_dt)
        data_dict[asset] = df
        click.echo(f"  Got {len(df)} bars")

    click.echo(f"\nRunning {strategy_name} backtest...")
    engine = BacktestEngine(initial_cash=cash)

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


@cli.group()
def optimize():
    """Optimization tools."""
    pass


@optimize.command("sweep")
@click.option("--strategy", "strategy_name", required=True)
@click.option("--assets", required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--param", "params", multiple=True, help="key:min-max:step")
def optimize_sweep(strategy_name, assets, start, end, params):
    """Run parameter sweep."""
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

    click.echo(f"Running parameter sweep...")
    sweep = ParameterSweep(strategy_class=strategy_cls, param_grid=param_grid, data=data_dict)
    result = sweep.run()
    click.echo(result.summary())
    cache.close()


@optimize.command("walk-forward")
@click.option("--strategy", "strategy_name", required=True)
@click.option("--assets", required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--param", "params", multiple=True)
@click.option("--train-bars", default=252)
@click.option("--test-bars", default=63)
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
    wf = WalkForward(strategy_class=strategy_cls, param_grid=param_grid, data=data_dict,
                     train_bars=train_bars, test_bars=test_bars, step_bars=test_bars)
    result = wf.run()
    click.echo(result.summary())
    cache.close()


@cli.command("dashboard")
def dashboard_command():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "quantflow/dashboard/app.py", "--server.headless", "true"])


@cli.command("chat")
@click.argument("message", required=False)
def chat_command(message):
    """Chat with the QuantFlow AI assistant."""
    if message:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            click.echo("Error: Set ANTHROPIC_API_KEY environment variable.")
            return
        from quantflow.assistant.provider import ClaudeProvider
        from quantflow.assistant.chat import ChatSession
        provider = ClaudeProvider(api_key=api_key)
        session = ChatSession(provider=provider)
        response = session.send(message)
        click.echo(response)
    else:
        from quantflow.assistant.chat import run_interactive_chat
        run_interactive_chat()


if __name__ == "__main__":
    cli()
