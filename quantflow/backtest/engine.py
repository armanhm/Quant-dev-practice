from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from quantflow.core.models import (
    Asset, Bar, Order, OrderSide, OrderType, OrderStatus,
    Position, Direction, Fill,
)
from quantflow.core.events import (
    EventBus, MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
)
from quantflow.backtest.execution import SimulatedExecution
from quantflow.strategies.base import Strategy


@dataclass
class Trade:
    """A completed round-trip trade."""
    asset: Asset
    entry_time: datetime
    exit_time: datetime
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float


@dataclass
class BacktestResult:
    """Contains all output from a backtest run."""
    equity_curve: list[float]
    timestamps: list[datetime]
    trades: list[Trade]
    signals: list[SignalEvent]
    benchmark_equity: list[float]
    initial_cash: float


class BacktestEngine:
    """Event-driven backtesting engine.

    Feeds historical bars one at a time through the event bus.
    Strategies emit signals, which become orders, which get filled.
    Tracks portfolio state bar-by-bar.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0,
        position_size_pct: float = 0.95,
    ) -> None:
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.position_size_pct = position_size_pct

    def run(
        self,
        data: dict[Asset, pd.DataFrame],
        strategy_factory: Callable[[EventBus, list[Asset]], Strategy],
    ) -> BacktestResult:
        bus = EventBus()
        assets = list(data.keys())

        # State
        cash = self.initial_cash
        positions: dict[Asset, Position] = {}
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        trades: list[Trade] = []
        signals: list[SignalEvent] = []
        current_prices: dict[Asset, float] = {}

        # Execution
        executor = SimulatedExecution(
            event_bus=bus,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )

        # Signal handler: convert signals to orders and fills
        def on_signal(event: SignalEvent):
            nonlocal cash
            signals.append(event)
            sig = event.signal
            asset = sig.asset
            price = current_prices.get(asset)
            if price is None or price <= 0:
                return

            current_pos = positions.get(asset)

            if sig.direction == Direction.LONG:
                if current_pos and current_pos.quantity > 0:
                    return  # Already long
                # Close short if exists
                if current_pos and current_pos.quantity < 0:
                    close_order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=abs(current_pos.quantity),
                        order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(close_order, price)
                    pnl = (current_pos.entry_price - fill.fill_price) * abs(current_pos.quantity)
                    # Buy back shares to close short: pay cash
                    cash -= fill.fill_price * fill.fill_quantity + fill.commission
                    trades.append(Trade(
                        asset=asset,
                        entry_time=sig.timestamp,
                        exit_time=sig.timestamp,
                        side="short",
                        entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price,
                        quantity=abs(current_pos.quantity),
                        pnl=pnl - fill.commission,
                        commission=fill.commission,
                    ))
                    del positions[asset]

                # Open long
                available = cash * self.position_size_pct
                quantity = available / price
                if quantity > 0:
                    order = Order(
                        asset=asset, side=OrderSide.BUY,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cost = fill.fill_price * fill.fill_quantity + fill.commission
                    cash -= cost
                    positions[asset] = Position(
                        asset=asset,
                        quantity=fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

            elif sig.direction == Direction.SHORT:
                if current_pos and current_pos.quantity < 0:
                    return  # Already short
                # Close long if exists
                if current_pos and current_pos.quantity > 0:
                    close_order = Order(
                        asset=asset, side=OrderSide.SELL,
                        quantity=current_pos.quantity,
                        order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(close_order, price)
                    pnl = (fill.fill_price - current_pos.entry_price) * current_pos.quantity
                    cash += fill.fill_price * fill.fill_quantity - fill.commission
                    trades.append(Trade(
                        asset=asset,
                        entry_time=sig.timestamp,
                        exit_time=sig.timestamp,
                        side="long",
                        entry_price=current_pos.entry_price,
                        exit_price=fill.fill_price,
                        quantity=current_pos.quantity,
                        pnl=pnl - fill.commission,
                        commission=fill.commission,
                    ))
                    del positions[asset]

                # Open short
                available = cash * self.position_size_pct
                quantity = available / price
                if quantity > 0:
                    order = Order(
                        asset=asset, side=OrderSide.SELL,
                        quantity=quantity, order_type=OrderType.MARKET,
                    )
                    fill = executor.execute(order, price)
                    cash += fill.fill_price * fill.fill_quantity - fill.commission
                    positions[asset] = Position(
                        asset=asset,
                        quantity=-fill.fill_quantity,
                        entry_price=fill.fill_price,
                    )

        bus.subscribe(SignalEvent, on_signal)

        # Create strategy (this subscribes to MarketDataEvent)
        strategy = strategy_factory(bus, assets)

        # Build unified timeline from all assets
        all_dates: set[datetime] = set()
        for df in data.values():
            all_dates.update(df.index.to_pydatetime())
        sorted_dates = sorted(all_dates)

        # Benchmark: buy-and-hold from first bar
        benchmark_equity: list[float] = []
        first_prices: dict[Asset, float] = {}

        # Main loop: emit bars in chronological order
        for ts in sorted_dates:
            for asset, df in data.items():
                if ts in df.index:
                    row = df.loc[ts]
                    bar = Bar(
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                    current_prices[asset] = bar.close
                    bus.emit(MarketDataEvent(asset=asset, bar=bar))

                    if asset not in first_prices:
                        first_prices[asset] = bar.close

            # Calculate equity
            equity = cash
            for asset, pos in positions.items():
                price = current_prices.get(asset, pos.entry_price)
                if pos.quantity > 0:
                    equity += pos.quantity * price
                else:
                    equity += pos.quantity * price

            equity_curve.append(equity)
            timestamps.append(ts)

            # Benchmark
            bench = self.initial_cash
            if first_prices:
                per_asset = self.initial_cash / len(first_prices)
                bench = sum(
                    per_asset * (current_prices.get(a, fp) / fp)
                    for a, fp in first_prices.items()
                )
            benchmark_equity.append(bench)

        return BacktestResult(
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            signals=signals,
            benchmark_equity=benchmark_equity,
            initial_cash=self.initial_cash,
        )
