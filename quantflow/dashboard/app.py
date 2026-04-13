# quantflow/dashboard/app.py
"""QuantFlow Streamlit Dashboard."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

from quantflow.core.models import Asset, AssetClass
from quantflow.core.events import EventBus
from quantflow.data.cache import DataCache
from quantflow.data.manager import DataManager
from quantflow.data.yahoo_fetcher import YahooFetcher
from quantflow.strategies.registry import get_strategy, list_strategies
from quantflow.backtest.engine import BacktestEngine
from quantflow.analytics.metrics import (
    total_return, sharpe_ratio, max_drawdown, cagr, sortino_ratio,
    win_rate, profit_factor, avg_win_loss_ratio,
)


st.set_page_config(page_title="QuantFlow", layout="wide")


def get_data_manager():
    if "data_manager" not in st.session_state:
        cache = DataCache()
        st.session_state["data_manager"] = DataManager(cache=cache, fetchers=[YahooFetcher()])
    return st.session_state["data_manager"]


def page_data_explorer():
    st.header("Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", value="AAPL")
    with col2:
        start = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col3:
        end = st.date_input("End Date", value=datetime(2024, 12, 31))

    if st.button("Fetch Data"):
        manager = get_data_manager()
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.min.time()).replace(tzinfo=timezone.utc)

        with st.spinner("Fetching..."):
            df = manager.get_ohlcv(asset, start_dt, end_dt)

        if not df.empty:
            st.session_state["current_data"] = df
            st.session_state["current_symbol"] = symbol
            st.success(f"Got {len(df)} bars")

    if "current_data" in st.session_state:
        df = st.session_state["current_data"]
        symbol = st.session_state.get("current_symbol", "")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
        ))
        fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date",
                          yaxis_title="Price ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bars", len(df))
        col2.metric("Close Range", f"${df['close'].min():.2f} - ${df['close'].max():.2f}")
        col3.metric("Avg Volume", f"{df['volume'].mean():,.0f}")
        col4.metric("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")


def page_strategy_lab():
    st.header("Strategy Lab")

    col1, col2 = st.columns(2)
    with col1:
        strategy_name = st.selectbox("Strategy", list_strategies())
        symbol = st.text_input("Symbol", value="AAPL", key="strat_symbol")
        start = st.date_input("Start", value=datetime(2020, 1, 1), key="strat_start")
        end = st.date_input("End", value=datetime(2024, 12, 31), key="strat_end")

    with col2:
        initial_cash = st.number_input("Initial Cash ($)", value=100000, step=10000)
        slippage = st.slider("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100

        st.subheader("Strategy Parameters")
        params = {}
        if strategy_name in ("sma_crossover",):
            params["fast_period"] = st.slider("Fast Period", 3, 50, 10)
            params["slow_period"] = st.slider("Slow Period", 10, 200, 50)
        elif strategy_name in ("mean_reversion",):
            params["bb_period"] = st.slider("BB Period", 5, 50, 20)
            params["num_std"] = st.slider("Num Std", 1.0, 3.0, 2.0, 0.1)
        elif strategy_name in ("rsi_macd",):
            params["rsi_period"] = st.slider("RSI Period", 5, 30, 14)
            params["rsi_oversold"] = st.slider("RSI Oversold", 10, 40, 30)
            params["rsi_overbought"] = st.slider("RSI Overbought", 60, 90, 70)

    if st.button("Run Backtest", type="primary"):
        manager = get_data_manager()
        asset = Asset(symbol=symbol, asset_class=AssetClass.EQUITY)
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.min.time()).replace(tzinfo=timezone.utc)

        with st.spinner("Running backtest..."):
            df = manager.get_ohlcv(asset, start_dt, end_dt)
            strategy_cls = get_strategy(strategy_name)
            engine = BacktestEngine(initial_cash=initial_cash, slippage_pct=slippage)

            def factory(bus, assets):
                return strategy_cls(event_bus=bus, assets=assets, **params)

            result = engine.run(data={asset: df}, strategy_factory=factory)
            st.session_state["backtest_result"] = result
            st.session_state["backtest_info"] = {
                "strategy": strategy_name, "symbol": symbol, "params": params,
            }

        st.success("Backtest complete!")

    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]
        info = st.session_state["backtest_info"]
        pnls = [t.pnl for t in result.trades]

        st.subheader(f"Results: {info['strategy']} on {info['symbol']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{total_return(result.equity_curve):.2%}")
        col2.metric("Sharpe Ratio", f"{sharpe_ratio(result.equity_curve):.2f}")
        col3.metric("Max Drawdown", f"{max_drawdown(result.equity_curve):.2%}")
        col4.metric("Total Trades", len(result.trades))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("CAGR", f"{cagr(result.equity_curve):.2%}")
        col6.metric("Sortino", f"{sortino_ratio(result.equity_curve):.2f}")
        col7.metric("Win Rate", f"{win_rate(pnls):.2%}")
        col8.metric("Profit Factor", f"{profit_factor(pnls):.2f}")

        # Equity curve
        import numpy as np
        eq = np.array(result.equity_curve)
        bench = np.array(result.benchmark_equity)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.timestamps, y=eq, name="Strategy", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=result.timestamps, y=bench, name="Buy & Hold",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="Equity Curve", xaxis_title="Date",
                          yaxis_title="Portfolio Value ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=result.timestamps, y=dd, fill="tozeroy",
                                  name="Drawdown", line=dict(color="red")))
        fig2.update_layout(title="Drawdown", xaxis_title="Date",
                           yaxis_title="Drawdown", height=300)
        st.plotly_chart(fig2, use_container_width=True)

        # Trade log
        if result.trades:
            st.subheader("Trade Log")
            trade_data = [{
                "Side": t.side, "Entry": f"${t.entry_price:.2f}",
                "Exit": f"${t.exit_price:.2f}", "Qty": f"{t.quantity:.2f}",
                "P&L": f"${t.pnl:.2f}",
            } for t in result.trades[-20:]]  # Last 20 trades
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)


def page_results_viewer():
    st.header("Results Viewer")

    if "backtest_result" not in st.session_state:
        st.info("Run a backtest in the Strategy Lab first.")
        return

    result = st.session_state["backtest_result"]
    info = st.session_state["backtest_info"]
    st.write(f"Showing results for **{info['strategy']}** on **{info['symbol']}**")
    st.write(f"Parameters: {info['params']}")

    pnls = [t.pnl for t in result.trades]
    metrics = {
        "Total Return": f"{total_return(result.equity_curve):.2%}",
        "CAGR": f"{cagr(result.equity_curve):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(result.equity_curve):.2f}",
        "Sortino Ratio": f"{sortino_ratio(result.equity_curve):.2f}",
        "Max Drawdown": f"{max_drawdown(result.equity_curve):.2%}",
        "Win Rate": f"{win_rate(pnls):.2%}",
        "Profit Factor": f"{profit_factor(pnls):.2f}",
        "Avg Win/Loss": f"{avg_win_loss_ratio(pnls):.2f}",
        "Total Trades": str(len(result.trades)),
        "Initial Cash": f"${result.initial_cash:,.2f}",
        "Final Equity": f"${result.equity_curve[-1]:,.2f}",
    }
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))


def main():
    st.sidebar.title("QuantFlow")
    page = st.sidebar.radio("Navigate", ["Data Explorer", "Strategy Lab", "Results Viewer"])

    if page == "Data Explorer":
        page_data_explorer()
    elif page == "Strategy Lab":
        page_strategy_lab()
    elif page == "Results Viewer":
        page_results_viewer()


if __name__ == "__main__":
    main()
