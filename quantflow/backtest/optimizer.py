"""Parameter sweep and grid search for strategy optimization."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import pandas as pd
from quantflow.core.models import Asset
from quantflow.core.events import EventBus
from quantflow.backtest.engine import BacktestEngine, BacktestResult
from quantflow.analytics.metrics import (
    total_return, sharpe_ratio, max_drawdown, cagr, sortino_ratio, win_rate, profit_factor,
)


@dataclass
class SweepResult:
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
    def __init__(self, strategy_class: type, param_grid: dict[str, list],
                 data: dict[Asset, pd.DataFrame], engine_kwargs: dict | None = None) -> None:
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
