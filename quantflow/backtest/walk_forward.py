"""Walk-forward optimization: rolling train/test backtesting."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from quantflow.core.models import Asset
from quantflow.backtest.optimizer import ParameterSweep
from quantflow.analytics.metrics import sharpe_ratio, total_return


@dataclass
class WalkForwardResult:
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
            lines.append(f"  Window {i+1}: IS Sharpe={w['is_sharpe']:.2f}, "
                         f"OOS Sharpe={w['oos_sharpe']:.2f}, Params={w['best_params']}")
        return "\n".join(lines)


class WalkForward:
    def __init__(self, strategy_class: type, param_grid: dict[str, list],
                 data: dict[Asset, pd.DataFrame], train_bars: int = 252,
                 test_bars: int = 63, step_bars: int = 63,
                 engine_kwargs: dict | None = None) -> None:
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.engine_kwargs = engine_kwargs or {"initial_cash": 100_000.0, "slippage_pct": 0.0, "commission_pct": 0.0}

    def run(self, metric: str = "sharpe_ratio") -> WalkForwardResult:
        first_df = next(iter(self.data.values()))
        total_bars = len(first_df)
        windows = []
        start = 0

        while start + self.train_bars + self.test_bars <= total_bars:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            train_data = {a: df.iloc[start:train_end] for a, df in self.data.items()}
            test_data = {a: df.iloc[train_end:test_end] for a, df in self.data.items()}

            sweep = ParameterSweep(
                strategy_class=self.strategy_class, param_grid=self.param_grid,
                data=train_data, engine_kwargs=self.engine_kwargs,
            )
            sweep_result = sweep.run(metric=metric)

            if sweep_result.best_params is None:
                start += self.step_bars
                continue

            from quantflow.backtest.engine import BacktestEngine
            from quantflow.core.events import EventBus

            engine = BacktestEngine(**self.engine_kwargs)
            best_params = sweep_result.best_params
            strategy_cls = self.strategy_class

            def factory(bus: EventBus, assets: list[Asset], _params=best_params):
                return strategy_cls(event_bus=bus, assets=assets, **_params)

            oos_result = engine.run(data=test_data, strategy_factory=factory)
            oos_sharpe_val = sharpe_ratio(oos_result.equity_curve) if len(oos_result.equity_curve) > 1 else 0.0
            oos_return_val = total_return(oos_result.equity_curve) if len(oos_result.equity_curve) > 1 else 0.0

            windows.append({
                "train_start": start, "train_end": train_end,
                "test_start": train_end, "test_end": test_end,
                "best_params": best_params, "is_sharpe": sweep_result.best_metric,
                "oos_sharpe": oos_sharpe_val, "oos_return": oos_return_val,
            })
            start += self.step_bars

        return WalkForwardResult(windows=windows)
