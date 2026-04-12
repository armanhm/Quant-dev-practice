from __future__ import annotations

from typing import Protocol

from quantflow.core.models import Asset


class PositionSizer(Protocol):
    def calculate_quantity(
        self, asset: Asset, price: float, equity: float, signal_strength: float,
    ) -> float: ...


class FixedFractional:
    """Risk a fixed fraction of equity per trade, scaled by signal strength."""

    def __init__(self, fraction: float = 0.02) -> None:
        self.fraction = fraction

    def calculate_quantity(
        self, asset: Asset, price: float, equity: float, signal_strength: float,
    ) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        risk_amount = equity * self.fraction * signal_strength
        return risk_amount / price


class KellyCriterion:
    """Kelly Criterion position sizing. f* = W - (1-W)/R"""

    def __init__(
        self, win_rate: float = 0.5, avg_win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.5,
    ) -> None:
        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.kelly_fraction = kelly_fraction

    @classmethod
    def from_trades(cls, pnls: list[float], kelly_fraction: float = 0.5) -> KellyCriterion:
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.5
        if wins and losses:
            avg_win_loss_ratio = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        else:
            avg_win_loss_ratio = 1.5
        return cls(win_rate=win_rate, avg_win_loss_ratio=avg_win_loss_ratio, kelly_fraction=kelly_fraction)

    def calculate_quantity(
        self, asset: Asset, price: float, equity: float, signal_strength: float,
    ) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        kelly_f = self.win_rate - (1 - self.win_rate) / self.avg_win_loss_ratio
        kelly_f = max(kelly_f, 0.0)
        kelly_f *= self.kelly_fraction
        kelly_f *= signal_strength
        risk_amount = equity * kelly_f
        return risk_amount / price
