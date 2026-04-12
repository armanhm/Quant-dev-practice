from __future__ import annotations

from quantflow.core.models import Asset, Position


class RiskManager:
    """Portfolio risk controls."""

    def __init__(
        self,
        max_drawdown: float | None = None,
        max_position_pct: float | None = None,
        max_open_positions: int | None = None,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self._killed = False

    def check_new_position(
        self,
        asset: Asset, quantity: float, price: float,
        equity: float, cash: float,
        positions: dict[Asset, Position],
        current_prices: dict[Asset, float],
        peak_equity: float,
    ) -> bool:
        if self._killed:
            return False

        if self.max_drawdown is not None and peak_equity > 0:
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < -self.max_drawdown:
                self._killed = True
                return False

        if self.max_position_pct is not None and equity > 0:
            position_value = abs(quantity) * price
            if position_value / equity > self.max_position_pct:
                return False

        if self.max_open_positions is not None:
            if asset not in positions and len(positions) >= self.max_open_positions:
                return False

        return True

    def adjust_quantity(
        self, asset: Asset, quantity: float, price: float, equity: float,
    ) -> float:
        if self.max_position_pct is not None and equity > 0 and price > 0:
            max_value = equity * self.max_position_pct
            max_qty = max_value / price
            quantity = min(quantity, max_qty)
        return quantity

    def reset(self) -> None:
        self._killed = False
