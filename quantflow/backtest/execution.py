from __future__ import annotations

from quantflow.core.models import Order, OrderSide, OrderStatus, Fill
from quantflow.core.events import EventBus, FillEvent


class SimulatedExecution:
    """Simulates order execution with configurable slippage and commission."""

    def __init__(
        self,
        event_bus: EventBus,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0,
    ) -> None:
        self.event_bus = event_bus
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

    def execute(self, order: Order, current_price: float) -> Fill:
        # Apply slippage: buys pay more, sells receive less
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1.0 + self.slippage_pct)
        else:
            fill_price = current_price * (1.0 - self.slippage_pct)

        slippage = abs(fill_price - current_price) * order.quantity

        # Commission based on notional value
        notional = order.quantity * fill_price
        commission = notional * self.commission_pct

        order.status = OrderStatus.FILLED

        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage,
        )

        self.event_bus.emit(FillEvent(fill=fill))
        return fill
