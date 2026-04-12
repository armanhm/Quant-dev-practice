import pytest
from quantflow.core.models import (
    Asset, AssetClass, Order, OrderSide, OrderType, OrderStatus,
)
from quantflow.core.events import EventBus, OrderEvent, FillEvent
from quantflow.backtest.execution import SimulatedExecution


class TestSimulatedExecution:
    def setup_method(self):
        self.bus = EventBus()
        self.fills = []
        self.bus.subscribe(FillEvent, lambda e: self.fills.append(e))
        self.asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)

    def test_market_order_fills_at_price_plus_slippage(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.001,  # 0.1%
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        assert len(self.fills) == 1
        fill = self.fills[0].fill
        assert fill.fill_quantity == 100.0
        # Buy slippage should increase price
        assert fill.fill_price == pytest.approx(150.0 * 1.001)
        assert fill.commission == 0.0
        assert fill.order.status == OrderStatus.FILLED

    def test_sell_order_slippage_decreases_price(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.001,
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.SELL,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        fill = self.fills[0].fill
        # Sell slippage should decrease price
        assert fill.fill_price == pytest.approx(150.0 * 0.999)

    def test_commission_calculated(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.0,
            commission_pct=0.001,  # 0.1%
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=150.0)

        fill = self.fills[0].fill
        # Commission = 0.1% of (100 * 150) = 15.0
        assert fill.commission == pytest.approx(15.0)

    def test_zero_slippage_and_commission(self):
        executor = SimulatedExecution(
            event_bus=self.bus,
            slippage_pct=0.0,
            commission_pct=0.0,
        )

        order = Order(
            asset=self.asset,
            side=OrderSide.BUY,
            quantity=50.0,
            order_type=OrderType.MARKET,
        )
        executor.execute(order, current_price=200.0)

        fill = self.fills[0].fill
        assert fill.fill_price == 200.0
        assert fill.commission == 0.0
