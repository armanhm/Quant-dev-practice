import pytest
from datetime import datetime, timezone
from quantflow.core.models import (
    Asset,
    AssetClass,
    Bar,
    Signal,
    Direction,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Fill,
    Position,
)


class TestAsset:
    def test_create_equity(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert asset.symbol == "AAPL"
        assert asset.asset_class == AssetClass.EQUITY
        assert asset.exchange is None

    def test_create_crypto(self):
        asset = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO, exchange="binance")
        assert asset.exchange == "binance"

    def test_asset_equality(self):
        a1 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        a2 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert a1 == a2

    def test_asset_hash(self):
        a1 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        a2 = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert hash(a1) == hash(a2)
        assert len({a1, a2}) == 1


class TestBar:
    def test_create_bar(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bar = Bar(
            timestamp=ts,
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000.0,
        )
        assert bar.close == 153.0
        assert bar.timestamp == ts


class TestSignal:
    def test_create_long_signal(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        signal = Signal(
            timestamp=ts,
            asset=asset,
            direction=Direction.LONG,
            strength=0.8,
        )
        assert signal.direction == Direction.LONG
        assert signal.strength == 0.8

    def test_signal_strength_clamped(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            Signal(timestamp=ts, asset=asset, direction=Direction.LONG, strength=1.5)


class TestOrder:
    def test_create_market_order(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        order = Order(
            asset=asset,
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 100.0


class TestFill:
    def test_create_fill(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        order = Order(asset=asset, side=OrderSide.BUY, quantity=100.0, order_type=OrderType.MARKET)
        fill = Fill(
            order=order,
            fill_price=150.0,
            fill_quantity=100.0,
            commission=0.0,
            slippage=0.075,
        )
        assert fill.fill_price == 150.0


class TestPosition:
    def test_create_position(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        assert pos.quantity == 100.0

    def test_unrealized_pnl(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        pnl = pos.unrealized_pnl(current_price=160.0)
        assert pnl == 1000.0  # (160 - 150) * 100

    def test_unrealized_pnl_short(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=-50.0, entry_price=150.0)
        pnl = pos.unrealized_pnl(current_price=140.0)
        assert pnl == 500.0  # (150 - 140) * 50, short profits when price drops

    def test_market_value(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        pos = Position(asset=asset, quantity=100.0, entry_price=150.0)
        assert pos.market_value(current_price=160.0) == 16000.0
