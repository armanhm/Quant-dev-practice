from datetime import datetime, timezone
from quantflow.core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventBus,
)
from quantflow.core.models import (
    Asset,
    AssetClass,
    Bar,
    Signal,
    Direction,
    Order,
    OrderSide,
    OrderType,
    Fill,
)


class TestEvents:
    def test_market_data_event(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        event = MarketDataEvent(asset=asset, bar=bar)
        assert event.asset == asset
        assert event.bar == bar

    def test_signal_event(self):
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            asset=asset, direction=Direction.LONG, strength=0.8,
        )
        event = SignalEvent(signal=signal)
        assert event.signal.direction == Direction.LONG


class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []

        def handler(event: MarketDataEvent):
            received.append(event)

        bus.subscribe(MarketDataEvent, handler)

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        event = MarketDataEvent(asset=asset, bar=bar)
        bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self):
        bus = EventBus()
        received_a = []
        received_b = []

        bus.subscribe(MarketDataEvent, lambda e: received_a.append(e))
        bus.subscribe(MarketDataEvent, lambda e: received_b.append(e))

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        bus.emit(MarketDataEvent(asset=asset, bar=bar))

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_no_crosstalk_between_event_types(self):
        bus = EventBus()
        received = []

        bus.subscribe(SignalEvent, lambda e: received.append(e))

        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        bus.emit(MarketDataEvent(asset=asset, bar=bar))

        assert len(received) == 0

    def test_emit_with_no_subscribers(self):
        bus = EventBus()
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        bar = Bar(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0, volume=1e6,
        )
        # Should not raise
        bus.emit(MarketDataEvent(asset=asset, bar=bar))
