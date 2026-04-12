# tests/test_portfolio/test_sizing.py
import pytest
from quantflow.core.models import Asset, AssetClass
from quantflow.portfolio.sizing import FixedFractional, KellyCriterion


class TestFixedFractional:
    def test_basic_sizing(self):
        sizer = FixedFractional(fraction=0.02)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=150.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty == pytest.approx(2000.0 / 150.0, rel=0.01)

    def test_signal_strength_scales_size(self):
        sizer = FixedFractional(fraction=0.02)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        full = sizer.calculate_quantity(asset=asset, price=150.0, equity=100_000.0, signal_strength=1.0)
        half = sizer.calculate_quantity(asset=asset, price=150.0, equity=100_000.0, signal_strength=0.5)
        assert half == pytest.approx(full * 0.5, rel=0.01)

    def test_zero_equity(self):
        sizer = FixedFractional(fraction=0.02)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(asset=asset, price=150.0, equity=0.0, signal_strength=1.0)
        assert qty == 0.0


class TestKellyCriterion:
    def test_basic_kelly(self):
        sizer = KellyCriterion(win_rate=0.6, avg_win_loss_ratio=2.0, kelly_fraction=1.0)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty == pytest.approx(400.0, rel=0.01)

    def test_half_kelly(self):
        sizer = KellyCriterion(win_rate=0.6, avg_win_loss_ratio=2.0, kelly_fraction=0.5)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty == pytest.approx(200.0, rel=0.01)

    def test_negative_kelly_returns_zero(self):
        sizer = KellyCriterion(win_rate=0.3, avg_win_loss_ratio=1.0)
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty == 0.0

    def test_kelly_with_trade_history(self):
        sizer = KellyCriterion.from_trades(pnls=[100, 200, -50, 150, -80])
        asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        qty = sizer.calculate_quantity(
            asset=asset, price=100.0, equity=100_000.0, signal_strength=1.0,
        )
        assert qty > 0
