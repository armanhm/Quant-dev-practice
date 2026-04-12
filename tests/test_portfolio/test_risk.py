import pytest
from quantflow.core.models import Asset, AssetClass, Position
from quantflow.portfolio.risk import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.asset_aapl = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.asset_goog = Asset(symbol="GOOG", asset_class=AssetClass.EQUITY)
        self.asset_btc = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO)

    def test_no_restrictions_by_default(self):
        rm = RiskManager()
        allowed = rm.check_new_position(
            asset=self.asset_aapl, quantity=100.0, price=150.0,
            equity=100_000.0, cash=50_000.0, positions={},
            current_prices={}, peak_equity=100_000.0,
        )
        assert allowed is True

    def test_drawdown_kill_switch(self):
        rm = RiskManager(max_drawdown=0.20)
        allowed = rm.check_new_position(
            asset=self.asset_aapl, quantity=100.0, price=150.0,
            equity=75_000.0, cash=75_000.0, positions={},
            current_prices={}, peak_equity=100_000.0,
        )
        assert allowed is False

    def test_drawdown_within_limit(self):
        rm = RiskManager(max_drawdown=0.20)
        allowed = rm.check_new_position(
            asset=self.asset_aapl, quantity=100.0, price=150.0,
            equity=85_000.0, cash=85_000.0, positions={},
            current_prices={}, peak_equity=100_000.0,
        )
        assert allowed is True

    def test_per_asset_exposure_limit(self):
        rm = RiskManager(max_position_pct=0.20)
        allowed = rm.check_new_position(
            asset=self.asset_aapl, quantity=200.0, price=150.0,
            equity=100_000.0, cash=50_000.0, positions={},
            current_prices={}, peak_equity=100_000.0,
        )
        assert allowed is False

    def test_per_asset_exposure_within_limit(self):
        rm = RiskManager(max_position_pct=0.20)
        allowed = rm.check_new_position(
            asset=self.asset_aapl, quantity=100.0, price=150.0,
            equity=100_000.0, cash=50_000.0, positions={},
            current_prices={}, peak_equity=100_000.0,
        )
        assert allowed is True

    def test_max_open_positions(self):
        rm = RiskManager(max_open_positions=2)
        positions = {
            self.asset_aapl: Position(asset=self.asset_aapl, quantity=10, entry_price=150),
            self.asset_goog: Position(asset=self.asset_goog, quantity=5, entry_price=140),
        }
        allowed = rm.check_new_position(
            asset=self.asset_btc, quantity=1.0, price=50_000.0,
            equity=100_000.0, cash=50_000.0, positions=positions,
            current_prices={self.asset_aapl: 155, self.asset_goog: 145},
            peak_equity=100_000.0,
        )
        assert allowed is False

    def test_adjust_quantity_to_fit_limit(self):
        rm = RiskManager(max_position_pct=0.20)
        adjusted = rm.adjust_quantity(
            asset=self.asset_aapl, quantity=200.0, price=150.0, equity=100_000.0,
        )
        assert adjusted == pytest.approx(20_000.0 / 150.0, rel=0.01)
