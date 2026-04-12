from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AssetClass(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    MACRO = "macro"
    OPTION = "option"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_class: AssetClass
    exchange: str | None = None


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    asset: Asset
    direction: Direction
    strength: float

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between 0 and 1, got {self.strength}")


@dataclass
class Order:
    asset: Asset
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING


@dataclass(frozen=True)
class Fill:
    order: Order
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float


@dataclass
class Position:
    asset: Asset
    quantity: float
    entry_price: float

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity

    def market_value(self, current_price: float) -> float:
        return abs(self.quantity) * current_price
