"""Maps strategy string names to their classes for CLI and config lookup."""
from __future__ import annotations

from quantflow.strategies.sma_crossover import SMACrossover
from quantflow.strategies.mean_reversion import MeanReversion
from quantflow.strategies.rsi_macd import RSIMACDCombo
from quantflow.strategies.pairs_trading import PairsTrading
from quantflow.strategies.macro_regime import MacroRegime
from quantflow.strategies.composite import CompositeStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossover,
    "mean_reversion": MeanReversion,
    "rsi_macd": RSIMACDCombo,
    "pairs_trading": PairsTrading,
    "macro_regime": MacroRegime,
    "composite": CompositeStrategy,
}


def get_strategy(name: str) -> type:
    """Look up a strategy class by name. Raises KeyError if not found."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def register_strategy(name: str, cls: type) -> None:
    """Register a new strategy class."""
    STRATEGY_REGISTRY[name] = cls
