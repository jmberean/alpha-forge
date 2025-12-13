"""
Strategy factory: Automated strategy generation and evolution.

Provides genetic programming, parameter optimization, and strategy orchestration.
"""

from alphaforge.factory.genetic import GeneticStrategyEvolver
from alphaforge.factory.orchestrator import StrategyFactory

__all__ = [
    "GeneticStrategyEvolver",
    "StrategyFactory",
]
