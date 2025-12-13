"""
Strategy factory: Strategy representation and signal generation.

This module provides:
- StrategyGenome: Universal strategy representation with JSON serialization
- StrategyTemplates: Built-in strategy templates (SMA crossover, momentum, etc.)
- SignalGenerator: Generate trading signals from strategies
"""

from alphaforge.strategy.genome import Filter, Rule, StrategyGenome
from alphaforge.strategy.signals import SignalGenerator
from alphaforge.strategy.templates import StrategyTemplates

__all__ = [
    "StrategyGenome",
    "Rule",
    "Filter",
    "StrategyTemplates",
    "SignalGenerator",
]
