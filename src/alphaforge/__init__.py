"""
AlphaForge: Production-grade platform for systematic trading strategy discovery,
validation, and deployment.

This package implements defense-in-depth against:
- Overfitting (false discovery) via CPCV + PBO + DSR validation
- Lookahead bias via bi-temporal data architecture
- Execution reality mismatch via realistic impact modeling
"""

__version__ = "0.1.0"

from alphaforge.data.loader import MarketDataLoader
from alphaforge.strategy.genome import StrategyGenome
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.validation.pipeline import ValidationPipeline

__all__ = [
    "__version__",
    "MarketDataLoader",
    "StrategyGenome",
    "BacktestEngine",
    "ValidationPipeline",
]
