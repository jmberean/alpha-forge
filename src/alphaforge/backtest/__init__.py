"""
Backtesting engine: Vectorized high-performance backtesting.

This module provides:
- BacktestEngine: Vectorized backtesting with realistic assumptions
- PerformanceMetrics: Sharpe, Sortino, max drawdown, etc.
- Portfolio: Portfolio simulation with position tracking
"""

from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.backtest.metrics import PerformanceMetrics
from alphaforge.backtest.portfolio import Portfolio

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "Portfolio",
]
