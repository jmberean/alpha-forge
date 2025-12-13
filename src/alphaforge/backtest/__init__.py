"""
Backtesting engine: Vectorized and event-driven backtesting.

This module provides:
- BacktestEngine: Vectorized backtesting with realistic assumptions
- EventDrivenEngine: Event-driven backtest with queue models and latency
- PerformanceMetrics: Sharpe, Sortino, max drawdown, etc.
- Portfolio: Portfolio simulation with position tracking
- calculate_implementation_shortfall: Compare vectorized vs event-driven results
"""

from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.backtest.event_driven import (
    EventDrivenEngine,
    ExecutionConfig,
    OrderSide,
    OrderStatus,
    calculate_implementation_shortfall,
)
from alphaforge.backtest.metrics import PerformanceMetrics
from alphaforge.backtest.portfolio import Portfolio

__all__ = [
    # Vectorized
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "Portfolio",
    # Event-driven (MVP7)
    "EventDrivenEngine",
    "ExecutionConfig",
    "OrderSide",
    "OrderStatus",
    "calculate_implementation_shortfall",
]
