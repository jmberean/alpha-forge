"""Strategy parameter optimization framework."""

from alphaforge.optimization.base import Optimizer, OptimizationResult, ParameterSpace
from alphaforge.optimization.grid import GridSearchOptimizer
from alphaforge.optimization.random import RandomSearchOptimizer

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "ParameterSpace",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
]

# Try to import Optuna optimizer (optional dependency)
try:
    from alphaforge.optimization.optuna import OptunaOptimizer

    __all__.append("OptunaOptimizer")
except ImportError:
    pass  # Optuna not installed
