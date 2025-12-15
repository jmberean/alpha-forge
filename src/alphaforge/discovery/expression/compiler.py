"""Compiler for expression trees to vectorized pandas operations.

Converts ExpressionTree to executable code that operates on DataFrames.

Performance optimizations:
- Numba JIT compilation for hot path temporal operators (10x speedup)
- Compiler singleton to avoid re-initialization overhead
- Data preparation caching to avoid redundant computations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd
import numpy as np
import numba

from alphaforge.discovery.expression.nodes import (
    Node,
    OperatorNode,
    TerminalNode,
    ConstantNode,
)
from alphaforge.discovery.expression.tree import ExpressionTree


# =============================================================================
# Numba JIT-compiled functions for hot path temporal operators
# These are called 60,000+ times per discovery run - JIT gives ~10x speedup
# =============================================================================

@numba.jit(nopython=True, cache=True)
def _numba_rank_in_window(arr: np.ndarray) -> float:
    """Numba-optimized rank calculation within window."""
    n = len(arr)
    if n < 2:
        return 0.5
    last_val = arr[-1]
    count = 0
    for i in range(n - 1):
        if last_val > arr[i]:
            count += 1
    return count / (n - 1)


@numba.jit(nopython=True, cache=True)
def _numba_argmax_in_window(arr: np.ndarray) -> float:
    """Numba-optimized days since maximum."""
    n = len(arr)
    max_idx = 0
    max_val = arr[0]
    for i in range(1, n):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return float(n - 1 - max_idx)


@numba.jit(nopython=True, cache=True)
def _numba_argmin_in_window(arr: np.ndarray) -> float:
    """Numba-optimized days since minimum."""
    n = len(arr)
    min_idx = 0
    min_val = arr[0]
    for i in range(1, n):
        if arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return float(n - 1 - min_idx)


@numba.jit(nopython=True, cache=True)
def _numba_mean_abs_deviation(arr: np.ndarray) -> float:
    """Numba-optimized mean absolute deviation."""
    n = len(arr)
    if n == 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n
    mad = 0.0
    for i in range(n):
        mad += abs(arr[i] - mean)
    return mad / n


# =============================================================================
# Data preparation cache - avoids redundant returns/vwap computation
# =============================================================================

_PREPARED_DATA_CACHE: dict[int, pd.DataFrame] = {}
_CACHE_MAX_SIZE = 10  # Keep last 10 prepared DataFrames


def _get_prepared_data(data: pd.DataFrame) -> pd.DataFrame:
    """Get prepared data from cache or compute and cache it."""
    cache_key = id(data)

    if cache_key in _PREPARED_DATA_CACHE:
        return _PREPARED_DATA_CACHE[cache_key]

    # Prepare the data
    prepared = data.copy()

    # Add returns if not present
    if "returns" not in prepared.columns and "close" in prepared.columns:
        prepared["returns"] = prepared["close"].pct_change()

    # Add VWAP if not present
    if "vwap" not in prepared.columns:
        if all(c in prepared.columns for c in ["high", "low", "close", "volume"]):
            typical_price = (prepared["high"] + prepared["low"] + prepared["close"]) / 3
            prepared["vwap"] = typical_price

    # Cache management - evict oldest if full
    if len(_PREPARED_DATA_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_PREPARED_DATA_CACHE))
        del _PREPARED_DATA_CACHE[oldest_key]

    _PREPARED_DATA_CACHE[cache_key] = prepared
    return prepared


def clear_data_cache() -> None:
    """Clear the prepared data cache."""
    _PREPARED_DATA_CACHE.clear()


# =============================================================================
# CompiledExpression dataclass
# =============================================================================

@dataclass
class CompiledExpression:
    """A compiled expression ready for evaluation.

    Attributes:
        tree: Original expression tree
        evaluate: Function that evaluates the expression on data
        required_columns: Columns needed from input DataFrame
    """

    tree: ExpressionTree
    evaluate: Callable[[pd.DataFrame], pd.Series]
    required_columns: frozenset[str]

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the expression on data."""
        return self.evaluate(data)


# =============================================================================
# ExpressionCompiler class
# =============================================================================

class ExpressionCompiler:
    """Compiles expression trees to executable pandas code.

    The compiler generates vectorized operations for efficiency.
    All temporal operations use trailing windows only (no lookahead).

    Performance features:
    - Numba JIT for temporal operators (ts_rank, ts_argmax, ts_argmin)
    - Cached data preparation (returns, vwap computed once)
    """

    # Operator implementations
    OPERATORS: dict[str, Callable[..., pd.Series]] = {}

    def __init__(self) -> None:
        self._register_operators()

    def _register_operators(self) -> None:
        """Register all operator implementations."""
        self.OPERATORS = {
            # Binary arithmetic
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y.replace(0, np.nan),

            # Unary arithmetic
            "abs": lambda x: x.abs(),
            "neg": lambda x: -x,
            "log": lambda x: np.log(x.clip(lower=1e-10)),
            "sqrt": lambda x: np.sqrt(x.clip(lower=0)),
            "sign": lambda x: np.sign(x),
            "power": lambda x, p: np.power(x, p),

            # Temporal operators (trailing windows only - PIT safe)
            # Using Numba JIT for hot path functions
            "delay": lambda x, d: x.shift(int(d)),
            "delta": lambda x, d: x - x.shift(int(d)),
            "ts_mean": lambda x, w: x.rolling(int(w), min_periods=1).mean(),
            "ts_std": lambda x, w: x.rolling(int(w), min_periods=2).std(),
            "ts_min": lambda x, w: x.rolling(int(w), min_periods=1).min(),
            "ts_max": lambda x, w: x.rolling(int(w), min_periods=1).max(),
            "ts_sum": lambda x, w: x.rolling(int(w), min_periods=1).sum(),
            "ts_rank": self._ts_rank,
            "ts_argmax": self._ts_argmax,
            "ts_argmin": self._ts_argmin,
            "ts_corr": lambda x, y, w: x.rolling(int(w), min_periods=int(w)//2).corr(y),
            "ts_cov": lambda x, y, w: x.rolling(int(w), min_periods=int(w)//2).cov(y),

            # Cross-sectional (operates on single series)
            "rank": lambda x: x.rank(pct=True),
            "scale": lambda x: x / x.abs().sum() if x.abs().sum() != 0 else x * 0,
            "zscore": lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0,

            # Comparison operators
            "gt": lambda x, y: (x > y).astype(float),
            "lt": lambda x, y: (x < y).astype(float),
            "gte": lambda x, y: (x >= y).astype(float),
            "lte": lambda x, y: (x <= y).astype(float),
            "eq": lambda x, y: (x == y).astype(float),

            # Logical operators
            "and_": lambda x, y: ((x > 0) & (y > 0)).astype(float),
            "or_": lambda x, y: ((x > 0) | (y > 0)).astype(float),
            "not_": lambda x: (x <= 0).astype(float),

            # Conditional
            "if_else": lambda cond, true_val, false_val: pd.Series(
                np.where(cond > 0, true_val, false_val),
                index=true_val.index,
            ),
        }

    @staticmethod
    def _ts_rank(x: pd.Series, w: int) -> pd.Series:
        """Rolling rank within window - Numba JIT optimized."""
        return x.rolling(int(w), min_periods=2).apply(
            _numba_rank_in_window, raw=True, engine="numba"
        )

    @staticmethod
    def _ts_argmax(x: pd.Series, w: int) -> pd.Series:
        """Days since maximum in window - Numba JIT optimized."""
        return x.rolling(int(w), min_periods=1).apply(
            _numba_argmax_in_window, raw=True, engine="numba"
        )

    @staticmethod
    def _ts_argmin(x: pd.Series, w: int) -> pd.Series:
        """Days since minimum in window - Numba JIT optimized."""
        return x.rolling(int(w), min_periods=1).apply(
            _numba_argmin_in_window, raw=True, engine="numba"
        )

    def compile(self, tree: ExpressionTree) -> CompiledExpression:
        """Compile an expression tree to executable code.

        Args:
            tree: The expression tree to compile

        Returns:
            CompiledExpression ready for evaluation
        """
        # Collect required columns
        required_columns = frozenset(tree.get_terminals())

        # Build evaluation function
        def evaluate(data: pd.DataFrame) -> pd.Series:
            # Validate required columns
            missing = required_columns - set(data.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            # Use cached data preparation
            prepared_data = _get_prepared_data(data)

            # Evaluate recursively
            try:
                result = self._evaluate_node(tree.root, prepared_data)
                return self._finalize(result)
            except Exception:
                # Return NaN series on error
                return pd.Series(np.nan, index=data.index)

        return CompiledExpression(
            tree=tree,
            evaluate=evaluate,
            required_columns=required_columns,
        )

    def _evaluate_node(self, node: Node, data: pd.DataFrame) -> pd.Series | float:
        """Recursively evaluate a node."""
        if isinstance(node, TerminalNode):
            return data[node.name].copy()

        elif isinstance(node, ConstantNode):
            return float(node.value)

        elif isinstance(node, OperatorNode):
            # Evaluate children first
            args = [self._evaluate_node(child, data) for child in node.children]

            # Get operator function
            op_func = self.OPERATORS.get(node.name)
            if op_func is None:
                raise ValueError(f"Unknown operator: {node.name}")

            # Execute operator
            return op_func(*args)

        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    def _finalize(self, result: pd.Series | float) -> pd.Series:
        """Finalize result series."""
        if isinstance(result, (int, float)):
            # Can't return scalar, need context
            raise ValueError("Expression evaluated to scalar, expected series")

        # Replace infinities with NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        return result


# =============================================================================
# Compiler singleton - avoids re-initialization overhead
# =============================================================================

_COMPILER: ExpressionCompiler | None = None


def get_compiler() -> ExpressionCompiler:
    """Get the singleton compiler instance."""
    global _COMPILER
    if _COMPILER is None:
        _COMPILER = ExpressionCompiler()
    return _COMPILER


def compile_tree(tree: ExpressionTree) -> CompiledExpression:
    """Convenience function to compile a tree using singleton compiler."""
    return get_compiler().compile(tree)


def evaluate_tree(tree: ExpressionTree, data: pd.DataFrame) -> pd.Series:
    """Convenience function to evaluate a tree on data."""
    compiled = compile_tree(tree)
    return compiled(data)
