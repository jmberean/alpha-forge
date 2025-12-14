"""Compiler for expression trees to vectorized pandas operations.

Converts ExpressionTree to executable code that operates on DataFrames.
"""

from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd
import numpy as np

from alphaforge.discovery.expression.nodes import (
    Node,
    OperatorNode,
    TerminalNode,
    ConstantNode,
)
from alphaforge.discovery.expression.tree import ExpressionTree


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


class ExpressionCompiler:
    """Compiles expression trees to executable pandas code.

    The compiler generates vectorized operations for efficiency.
    All temporal operations use trailing windows only (no lookahead).
    """

    # Operator implementations
    OPERATORS: dict[str, Callable[..., pd.Series]] = {}

    def __init__(self) -> None:
        self._register_operators()

    def _register_operators(self) -> None:
        """Register all operator implementations."""
        # Arithmetic operators
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
        """Rolling rank within window."""
        def rank_in_window(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.5
            return (arr[-1:] > arr[:-1]).sum() / (len(arr) - 1)
        return x.rolling(int(w), min_periods=2).apply(rank_in_window, raw=True)

    @staticmethod
    def _ts_argmax(x: pd.Series, w: int) -> pd.Series:
        """Days since maximum in window."""
        def argmax_in_window(arr: np.ndarray) -> float:
            return len(arr) - 1 - np.argmax(arr)
        return x.rolling(int(w), min_periods=1).apply(argmax_in_window, raw=True)

    @staticmethod
    def _ts_argmin(x: pd.Series, w: int) -> pd.Series:
        """Days since minimum in window."""
        def argmin_in_window(arr: np.ndarray) -> float:
            return len(arr) - 1 - np.argmin(arr)
        return x.rolling(int(w), min_periods=1).apply(argmin_in_window, raw=True)

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

            # Add derived columns if needed
            data = self._prepare_data(data)

            # Evaluate recursively
            try:
                result = self._evaluate_node(tree.root, data)
                return self._finalize(result)
            except Exception as e:
                # Return NaN series on error
                return pd.Series(np.nan, index=data.index)

        return CompiledExpression(
            tree=tree,
            evaluate=evaluate,
            required_columns=required_columns,
        )

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with derived columns."""
        data = data.copy()

        # Add returns if not present
        if "returns" not in data.columns and "close" in data.columns:
            data["returns"] = data["close"].pct_change()

        # Add VWAP if not present
        if "vwap" not in data.columns:
            if all(c in data.columns for c in ["high", "low", "close", "volume"]):
                typical_price = (data["high"] + data["low"] + data["close"]) / 3
                data["vwap"] = typical_price  # Simplified VWAP

        return data

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


def compile_tree(tree: ExpressionTree) -> CompiledExpression:
    """Convenience function to compile a tree."""
    compiler = ExpressionCompiler()
    return compiler.compile(tree)


def evaluate_tree(tree: ExpressionTree, data: pd.DataFrame) -> pd.Series:
    """Convenience function to evaluate a tree on data."""
    compiled = compile_tree(tree)
    return compiled(data)
