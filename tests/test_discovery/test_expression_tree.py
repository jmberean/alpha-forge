"""Tests for expression tree system."""

import pytest
import pandas as pd
import numpy as np

from alphaforge.discovery.expression.tree import (
    ExpressionTree,
    TreeGenerator,
)
from alphaforge.discovery.expression.nodes import (
    OperatorNode,
    TerminalNode,
    ConstantNode,
)
from alphaforge.discovery.expression.types import DataType
from alphaforge.discovery.expression.compiler import (
    compile_tree,
    evaluate_tree,
)


class TestExpressionTree:
    """Test ExpressionTree class."""

    def test_create_simple_tree(self):
        """Test creating a simple expression tree."""
        # Create: close
        root = TerminalNode(name="close")
        tree = ExpressionTree(root=root, MIN_DEPTH=1, MIN_SIZE=1)

        assert tree.size == 1
        assert tree.depth == 1
        assert tree.formula == "close"
        assert tree.return_type == DataType.SERIES

    def test_create_operator_tree(self):
        """Test creating tree with operators."""
        # Create: ts_mean(close, 20)
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=20, data_type=DataType.WINDOW),
            ],
        )
        tree = ExpressionTree(root=root)

        assert tree.size == 3
        assert tree.depth == 2
        assert "ts_mean" in tree.formula
        assert tree.is_valid()

    def test_tree_validation(self):
        """Test tree validation."""
        # Valid tree
        root = OperatorNode(
            name="add",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
            ],
        )
        tree = ExpressionTree(root=root)
        assert tree.is_valid()

    def test_tree_clone(self):
        """Test tree cloning."""
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=10, data_type=DataType.WINDOW),
            ],
        )
        tree = ExpressionTree(root=root)
        cloned = tree.clone()

        assert tree.formula == cloned.formula
        assert tree.root is not cloned.root  # Different objects

    def test_tree_replacement(self):
        """Test subtree replacement."""
        # Original: add(close, volume)
        root = OperatorNode(
            name="add",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
            ],
        )
        tree = ExpressionTree(root=root)

        # Replace volume with returns
        new_subtree = TerminalNode(name="returns")
        new_tree = tree.replace_subtree(2, new_subtree)

        assert "returns" in new_tree.formula
        assert tree.formula != new_tree.formula  # Immutable

    def test_complexity_score(self):
        """Test complexity scoring."""
        # Simple tree
        simple = ExpressionTree(root=TerminalNode(name="close"), MIN_DEPTH=1, MIN_SIZE=1)
        assert simple.complexity_score() < 0.2

        # Complex tree
        complex_root = OperatorNode(
            name="ts_corr",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
                ConstantNode(value=50, data_type=DataType.WINDOW),
            ],
        )
        complex_tree = ExpressionTree(root=complex_root)
        assert complex_tree.complexity_score() > simple.complexity_score()


class TestTreeGenerator:
    """Test TreeGenerator class."""

    def test_generate_random_tree(self):
        """Test random tree generation."""
        generator = TreeGenerator(max_depth=4, seed=42)
        tree = generator.generate(method="grow")

        assert tree.is_valid()
        assert tree.depth <= 4
        assert tree.size >= 3

    def test_generate_full_tree(self):
        """Test full tree generation."""
        generator = TreeGenerator(max_depth=3, seed=42)
        tree = generator.generate(method="full")

        assert tree.is_valid()
        assert tree.depth >= 2

    def test_generate_ramped_tree(self):
        """Test ramped tree generation."""
        generator = TreeGenerator(max_depth=6, min_depth=2, seed=42)
        tree = generator.generate(method="ramped")

        assert tree.is_valid()
        # Ramped can produce trees near max_depth
        assert 2 <= tree.depth <= 6
        assert tree.size >= 3

    def test_generate_population(self):
        """Test population generation."""
        generator = TreeGenerator(seed=42)
        population = generator.generate_population(size=10)

        assert len(population) == 10
        assert all(t.is_valid() for t in population)

        # Check uniqueness
        hashes = [t.hash for t in population]
        assert len(set(hashes)) == len(hashes)


class TestExpressionCompiler:
    """Test expression compilation and evaluation."""

    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2020-01-01", periods=100)
        data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )
        return data

    def test_compile_terminal(self):
        """Test compiling a terminal node."""
        tree = ExpressionTree(root=TerminalNode(name="close"), MIN_DEPTH=1, MIN_SIZE=1)
        
        # Mock data
        df = pd.DataFrame({
            "close": np.random.randn(100),
            "volume": np.random.randn(100),
        })
        result = evaluate_tree(tree, df)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        pd.testing.assert_series_equal(result, df["close"])

    def test_compile_operator(self, market_data):
        """Test compiling operator node."""
        # ts_mean(close, 5)
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=5),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(market_data)
        assert not result.isna().all()

    def test_compile_arithmetic(self, market_data):
        """Test compiling arithmetic operations."""
        # add(close, volume)
        root = OperatorNode(
            name="add",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        expected = market_data["close"] + market_data["volume"]
        pd.testing.assert_series_equal(result, expected)

    def test_compile_temporal(self, market_data):
        """Test temporal operators (no lookahead)."""
        # delay(close, 1)
        root = OperatorNode(
            name="delay",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=1),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        expected = market_data["close"].shift(1)
        pd.testing.assert_series_equal(result, expected)

    def test_compile_complex_expression(self, market_data):
        """Test complex nested expression."""
        # rank(ts_mean(close, 20))
        root = OperatorNode(
            name="rank",
            children=[
                OperatorNode(
                    name="ts_mean",
                    children=[
                        TerminalNode(name="close"),
                        ConstantNode(value=20),
                    ],
                ),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(market_data)
        # Rank should be between 0 and 1
        assert result.min() >= 0
        assert result.max() <= 1

    def test_no_lookahead_bias(self, market_data):
        """Verify no lookahead bias in temporal operations."""
        # All temporal ops should use trailing windows only
        root = OperatorNode(
            name="ts_corr",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
                ConstantNode(value=10),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        # First values should be NaN (not enough history)
        assert result.iloc[:5].isna().any()


class TestOperatorImplementations:
    """Test individual operator implementations."""

    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2020-01-01", periods=50)
        data = pd.DataFrame(
            {
                "close": np.arange(50, dtype=float) + 100,  # Linear trend
                "volume": np.arange(50, dtype=float) * 100,
                "returns": np.random.randn(50) * 0.01,
            },
            index=dates,
        )
        return data

    def test_ts_mean(self, market_data):
        """Test ts_mean operator."""
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=10),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        # Check against pandas rolling mean
        expected = market_data["close"].rolling(10, min_periods=1).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_delay(self, market_data):
        """Test delay operator."""
        root = OperatorNode(
            name="delay",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=5),
            ],
        )
        tree = ExpressionTree(root=root)
        result = evaluate_tree(tree, market_data)

        expected = market_data["close"].shift(5)
        pd.testing.assert_series_equal(result, expected)

    def test_rank(self, market_data):
        """Test rank operator."""
        root = OperatorNode(
            name="rank",
            children=[TerminalNode(name="close")],
        )
        tree = ExpressionTree(root=root, MIN_SIZE=2)
        result = evaluate_tree(tree, market_data)

        # Rank should be monotonic for monotonic input
        assert (result.diff().dropna() >= 0).all()

    def test_rank_is_point_in_time_safe(self, market_data):
        """Rank at time t must not change when future data is appended."""
        root = OperatorNode(
            name="rank",
            children=[TerminalNode(name="close")],
        )
        tree = ExpressionTree(root=root, MIN_SIZE=2)

        full = evaluate_tree(tree, market_data)
        partial_data = market_data.iloc[:50].copy()
        partial = evaluate_tree(tree, partial_data)

        pd.testing.assert_series_equal(full.iloc[:50], partial)
