"""Tests for parameter optimizers."""

import numpy as np
import pytest

from alphaforge.optimization import (
    GridSearchOptimizer,
    OptimizationResult,
    ParameterSpace,
    RandomSearchOptimizer,
)


def simple_objective(params):
    """Simple quadratic objective function for testing."""
    x = params["x"]
    y = params["y"]
    # Optimum at x=5, y=10
    return -(x - 5) ** 2 - (y - 10) ** 2


def single_param_objective(params):
    """Single parameter objective."""
    x = params["x"]
    return -(x - 3) ** 2


class TestParameterSpace:
    """Test ParameterSpace class."""

    def test_float_parameter(self):
        """Test float parameter space."""
        param = ParameterSpace(name="learning_rate", type="float", low=0.001, high=0.1)

        assert param.name == "learning_rate"
        assert param.type == "float"
        assert param.low == 0.001
        assert param.high == 0.1

    def test_int_parameter(self):
        """Test integer parameter space."""
        param = ParameterSpace(name="n_layers", type="int", low=1, high=10)

        assert param.type == "int"
        assert param.low == 1
        assert param.high == 10

    def test_categorical_parameter(self):
        """Test categorical parameter space."""
        param = ParameterSpace(
            name="optimizer", type="categorical", choices=["adam", "sgd", "rmsprop"]
        )

        assert param.type == "categorical"
        assert param.choices == ["adam", "sgd", "rmsprop"]

    def test_log_scale_parameter(self):
        """Test log scale parameter."""
        param = ParameterSpace(
            name="learning_rate", type="float", low=1e-4, high=1e-1, log_scale=True
        )

        assert param.log_scale is True

    def test_sample_float(self):
        """Test sampling from float parameter."""
        param = ParameterSpace(name="x", type="float", low=0.0, high=1.0)

        samples = [param.sample() for _ in range(100)]

        assert all(0.0 <= s <= 1.0 for s in samples)
        assert all(isinstance(s, float) for s in samples)

    def test_sample_int(self):
        """Test sampling from int parameter."""
        param = ParameterSpace(name="x", type="int", low=1, high=10)

        samples = [param.sample() for _ in range(100)]

        assert all(1 <= s <= 10 for s in samples)
        assert all(isinstance(s, (int, np.integer)) for s in samples)

    def test_sample_categorical(self):
        """Test sampling from categorical parameter."""
        param = ParameterSpace(name="x", type="categorical", choices=["a", "b", "c"])

        samples = [param.sample() for _ in range(100)]

        assert all(s in ["a", "b", "c"] for s in samples)

    def test_grid_values_float(self):
        """Test grid values for float parameter."""
        param = ParameterSpace(name="x", type="float", low=0.0, high=1.0)

        grid = param.grid_values()

        assert len(grid) == 10  # Default
        assert grid[0] == 0.0
        assert grid[-1] == 1.0

    def test_grid_values_int(self):
        """Test grid values for int parameter."""
        param = ParameterSpace(name="x", type="int", low=1, high=5)

        grid = param.grid_values()

        assert all(isinstance(v, int) for v in grid)
        assert 1 in grid
        assert 5 in grid

    def test_grid_values_categorical(self):
        """Test grid values for categorical parameter."""
        param = ParameterSpace(name="x", type="categorical", choices=["a", "b", "c"])

        grid = param.grid_values()

        assert grid == ["a", "b", "c"]

    def test_grid_values_with_step(self):
        """Test grid values with explicit step."""
        param = ParameterSpace(name="x", type="float", low=0.0, high=10.0, step=2.0)

        grid = param.grid_values()

        assert len(grid) == 6  # 0, 2, 4, 6, 8, 10

    def test_validation_missing_bounds(self):
        """Test validation fails when bounds missing for numerical params."""
        with pytest.raises(ValueError, match="low and high must be specified"):
            ParameterSpace(name="x", type="float", low=0.0)

    def test_validation_invalid_bounds(self):
        """Test validation fails when low >= high."""
        with pytest.raises(ValueError, match="low must be < high"):
            ParameterSpace(name="x", type="float", low=5.0, high=1.0)

    def test_validation_missing_choices(self):
        """Test validation fails when choices missing for categorical."""
        with pytest.raises(ValueError, match="choices must be specified"):
            ParameterSpace(name="x", type="categorical")

    def test_validation_invalid_type(self):
        """Test validation fails for invalid parameter type."""
        with pytest.raises(ValueError, match="invalid type"):
            ParameterSpace(name="x", type="invalid", low=0.0, high=1.0)


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer."""

    def test_initialization(self):
        """Test grid search initialization."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0, step=2.0),
            ParameterSpace(name="y", type="float", low=0.0, high=10.0, step=2.0),
        ]

        optimizer = GridSearchOptimizer(
            parameter_space=param_space, objective=simple_objective
        )

        assert optimizer.maximize is True
        assert len(optimizer.parameter_space) == 2

    def test_optimize_simple(self):
        """Test grid search on simple objective."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0, step=1.0),
            ParameterSpace(name="y", type="float", low=0.0, high=15.0, step=1.0),
        ]

        optimizer = GridSearchOptimizer(
            parameter_space=param_space, objective=simple_objective
        )

        result = optimizer.optimize()

        # Should find optimum at (5, 10)
        assert result.best_params["x"] == 5.0
        assert result.best_params["y"] == 10.0
        assert result.best_score == 0.0  # Maximum at (5, 10)
        assert result.n_trials > 0

    def test_optimize_minimization(self):
        """Test grid search with minimization."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0, step=1.0),
        ]

        # Use negative objective to test minimization
        def minimize_obj(params):
            return (params["x"] - 3) ** 2

        optimizer = GridSearchOptimizer(
            parameter_space=param_space, objective=minimize_obj, maximize=False
        )

        result = optimizer.optimize()

        # Should find minimum at x=3
        assert result.best_params["x"] == 3.0
        assert result.n_trials == 11  # 0 to 10

    def test_categorical_parameters(self):
        """Test grid search with categorical parameters."""
        param_space = [
            ParameterSpace(name="method", type="categorical", choices=["a", "b", "c"]),
        ]

        def cat_objective(params):
            scores = {"a": 1.0, "b": 2.0, "c": 3.0}
            return scores[params["method"]]

        optimizer = GridSearchOptimizer(
            parameter_space=param_space, objective=cat_objective
        )

        result = optimizer.optimize()

        assert result.best_params["method"] == "c"
        assert result.best_score == 3.0

    def test_result_fields(self):
        """Test that result contains all expected fields."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=5.0, step=1.0),
        ]

        optimizer = GridSearchOptimizer(
            parameter_space=param_space, objective=single_param_objective
        )

        result = optimizer.optimize()

        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "trials")
        assert hasattr(result, "scores")
        assert hasattr(result, "n_trials")
        assert result.optimizer_name == "GridSearch"


class TestRandomSearchOptimizer:
    """Test RandomSearchOptimizer."""

    def test_initialization(self):
        """Test random search initialization."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0),
        ]

        optimizer = RandomSearchOptimizer(
            parameter_space=param_space, objective=simple_objective, seed=42
        )

        assert optimizer.seed == 42

    def test_optimize_simple(self):
        """Test random search on simple objective."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0),
            ParameterSpace(name="y", type="float", low=0.0, high=15.0),
        ]

        optimizer = RandomSearchOptimizer(
            parameter_space=param_space, objective=simple_objective, seed=42
        )

        result = optimizer.optimize(n_trials=100)

        # Should find value close to optimum (5, 10)
        assert 3.0 <= result.best_params["x"] <= 7.0
        assert 8.0 <= result.best_params["y"] <= 12.0
        assert result.n_trials == 100

    def test_optimize_with_seed(self):
        """Test that seed produces reproducible results."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0),
        ]

        # Reset numpy seed before each optimization
        np.random.seed(42)
        optimizer1 = RandomSearchOptimizer(
            parameter_space=param_space, objective=single_param_objective, seed=42
        )
        result1 = optimizer1.optimize(n_trials=10)

        np.random.seed(42)
        optimizer2 = RandomSearchOptimizer(
            parameter_space=param_space, objective=single_param_objective, seed=42
        )
        result2 = optimizer2.optimize(n_trials=10)

        # Results should be identical with same seed
        assert abs(result1.best_score - result2.best_score) < 1e-10
        assert abs(result1.best_params["x"] - result2.best_params["x"]) < 1e-10

    def test_result_optimizer_name(self):
        """Test that result has correct optimizer name."""
        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=5.0),
        ]

        optimizer = RandomSearchOptimizer(
            parameter_space=param_space, objective=single_param_objective
        )

        result = optimizer.optimize(n_trials=10)

        assert result.optimizer_name == "RandomSearch"


class TestOptimizationResult:
    """Test OptimizationResult class."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = OptimizationResult(
            best_params={"x": 5.0, "y": 10.0},
            best_score=100.0,
            trials=[{"params": {"x": 1.0}, "score": 50.0}],
            scores=[50.0, 100.0],
            n_trials=2,
            optimizer_name="TestOptimizer",
        )

        result_dict = result.to_dict()

        assert result_dict["best_params"] == {"x": 5.0, "y": 10.0}
        assert result_dict["best_score"] == 100.0
        assert result_dict["n_trials"] == 2
        assert result_dict["optimizer_name"] == "TestOptimizer"

    def test_summary(self):
        """Test summary string generation."""
        result = OptimizationResult(
            best_params={"x": 5.0},
            best_score=100.0,
            n_trials=50,
            optimizer_name="TestOptimizer",
        )

        summary = result.summary()

        assert "TestOptimizer" in summary
        assert "100.0" in summary
        assert "x=5.0" in summary
        assert "50" in summary


class TestOptunaOptimizer:
    """Test OptunaOptimizer (if available)."""

    def test_import(self):
        """Test that OptunaOptimizer can be imported."""
        try:
            from alphaforge.optimization import OptunaOptimizer

            assert OptunaOptimizer is not None
        except ImportError:
            pytest.skip("Optuna not installed")

    def test_optimize_simple(self):
        """Test Optuna optimization."""
        try:
            from alphaforge.optimization import OptunaOptimizer
        except ImportError:
            pytest.skip("Optuna not installed")

        param_space = [
            ParameterSpace(name="x", type="float", low=0.0, high=10.0),
            ParameterSpace(name="y", type="float", low=0.0, high=15.0),
        ]

        optimizer = OptunaOptimizer(
            parameter_space=param_space, objective=simple_objective, seed=42
        )

        result = optimizer.optimize(n_trials=50)

        # Should find value close to optimum (5, 10)
        assert 3.0 <= result.best_params["x"] <= 7.0
        assert 8.0 <= result.best_params["y"] <= 12.0
        assert result.n_trials == 50

    def test_categorical_params(self):
        """Test Optuna with categorical parameters."""
        try:
            from alphaforge.optimization import OptunaOptimizer
        except ImportError:
            pytest.skip("Optuna not installed")

        param_space = [
            ParameterSpace(name="method", type="categorical", choices=["a", "b", "c"]),
            ParameterSpace(name="value", type="int", low=1, high=10),
        ]

        def mixed_objective(params):
            method_scores = {"a": 1.0, "b": 2.0, "c": 3.0}
            return method_scores[params["method"]] * params["value"]

        optimizer = OptunaOptimizer(
            parameter_space=param_space, objective=mixed_objective, seed=42
        )

        result = optimizer.optimize(n_trials=30)

        # Best should be method="c" and value=10
        assert result.best_params["method"] == "c"
        assert result.best_params["value"] == 10
        assert result.best_score == 30.0

    def test_log_scale_param(self):
        """Test Optuna with log scale parameter."""
        try:
            from alphaforge.optimization import OptunaOptimizer
        except ImportError:
            pytest.skip("Optuna not installed")

        param_space = [
            ParameterSpace(
                name="lr", type="float", low=1e-4, high=1e-1, log_scale=True
            ),
        ]

        def lr_objective(params):
            # Optimum at lr=0.01
            return -abs(params["lr"] - 0.01)

        optimizer = OptunaOptimizer(
            parameter_space=param_space, objective=lr_objective, seed=42
        )

        result = optimizer.optimize(n_trials=50)

        # Should find value close to 0.01
        assert 0.005 <= result.best_params["lr"] <= 0.02
