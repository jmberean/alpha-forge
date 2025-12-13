"""
Base optimizer interface and parameter space definition.

Provides foundation for strategy parameter optimization using different algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParameterSpace:
    """
    Definition of parameter search space.

    Supports different parameter types: integer, float, categorical.
    """

    name: str
    type: str  # 'int', 'float', 'categorical'

    # For numerical parameters
    low: float | None = None
    high: float | None = None
    step: float | None = None  # For grid search

    # For categorical parameters
    choices: list[Any] | None = None

    # Log scale (for parameters like learning rate)
    log_scale: bool = False

    def __post_init__(self):
        """Validate parameter space definition."""
        if self.type in ['int', 'float']:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter '{self.name}': low and high must be specified for {self.type}")
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be < high")
        elif self.type == 'categorical':
            if not self.choices:
                raise ValueError(f"Parameter '{self.name}': choices must be specified for categorical")
        else:
            raise ValueError(f"Parameter '{self.name}': invalid type '{self.type}'")

    def sample(self) -> Any:
        """
        Sample a random value from this parameter space.

        Returns:
            Random sample from the parameter space
        """
        if self.type == 'categorical':
            return np.random.choice(self.choices)
        elif self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            value = np.exp(np.random.uniform(log_low, log_high))
        else:
            value = np.random.uniform(self.low, self.high)

        if self.type == 'int':
            return int(np.round(value))
        return value

    def grid_values(self) -> list[Any]:
        """
        Generate grid of values for grid search.

        Returns:
            List of parameter values for grid search
        """
        if self.type == 'categorical':
            return list(self.choices)

        if self.step is None:
            # Default: 10 values
            num_values = 10
        else:
            num_values = int((self.high - self.low) / self.step) + 1

        if self.log_scale:
            values = np.logspace(
                np.log10(self.low),
                np.log10(self.high),
                num_values
            )
        else:
            values = np.linspace(self.low, self.high, num_values)

        if self.type == 'int':
            values = [int(np.round(v)) for v in values]
            # Remove duplicates while preserving order
            seen = set()
            values = [v for v in values if not (v in seen or seen.add(v))]
        else:
            values = list(values)

        return values


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""

    # Best parameters found
    best_params: dict[str, Any]

    # Best objective value
    best_score: float

    # All trials
    trials: list[dict[str, Any]] = field(default_factory=list)

    # Optimization history
    scores: list[float] = field(default_factory=list)

    # Metadata
    n_trials: int = 0
    optimizer_name: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "optimizer_name": self.optimizer_name,
            "trials": self.trials,
            "scores": self.scores,
        }

    def summary(self) -> str:
        """Generate summary string."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        return (
            f"Optimization Result ({self.optimizer_name}):\n"
            f"Best Score: {self.best_score:.4f}\n"
            f"Best Parameters: {param_str}\n"
            f"Total Trials: {self.n_trials}"
        )


class Optimizer(ABC):
    """
    Base class for parameter optimizers.

    Optimizers search for best parameter combinations using different strategies.
    """

    def __init__(
        self,
        parameter_space: list[ParameterSpace],
        objective: Callable[[dict], float],
        maximize: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            parameter_space: List of ParameterSpace defining search space
            objective: Function to optimize (takes params dict, returns score)
            maximize: If True, maximize objective; if False, minimize
        """
        self.parameter_space = parameter_space
        self.objective = objective
        self.maximize = maximize

        # Validate parameter space
        param_names = [p.name for p in parameter_space]
        if len(param_names) != len(set(param_names)):
            raise ValueError("Duplicate parameter names in parameter space")

    @abstractmethod
    def optimize(self, n_trials: int, **kwargs) -> OptimizationResult:
        """
        Run optimization to find best parameters.

        Args:
            n_trials: Number of trials to run
            **kwargs: Optimizer-specific parameters

        Returns:
            OptimizationResult with best parameters and scores
        """
        pass

    def _evaluate(self, params: dict[str, Any]) -> float:
        """
        Evaluate objective function with given parameters.

        Args:
            params: Parameter values to evaluate

        Returns:
            Objective function value (always maximization convention)
        """
        score = self.objective(params)

        # Convert to maximization convention
        if not self.maximize:
            score = -score

        return score

    def _sample_params(self) -> dict[str, Any]:
        """
        Sample random parameters from parameter space.

        Returns:
            Dictionary of parameter values
        """
        return {p.name: p.sample() for p in self.parameter_space}
