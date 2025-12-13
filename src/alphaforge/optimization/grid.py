"""
Grid search optimizer.

Exhaustively searches all combinations of parameters in a grid.
"""

import itertools
from typing import Any

from alphaforge.optimization.base import Optimizer, OptimizationResult


class GridSearchOptimizer(Optimizer):
    """
    Grid search parameter optimizer.

    Searches all combinations of parameters in a predefined grid.
    Best for small parameter spaces with discrete values.
    """

    def optimize(self, n_trials: int | None = None, **kwargs) -> OptimizationResult:
        """
        Run grid search optimization.

        Args:
            n_trials: Ignored for grid search (searches all combinations)
            **kwargs: Additional arguments (ignored)

        Returns:
            OptimizationResult with best parameters
        """
        # Generate grid of values for each parameter
        param_grids = {
            p.name: p.grid_values() for p in self.parameter_space
        }

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values_list = list(param_grids.values())

        all_combinations = list(itertools.product(*param_values_list))

        # Evaluate all combinations
        best_score = float('-inf')
        best_params = None
        trials = []
        scores = []

        for combination in all_combinations:
            params = dict(zip(param_names, combination))

            try:
                score = self._evaluate(params)
            except Exception as e:
                # Skip invalid parameter combinations
                score = float('-inf')

            scores.append(score)
            trials.append({
                "params": params.copy(),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()

        # Convert back if minimizing
        if not self.maximize:
            best_score = -best_score
            scores = [-s for s in scores]

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            scores=scores,
            n_trials=len(all_combinations),
            optimizer_name="GridSearch",
        )
