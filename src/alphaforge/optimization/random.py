"""
Random search optimizer.

Randomly samples parameters from the search space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alphaforge.optimization.base import Optimizer, OptimizationResult


class RandomSearchOptimizer(Optimizer):
    """
    Random search parameter optimizer.

    Randomly samples parameters from the search space.
    More efficient than grid search for high-dimensional spaces.
    """

    def __init__(self, *args, seed: int | None = None, **kwargs):
        """
        Initialize random search optimizer.

        Args:
            *args: Passed to Optimizer
            seed: Random seed for reproducibility
            **kwargs: Passed to Optimizer
        """
        super().__init__(*args, **kwargs)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def optimize(self, n_trials: int, **kwargs) -> OptimizationResult:
        """
        Run random search optimization.

        Args:
            n_trials: Number of random parameter combinations to try
            **kwargs: Additional arguments (ignored)

        Returns:
            OptimizationResult with best parameters
        """
        best_score = float('-inf')
        best_params = None
        trials = []
        scores = []

        for i in range(n_trials):
            # Sample random parameters
            params = self._sample_params()

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
            n_trials=n_trials,
            optimizer_name="RandomSearch",
        )
