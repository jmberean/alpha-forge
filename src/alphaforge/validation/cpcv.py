"""
Combinatorially Purged Cross-Validation (CPCV) implementation.

From Lopez de Prado "Advances in Financial Machine Learning" (2018).

CPCV addresses the limitations of standard k-fold CV for time series:
- Temporal leakage between train/test sets
- Single train/test split bias

By testing ALL C(N, K) combinations with purging, we get a robust
estimate of out-of-sample performance.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Callable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


@dataclass
class CPCVResult:
    """Result of CPCV validation."""

    sharpe_distribution: list[float]
    mean_sharpe: float
    std_sharpe: float
    pbo: float  # Probability of Backtest Overfitting
    n_combinations: int
    n_negative: int
    n_splits: int
    test_splits: int
    embargo_pct: float
    passed: bool

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"CPCVResult({status})\n"
            f"  Mean Sharpe: {self.mean_sharpe:.3f} ± {self.std_sharpe:.3f}\n"
            f"  PBO: {self.pbo:.4f}\n"
            f"  Combinations: {self.n_combinations}\n"
            f"  Negative OOS: {self.n_negative} ({100*self.n_negative/self.n_combinations:.1f}%)"
        )


class CombinatorialPurgedCV:
    """
    Combinatorially Purged Cross-Validation.

    Tests ALL possible combinations of train/test splits with:
    - Temporal purging (remove data near boundaries)
    - Embargo periods (gap between train and test)

    Standard configuration:
    - n_splits = 16 (divide data into 16 blocks)
    - test_splits = 8 (use 8 blocks for testing)
    - This gives C(16,8) = 12,870 unique train/test combinations
    """

    def __init__(
        self,
        n_splits: int = 16,
        test_splits: int = 8,
        embargo_pct: float = 0.02,
        pbo_threshold: float = 0.05,
    ) -> None:
        """
        Initialize CPCV.

        Args:
            n_splits: Number of time blocks to divide data into
            test_splits: Number of blocks to use for testing
            embargo_pct: Percentage of data to purge at boundaries
            pbo_threshold: PBO threshold for passing (default 0.05 = 5%)
        """
        self.n_splits = n_splits
        self.test_splits = test_splits
        self.embargo_pct = embargo_pct
        self.pbo_threshold = pbo_threshold

        # Calculate number of combinations
        self.n_combinations = self._n_choose_k(n_splits, test_splits)

    def _n_choose_k(self, n: int, k: int) -> int:
        """Calculate binomial coefficient C(n, k)."""
        from math import factorial

        return factorial(n) // (factorial(k) * factorial(n - k))

    def validate(
        self,
        data: pd.DataFrame,
        backtest_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        max_combinations: Optional[int] = None,
        n_jobs: int = 1,
    ) -> CPCVResult:
        """
        Run CPCV validation.

        Args:
            data: DataFrame with OHLCV data (DatetimeIndex)
            backtest_fn: Function that takes (train_df, test_df) and returns Sharpe
            max_combinations: Limit number of combinations (for speed)
            n_jobs: Number of parallel jobs

        Returns:
            CPCVResult with full statistics
        """
        # Create time-based splits
        splits = np.array_split(data, self.n_splits)
        split_indices = list(range(self.n_splits))

        # Generate all test combinations
        all_combinations = list(combinations(split_indices, self.test_splits))

        # Optionally limit combinations
        if max_combinations and len(all_combinations) > max_combinations:
            np.random.seed(42)  # Reproducible sampling
            indices = np.random.choice(
                len(all_combinations), max_combinations, replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        logger.info(f"Running CPCV with {len(all_combinations)} combinations")

        # Run backtests
        if n_jobs == 1:
            results = self._run_sequential(splits, all_combinations, backtest_fn)
        else:
            results = self._run_parallel(splits, all_combinations, backtest_fn, n_jobs)

        # Calculate statistics
        sharpe_distribution = [r for r in results if not np.isnan(r)]

        if len(sharpe_distribution) == 0:
            raise ValueError("All CPCV combinations failed")

        mean_sharpe = np.mean(sharpe_distribution)
        std_sharpe = np.std(sharpe_distribution)
        n_negative = sum(1 for s in sharpe_distribution if s < 0)

        # PBO = probability of backtest overfitting
        # Fraction of test combinations with negative Sharpe
        pbo = n_negative / len(sharpe_distribution)

        return CPCVResult(
            sharpe_distribution=sharpe_distribution,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            pbo=pbo,
            n_combinations=len(sharpe_distribution),
            n_negative=n_negative,
            n_splits=self.n_splits,
            test_splits=self.test_splits,
            embargo_pct=self.embargo_pct,
            passed=pbo < self.pbo_threshold,
        )

    def _run_sequential(
        self,
        splits: list[pd.DataFrame],
        combinations_list: list[tuple],
        backtest_fn: Callable,
    ) -> list[float]:
        """Run combinations sequentially."""
        results = []

        for i, test_indices in enumerate(combinations_list):
            if i % 1000 == 0:
                logger.debug(f"CPCV progress: {i}/{len(combinations_list)}")

            train_indices = [j for j in range(self.n_splits) if j not in test_indices]

            # Purge and concatenate
            train_data = self._purge_and_concat(splits, train_indices, test_indices)
            test_data = self._concat_splits(splits, test_indices)

            try:
                sharpe = backtest_fn(train_data, test_data)
                results.append(sharpe)
            except Exception as e:
                logger.warning(f"CPCV combination {i} failed: {e}")
                results.append(np.nan)

        return results

    def _run_parallel(
        self,
        splits: list[pd.DataFrame],
        combinations_list: list[tuple],
        backtest_fn: Callable,
        n_jobs: int,
    ) -> list[float]:
        """Run combinations in parallel."""
        results = [np.nan] * len(combinations_list)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}

            for i, test_indices in enumerate(combinations_list):
                train_indices = [
                    j for j in range(self.n_splits) if j not in test_indices
                ]

                train_data = self._purge_and_concat(splits, train_indices, test_indices)
                test_data = self._concat_splits(splits, test_indices)

                future = executor.submit(backtest_fn, train_data, test_data)
                futures[future] = i

            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.warning(f"CPCV combination {i} failed: {e}")

        return results

    def _purge_and_concat(
        self,
        splits: list[pd.DataFrame],
        train_indices: list[int],
        test_indices: list[int],
    ) -> pd.DataFrame:
        """
        Concatenate training splits with purging near test boundaries.

        Purging removes data points that might have information leakage
        with the test set.
        """
        train_dfs = []

        for idx in train_indices:
            split = splits[idx].copy()

            # Calculate embargo size
            embargo_size = max(1, int(len(split) * self.embargo_pct))

            # Purge end if next split is in test
            if idx + 1 in test_indices:
                split = split.iloc[:-embargo_size]

            # Purge start if previous split is in test
            if idx - 1 in test_indices:
                split = split.iloc[embargo_size:]

            if len(split) > 0:
                train_dfs.append(split)

        if not train_dfs:
            raise ValueError("No training data after purging")

        return pd.concat(train_dfs)

    def _concat_splits(
        self, splits: list[pd.DataFrame], indices: list[int]
    ) -> pd.DataFrame:
        """Concatenate splits at given indices."""
        return pd.concat([splits[i] for i in indices])

    def get_split_info(self, data: pd.DataFrame) -> dict:
        """
        Get information about how data will be split.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with split information
        """
        splits = np.array_split(data, self.n_splits)

        split_info = []
        for i, split in enumerate(splits):
            split_info.append(
                {
                    "index": i,
                    "start": split.index[0],
                    "end": split.index[-1],
                    "rows": len(split),
                }
            )

        return {
            "n_splits": self.n_splits,
            "test_splits": self.test_splits,
            "n_combinations": self.n_combinations,
            "embargo_pct": self.embargo_pct,
            "splits": split_info,
        }


def combinatorial_purged_cv(
    data: pd.DataFrame,
    backtest_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    n_splits: int = 16,
    test_splits: int = 8,
    embargo_pct: float = 0.02,
    max_combinations: Optional[int] = None,
) -> CPCVResult:
    """
    Convenience function for CPCV validation.

    Args:
        data: OHLCV DataFrame
        backtest_fn: Function(train_df, test_df) -> sharpe
        n_splits: Number of time blocks
        test_splits: Number of test blocks
        embargo_pct: Purge percentage
        max_combinations: Optional limit on combinations

    Returns:
        CPCVResult
    """
    cpcv = CombinatorialPurgedCV(
        n_splits=n_splits,
        test_splits=test_splits,
        embargo_pct=embargo_pct,
    )
    return cpcv.validate(data, backtest_fn, max_combinations)
