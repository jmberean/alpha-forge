"""
Probability of Backtest Overfitting (PBO) implementation.

From Bailey et al. (2017):
"The Probability of Backtest Overfitting"

PBO measures the likelihood that a strategy selected via optimization
will underperform out-of-sample.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PBOResult:
    """Result of PBO calculation."""

    pbo: float  # Main metric: probability of backtest overfitting
    logit_pbo: float  # Logit of PBO for statistical properties
    n_combinations: int
    n_overfit: int  # Combinations where IS rank != OOS rank
    rank_correlation: float  # Spearman correlation of IS/OOS ranks
    passed: bool

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"PBOResult({status})\n"
            f"  PBO: {self.pbo:.4f} ({self.pbo*100:.1f}%)\n"
            f"  Rank Correlation: {self.rank_correlation:.3f}\n"
            f"  Overfit combinations: {self.n_overfit}/{self.n_combinations}"
        )


class ProbabilityOfOverfitting:
    """
    Calculate Probability of Backtest Overfitting.

    PBO estimates the probability that a strategy selected based on
    in-sample (backtest) performance will have negative out-of-sample
    performance.

    The key insight is that with multiple strategy variations, even if
    ALL strategies have zero expected return, optimization will select
    one that looks good in-sample but fails out-of-sample.

    Method:
    1. Generate multiple train/test splits
    2. For each split, rank strategies by IS (train) performance
    3. Check if top IS performer has positive OOS (test) return
    4. PBO = fraction of splits where top IS performer has OOS rank below median
    """

    def __init__(self, pbo_threshold: float = 0.05) -> None:
        """
        Initialize PBO calculator.

        Args:
            pbo_threshold: PBO threshold for passing (default 0.05 = 5%)
        """
        self.pbo_threshold = pbo_threshold

    def calculate(
        self,
        is_performance: np.ndarray,
        oos_performance: np.ndarray,
    ) -> PBOResult:
        """
        Calculate PBO from in-sample and out-of-sample performance arrays.

        Args:
            is_performance: 2D array [n_combinations x n_strategies] of IS performance
            oos_performance: 2D array [n_combinations x n_strategies] of OOS performance

        Returns:
            PBOResult
        """
        n_combinations, n_strategies = is_performance.shape

        if oos_performance.shape != is_performance.shape:
            raise ValueError("IS and OOS arrays must have same shape")

        # For each combination, find the strategy with best IS performance
        best_is_idx = np.argmax(is_performance, axis=1)

        # Get OOS performance of IS-selected strategy
        selected_oos = np.array(
            [oos_performance[i, best_is_idx[i]] for i in range(n_combinations)]
        )

        # Calculate relative rank of selected strategy in OOS
        oos_ranks = np.zeros(n_combinations)
        for i in range(n_combinations):
            # Rank all strategies by OOS (1 = best)
            ranks = stats.rankdata(-oos_performance[i])
            oos_ranks[i] = ranks[best_is_idx[i]]

        # PBO = probability that IS-selected strategy is below median in OOS
        median_rank = n_strategies / 2
        n_overfit = np.sum(oos_ranks > median_rank)
        pbo = n_overfit / n_combinations

        # Logit of PBO (for better statistical properties)
        if pbo == 0:
            logit_pbo = -np.inf
        elif pbo == 1:
            logit_pbo = np.inf
        else:
            logit_pbo = np.log(pbo / (1 - pbo))

        # Rank correlation between IS and OOS rankings
        is_ranks_all = []
        oos_ranks_all = []
        for i in range(n_combinations):
            is_ranks_all.extend(stats.rankdata(-is_performance[i]))
            oos_ranks_all.extend(stats.rankdata(-oos_performance[i]))

        rank_corr, _ = stats.spearmanr(is_ranks_all, oos_ranks_all)

        return PBOResult(
            pbo=pbo,
            logit_pbo=logit_pbo,
            n_combinations=n_combinations,
            n_overfit=int(n_overfit),
            rank_correlation=rank_corr,
            passed=pbo < self.pbo_threshold,
        )

    def calculate_from_sharpes(
        self,
        sharpe_matrix: np.ndarray,
        n_is_periods: int,
        n_oos_periods: int,
    ) -> PBOResult:
        """
        Calculate PBO from a matrix of Sharpe ratios across time periods.

        Args:
            sharpe_matrix: 2D array [n_periods x n_strategies]
            n_is_periods: Number of periods to use for in-sample
            n_oos_periods: Number of periods for out-of-sample

        Returns:
            PBOResult
        """
        n_periods, n_strategies = sharpe_matrix.shape
        total_periods = n_is_periods + n_oos_periods

        if n_periods < total_periods:
            raise ValueError(
                f"Need at least {total_periods} periods, got {n_periods}"
            )

        # Generate combinations
        n_combinations = n_periods - total_periods + 1

        is_performance = np.zeros((n_combinations, n_strategies))
        oos_performance = np.zeros((n_combinations, n_strategies))

        for i in range(n_combinations):
            is_start = i
            is_end = i + n_is_periods
            oos_end = is_end + n_oos_periods

            # Average Sharpe in IS and OOS windows
            is_performance[i] = sharpe_matrix[is_start:is_end].mean(axis=0)
            oos_performance[i] = sharpe_matrix[is_end:oos_end].mean(axis=0)

        return self.calculate(is_performance, oos_performance)


def calculate_pbo_from_cpcv(
    sharpe_distribution: list[float],
    threshold: float = 0.05,
) -> PBOResult:
    """
    Simplified PBO from CPCV Sharpe distribution.

    This is a simplified version that uses the fraction of negative
    OOS Sharpes as the PBO estimate.

    Args:
        sharpe_distribution: List of OOS Sharpe ratios from CPCV
        threshold: PBO threshold for passing

    Returns:
        PBOResult
    """
    sharpes = np.array(sharpe_distribution)
    n_combinations = len(sharpes)
    n_negative = np.sum(sharpes < 0)

    pbo = n_negative / n_combinations

    if pbo == 0:
        logit_pbo = -np.inf
    elif pbo == 1:
        logit_pbo = np.inf
    else:
        logit_pbo = np.log(pbo / (1 - pbo))

    return PBOResult(
        pbo=pbo,
        logit_pbo=logit_pbo,
        n_combinations=n_combinations,
        n_overfit=n_negative,
        rank_correlation=np.nan,  # Not applicable for single strategy
        passed=pbo < threshold,
    )


def minimum_track_record_length(
    sharpe_ratio: float,
    target_pbo: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """
    Calculate minimum track record length for given PBO target.

    From Bailey et al.: How long should a backtest be to have
    a given probability of overfitting?

    Args:
        sharpe_ratio: Observed Sharpe ratio
        target_pbo: Target PBO (default 5%)
        skewness: Return skewness
        kurtosis: Return kurtosis

    Returns:
        Minimum number of observations needed
    """
    # Z-score for target PBO
    z = stats.norm.ppf(1 - target_pbo)

    # Variance of Sharpe ratio estimator
    # SE[SR] = sqrt((1 + 0.5*SR² - skew*SR + (kurt-3)/4 * SR²) / n)

    # Solve for n given: SR / SE[SR] = z
    # SR² * n / variance_factor = z²

    excess_kurt = kurtosis - 3
    variance_factor = 1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio
    variance_factor += (excess_kurt / 4) * sharpe_ratio**2

    if sharpe_ratio == 0:
        return np.inf

    n = (z**2 * variance_factor) / (sharpe_ratio**2)

    return int(np.ceil(n))
