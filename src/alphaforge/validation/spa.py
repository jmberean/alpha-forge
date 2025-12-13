"""
Hansen's Superior Predictive Ability (SPA) Test.

Tests whether a strategy has statistically significant superior
predictive ability compared to a benchmark.

Uses the arch library for production-tested bootstrap implementation.

Reference:
- Hansen (2005): "A Test for Superior Predictive Ability"
- White (2000): "A Reality Check for Data Snooping"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from arch.bootstrap import SPA


@dataclass
class SPAResult:
    """
    Result from Hansen's SPA test.

    The test determines if a strategy has superior predictive ability
    compared to a benchmark (typically buy-and-hold).
    """

    pvalue: float  # P-value from SPA test
    test_statistic: float  # Test statistic value
    benchmark_name: str  # Name of benchmark (e.g., "SPY")
    passed: bool  # True if p-value < threshold

    # Additional statistics
    strategy_mean_return: float = 0.0
    benchmark_mean_return: float = 0.0
    outperformance: float = 0.0  # Avg daily outperformance

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Hansen's SPA Test: {'PASSED' if self.passed else 'FAILED'}",
            f"",
            f"P-value: {self.pvalue:.4f}",
            f"Test Statistic: {self.test_statistic:.4f}",
            f"Benchmark: {self.benchmark_name}",
            f"",
            f"Strategy Mean Return: {self.strategy_mean_return:.4%} daily",
            f"Benchmark Mean Return: {self.benchmark_mean_return:.4%} daily",
            f"Outperformance: {self.outperformance:.4%} daily",
            f"",
        ]

        if self.passed:
            lines.append("✓ Strategy has superior predictive ability vs benchmark")
        else:
            lines.append("✗ Strategy does not show superior predictive ability")
            lines.append("  (May be due to luck or data snooping)")

        return "\n".join(lines)


class SPATest:
    """
    Hansen's Superior Predictive Ability test.

    Tests if a trading strategy has genuinely superior performance
    compared to a benchmark, accounting for data snooping.

    Usage:
        >>> spa = SPATest(pvalue_threshold=0.05)
        >>> result = spa.test(strategy_returns, benchmark_returns)
        >>> if result.passed:
        ...     print("Strategy beats benchmark!")
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        block_size: int = 10,
        bootstrap_reps: int = 1000,
        bootstrap_type: str = "stationary",
    ):
        """
        Initialize SPA test.

        Args:
            pvalue_threshold: Significance level (default 0.05)
            block_size: Block size for block bootstrap (default 10)
            bootstrap_reps: Number of bootstrap replications (default 1000)
            bootstrap_type: Bootstrap type ('stationary' or 'circular')
        """
        self.pvalue_threshold = pvalue_threshold
        self.block_size = block_size
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_type = bootstrap_type

    def test(
        self,
        strategy_returns: np.ndarray | pd.Series,
        benchmark_returns: np.ndarray | pd.Series,
        benchmark_name: str = "Benchmark",
    ) -> SPAResult:
        """
        Run SPA test comparing strategy to benchmark.

        Args:
            strategy_returns: Daily returns of strategy
            benchmark_returns: Daily returns of benchmark
            benchmark_name: Name of benchmark for reporting

        Returns:
            SPAResult with p-value and pass/fail

        Example:
            >>> spa = SPATest()
            >>> # Compare strategy to buy-and-hold SPY
            >>> result = spa.test(strategy_returns, spy_returns, "SPY")
            >>> print(result.summary())

        Note:
            The null hypothesis is that the benchmark is as good as
            the strategy. We want to reject this (p < 0.05) to claim
            superior performance.
        """
        # Convert to numpy arrays
        if isinstance(strategy_returns, pd.Series):
            strategy_returns = strategy_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values

        # Align lengths (take minimum)
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Calculate losses (SPA uses losses, not returns)
        # Lower loss = better performance
        strategy_losses = -strategy_returns
        benchmark_losses = -benchmark_returns

        # Reshape for SPA (needs 2D array for models)
        benchmark_losses_2d = benchmark_losses.reshape(-1, 1)
        strategy_losses_2d = strategy_losses.reshape(-1, 1)

        # Run SPA test using arch library
        spa = SPA(
            benchmark_losses_2d,
            strategy_losses_2d,
            block_size=self.block_size,
            bootstrap=self.bootstrap_type,
            reps=self.bootstrap_reps,
        )

        # Compute() stores results in the SPA object, doesn't return them
        spa.compute()

        # Extract results
        # pvalues is a Series with 'lower', 'consistent', 'upper'
        # We use 'consistent' as the main p-value
        pvalue = float(spa.pvalues["consistent"])
        test_stat = float(spa.t) if hasattr(spa, 't') else 0.0

        # Calculate additional statistics
        strategy_mean = float(np.mean(strategy_returns))
        benchmark_mean = float(np.mean(benchmark_returns))
        outperformance = strategy_mean - benchmark_mean

        return SPAResult(
            pvalue=pvalue,
            test_statistic=test_stat,
            benchmark_name=benchmark_name,
            passed=pvalue < self.pvalue_threshold,
            strategy_mean_return=strategy_mean,
            benchmark_mean_return=benchmark_mean,
            outperformance=outperformance,
        )

    def test_multiple(
        self,
        strategy_returns: np.ndarray | pd.Series,
        benchmarks: dict[str, np.ndarray | pd.Series],
    ) -> dict[str, SPAResult]:
        """
        Test strategy against multiple benchmarks.

        Args:
            strategy_returns: Daily returns of strategy
            benchmarks: Dict mapping benchmark name to returns

        Returns:
            Dict mapping benchmark name to SPAResult

        Example:
            >>> benchmarks = {
            ...     "SPY": spy_returns,
            ...     "QQQ": qqq_returns,
            ...     "IWM": iwm_returns,
            ... }
            >>> results = spa.test_multiple(strategy_returns, benchmarks)
            >>> for name, result in results.items():
            ...     print(f"{name}: {result.passed}")
        """
        results = {}

        for benchmark_name, benchmark_returns in benchmarks.items():
            result = self.test(strategy_returns, benchmark_returns, benchmark_name)
            results[benchmark_name] = result

        return results


# Convenience function
def superior_predictive_ability(
    strategy_returns: np.ndarray | pd.Series,
    benchmark_returns: np.ndarray | pd.Series,
    benchmark_name: str = "Benchmark",
    pvalue_threshold: float = 0.05,
) -> SPAResult:
    """
    Convenience function to run SPA test.

    Args:
        strategy_returns: Daily returns of strategy
        benchmark_returns: Daily returns of benchmark
        benchmark_name: Name of benchmark
        pvalue_threshold: Significance level

    Returns:
        SPAResult

    Example:
        >>> result = superior_predictive_ability(
        ...     strategy_returns,
        ...     spy_returns,
        ...     benchmark_name="SPY"
        ... )
        >>> if result.passed:
        ...     print("Strategy beats SPY!")
    """
    spa = SPATest(pvalue_threshold=pvalue_threshold)
    return spa.test(strategy_returns, benchmark_returns, benchmark_name)
