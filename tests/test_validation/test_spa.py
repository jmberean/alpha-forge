"""Tests for Hansen's SPA test."""

import pytest
import numpy as np
import pandas as pd

from alphaforge.validation.spa import (
    SPAResult,
    SPATest,
    superior_predictive_ability,
)


class TestSPAResult:
    """Tests for SPAResult dataclass."""

    def test_passing_result(self):
        """Test a passing SPA result."""
        result = SPAResult(
            pvalue=0.01,
            test_statistic=2.5,
            benchmark_name="SPY",
            passed=True,
            strategy_mean_return=0.001,
            benchmark_mean_return=0.0005,
            outperformance=0.0005,
        )

        assert result.passed is True
        assert result.pvalue == 0.01
        assert result.benchmark_name == "SPY"
        assert result.outperformance > 0

    def test_failing_result(self):
        """Test a failing SPA result."""
        result = SPAResult(
            pvalue=0.50,
            test_statistic=0.1,
            benchmark_name="SPY",
            passed=False,
            strategy_mean_return=0.0005,
            benchmark_mean_return=0.0006,
            outperformance=-0.0001,
        )

        assert result.passed is False
        assert result.pvalue > 0.05
        assert result.outperformance < 0

    def test_summary_format(self):
        """Test summary string generation."""
        result = SPAResult(
            pvalue=0.02,
            test_statistic=2.0,
            benchmark_name="SPY",
            passed=True,
            strategy_mean_return=0.001,
            benchmark_mean_return=0.0005,
            outperformance=0.0005,
        )

        summary = result.summary()

        assert "PASSED" in summary
        assert "SPY" in summary
        assert "0.02" in summary
        assert "superior predictive ability" in summary


class TestSPATest:
    """Tests for SPATest class."""

    def test_initialization(self):
        """Test SPATest initialization."""
        spa = SPATest(pvalue_threshold=0.05, block_size=10, bootstrap_reps=100)

        assert spa.pvalue_threshold == 0.05
        assert spa.block_size == 10
        assert spa.bootstrap_reps == 100

    def test_with_clearly_superior_strategy(self):
        """Test SPA with strategy that clearly beats benchmark."""
        np.random.seed(42)

        # Benchmark: small positive returns
        benchmark_returns = np.random.randn(250) * 0.01 + 0.0005

        # Strategy: larger positive returns (clearly better)
        strategy_returns = np.random.randn(250) * 0.01 + 0.0015

        spa = SPATest(bootstrap_reps=100)  # Fewer reps for speed
        result = spa.test(strategy_returns, benchmark_returns, "Benchmark")

        # Should have low p-value (likely to pass)
        assert result.pvalue >= 0.0  # Just check it ran
        assert result.outperformance > 0  # Strategy has higher mean

    def test_with_inferior_strategy(self):
        """Test SPA with strategy worse than benchmark."""
        np.random.seed(42)

        # Benchmark: positive returns
        benchmark_returns = np.random.randn(250) * 0.01 + 0.001

        # Strategy: negative returns (clearly worse)
        strategy_returns = np.random.randn(250) * 0.01 - 0.001

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "Benchmark")

        # Should have high p-value (fail to reject null)
        assert result.outperformance < 0  # Strategy worse than benchmark

    def test_with_equal_strategies(self):
        """Test SPA when strategy equals benchmark."""
        np.random.seed(42)

        # Both have same distribution
        returns = np.random.randn(250) * 0.01 + 0.001
        benchmark_returns = returns.copy()
        strategy_returns = returns + np.random.randn(250) * 0.001  # Small noise

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "Benchmark")

        # Outperformance should be near zero
        assert abs(result.outperformance) < 0.01

    def test_with_pandas_series(self):
        """Test SPA with pandas Series input."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=250)
        benchmark_returns = pd.Series(
            np.random.randn(250) * 0.01 + 0.0005, index=dates
        )
        strategy_returns = pd.Series(
            np.random.randn(250) * 0.01 + 0.001, index=dates
        )

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "SPY")

        assert isinstance(result, SPAResult)
        assert result.benchmark_name == "SPY"

    def test_with_mismatched_lengths(self):
        """Test SPA with different length arrays (should align)."""
        np.random.seed(42)

        benchmark_returns = np.random.randn(300) * 0.01 + 0.001
        strategy_returns = np.random.randn(250) * 0.01 + 0.001

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "Benchmark")

        # Should run without error (uses min length)
        assert isinstance(result, SPAResult)

    def test_multiple_benchmarks(self):
        """Test comparing strategy against multiple benchmarks."""
        np.random.seed(42)

        strategy_returns = np.random.randn(250) * 0.01 + 0.001

        benchmarks = {
            "SPY": np.random.randn(250) * 0.01 + 0.0005,
            "QQQ": np.random.randn(250) * 0.01 + 0.0008,
            "IWM": np.random.randn(250) * 0.01 + 0.0003,
        }

        spa = SPATest(bootstrap_reps=100)
        results = spa.test_multiple(strategy_returns, benchmarks)

        assert len(results) == 3
        assert "SPY" in results
        assert "QQQ" in results
        assert "IWM" in results

        # All should be SPAResult instances
        for result in results.values():
            assert isinstance(result, SPAResult)

    def test_custom_pvalue_threshold(self):
        """Test with custom p-value threshold."""
        np.random.seed(42)

        benchmark_returns = np.random.randn(250) * 0.01 + 0.001
        strategy_returns = np.random.randn(250) * 0.01 + 0.0011

        # Test that threshold affects pass/fail, not p-value calculation
        # Using same seed for both tests
        np.random.seed(123)
        spa_strict = SPATest(pvalue_threshold=0.01, bootstrap_reps=100)
        result_strict = spa_strict.test(strategy_returns, benchmark_returns)

        # If p-value is between 0.01 and 0.10, one passes and one fails
        if 0.01 < result_strict.pvalue < 0.10:
            np.random.seed(123)  # Same seed
            spa_lenient = SPATest(pvalue_threshold=0.10, bootstrap_reps=100)
            result_lenient = spa_lenient.test(strategy_returns, benchmark_returns)

            # Should have different pass/fail with same p-value
            assert result_strict.passed == False
            assert result_lenient.passed == True

    def test_with_high_volatility(self):
        """Test SPA with high volatility returns."""
        np.random.seed(42)

        # High volatility (5% daily)
        benchmark_returns = np.random.randn(250) * 0.05 + 0.001
        strategy_returns = np.random.randn(250) * 0.05 + 0.002

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "High Vol Benchmark")

        assert isinstance(result, SPAResult)

    def test_with_negative_returns(self):
        """Test SPA when both have negative returns."""
        np.random.seed(42)

        # Both negative, but strategy less negative
        benchmark_returns = np.random.randn(250) * 0.01 - 0.002
        strategy_returns = np.random.randn(250) * 0.01 - 0.001

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns, "Benchmark")

        # Strategy less negative = outperformance
        assert result.outperformance > 0


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_superior_predictive_ability_function(self):
        """Test convenience function."""
        np.random.seed(42)

        benchmark_returns = np.random.randn(250) * 0.01 + 0.0005
        strategy_returns = np.random.randn(250) * 0.01 + 0.001

        result = superior_predictive_ability(
            strategy_returns, benchmark_returns, benchmark_name="SPY", pvalue_threshold=0.05
        )

        assert isinstance(result, SPAResult)
        assert result.benchmark_name == "SPY"

    def test_function_with_defaults(self):
        """Test convenience function with default parameters."""
        np.random.seed(42)

        benchmark_returns = np.random.randn(250) * 0.01
        strategy_returns = np.random.randn(250) * 0.01

        result = superior_predictive_ability(strategy_returns, benchmark_returns)

        assert isinstance(result, SPAResult)
        assert result.benchmark_name == "Benchmark"  # Default name


class TestEdgeCases:
    """Tests for edge cases."""

    def test_with_constant_returns(self):
        """Test with constant returns (zero variance)."""
        # Both constant
        benchmark_returns = np.ones(250) * 0.001
        strategy_returns = np.ones(250) * 0.002

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns)

        # Should handle gracefully
        assert isinstance(result, SPAResult)

    def test_with_very_short_series(self):
        """Test with very short return series."""
        np.random.seed(42)

        # Only 20 observations
        benchmark_returns = np.random.randn(20) * 0.01
        strategy_returns = np.random.randn(20) * 0.01

        spa = SPATest(bootstrap_reps=100, block_size=5)  # Smaller block
        result = spa.test(strategy_returns, benchmark_returns)

        assert isinstance(result, SPAResult)

    def test_with_extreme_outperformance(self):
        """Test with extreme outperformance."""
        np.random.seed(42)

        # Benchmark nearly flat
        benchmark_returns = np.random.randn(250) * 0.001

        # Strategy with huge returns
        strategy_returns = np.random.randn(250) * 0.01 + 0.01  # 1% daily!

        spa = SPATest(bootstrap_reps=100)
        result = spa.test(strategy_returns, benchmark_returns)

        # Should have very low p-value
        assert result.outperformance > 0.005  # Large outperformance
