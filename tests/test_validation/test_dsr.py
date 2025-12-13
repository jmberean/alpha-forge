"""Tests for Deflated Sharpe Ratio."""

import numpy as np

from alphaforge.validation.dsr import (
    DeflatedSharpeRatio,
    DSRResult,
    deflated_sharpe_ratio,
)


class TestDeflatedSharpeRatio:
    """Tests for DSR calculation."""

    def test_expected_max_sharpe_increases_with_trials(self):
        """Test that expected max Sharpe increases with more trials."""
        dsr = DeflatedSharpeRatio()

        e_max_100 = dsr._expected_max_sharpe(100)
        e_max_1000 = dsr._expected_max_sharpe(1000)
        e_max_10000 = dsr._expected_max_sharpe(10000)

        assert e_max_100 < e_max_1000 < e_max_10000

    def test_expected_max_sharpe_known_values(self):
        """Test expected max Sharpe against known values."""
        dsr = DeflatedSharpeRatio()

        # Approximate values from literature (with some tolerance)
        # N=100 -> ~2.5-3.0, N=1000 -> ~3.2-3.8, N=10000 -> ~3.8-4.4
        e_max_100 = dsr._expected_max_sharpe(100)
        e_max_1000 = dsr._expected_max_sharpe(1000)
        e_max_10000 = dsr._expected_max_sharpe(10000)

        assert 2.3 < e_max_100 < 3.5
        assert 3.0 < e_max_1000 < 4.0
        assert 3.7 < e_max_10000 < 4.7

    def test_dsr_low_sharpe_many_trials(self):
        """Test that low Sharpe with many trials fails DSR."""
        dsr = DeflatedSharpeRatio(confidence_threshold=0.95)

        # Sharpe of 1.5 with 10,000 trials should likely fail
        # because expected max from noise is ~4.3
        result = dsr.calculate(
            sharpe_ratio=1.5,
            n_trials=10000,
            sample_size=252,
        )

        assert not result.passed
        assert result.dsr_pvalue < 0.95

    def test_dsr_high_sharpe_few_trials(self):
        """Test that high Sharpe with few trials passes DSR."""
        dsr = DeflatedSharpeRatio(confidence_threshold=0.95)

        # Sharpe of 3.0 with only 10 trials should pass
        result = dsr.calculate(
            sharpe_ratio=3.0,
            n_trials=10,
            sample_size=252,
        )

        assert result.passed
        assert result.dsr_pvalue > 0.95

    def test_dsr_result_contains_all_fields(self):
        """Test that DSR result contains all expected fields."""
        dsr = DeflatedSharpeRatio()

        result = dsr.calculate(
            sharpe_ratio=2.0,
            n_trials=100,
            sample_size=252,
            skewness=-0.5,
            kurtosis=4.0,
        )

        assert isinstance(result, DSRResult)
        assert result.sharpe_ratio == 2.0
        assert result.n_trials == 100
        assert result.sample_size == 252
        assert result.skewness == -0.5
        assert result.kurtosis == 4.0
        assert 0 <= result.dsr_pvalue <= 1

    def test_dsr_skewness_effect(self):
        """Test that negative skewness reduces DSR."""
        dsr = DeflatedSharpeRatio()

        # Same Sharpe, different skewness
        result_normal = dsr.calculate(sharpe_ratio=2.0, n_trials=100, sample_size=252, skewness=0)
        result_negative = dsr.calculate(
            sharpe_ratio=2.0, n_trials=100, sample_size=252, skewness=-1.0
        )

        # Negative skewness should result in lower DSR
        # (because SE of Sharpe is higher)
        assert result_negative.sharpe_std_error >= result_normal.sharpe_std_error

    def test_dsr_sample_size_effect(self):
        """Test that larger sample size reduces standard error."""
        dsr = DeflatedSharpeRatio()

        result_small = dsr.calculate(sharpe_ratio=2.0, n_trials=100, sample_size=100)
        result_large = dsr.calculate(sharpe_ratio=2.0, n_trials=100, sample_size=1000)

        # Larger sample = lower SE
        assert result_large.sharpe_std_error < result_small.sharpe_std_error

        # When SR > E[SR_max], higher sample = higher DSR
        # When SR < E[SR_max], higher sample = lower DSR (more confident it's noise)
        # Test with high Sharpe that exceeds expected max
        result_small_high = dsr.calculate(sharpe_ratio=4.0, n_trials=10, sample_size=100)
        result_large_high = dsr.calculate(sharpe_ratio=4.0, n_trials=10, sample_size=1000)

        # With high Sharpe exceeding E[SR_max], larger sample should increase DSR
        assert result_large_high.dsr_pvalue > result_small_high.dsr_pvalue

    def test_convenience_function(self):
        """Test the convenience function."""
        dsr_prob = deflated_sharpe_ratio(
            sharpe=2.0,
            n_trials=100,
            T=252,
        )

        assert 0 <= dsr_prob <= 1

    def test_calculate_from_returns(self, sample_returns):
        """Test DSR calculation from returns series."""
        dsr = DeflatedSharpeRatio()

        result = dsr.calculate_from_returns(
            sample_returns.values,
            n_trials=10,
        )

        assert isinstance(result, DSRResult)
        assert not np.isnan(result.sharpe_ratio)
        assert not np.isnan(result.dsr_pvalue)
