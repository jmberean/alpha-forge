"""Tests for Combinatorially Purged Cross-Validation."""

import pytest
import numpy as np
import pandas as pd
from math import factorial

from alphaforge.validation.cpcv import (
    CombinatorialPurgedCV,
    CPCVResult,
    combinatorial_purged_cv,
)


class TestCombinatorialPurgedCV:
    """Tests for CPCV implementation."""

    def test_n_combinations_calculation(self):
        """Test that n_combinations is calculated correctly."""
        cpcv = CombinatorialPurgedCV(n_splits=16, test_splits=8)

        # C(16, 8) = 12870
        expected = factorial(16) // (factorial(8) * factorial(8))
        assert cpcv.n_combinations == expected
        assert cpcv.n_combinations == 12870

    def test_n_combinations_different_configs(self):
        """Test n_combinations for different configurations."""
        # C(10, 5) = 252
        cpcv1 = CombinatorialPurgedCV(n_splits=10, test_splits=5)
        assert cpcv1.n_combinations == 252

        # C(8, 4) = 70
        cpcv2 = CombinatorialPurgedCV(n_splits=8, test_splits=4)
        assert cpcv2.n_combinations == 70

    def test_split_info(self, spy_df):
        """Test split information is correct."""
        cpcv = CombinatorialPurgedCV(n_splits=8, test_splits=4)
        info = cpcv.get_split_info(spy_df)

        assert info["n_splits"] == 8
        assert info["test_splits"] == 4
        assert info["n_combinations"] == 70
        assert len(info["splits"]) == 8

    def test_validate_returns_result(self, spy_df):
        """Test that validate returns CPCVResult."""

        def dummy_backtest(train_df, test_df):
            """Dummy backtest returning constant Sharpe."""
            return 1.5

        cpcv = CombinatorialPurgedCV(n_splits=4, test_splits=2)
        result = cpcv.validate(spy_df, dummy_backtest, max_combinations=10)

        assert isinstance(result, CPCVResult)
        assert result.n_splits == 4
        assert result.test_splits == 2
        assert len(result.sharpe_distribution) <= 10

    def test_pbo_calculation(self, spy_df):
        """Test PBO is calculated correctly."""

        def negative_backtest(train_df, test_df):
            """Returns negative Sharpe 50% of time."""
            return np.random.choice([-1.0, 1.0])

        np.random.seed(42)
        cpcv = CombinatorialPurgedCV(n_splits=4, test_splits=2, pbo_threshold=0.5)
        result = cpcv.validate(spy_df, negative_backtest, max_combinations=100)

        # PBO should be approximately 0.5 for this dummy strategy
        assert 0.2 < result.pbo < 0.8

    def test_positive_sharpe_low_pbo(self, spy_df):
        """Test that consistently positive Sharpe has low PBO."""

        def positive_backtest(train_df, test_df):
            """Always returns positive Sharpe."""
            return 2.0 + np.random.uniform(0, 0.5)

        np.random.seed(42)
        cpcv = CombinatorialPurgedCV(n_splits=4, test_splits=2)
        result = cpcv.validate(spy_df, positive_backtest, max_combinations=50)

        assert result.pbo == 0.0  # All positive = 0% negative
        assert result.passed

    def test_negative_sharpe_high_pbo(self, spy_df):
        """Test that consistently negative Sharpe has high PBO."""

        def negative_backtest(train_df, test_df):
            """Always returns negative Sharpe."""
            return -1.0 - np.random.uniform(0, 0.5)

        np.random.seed(42)
        cpcv = CombinatorialPurgedCV(n_splits=4, test_splits=2, pbo_threshold=0.05)
        result = cpcv.validate(spy_df, negative_backtest, max_combinations=50)

        assert result.pbo == 1.0  # All negative = 100% negative
        assert not result.passed

    def test_embargo_applied(self, spy_df):
        """Test that embargo removes data at boundaries."""
        cpcv = CombinatorialPurgedCV(n_splits=4, test_splits=2, embargo_pct=0.10)

        # Get split info
        splits = np.array_split(spy_df, 4)
        original_len = sum(len(s) for s in splits)

        # Simulate purging for train indices [0, 3] when test is [1, 2]
        train_data = cpcv._purge_and_concat(splits, [0, 3], [1, 2])

        # Purged data should be smaller due to embargo
        # Split 0 has test neighbor at 1, so end is purged
        # Split 3 has test neighbor at 2, so start is purged
        assert len(train_data) < sum(len(splits[i]) for i in [0, 3])

    def test_convenience_function(self, spy_df):
        """Test the convenience function."""

        def simple_backtest(train_df, test_df):
            return 1.0

        result = combinatorial_purged_cv(
            spy_df,
            simple_backtest,
            n_splits=4,
            test_splits=2,
            max_combinations=10,
        )

        assert isinstance(result, CPCVResult)
