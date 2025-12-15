
import numpy as np
import pytest
from alphaforge.validation.pbo import (
    calculate_probability_of_loss,
    ProbabilityOfOverfitting,
    PBOResult
)

class TestPBOLogic:
    def test_probability_of_loss(self):
        """Test the simplified single-strategy metric."""
        # Case 1: All positive sharpes -> 0% prob loss
        sharpes_good = [1.0, 1.2, 0.8, 1.5]
        res_good = calculate_probability_of_loss(sharpes_good)
        assert res_good.pbo == 0.0
        assert bool(res_good.passed) is True

        # Case 2: Mix -> 50% prob loss
        sharpes_mixed = [1.0, -0.5, 0.8, -0.2]
        res_mixed = calculate_probability_of_loss(sharpes_mixed)
        assert res_mixed.pbo == 0.5
        assert bool(res_mixed.passed) is False

    def test_true_pbo_matrix_method(self):
        """Test the Bailey et al. matrix method for selection bias."""
        pbo_calc = ProbabilityOfOverfitting()
        
        # Scenario A: Robust strategies
        # 10 splits, 5 strategies
        # IS performance is perfectly predictive of OOS performance
        n_splits = 10
        n_strategies = 5
        
        # Strategy 0 is best, Strategy 4 is worst
        perf_pattern = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        is_matrix = np.tile(perf_pattern, (n_splits, 1))
        oos_matrix = np.tile(perf_pattern, (n_splits, 1))
        
        # Add tiny noise to avoid exact ties issues if any
        is_matrix += np.random.normal(0, 0.01, is_matrix.shape)
        oos_matrix += np.random.normal(0, 0.01, oos_matrix.shape)
        
        result_robust = pbo_calc.calculate(is_matrix, oos_matrix)
        
        # Since best IS is always best OOS, it should never be below median
        # PBO should be 0
        assert result_robust.pbo == 0.0
        assert result_robust.n_overfit == 0

        # Scenario B: Overfit strategies (Random/Noise)
        # IS rankings are random, OOS rankings are random
        np.random.seed(42)
        is_random = np.random.randn(n_splits, n_strategies)
        oos_random = np.random.randn(n_splits, n_strategies)
        
        result_random = pbo_calc.calculate(is_random, oos_random)
        
        # PBO should be roughly 0.5 (coin flip)
        # With small N, it varies, but should be > 0
        assert result_random.pbo > 0.0
