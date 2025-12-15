
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from alphaforge.discovery.orchestrator import DiscoveryOrchestrator, DiscoveryConfig
from alphaforge.backtest.metrics import PerformanceMetrics

class TestOrchestratorLogic:
    @pytest.fixture
    def mock_data(self):
        # Create dummy OHLCV data
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame({
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0 + np.random.randn(100).cumsum(),
            "volume": 1000000
        }, index=dates)
        return df

    @pytest.fixture
    def orchestrator(self, mock_data):
        config = DiscoveryConfig(min_trades=5, min_volatility=0.01)
        return DiscoveryOrchestrator(market_data=mock_data, config=config)

    def test_zero_trade_penalty(self, orchestrator):
        """Test that strategies with too few trades are penalized."""
        # Mock metrics to return 0 trades
        mock_metrics = MagicMock()
        mock_metrics.num_trades = 0
        mock_metrics.volatility = 0.0
        mock_metrics.sharpe_ratio = 5.0 # Even with high Sharpe, should fail

        # Mock the backtest process parts
        with patch("alphaforge.discovery.orchestrator.PerformanceMetrics.from_returns", return_value=mock_metrics):
            # We need to mock _evaluate_tree and _signals_to_positions to run _compute_all_fitness
            # But _compute_all_fitness is what we want to test.
            # Let's bypass the internal calls and test the logic block directly if possible,
            # or just mock the dependencies.
            
            orchestrator._evaluate_tree = MagicMock(return_value=pd.Series([1]*100)) # Dummy signal
            orchestrator._signals_to_positions = MagicMock(return_value=pd.Series([1]*100)) # Dummy positions
            
            # Create a dummy tree
            dummy_tree = MagicMock()
            dummy_tree.complexity_score.return_value = 0.1
            
            # Run
            fitness = orchestrator._compute_all_fitness(dummy_tree)
            
            # Assert penalty
            assert fitness["sharpe"] == -999.0
            assert fitness["drawdown"] == -999.0

    def test_lookahead_invariance(self, orchestrator):
        """
        Test that signal normalization is point-in-time safe.
        The position at time T should not depend on data at T+k.
        """
        # Generate a random signal
        np.random.seed(42)
        # Use market_data index which has full 100 points
        signal_full = pd.Series(np.random.randn(100), index=orchestrator.market_data.index[:100])
        
        # Scenario 1: Calculate positions on full dataset
        positions_full = orchestrator._signals_to_positions(signal_full)
        
        # Scenario 2: Calculate positions on partial dataset (first 50)
        signal_partial = signal_full.iloc[:50]
        positions_partial = orchestrator._signals_to_positions(signal_partial)
        
        # Check index 49 (last point of partial)
        # It must be identical in both cases
        pos_at_49_full = positions_full.iloc[49]
        pos_at_49_partial = positions_partial.iloc[49]
        
        assert pos_at_49_full == pos_at_49_partial, \
            f"Lookahead detected! Position at t=49 changed when future data was added. {pos_at_49_partial} vs {pos_at_49_full}"
            
        # Also check the whole overlap
        pd.testing.assert_series_equal(positions_full.iloc[:50], positions_partial)

    def test_expanding_window_behavior(self, orchestrator):
        """Test that we are using expanding window statistics."""
        # Create a step signal: 0 for 20 days, then 100
        signal = pd.Series([0]*25 + [100]*25)
        
        positions = orchestrator._signals_to_positions(signal)
        
        # First 20 should be 0 (min_periods)
        assert (positions.iloc[:20] == 0).all()
        
        # At step change, z-score should spike
        # We don't check exact values, just that it runs without error and produces output
        assert len(positions) == 50
