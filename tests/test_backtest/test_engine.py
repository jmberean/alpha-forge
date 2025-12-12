"""Tests for backtest engine."""

import pytest
import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestEngine, BacktestResult, quick_backtest
from alphaforge.backtest.metrics import PerformanceMetrics, calculate_sharpe
from alphaforge.strategy.templates import StrategyTemplates


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_backtest_returns_result(self, spy_data, sample_strategy):
        """Test that backtest returns BacktestResult."""
        engine = BacktestEngine()
        result = engine.run(sample_strategy, spy_data)

        assert isinstance(result, BacktestResult)
        assert result.strategy == sample_strategy
        assert result.data_symbol == "SPY"

    def test_backtest_result_has_metrics(self, spy_data, sample_strategy):
        """Test that result contains all expected metrics."""
        engine = BacktestEngine()
        result = engine.run(sample_strategy, spy_data)

        metrics = result.metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "annualized_return")
        assert hasattr(metrics, "volatility")

    def test_backtest_returns_series(self, spy_data, sample_strategy):
        """Test that returns series has correct length."""
        engine = BacktestEngine()
        result = engine.run(sample_strategy, spy_data)

        assert len(result.returns) == len(spy_data)
        assert isinstance(result.returns, pd.Series)

    def test_backtest_positions_series(self, spy_data, sample_strategy):
        """Test that positions series is valid."""
        engine = BacktestEngine()
        result = engine.run(sample_strategy, spy_data)

        assert len(result.positions) == len(spy_data)
        # Positions should be 0 or 1 for long-only
        assert set(result.positions.unique()).issubset({0, 1})

    def test_backtest_equity_curve(self, spy_data, sample_strategy):
        """Test equity curve starts at initial capital."""
        initial_capital = 100000.0
        engine = BacktestEngine(initial_capital=initial_capital)
        result = engine.run(sample_strategy, spy_data)

        # First equity value should be close to initial capital
        first_equity = result.equity_curve.iloc[0]
        assert abs(first_equity - initial_capital) < initial_capital * 0.01

    def test_different_strategies_different_results(self, spy_data):
        """Test that different strategies produce different results."""
        engine = BacktestEngine()

        # Use strategies with indicators that are computed by compute_all
        strategy1 = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)
        strategy2 = StrategyTemplates.rsi_mean_reversion(oversold=30, overbought=70)

        result1 = engine.run(strategy1, spy_data)
        result2 = engine.run(strategy2, spy_data)

        # Results should be different
        assert result1.metrics.sharpe_ratio != result2.metrics.sharpe_ratio

    def test_commission_reduces_returns(self, spy_data, sample_strategy):
        """Test that higher commission reduces returns."""
        engine_low = BacktestEngine(commission_pct=0.0001)
        engine_high = BacktestEngine(commission_pct=0.01)

        result_low = engine_low.run(sample_strategy, spy_data)
        result_high = engine_high.run(sample_strategy, spy_data)

        # Higher commission should result in lower returns
        assert result_high.metrics.total_return <= result_low.metrics.total_return

    def test_run_multiple_strategies(self, spy_data):
        """Test running multiple strategies at once."""
        engine = BacktestEngine()

        strategies = [
            StrategyTemplates.sma_crossover(),
            StrategyTemplates.rsi_mean_reversion(),
            StrategyTemplates.macd_crossover(),
        ]

        results = engine.run_multiple(strategies, spy_data)

        assert len(results) == 3
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_quick_backtest(self, spy_data, sample_strategy):
        """Test quick backtest function."""
        sharpe = quick_backtest(sample_strategy, spy_data)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_to_dict_serialization(self, spy_data, sample_strategy):
        """Test that result can be serialized."""
        engine = BacktestEngine()
        result = engine.run(sample_strategy, spy_data)

        result_dict = result.to_dict()

        assert "strategy_id" in result_dict
        assert "strategy_name" in result_dict
        assert "metrics" in result_dict
        assert "sharpe_ratio" in result_dict["metrics"]


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_from_returns_positive(self, sample_returns):
        """Test metrics calculation from returns."""
        metrics = PerformanceMetrics.from_returns(sample_returns)

        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown >= 0
        assert metrics.trading_days == len(sample_returns)

    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation matches manual."""
        # Create simple returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)

        sharpe = calculate_sharpe(returns, periods_per_year=252)

        # Manual calculation
        expected = returns.mean() / returns.std() * np.sqrt(252)
        np.testing.assert_almost_equal(sharpe, expected, decimal=5)

    def test_drawdown_calculation(self, sample_returns):
        """Test max drawdown is correctly calculated."""
        metrics = PerformanceMetrics.from_returns(sample_returns)

        # Manually calculate
        cumulative = (1 + sample_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        expected_max_dd = abs(drawdown.min())

        np.testing.assert_almost_equal(
            metrics.max_drawdown, expected_max_dd, decimal=5
        )

    def test_win_rate_with_positions(self, sample_returns):
        """Test win rate calculation with positions."""
        # Create positions that alternate
        positions = pd.Series([1, 1, 0, 1, 1, 0] * (len(sample_returns) // 6))
        positions = positions[: len(sample_returns)]

        metrics = PerformanceMetrics.from_returns(sample_returns, positions=positions)

        assert 0 <= metrics.win_rate <= 1

    def test_zero_std_returns(self):
        """Test handling of zero standard deviation returns."""
        returns = pd.Series([0.0] * 100)
        metrics = PerformanceMetrics.from_returns(returns)

        assert metrics.sharpe_ratio == 0.0
        assert metrics.volatility == 0.0
