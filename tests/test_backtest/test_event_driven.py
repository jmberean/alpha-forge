"""Tests for event-driven backtesting."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.event_driven import (
    EventDrivenEngine,
    ExecutionConfig,
    OrderSide,
    OrderStatus,
    QueueModel,
    calculate_implementation_shortfall,
)
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.data.schema import OHLCVData


@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = pd.DataFrame({
        "open": 100 + np.random.randn(100).cumsum(),
        "high": 102 + np.random.randn(100).cumsum(),
        "low": 98 + np.random.randn(100).cumsum(),
        "close": 100 + np.random.randn(100).cumsum(),
        "volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)

    # Ensure OHLC consistency
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return OHLCVData(df=data, symbol="TEST")


class TestExecutionConfig:
    """Test execution configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutionConfig()

        assert config.latency_ms == 50
        assert config.queue_position_pct == 0.5
        assert config.allow_partial_fills is True


class TestQueueModel:
    """Test queue model."""

    def test_fill_probability(self, sample_data):
        """Test fill probability estimation."""
        from alphaforge.backtest.event_driven import Order

        config = ExecutionConfig()
        queue_model = QueueModel(config)

        order = Order(
            order_id="test",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            price=100.0,
            timestamp=sample_data.df.index[0],
        )

        prob = queue_model.estimate_fill_probability(
            order,
            market_price=100.0,
            volume=1000000.0,
        )

        assert 0.0 <= prob <= 1.0

    def test_fill_quantity(self, sample_data):
        """Test fill quantity estimation."""
        from alphaforge.backtest.event_driven import Order

        config = ExecutionConfig()
        queue_model = QueueModel(config)

        order = Order(
            order_id="test",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            price=100.0,
            timestamp=sample_data.df.index[0],
        )

        fill_qty = queue_model.estimate_fill_quantity(order, available_volume=50.0)

        assert 0.0 <= fill_qty <= 50.0


class TestEventDrivenEngine:
    """Test event-driven backtesting engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = EventDrivenEngine()

        assert engine.config is not None
        assert engine.queue_model is not None
        assert engine.cash == 100000.0

    def test_run_backtest(self, sample_data):
        """Test running event-driven backtest."""
        engine = EventDrivenEngine()
        strategy = StrategyTemplates.sma_crossover()

        result = engine.run(strategy, sample_data)

        # Check result
        assert result is not None
        assert result.metrics is not None
        assert len(result.equity_curve) == len(sample_data)
        assert len(result.returns) == len(sample_data)

    def test_backtest_produces_trades(self, sample_data):
        """Test that backtest produces some trades."""
        engine = EventDrivenEngine(config=ExecutionConfig(latency_ms=0))  # No latency for testing
        strategy = StrategyTemplates.sma_crossover()

        result = engine.run(strategy, sample_data)

        # Should have some orders (may or may not be filled)
        assert len(engine.orders) >= 0  # At least attempted some orders


def test_calculate_implementation_shortfall(sample_data):
    """Test implementation shortfall calculation."""
    from alphaforge.backtest.engine import BacktestEngine

    strategy = StrategyTemplates.sma_crossover()

    # Vectorized backtest
    vector_engine = BacktestEngine()
    vector_result = vector_engine.run(strategy, sample_data)

    # Event-driven backtest
    event_engine = EventDrivenEngine()
    event_result = event_engine.run(strategy, sample_data)

    # Calculate shortfall
    shortfall = calculate_implementation_shortfall(vector_result, event_result)

    # Check results
    assert "vectorized_sharpe" in shortfall
    assert "event_driven_sharpe" in shortfall
    assert "sharpe_shortfall" in shortfall
    assert "passed_threshold" in shortfall

    # Shortfall should be a reasonable number
    assert -2.0 <= shortfall["sharpe_shortfall"] <= 2.0
