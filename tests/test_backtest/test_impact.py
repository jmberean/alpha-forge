"""Tests for market impact modeling."""

import pytest
import numpy as np

from alphaforge.backtest.impact import (
    MarketImpactModel,
    ExecutionModel,
    ExecutionCost,
    calculate_market_impact,
)


class TestMarketImpactModel:
    """Tests for Almgren-Chriss market impact model."""

    def test_basic_impact_calculation(self):
        """Test basic impact calculation with typical values."""
        model = MarketImpactModel()

        # $100k order, $50M daily volume, 1% volatility
        impact = model.calculate_impact(
            order_size=100_000, daily_volume=50_000_000, volatility=0.01
        )

        # Should be positive
        assert impact > 0

        # Should be small (<0.1%) for 0.2% participation
        assert impact < 0.001

    def test_impact_increases_with_order_size(self):
        """Test that impact increases with order size."""
        model = MarketImpactModel()

        daily_volume = 50_000_000
        volatility = 0.01

        impact_small = model.calculate_impact(100_000, daily_volume, volatility)
        impact_medium = model.calculate_impact(500_000, daily_volume, volatility)
        impact_large = model.calculate_impact(1_000_000, daily_volume, volatility)

        assert impact_small < impact_medium < impact_large

    def test_impact_decreases_with_daily_volume(self):
        """Test that impact decreases with higher daily volume."""
        model = MarketImpactModel()

        order_size = 100_000
        volatility = 0.01

        impact_low_vol = model.calculate_impact(order_size, 10_000_000, volatility)
        impact_med_vol = model.calculate_impact(order_size, 50_000_000, volatility)
        impact_high_vol = model.calculate_impact(order_size, 100_000_000, volatility)

        assert impact_low_vol > impact_med_vol > impact_high_vol

    def test_impact_increases_with_volatility(self):
        """Test that impact increases with volatility."""
        model = MarketImpactModel()

        order_size = 100_000
        daily_volume = 50_000_000

        impact_low = model.calculate_impact(order_size, daily_volume, 0.005)
        impact_med = model.calculate_impact(order_size, daily_volume, 0.01)
        impact_high = model.calculate_impact(order_size, daily_volume, 0.02)

        assert impact_low < impact_med < impact_high

    def test_impact_decreases_with_time_horizon(self):
        """Test that impact decreases with longer execution time."""
        model = MarketImpactModel()

        order_size = 100_000
        daily_volume = 50_000_000
        volatility = 0.01

        impact_fast = model.calculate_impact(
            order_size, daily_volume, volatility, time_horizon=0.25
        )
        impact_normal = model.calculate_impact(
            order_size, daily_volume, volatility, time_horizon=1.0
        )
        impact_slow = model.calculate_impact(
            order_size, daily_volume, volatility, time_horizon=5.0
        )

        # Temporary impact decreases with time, permanent stays same
        # So total should decrease
        assert impact_fast > impact_normal > impact_slow

    def test_component_breakdown(self):
        """Test permanent vs temporary impact breakdown."""
        model = MarketImpactModel()

        permanent, temporary = model.calculate_components(
            order_size=100_000, daily_volume=50_000_000, volatility=0.01
        )

        # Both should be positive
        assert permanent > 0
        assert temporary > 0

        # Total should equal sum
        total = model.calculate_impact(100_000, 50_000_000, 0.01)
        assert abs(total - (permanent + temporary)) < 1e-10

    def test_zero_volatility(self):
        """Test that zero volatility gives zero impact."""
        model = MarketImpactModel()

        impact = model.calculate_impact(
            order_size=100_000, daily_volume=50_000_000, volatility=0.0
        )

        assert impact == 0.0

    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        model = MarketImpactModel()

        # Negative daily volume
        with pytest.raises(ValueError, match="daily_volume must be positive"):
            model.calculate_impact(100_000, -1000, 0.01)

        # Zero daily volume
        with pytest.raises(ValueError, match="daily_volume must be positive"):
            model.calculate_impact(100_000, 0, 0.01)

        # Negative volatility
        with pytest.raises(ValueError, match="volatility must be non-negative"):
            model.calculate_impact(100_000, 50_000_000, -0.01)

        # Zero time horizon
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            model.calculate_impact(100_000, 50_000_000, 0.01, time_horizon=0)

        # Negative time horizon
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            model.calculate_impact(100_000, 50_000_000, 0.01, time_horizon=-1)

    def test_custom_parameters(self):
        """Test model with custom parameters."""
        # More aggressive impact model
        aggressive_model = MarketImpactModel(
            permanent_impact=0.5, temporary_impact=0.3
        )

        # Less aggressive impact model
        conservative_model = MarketImpactModel(
            permanent_impact=0.1, temporary_impact=0.05
        )

        order_size = 100_000
        daily_volume = 50_000_000
        volatility = 0.01

        aggressive_impact = aggressive_model.calculate_impact(
            order_size, daily_volume, volatility
        )
        conservative_impact = conservative_model.calculate_impact(
            order_size, daily_volume, volatility
        )

        assert aggressive_impact > conservative_impact

    def test_realistic_scenario(self):
        """Test with realistic market scenario."""
        model = MarketImpactModel()

        # SPY-like stock: $500 price, $5M daily volume (10k shares)
        # Order: 100 shares = $50k
        # Daily volatility: ~1%
        impact = model.calculate_impact(
            order_size=50_000, daily_volume=5_000_000, volatility=0.01
        )

        # For 1% participation, impact should be ~5-15 bps (empirically reasonable)
        assert 0.0003 < impact < 0.0020  # 3 to 20 bps


class TestExecutionModel:
    """Tests for complete execution cost model."""

    def test_execution_cost_components(self):
        """Test that execution cost includes all components."""
        model = ExecutionModel(base_commission=1.0, spread_bps=2.0)

        cost = model.calculate_total_cost(
            order_value=10_000, daily_volume=50_000_000, volatility=0.01
        )

        assert cost.commission == 1.0
        assert cost.spread_cost > 0
        assert cost.impact_cost > 0
        assert cost.total_cost == (
            cost.commission + cost.spread_cost + cost.impact_cost
        )

    def test_zero_commission(self):
        """Test with zero commission (typical for retail)."""
        model = ExecutionModel(base_commission=0.0, spread_bps=2.0)

        cost = model.calculate_total_cost(
            order_value=10_000, daily_volume=50_000_000, volatility=0.01
        )

        assert cost.commission == 0.0
        assert cost.total_cost == cost.spread_cost + cost.impact_cost

    def test_spread_cost_scales_with_order_value(self):
        """Test that spread cost scales linearly with order value."""
        model = ExecutionModel(base_commission=0.0, spread_bps=2.0)

        cost_small = model.calculate_total_cost(
            order_value=10_000, daily_volume=50_000_000, volatility=0.01
        )
        cost_large = model.calculate_total_cost(
            order_value=20_000, daily_volume=50_000_000, volatility=0.01
        )

        # Spread cost should double
        assert abs(cost_large.spread_cost - 2 * cost_small.spread_cost) < 0.01

    def test_no_impact_model(self):
        """Test execution cost without market impact."""
        model = ExecutionModel(
            base_commission=1.0, spread_bps=2.0, impact_model=False
        )

        cost = model.calculate_total_cost(
            order_value=10_000, daily_volume=50_000_000, volatility=0.01
        )

        assert cost.impact_cost == 0.0
        assert cost.total_cost == cost.commission + cost.spread_cost

    def test_zero_daily_volume(self):
        """Test handling of zero daily volume."""
        model = ExecutionModel(base_commission=1.0, spread_bps=2.0)

        # Should not crash, impact should be zero
        cost = model.calculate_total_cost(
            order_value=10_000, daily_volume=0, volatility=0.01
        )

        assert cost.impact_cost == 0.0
        assert cost.total_cost == cost.commission + cost.spread_cost


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_calculate_market_impact_function(self):
        """Test convenience function matches model."""
        order_size = 100_000
        daily_volume = 50_000_000
        volatility = 0.01

        # Using convenience function
        impact_function = calculate_market_impact(order_size, daily_volume, volatility)

        # Using model directly
        model = MarketImpactModel()
        impact_model = model.calculate_impact(order_size, daily_volume, volatility)

        assert abs(impact_function - impact_model) < 1e-10

    def test_custom_parameters_in_function(self):
        """Test convenience function with custom parameters."""
        impact = calculate_market_impact(
            order_size=100_000,
            daily_volume=50_000_000,
            volatility=0.01,
            time_horizon=2.0,
            permanent_impact=0.5,
            temporary_impact=0.3,
        )

        assert impact > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_order(self):
        """Test with order size equal to daily volume."""
        model = MarketImpactModel()

        # 100% participation
        impact = model.calculate_impact(
            order_size=50_000_000, daily_volume=50_000_000, volatility=0.01
        )

        # Impact should be substantial (>1%)
        assert impact > 0.01

    def test_very_small_order(self):
        """Test with tiny order size."""
        model = MarketImpactModel()

        # 0.0001% participation
        impact = model.calculate_impact(
            order_size=50, daily_volume=50_000_000, volatility=0.01
        )

        # Impact should be negligible
        assert impact < 0.00001

    def test_high_volatility(self):
        """Test with very high volatility (10%)."""
        model = MarketImpactModel()

        impact = model.calculate_impact(
            order_size=100_000, daily_volume=50_000_000, volatility=0.10
        )

        # Impact should be higher than with 1% vol
        impact_normal = model.calculate_impact(
            order_size=100_000, daily_volume=50_000_000, volatility=0.01
        )

        assert impact > impact_normal
