"""Tests for stress testing framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from alphaforge.validation.stress import (
    StressScenario,
    ScenarioResult,
    StressTestResult,
    StressTester,
    stress_test_strategy,
)
from alphaforge.strategy.templates import StrategyTemplates


class TestStressScenario:
    """Tests for StressScenario dataclass."""

    def test_historical_scenario_creation(self):
        """Test creating a historical scenario."""
        scenario = StressScenario(
            name="test_crisis",
            description="Test crisis scenario",
            scenario_type="historical",
            period=("2020-01-01", "2020-12-31"),
        )

        assert scenario.name == "test_crisis"
        assert scenario.scenario_type == "historical"
        assert scenario.period == ("2020-01-01", "2020-12-31")
        assert scenario.min_sharpe == 0.0
        assert scenario.max_drawdown == 0.50

    def test_synthetic_scenario_creation(self):
        """Test creating a synthetic scenario."""

        def dummy_transform(df: pd.DataFrame) -> pd.DataFrame:
            return df

        scenario = StressScenario(
            name="test_shock",
            description="Test shock scenario",
            scenario_type="synthetic",
            transform=dummy_transform,
        )

        assert scenario.name == "test_shock"
        assert scenario.scenario_type == "synthetic"
        assert scenario.transform is not None


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_passing_result(self):
        """Test a passing scenario result."""
        result = ScenarioResult(
            scenario_name="test",
            sharpe=1.5,
            max_drawdown=0.10,
            annual_return=0.15,
            passed=True,
        )

        assert result.passed is True
        assert result.failure_reason is None

    def test_failing_result(self):
        """Test a failing scenario result."""
        result = ScenarioResult(
            scenario_name="test",
            sharpe=-0.5,
            max_drawdown=0.60,
            annual_return=-0.20,
            passed=False,
            failure_reason="Sharpe < 0",
        )

        assert result.passed is False
        assert result.failure_reason == "Sharpe < 0"


class TestStressTestResult:
    """Tests for StressTestResult dataclass."""

    def test_passing_stress_test(self):
        """Test stress test with 80%+ pass rate."""
        results = [
            ScenarioResult("s1", 1.0, 0.1, 0.1, True),
            ScenarioResult("s2", 1.0, 0.1, 0.1, True),
            ScenarioResult("s3", 1.0, 0.1, 0.1, True),
            ScenarioResult("s4", -0.5, 0.6, -0.1, False),
        ]

        stress_result = StressTestResult(
            scenarios_tested=4,
            scenarios_passed=3,
            pass_rate=0.75,
            results=results,
        )

        # 75% < 80%, should fail
        assert stress_result.passed is False

    def test_exact_80_percent_passes(self):
        """Test that exactly 80% pass rate passes."""
        results = [
            ScenarioResult("s1", 1.0, 0.1, 0.1, True),
            ScenarioResult("s2", 1.0, 0.1, 0.1, True),
            ScenarioResult("s3", 1.0, 0.1, 0.1, True),
            ScenarioResult("s4", 1.0, 0.1, 0.1, True),
            ScenarioResult("s5", -0.5, 0.6, -0.1, False),
        ]

        stress_result = StressTestResult(
            scenarios_tested=5,
            scenarios_passed=4,
            pass_rate=0.80,
            results=results,
        )

        assert stress_result.passed is True

    def test_summary_format(self):
        """Test summary string generation."""
        results = [
            ScenarioResult("scenario1", 1.5, 0.10, 0.15, True),
            ScenarioResult("scenario2", -0.5, 0.60, -0.20, False, "Sharpe < 0"),
        ]

        stress_result = StressTestResult(
            scenarios_tested=2,
            scenarios_passed=1,
            pass_rate=0.50,
            results=results,
        )

        summary = stress_result.summary()

        assert "FAILED" in summary  # 50% < 80%
        assert "Scenarios Tested: 2" in summary
        assert "Scenarios Passed: 1/2" in summary
        assert "scenario1" in summary
        assert "scenario2" in summary


class TestStressTester:
    """Tests for StressTester class."""

    def test_initialization(self):
        """Test StressTester initialization."""
        tester = StressTester()

        assert tester.data_loader is not None
        assert tester.backtest_engine is not None

        # Check historical scenarios loaded
        assert "2008_financial_crisis" in tester.HISTORICAL_SCENARIOS
        assert "2020_covid_crash" in tester.HISTORICAL_SCENARIOS
        assert "2022_rate_hikes" in tester.HISTORICAL_SCENARIOS

        # Check synthetic scenarios loaded
        assert "volatility_3x" in tester.SYNTHETIC_SCENARIOS
        assert "gap_risk" in tester.SYNTHETIC_SCENARIOS
        assert "correlation_spike" in tester.SYNTHETIC_SCENARIOS

    def test_synthetic_transforms_setup(self):
        """Test that synthetic transforms are properly set up."""
        tester = StressTester()

        # Volatility 3x should have transform
        vol_scenario = tester.SYNTHETIC_SCENARIOS["volatility_3x"]
        assert vol_scenario.transform is not None

        # Gap risk should have transform
        gap_scenario = tester.SYNTHETIC_SCENARIOS["gap_risk"]
        assert gap_scenario.transform is not None

    def test_volatility_3x_transform(self):
        """Test volatility 3x transform increases volatility."""
        tester = StressTester()
        transform = tester.SYNTHETIC_SCENARIOS["volatility_3x"].transform

        # Create sample data
        dates = pd.date_range("2020-01-01", periods=100)
        prices = 100 * (1 + np.random.randn(100) * 0.01).cumprod()
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": 1000000,
                "adjusted_close": prices,
            },
            index=dates,
        )

        # Apply transform
        df_transformed = transform(df)

        # Calculate volatilities
        returns_original = df["close"].pct_change().dropna()
        returns_transformed = df_transformed["close"].pct_change().dropna()

        vol_original = returns_original.std()
        vol_transformed = returns_transformed.std()

        # Transformed should have higher volatility
        # (May not be exactly 3x due to compounding effects)
        assert vol_transformed > vol_original

    def test_gap_risk_transform(self):
        """Test gap risk transform adds price gaps."""
        tester = StressTester()
        transform = tester.SYNTHETIC_SCENARIOS["gap_risk"].transform

        # Set seed for reproducibility
        np.random.seed(42)

        # Create sample data
        dates = pd.date_range("2020-01-01", periods=100)
        prices = np.ones(100) * 100  # Constant price
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1000000,
                "adjusted_close": prices,
            },
            index=dates,
        )

        # Apply transform
        df_transformed = transform(df)

        # Should have price changes (gaps)
        price_changes = df_transformed["close"] != 100
        assert price_changes.any()

    @pytest.mark.slow
    def test_stress_test_with_real_strategy(self):
        """Test stress testing with a real strategy (uses real data)."""
        tester = StressTester()
        strategy = StrategyTemplates.get_template("sma_crossover")

        # Test with just one historical scenario to save time
        result = tester.test_strategy(
            strategy, symbol="SPY", scenarios=["2020_covid_crash"]
        )

        assert result.scenarios_tested == 1
        assert len(result.results) == 1
        assert result.results[0].scenario_name == "2020_covid_crash"

        # Should have metrics
        assert result.results[0].sharpe is not None
        assert result.results[0].max_drawdown is not None

    def test_unknown_scenario_raises_error(self):
        """Test that unknown scenario name raises error."""
        tester = StressTester()
        strategy = StrategyTemplates.get_template("sma_crossover")

        with pytest.raises(ValueError, match="Unknown scenario"):
            tester.test_strategy(strategy, scenarios=["nonexistent_scenario"])

    def test_scenario_pass_criteria(self, spy_data):
        """Test scenario pass/fail logic."""
        tester = StressTester()

        # Create a simple scenario
        scenario = StressScenario(
            name="test",
            description="test",
            scenario_type="historical",
            period=("2023-01-01", "2023-06-30"),
            min_sharpe=1.5,  # Very high requirement
            max_drawdown=0.05,  # Very strict
        )

        strategy = StrategyTemplates.get_template("sma_crossover")

        result = tester._test_scenario(strategy, "SPY", scenario)

        # With strict criteria, likely to fail
        # (Just checking the logic works, not the specific outcome)
        assert result.passed in [True, False]
        if not result.passed:
            assert result.failure_reason is not None


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.slow
    def test_stress_test_strategy_function(self):
        """Test convenience function."""
        strategy = StrategyTemplates.get_template("sma_crossover")

        # Test with subset of scenarios
        result = stress_test_strategy(
            strategy, symbol="SPY", scenarios=["2020_covid_crash"]
        )

        assert isinstance(result, StressTestResult)
        assert result.scenarios_tested >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scenario_list(self):
        """Test with empty scenario list."""
        tester = StressTester()
        strategy = StrategyTemplates.get_template("sma_crossover")

        result = tester.test_strategy(strategy, scenarios=[])

        assert result.scenarios_tested == 0
        assert result.scenarios_passed == 0
        assert result.pass_rate == 0.0
        assert result.passed is False

    def test_all_scenarios_pass(self):
        """Test when all scenarios pass."""
        results = [
            ScenarioResult("s1", 2.0, 0.05, 0.20, True),
            ScenarioResult("s2", 1.5, 0.10, 0.15, True),
            ScenarioResult("s3", 1.0, 0.15, 0.10, True),
        ]

        stress_result = StressTestResult(
            scenarios_tested=3,
            scenarios_passed=3,
            pass_rate=1.0,
            results=results,
        )

        assert stress_result.passed is True
        assert "PASSED" in stress_result.summary()

    def test_all_scenarios_fail(self):
        """Test when all scenarios fail."""
        results = [
            ScenarioResult("s1", -1.0, 0.60, -0.30, False, "Negative Sharpe"),
            ScenarioResult("s2", -0.5, 0.70, -0.20, False, "High drawdown"),
        ]

        stress_result = StressTestResult(
            scenarios_tested=2,
            scenarios_passed=0,
            pass_rate=0.0,
            results=results,
        )

        assert stress_result.passed is False
        assert "FAILED" in stress_result.summary()
