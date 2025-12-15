"""
Stress testing framework for strategy robustness validation.

Tests strategies under extreme market conditions:
- Historical replays (2008 crisis, 2020 COVID, 2022 rate hikes)
- Synthetic shocks (correlation spike, volatility surge, liquidity drain)

A strategy must pass 80% of scenarios to be considered robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

from alphaforge.data.loader import MarketDataLoader
from alphaforge.strategy.genome import StrategyGenome
from alphaforge.backtest.engine import BacktestEngine


@dataclass
class StressScenario:
    """
    A stress test scenario.

    Can be either:
    - Historical: Replay actual market crisis
    - Synthetic: Apply mathematical transform to returns
    """

    name: str
    description: str
    scenario_type: Literal["historical", "synthetic"]

    # For historical scenarios
    period: tuple[str, str] | None = None  # (start_date, end_date)

    # For synthetic scenarios
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    # Pass criteria
    min_sharpe: float = 0.0  # Must still make money
    max_drawdown: float = 0.50  # Can't lose more than 50%


@dataclass
class ScenarioResult:
    """Result from a single stress scenario."""

    scenario_name: str
    sharpe: float
    max_drawdown: float
    annual_return: float
    passed: bool
    failure_reason: str | None = None


@dataclass
class StressTestResult:
    """
    Complete stress test result.

    A strategy must pass 80% of scenarios to be considered robust.
    """

    scenarios_tested: int
    scenarios_passed: int
    pass_rate: float
    results: list[ScenarioResult] = field(default_factory=list)
    passed: bool = False

    def __post_init__(self) -> None:
        """Calculate pass/fail after initialization."""
        self.passed = self.pass_rate >= 0.80

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Stress Test Result: {'PASSED' if self.passed else 'FAILED'}",
            f"",
            f"Scenarios Tested: {self.scenarios_tested}",
            f"Scenarios Passed: {self.scenarios_passed}/{self.scenarios_tested}",
            f"Pass Rate: {self.pass_rate:.1%}",
            f"",
            "Individual Results:",
        ]

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(
                f"  {status} | {result.scenario_name}: "
                f"Sharpe={result.sharpe:.2f}, DD={result.max_drawdown:.1%}"
            )
            if result.failure_reason:
                lines.append(f"         Reason: {result.failure_reason}")

        return "\n".join(lines)


class StressTester:
    """
    Stress test strategies under extreme market conditions.

    Usage:
        >>> tester = StressTester()
        >>> result = tester.test_strategy(my_strategy, "SPY")
        >>> print(result.summary())
    """

    # Historical crisis scenarios
    HISTORICAL_SCENARIOS = {
        "2008_financial_crisis": StressScenario(
            name="2008_financial_crisis",
            description="Lehman collapse and credit freeze (Sep-Dec 2008)",
            scenario_type="historical",
            period=("2008-09-01", "2008-12-31"),
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
        "2020_covid_crash": StressScenario(
            name="2020_covid_crash",
            description="COVID-19 market crash (Feb-Mar 2020)",
            scenario_type="historical",
            period=("2020-02-19", "2020-03-23"),
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
        "2022_rate_hikes": StressScenario(
            name="2022_rate_hikes",
            description="Fed rate hikes, bond-equity correlation breakdown",
            scenario_type="historical",
            period=("2022-01-01", "2022-10-31"),
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
    }

    # Synthetic shock scenarios
    SYNTHETIC_SCENARIOS = {
        "correlation_spike": StressScenario(
            name="correlation_spike",
            description="All assets correlate 0.95 (crisis behavior)",
            scenario_type="synthetic",
            transform=None,  # Will be set in __init__
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
        "volatility_3x": StressScenario(
            name="volatility_3x",
            description="Triple normal volatility",
            scenario_type="synthetic",
            transform=None,  # Will be set in __init__
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
        "gap_risk": StressScenario(
            name="gap_risk",
            description="5% overnight gaps (flash crash simulation)",
            scenario_type="synthetic",
            transform=None,  # Will be set in __init__
            min_sharpe=0.0,
            max_drawdown=0.50,
        ),
    }

    def __init__(self, data_loader: MarketDataLoader | None = None):
        """
        Initialize stress tester.

        Args:
            data_loader: Optional custom data loader
        """
        self.data_loader = data_loader or MarketDataLoader()
        self.backtest_engine = BacktestEngine()

        # Set up synthetic transforms
        self._setup_synthetic_transforms()

    def _setup_synthetic_transforms(self) -> None:
        """Set up transform functions for synthetic scenarios."""

        # Volatility 3x
        def volatility_3x(df: pd.DataFrame) -> pd.DataFrame:
            """Triple the volatility by scaling returns."""
            df_copy = df.copy()
            returns = df_copy["close"].pct_change()
            mean_return = returns.mean()

            # Scale returns to 3x volatility
            scaled_returns = (returns - mean_return) * 3.0 + mean_return

            # Reconstruct prices
            df_copy["close"] = (1 + scaled_returns).cumprod() * df["close"].iloc[0]
            df_copy["high"] = df_copy["close"] * 1.01  # Approximate
            df_copy["low"] = df_copy["close"] * 0.99
            df_copy["open"] = df_copy["close"].shift(1).fillna(df_copy["close"].iloc[0])

            return df_copy

        # Gap risk
        def gap_risk(df: pd.DataFrame) -> pd.DataFrame:
            """Add random 5% gaps to simulate flash crashes."""
            df_copy = df.copy()

            # Add gaps to 10% of days
            n_gaps = int(len(df) * 0.10)
            gap_days = np.random.choice(len(df), n_gaps, replace=False)

            for day in gap_days:
                if day > 0:
                    gap = np.random.choice([-0.05, 0.05])  # +/- 5%
                    df_copy.loc[df_copy.index[day]:, "close"] *= (1 + gap)
                    df_copy.loc[df_copy.index[day]:, "high"] *= (1 + gap)
                    df_copy.loc[df_copy.index[day]:, "low"] *= (1 + gap)
                    df_copy.loc[df_copy.index[day]:, "open"] *= (1 + gap)

            return df_copy

        # Correlation spike (note: this is simplified, real impl needs multi-asset)
        def correlation_spike(df: pd.DataFrame) -> pd.DataFrame:
            """Increase serial correlation (simplified single-asset version)."""
            df_copy = df.copy()
            returns = df_copy["close"].pct_change()

            # Add strong autocorrelation
            smoothed_returns = returns.rolling(3).mean().fillna(returns)

            # Reconstruct prices
            df_copy["close"] = (1 + smoothed_returns).cumprod() * df["close"].iloc[0]
            df_copy["high"] = df_copy["close"] * 1.01
            df_copy["low"] = df_copy["close"] * 0.99
            df_copy["open"] = df_copy["close"].shift(1).fillna(df_copy["close"].iloc[0])

            return df_copy

        # Assign transforms
        self.SYNTHETIC_SCENARIOS["volatility_3x"].transform = volatility_3x
        self.SYNTHETIC_SCENARIOS["gap_risk"].transform = gap_risk
        self.SYNTHETIC_SCENARIOS["correlation_spike"].transform = correlation_spike

    def test_strategy(
        self,
        strategy: StrategyGenome,
        symbol: str = "SPY",
        scenarios: list[str] | None = None,
    ) -> StressTestResult:
        """
        Test strategy across stress scenarios.

        Args:
            strategy: Strategy to test
            symbol: Symbol to test with (default SPY)
            scenarios: List of scenario names, or None for all

        Returns:
            StressTestResult with pass/fail for each scenario

        Example:
            >>> tester = StressTester()
            >>> result = tester.test_strategy(my_strategy, "SPY")
            >>> if result.passed:
            ...     print("Strategy is robust!")
        """
        if scenarios is None:
            # Test all scenarios
            scenario_names = (
                list(self.HISTORICAL_SCENARIOS.keys()) +
                list(self.SYNTHETIC_SCENARIOS.keys())
            )
        else:
            scenario_names = scenarios

        results = []

        for scenario_name in scenario_names:
            # Get scenario
            if scenario_name in self.HISTORICAL_SCENARIOS:
                scenario = self.HISTORICAL_SCENARIOS[scenario_name]
            elif scenario_name in self.SYNTHETIC_SCENARIOS:
                scenario = self.SYNTHETIC_SCENARIOS[scenario_name]
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")

            # Run scenario
            result = self._test_scenario(strategy, symbol, scenario)
            results.append(result)

        # Calculate overall pass rate
        scenarios_passed = sum(1 for r in results if r.passed)
        pass_rate = scenarios_passed / len(results) if results else 0.0

        return StressTestResult(
            scenarios_tested=len(results),
            scenarios_passed=scenarios_passed,
            pass_rate=pass_rate,
            results=results,
        )

    def _test_scenario(
        self,
        strategy: StrategyGenome,
        symbol: str,
        scenario: StressScenario,
    ) -> ScenarioResult:
        """Test strategy on a single scenario."""

        if scenario.scenario_type == "historical":
            # Historical replay
            if scenario.period is None:
                raise ValueError(f"Historical scenario {scenario.name} missing period")

            start, end = scenario.period
            data = self.data_loader.load(symbol, start, end)

        elif scenario.scenario_type == "synthetic":
            # Synthetic shock
            if scenario.transform is None:
                raise ValueError(f"Synthetic scenario {scenario.name} missing transform")

            # Load baseline data (last 2 years)
            from datetime import timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=2*365)

            data = self.data_loader.load(symbol, start_date, end_date)

            # Apply transform
            data.df = scenario.transform(data.df)

        else:
            raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")

        # Run backtest
        backtest_result = self.backtest_engine.run(strategy, data)

        # Check pass criteria
        passed = True
        failure_reason = None

        if backtest_result.metrics.sharpe_ratio < scenario.min_sharpe:
            passed = False
            failure_reason = f"Sharpe {backtest_result.metrics.sharpe_ratio:.2f} < {scenario.min_sharpe:.2f}"

        if backtest_result.metrics.max_drawdown > scenario.max_drawdown:
            passed = False
            if failure_reason:
                failure_reason += f"; Drawdown {backtest_result.metrics.max_drawdown:.1%} > {scenario.max_drawdown:.1%}"
            else:
                failure_reason = f"Drawdown {backtest_result.metrics.max_drawdown:.1%} > {scenario.max_drawdown:.1%}"

        return ScenarioResult(
            scenario_name=scenario.name,
            sharpe=backtest_result.metrics.sharpe_ratio,
            max_drawdown=backtest_result.metrics.max_drawdown,
            annual_return=backtest_result.metrics.annualized_return,
            passed=passed,
            failure_reason=failure_reason,
        )


# Convenience function
def stress_test_strategy(
    strategy: StrategyGenome,
    symbol: str = "SPY",
    scenarios: list[str] | None = None,
) -> StressTestResult:
    """
    Convenience function to stress test a strategy.

    Args:
        strategy: Strategy to test
        symbol: Symbol to test with
        scenarios: List of scenario names, or None for all

    Returns:
        StressTestResult

    Example:
        >>> result = stress_test_strategy(my_strategy)
        >>> print(result.summary())
    """
    tester = StressTester()
    return tester.test_strategy(strategy, symbol, scenarios)
