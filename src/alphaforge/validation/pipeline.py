"""
Full validation pipeline.

Orchestrates the complete validation process:
1. Vectorized screening with DSR
2. CPCV validation
3. PBO calculation
4. Hansen's SPA test (optional)
5. Stress testing (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from alphaforge.backtest.engine import BacktestEngine, BacktestResult, backtest_on_split
from alphaforge.data.schema import OHLCVData
from alphaforge.features.technical import TechnicalIndicators
from alphaforge.strategy.genome import StrategyGenome
from alphaforge.validation.cpcv import CombinatorialPurgedCV, CPCVResult
from alphaforge.validation.dsr import DeflatedSharpeRatio, DSRResult
from alphaforge.validation.pbo import (
    PBOResult,
    ProbabilityOfOverfitting,
    calculate_probability_of_loss,
)
from alphaforge.validation.spa import SPAResult, SPATest
from alphaforge.validation.stress import StressTestResult, StressTester

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Validation threshold configuration."""

    # DSR thresholds
    dsr_confidence: float = 0.95  # Minimum DSR for passing

    # PBO thresholds
    pbo_deploy: float = 0.05  # Maximum PBO to deploy
    pbo_auto_accept: float = 0.02  # Maximum PBO for auto-accept

    # Sharpe thresholds
    min_sharpe: float = 1.0  # Minimum Sharpe for deployment
    min_sharpe_auto: float = 1.5  # Minimum for auto-accept

    # SPA test thresholds
    spa_pvalue_threshold: float = 0.05  # P-value threshold for SPA test
    spa_bootstrap_reps: int = 1000  # Bootstrap replications

    # Stress testing thresholds
    stress_pass_rate: float = 0.80  # Minimum scenario pass rate (80%)
    stress_min_sharpe: float = 0.0  # Minimum Sharpe during stress
    stress_max_drawdown: float = 0.50  # Maximum drawdown during stress

    # Implementation shortfall (for future use)
    max_shortfall: float = 0.20  # Max backtest vs live gap

    # CPCV configuration
    cpcv_n_splits: int = 16
    cpcv_test_splits: int = 8
    cpcv_embargo_pct: float = 0.02
    cpcv_max_combinations: int | None = 1000  # Limit for speed


@dataclass
class ValidationResult:
    """Complete validation result."""

    strategy: StrategyGenome
    passed: bool
    auto_accept: bool

    # Component results
    backtest_result: BacktestResult
    dsr_result: DSRResult
    cpcv_result: CPCVResult | None
    pbo_result: PBOResult | None
    spa_result: SPAResult | None = None
    stress_result: StressTestResult | None = None

    # Metadata
    validated_at: str = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None).isoformat())
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    n_trials: int = 1  # Total strategies tested

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy.id,
            "strategy_name": self.strategy.name,
            "passed": self.passed,
            "auto_accept": self.auto_accept,
            "metrics": {
                "sharpe": self.backtest_result.metrics.sharpe_ratio,
                "dsr": self.dsr_result.dsr_pvalue,
                "pbo": self.pbo_result.pbo if self.pbo_result else None,
                "cpcv_mean_sharpe": self.cpcv_result.mean_sharpe
                if self.cpcv_result
                else None,
                "spa_pvalue": self.spa_result.pvalue if self.spa_result else None,
                "stress_pass_rate": self.stress_result.pass_rate if self.stress_result else None,
            },
            "validated_at": self.validated_at,
            "n_trials": self.n_trials,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        if self.passed and self.auto_accept:
            status = "AUTO-ACCEPT"

        lines = [
            f"Validation Result: {status}",
            f"Strategy: {self.strategy.name} ({self.strategy.id})",
            "",
            "Backtest Metrics:",
            f"  Sharpe Ratio: {self.backtest_result.metrics.sharpe_ratio:.2f}",
            f"  Annual Return: {self.backtest_result.metrics.annualized_return:.1%}",
            f"  Max Drawdown: {self.backtest_result.metrics.max_drawdown:.1%}",
            "",
            "Statistical Validation:",
            f"  DSR: {self.dsr_result.dsr_pvalue:.4f} "
            f"({'PASS' if self.dsr_result.passed else 'FAIL'})",
            f"  Expected Max Sharpe (noise): {self.dsr_result.expected_max_sharpe:.2f}",
        ]

        if self.cpcv_result:
            lines.extend(
                [
                    "",
                    f"CPCV Validation ({self.cpcv_result.n_combinations} combinations):",
                    f"  Mean OOS Sharpe: {self.cpcv_result.mean_sharpe:.2f} ± {self.cpcv_result.std_sharpe:.2f}",
                    f"  PBO: {self.cpcv_result.pbo:.4f} "
                    f"({'PASS' if self.cpcv_result.passed else 'FAIL'})",
                ]
            )

        if self.pbo_result:
            lines.extend(
                [
                    "",
                    "PBO Analysis:",
                    f"  PBO: {self.pbo_result.pbo:.4f} ({self.pbo_result.pbo*100:.1f}%)",
                    f"  Verdict: {'PASS' if self.pbo_result.passed else 'FAIL'}",
                ]
            )

        if self.spa_result:
            lines.extend(
                [
                    "",
                    "Hansen's SPA Test:",
                    f"  P-value: {self.spa_result.pvalue:.4f}",
                    f"  Benchmark: {self.spa_result.benchmark_name}",
                    f"  Outperformance: {self.spa_result.outperformance:.4%} daily",
                    f"  Verdict: {'PASS' if self.spa_result.passed else 'FAIL'}",
                ]
            )

        if self.stress_result:
            lines.extend(
                [
                    "",
                    "Stress Testing:",
                    f"  Scenarios Passed: {self.stress_result.scenarios_passed}/{self.stress_result.scenarios_tested}",
                    f"  Pass Rate: {self.stress_result.pass_rate:.1%}",
                    f"  Verdict: {'PASS' if self.stress_result.passed else 'FAIL'}",
                ]
            )

        return "\n".join(lines)


class ValidationPipeline:
    """
    Complete validation pipeline for strategy evaluation.

    Stages:
    1. Initial backtest and metrics calculation
    2. DSR screening (accounts for multiple testing)
    3. CPCV validation (optional, compute-intensive)
    4. PBO calculation
    5. Hansen's SPA test (optional)
    6. Stress testing (optional)
    """

    def __init__(
        self,
        thresholds: ValidationThresholds | None = None,
        initial_capital: float = 100000.0,
    ) -> None:
        """
        Initialize validation pipeline.

        Args:
            thresholds: Validation thresholds
            initial_capital: Starting capital for backtests
        """
        self.thresholds = thresholds or ValidationThresholds()
        self.initial_capital = initial_capital

        # Component validators
        self.dsr_calculator = DeflatedSharpeRatio(
            confidence_threshold=self.thresholds.dsr_confidence
        )
        self.cpcv_validator = CombinatorialPurgedCV(
            n_splits=self.thresholds.cpcv_n_splits,
            test_splits=self.thresholds.cpcv_test_splits,
            embargo_pct=self.thresholds.cpcv_embargo_pct,
            pbo_threshold=self.thresholds.pbo_deploy,
        )
        self.pbo_calculator = ProbabilityOfOverfitting(
            pbo_threshold=self.thresholds.pbo_deploy
        )
        self.spa_test = SPATest(
            pvalue_threshold=self.thresholds.spa_pvalue_threshold,
            bootstrap_reps=self.thresholds.spa_bootstrap_reps,
        )
        self.stress_tester = StressTester()

        # Backtest engine
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital)

    def validate(
        self,
        strategy: StrategyGenome,
        data: OHLCVData,
        n_trials: int = 1,
        run_cpcv: bool = True,
        run_spa: bool = False,
        run_stress: bool = False,
        benchmark_returns: pd.Series | None = None,
        benchmark_name: str = "SPY",
        stress_scenarios: list[str] | None = None,
    ) -> ValidationResult:
        """
        Run full validation pipeline.

        Args:
            strategy: Strategy to validate
            data: Market data
            n_trials: Total number of strategies tested (for DSR)
            run_cpcv: Whether to run CPCV (slower but more rigorous)
            run_spa: Whether to run Hansen's SPA test
            run_stress: Whether to run stress testing
            benchmark_returns: Returns for SPA test (if run_spa=True)
            benchmark_name: Name of benchmark for SPA test
            stress_scenarios: List of scenario names to run (None = all)

        Returns:
            ValidationResult
        """
        logger.info(f"Validating strategy: {strategy.name} ({strategy.id})")

        # Stage 1: Initial backtest
        logger.info("Stage 1: Running backtest...")
        backtest_result = self.backtest_engine.run(strategy, data)

        # Stage 2: DSR screening
        logger.info("Stage 2: Calculating DSR...")
        dsr_result = self.dsr_calculator.calculate_from_returns(
            backtest_result.returns.values,
            n_trials=n_trials,
        )

        # Stage 3: CPCV (if requested)
        cpcv_result = None
        if run_cpcv:
            logger.info("Stage 3: Running CPCV...")
            cpcv_result = self._run_cpcv(strategy, data)

        # Stage 4: PBO calculation
        pbo_result = None
        if cpcv_result:
            logger.info("Stage 4: Calculating Probability of Loss (single-strategy PBO proxy)...")
            pbo_result = calculate_probability_of_loss(
                cpcv_result.sharpe_distribution,
                threshold=self.thresholds.pbo_deploy,
            )

        # Stage 5: SPA test (if requested)
        spa_result = None
        if run_spa:
            logger.info("Stage 5: Running Hansen's SPA test...")
            spa_result = self._run_spa(
                backtest_result.returns,
                benchmark_returns,
                benchmark_name,
            )

        # Stage 6: Stress testing (if requested)
        stress_result = None
        if run_stress:
            logger.info("Stage 6: Running stress tests...")
            stress_result = self._run_stress(strategy, data.symbol, stress_scenarios)

        # Determine pass/fail
        passed = self._evaluate_results(
            backtest_result, dsr_result, cpcv_result, pbo_result, spa_result, stress_result
        )

        # Determine auto-accept
        auto_accept = self._check_auto_accept(
            backtest_result, dsr_result, cpcv_result, pbo_result
        )

        return ValidationResult(
            strategy=strategy,
            passed=passed,
            auto_accept=auto_accept,
            backtest_result=backtest_result,
            dsr_result=dsr_result,
            cpcv_result=cpcv_result,
            pbo_result=pbo_result,
            spa_result=spa_result,
            stress_result=stress_result,
            thresholds=self.thresholds,
            n_trials=n_trials,
        )

    def _run_cpcv(
        self,
        strategy: StrategyGenome,
        data: OHLCVData,
    ) -> CPCVResult:
        """Run CPCV validation."""
        # Prepare features once
        df = TechnicalIndicators.compute_all(data.df)

        def backtest_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
            """Backtest function for CPCV."""
            result = backtest_on_split(strategy, train_df, test_df, data.symbol)
            return result["sharpe"]

        return self.cpcv_validator.validate(
            df,
            backtest_fn,
            max_combinations=self.thresholds.cpcv_max_combinations,
        )

    def _run_spa(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None,
        benchmark_name: str,
    ) -> SPAResult:
        """Run Hansen's SPA test."""
        if benchmark_returns is None:
            raise ValueError("benchmark_returns required for SPA test")

        return self.spa_test.test(
            strategy_returns,
            benchmark_returns,
            benchmark_name,
        )

    def _run_stress(
        self,
        strategy: StrategyGenome,
        symbol: str,
        scenarios: list[str] | None = None,
    ) -> StressTestResult:
        """Run stress testing."""
        return self.stress_tester.test_strategy(strategy, symbol, scenarios=scenarios)

    def _evaluate_results(
        self,
        backtest: BacktestResult,
        dsr: DSRResult,
        cpcv: CPCVResult | None,
        pbo: PBOResult | None,
        spa: SPAResult | None,
        stress: StressTestResult | None,
    ) -> bool:
        """Evaluate if strategy passes all criteria."""
        # Must pass DSR
        if not dsr.passed:
            logger.info(f"Failed DSR: {dsr.dsr_pvalue:.4f} < {self.thresholds.dsr_confidence}")
            return False

        # Must meet minimum Sharpe
        if backtest.metrics.sharpe_ratio < self.thresholds.min_sharpe:
            logger.info(
                f"Failed Sharpe: {backtest.metrics.sharpe_ratio:.2f} < {self.thresholds.min_sharpe}"
            )
            return False

        # If CPCV was run, must pass
        if cpcv and not cpcv.passed:
            logger.info(f"Failed CPCV PBO: {cpcv.pbo:.4f} > {self.thresholds.pbo_deploy}")
            return False

        # If PBO was calculated, must pass
        if pbo and not pbo.passed:
            logger.info(f"Failed PBO: {pbo.pbo:.4f} > {self.thresholds.pbo_deploy}")
            return False

        # If SPA was run, must pass
        if spa and not spa.passed:
            logger.info(f"Failed SPA: p-value {spa.pvalue:.4f} >= {self.thresholds.spa_pvalue_threshold}")
            return False

        # If stress testing was run, must pass
        if stress and not stress.passed:
            logger.info(f"Failed stress testing: {stress.pass_rate:.1%} < {self.thresholds.stress_pass_rate}")
            return False

        return True

    def _check_auto_accept(
        self,
        backtest: BacktestResult,
        dsr: DSRResult,
        cpcv: CPCVResult | None,
        pbo: PBOResult | None,
    ) -> bool:
        """Check if strategy meets auto-accept criteria."""
        # Higher DSR threshold
        if dsr.dsr_pvalue < 0.98:  # Stricter than pass threshold
            return False

        # Higher Sharpe threshold
        if backtest.metrics.sharpe_ratio < self.thresholds.min_sharpe_auto:
            return False

        # Lower PBO threshold
        if pbo and pbo.pbo > self.thresholds.pbo_auto_accept:
            return False

        return True

    def validate_multiple(
        self,
        strategies: list[StrategyGenome],
        data: OHLCVData,
        run_cpcv: bool = False,  # Default off for speed with multiple
    ) -> list[ValidationResult]:
        """
        Validate multiple strategies.

        DSR is calculated with n_trials = total number of strategies.

        Args:
            strategies: List of strategies to validate
            data: Market data
            run_cpcv: Whether to run CPCV for each

        Returns:
            List of ValidationResult
        """
        n_trials = len(strategies)
        results = []

        for i, strategy in enumerate(strategies):
            logger.info(f"Validating strategy {i+1}/{n_trials}: {strategy.name}")
            result = self.validate(strategy, data, n_trials=n_trials, run_cpcv=run_cpcv)
            results.append(result)

        # Summary
        n_passed = sum(1 for r in results if r.passed)
        n_auto = sum(1 for r in results if r.auto_accept)
        logger.info(f"Validation complete: {n_passed}/{n_trials} passed, {n_auto} auto-accept")

        return results


def quick_validate(
    strategy: StrategyGenome,
    data: OHLCVData,
    n_trials: int = 1,
) -> tuple[bool, float, float]:
    """
    Quick validation returning (passed, sharpe, dsr).

    Useful for screening large numbers of strategies.

    Args:
        strategy: Strategy to validate
        data: Market data
        n_trials: Number of strategies tested

    Returns:
        Tuple of (passed, sharpe_ratio, dsr_probability)
    """
    pipeline = ValidationPipeline()
    result = pipeline.validate(strategy, data, n_trials=n_trials, run_cpcv=False)

    return (
        result.passed,
        result.backtest_result.metrics.sharpe_ratio,
        result.dsr_result.dsr_pvalue,
    )
