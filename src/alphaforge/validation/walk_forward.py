"""
Walk-forward analysis for strategy validation.

Simulates realistic parameter optimization by testing on out-of-sample data
after optimizing on in-sample windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.data.schema import OHLCVData
from alphaforge.optimization.base import Optimizer, ParameterSpace
from alphaforge.strategy.genome import StrategyGenome


@dataclass
class WalkForwardPeriod:
    """Single walk-forward period (in-sample + out-of-sample)."""

    # Period identification
    period_id: int
    start_date: datetime
    end_date: datetime

    # In-sample (optimization) period
    is_start: datetime
    is_end: datetime
    is_result: BacktestResult | None = None
    is_params: dict | None = None

    # Out-of-sample (testing) period
    oos_start: datetime
    oos_end: datetime
    oos_result: BacktestResult | None = None

    # Metrics
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    degradation: float = 0.0  # (IS - OOS) / IS


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    # All periods
    periods: list[WalkForwardPeriod] = field(default_factory=list)

    # Aggregate metrics
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_degradation: float = 0.0

    # Walk-forward efficiency (OOS/IS performance ratio)
    wf_efficiency: float = 0.0

    # Combined out-of-sample results
    combined_oos_sharpe: float = 0.0
    combined_oos_return: float = 0.0
    combined_oos_drawdown: float = 0.0

    # Metadata
    n_periods: int = 0
    window_type: str = ""  # 'rolling' or 'anchored'

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n_periods": self.n_periods,
            "window_type": self.window_type,
            "avg_is_sharpe": self.avg_is_sharpe,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "avg_degradation": self.avg_degradation,
            "wf_efficiency": self.wf_efficiency,
            "combined_oos_sharpe": self.combined_oos_sharpe,
            "combined_oos_return": self.combined_oos_return,
            "combined_oos_drawdown": self.combined_oos_drawdown,
            "periods": [
                {
                    "period_id": p.period_id,
                    "is_sharpe": p.is_sharpe,
                    "oos_sharpe": p.oos_sharpe,
                    "degradation": p.degradation,
                }
                for p in self.periods
            ],
        }

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Walk-Forward Analysis ({self.window_type}):\n"
            f"Periods: {self.n_periods}\n"
            f"Avg IS Sharpe: {self.avg_is_sharpe:.2f}\n"
            f"Avg OOS Sharpe: {self.avg_oos_sharpe:.2f}\n"
            f"WF Efficiency: {self.wf_efficiency:.1%}\n"
            f"Combined OOS Sharpe: {self.combined_oos_sharpe:.2f}\n"
            f"Combined OOS Return: {self.combined_oos_return:.1%}\n"
            f"Combined OOS Drawdown: {self.combined_oos_drawdown:.1%}"
        )


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Simulates realistic parameter optimization by repeatedly:
    1. Optimizing parameters on an in-sample window
    2. Testing those parameters on the next out-of-sample window
    3. Rolling forward and repeating
    """

    def __init__(
        self,
        is_window: int = 252,  # In-sample days
        oos_window: int = 63,  # Out-of-sample days
        window_type: str = "rolling",  # 'rolling' or 'anchored'
        min_periods: int = 100,  # Minimum data points required
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            is_window: In-sample window size (days)
            oos_window: Out-of-sample window size (days)
            window_type: 'rolling' (fixed window) or 'anchored' (expanding window)
            min_periods: Minimum data points required for optimization
        """
        self.is_window = is_window
        self.oos_window = oos_window
        self.window_type = window_type
        self.min_periods = min_periods

        if window_type not in ["rolling", "anchored"]:
            raise ValueError("window_type must be 'rolling' or 'anchored'")

    def analyze(
        self,
        strategy_template: StrategyGenome,
        data: OHLCVData,
        optimizer: Optimizer,
        n_trials: int = 50,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            strategy_template: Strategy template with parameters to optimize
            data: Full dataset
            optimizer: Optimizer to use for parameter search
            n_trials: Number of optimization trials per period

        Returns:
            WalkForwardResult with all periods and metrics
        """
        # Generate walk-forward periods
        periods = self._generate_periods(data)

        # Run optimization and testing for each period
        engine = BacktestEngine()

        for period in periods:
            # Extract in-sample data
            is_data = self._extract_period(data, period.is_start, period.is_end)

            # Optimize on in-sample data
            opt_result = optimizer.optimize(n_trials=n_trials)
            period.is_params = opt_result.best_params

            # Backtest with optimized params on in-sample
            is_strategy = self._apply_params(strategy_template, opt_result.best_params)
            period.is_result = engine.run(is_strategy, is_data)
            period.is_sharpe = period.is_result.metrics.sharpe_ratio

            # Extract out-of-sample data
            oos_data = self._extract_period(data, period.oos_start, period.oos_end)

            # Test on out-of-sample data (no optimization)
            period.oos_result = engine.run(is_strategy, oos_data)
            period.oos_sharpe = period.oos_result.metrics.sharpe_ratio

            # Calculate degradation
            if period.is_sharpe != 0:
                period.degradation = (
                    period.is_sharpe - period.oos_sharpe
                ) / abs(period.is_sharpe)
            else:
                period.degradation = 0.0

        # Calculate aggregate metrics
        result = self._aggregate_results(periods)

        return result

    def _generate_periods(self, data: OHLCVData) -> list[WalkForwardPeriod]:
        """Generate walk-forward periods from data."""
        periods = []
        dates = data.df.index
        total_days = len(dates)

        period_id = 0
        current_idx = 0

        while current_idx + self.is_window + self.oos_window <= total_days:
            # In-sample period
            if self.window_type == "anchored":
                is_start_idx = 0  # Always start from beginning
            else:
                is_start_idx = current_idx

            is_end_idx = current_idx + self.is_window

            # Out-of-sample period
            oos_start_idx = is_end_idx
            oos_end_idx = min(oos_start_idx + self.oos_window, total_days)

            period = WalkForwardPeriod(
                period_id=period_id,
                start_date=dates[is_start_idx],
                end_date=dates[oos_end_idx - 1],
                is_start=dates[is_start_idx],
                is_end=dates[is_end_idx - 1],
                oos_start=dates[oos_start_idx],
                oos_end=dates[oos_end_idx - 1],
            )

            periods.append(period)

            # Move forward by OOS window
            current_idx += self.oos_window
            period_id += 1

        return periods

    def _extract_period(
        self, data: OHLCVData, start: datetime, end: datetime
    ) -> OHLCVData:
        """Extract data for a specific period."""
        mask = (data.df.index >= start) & (data.df.index <= end)
        period_df = data.df[mask].copy()

        return OHLCVData(
            df=period_df,
            symbol=data.symbol,
            start_date=start,
            end_date=end,
        )

    def _apply_params(
        self, strategy: StrategyGenome, params: dict
    ) -> StrategyGenome:
        """
        Apply optimized parameters to strategy.

        For now, this is a placeholder. In practice, you would:
        1. Clone the strategy
        2. Update its parameters based on the params dict
        3. Return the modified strategy

        Args:
            strategy: Original strategy template
            params: Parameter values to apply

        Returns:
            Strategy with updated parameters
        """
        # For now, just return the original strategy
        # TODO: Implement parameter application logic
        return strategy

    def _aggregate_results(
        self, periods: list[WalkForwardPeriod]
    ) -> WalkForwardResult:
        """Aggregate results from all periods."""
        n_periods = len(periods)

        if n_periods == 0:
            return WalkForwardResult(
                n_periods=0,
                window_type=self.window_type,
            )

        # Calculate averages
        avg_is_sharpe = np.mean([p.is_sharpe for p in periods])
        avg_oos_sharpe = np.mean([p.oos_sharpe for p in periods])
        avg_degradation = np.mean([p.degradation for p in periods])

        # Walk-forward efficiency
        if avg_is_sharpe != 0:
            wf_efficiency = avg_oos_sharpe / avg_is_sharpe
        else:
            wf_efficiency = 0.0

        # Combine OOS results
        oos_returns = []
        oos_equity = 1.0

        for period in periods:
            if period.oos_result is not None:
                period_return = period.oos_result.metrics.annualized_return
                # Simple approximation of compounding
                oos_equity *= 1 + (period_return * (self.oos_window / 252))
                oos_returns.append(period_return)

        combined_oos_return = oos_equity - 1.0
        combined_oos_sharpe = avg_oos_sharpe  # Simplified
        combined_oos_drawdown = np.mean([
            p.oos_result.metrics.max_drawdown
            for p in periods
            if p.oos_result is not None
        ])

        return WalkForwardResult(
            periods=periods,
            n_periods=n_periods,
            window_type=self.window_type,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_degradation=avg_degradation,
            wf_efficiency=wf_efficiency,
            combined_oos_sharpe=combined_oos_sharpe,
            combined_oos_return=combined_oos_return,
            combined_oos_drawdown=combined_oos_drawdown,
        )


def walk_forward_analysis(
    strategy: StrategyGenome,
    data: OHLCVData,
    optimizer: Optimizer,
    is_window: int = 252,
    oos_window: int = 63,
    n_trials: int = 50,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward analysis.

    Args:
        strategy: Strategy template
        data: Full dataset
        optimizer: Optimizer for parameter search
        is_window: In-sample window size (days)
        oos_window: Out-of-sample window size (days)
        n_trials: Optimization trials per period

    Returns:
        WalkForwardResult

    Example:
        >>> from alphaforge.optimization import GridSearchOptimizer
        >>> optimizer = GridSearchOptimizer(param_space, objective)
        >>> result = walk_forward_analysis(strategy, data, optimizer)
        >>> print(f"WF Efficiency: {result.wf_efficiency:.1%}")
    """
    analyzer = WalkForwardAnalyzer(
        is_window=is_window,
        oos_window=oos_window,
    )

    return analyzer.analyze(strategy, data, optimizer, n_trials)
