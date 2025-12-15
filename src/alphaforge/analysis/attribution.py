"""
Performance attribution analysis.

Breaks down returns by regime, time period, and other factors to understand
where strategy performance comes from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.validation.regime import RegimeDetector, RegimeType


@dataclass
class RegimeAttribution:
    """Performance attribution by market regime."""

    regime: RegimeType
    n_periods: int
    total_return: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


@dataclass
class PerformanceAttribution:
    """
    Comprehensive performance attribution.

    Breaks down returns by different factors to understand
    sources of performance.
    """

    # Regime-based attribution
    regime_breakdown: list[RegimeAttribution]

    # Time-based metrics
    monthly_returns: pd.Series
    yearly_returns: pd.Series

    # Rolling metrics
    rolling_sharpe_12m: pd.Series
    rolling_dd: pd.Series

    # Contribution analysis
    best_month: tuple[str, float]  # (date, return)
    worst_month: tuple[str, float]
    best_year: tuple[int, float]
    worst_year: tuple[int, float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime_breakdown": [
                {
                    "regime": r.regime.value,
                    "n_periods": r.n_periods,
                    "total_return": r.total_return,
                    "avg_return": r.avg_return,
                    "sharpe_ratio": r.sharpe_ratio,
                    "max_drawdown": r.max_drawdown,
                    "win_rate": r.win_rate,
                }
                for r in self.regime_breakdown
            ],
            "best_month": {
                "date": self.best_month[0],
                "return": self.best_month[1],
            },
            "worst_month": {
                "date": self.worst_month[0],
                "return": self.worst_month[1],
            },
            "best_year": {
                "year": self.best_year[0],
                "return": self.best_year[1],
            },
            "worst_year": {
                "year": self.worst_year[0],
                "return": self.worst_year[1],
            },
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Performance Attribution:\n"]

        lines.append("Regime Breakdown:")
        for r in self.regime_breakdown:
            lines.append(
                f"  {r.regime.value:12s}: Return={r.avg_return:>7.1%}, Sharpe={r.sharpe_ratio:>5.2f}, "
                f"DD={r.max_drawdown:>6.1%}, WinRate={r.win_rate:>5.1%}"
            )

        lines.append(f"\nBest Month:  {self.best_month[0]} ({self.best_month[1]:.1%})")
        lines.append(f"Worst Month: {self.worst_month[0]} ({self.worst_month[1]:.1%})")
        lines.append(f"Best Year:   {self.best_year[0]} ({self.best_year[1]:.1%})")
        lines.append(f"Worst Year:  {self.worst_year[0]} ({self.worst_year[1]:.1%})")

        return "\n".join(lines)


class PerformanceAnalyzer:
    """Analyze strategy performance with attribution."""

    @staticmethod
    def analyze(
        returns: pd.Series,
        prices: pd.Series,
        positions: pd.Series | None = None,
    ) -> PerformanceAttribution:
        """
        Perform comprehensive performance attribution.

        Args:
            returns: Daily returns series
            prices: Price series (for regime detection)
            positions: Optional position series

        Returns:
            PerformanceAttribution with breakdown by regime and time
        """
        # Regime-based attribution
        regime_breakdown = PerformanceAnalyzer._regime_attribution(returns, prices)

        # Time-based metrics
        monthly_returns = PerformanceAnalyzer._monthly_returns(returns)
        yearly_returns = PerformanceAnalyzer._yearly_returns(returns)

        # Rolling metrics
        rolling_sharpe_12m = PerformanceAnalyzer._rolling_sharpe(returns, window=252)
        rolling_dd = PerformanceAnalyzer._rolling_drawdown(returns)

        # Best/worst periods
        best_month_idx = monthly_returns.idxmax()
        worst_month_idx = monthly_returns.idxmin()
        best_year_idx = yearly_returns.idxmax()
        worst_year_idx = yearly_returns.idxmin()

        best_month = (
            str(best_month_idx.strftime("%Y-%m")) if not pd.isna(best_month_idx) else "N/A",
            monthly_returns[best_month_idx] if not pd.isna(best_month_idx) else 0.0,
        )
        worst_month = (
            str(worst_month_idx.strftime("%Y-%m")) if not pd.isna(worst_month_idx) else "N/A",
            monthly_returns[worst_month_idx] if not pd.isna(worst_month_idx) else 0.0,
        )
        best_year = (
            int(best_year_idx) if not pd.isna(best_year_idx) else 0,
            yearly_returns[best_year_idx] if not pd.isna(best_year_idx) else 0.0,
        )
        worst_year = (
            int(worst_year_idx) if not pd.isna(worst_year_idx) else 0,
            yearly_returns[worst_year_idx] if not pd.isna(worst_year_idx) else 0.0,
        )

        return PerformanceAttribution(
            regime_breakdown=regime_breakdown,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            rolling_sharpe_12m=rolling_sharpe_12m,
            rolling_dd=rolling_dd,
            best_month=best_month,
            worst_month=worst_month,
            best_year=best_year,
            worst_year=worst_year,
        )

    @staticmethod
    def _regime_attribution(
        returns: pd.Series, prices: pd.Series
    ) -> list[RegimeAttribution]:
        """Calculate performance attribution by regime."""
        detector = RegimeDetector()

        # Detect regime for each period
        regimes = []
        for date in prices.index:
            detection = detector.detect(prices, as_of_date=date)
            regimes.append(detection.regime)

        regime_series = pd.Series(regimes, index=prices.index)

        # Calculate metrics for each regime
        attributions = []

        for regime_type in RegimeType:
            regime_mask = regime_series == regime_type
            regime_returns = returns[regime_mask]

            if len(regime_returns) == 0:
                continue

            # Calculate metrics
            n_periods = len(regime_returns)
            total_return = (1 + regime_returns).prod() - 1
            avg_return = regime_returns.mean()

            # Sharpe ratio
            if regime_returns.std() > 0:
                sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
            else:
                sharpe = 0.0

            # Max drawdown
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()

            # Win rate
            win_rate = (regime_returns > 0).sum() / len(regime_returns)

            attribution = RegimeAttribution(
                regime=regime_type,
                n_periods=n_periods,
                total_return=total_return,
                avg_return=avg_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
            )

            attributions.append(attribution)

        return attributions

    @staticmethod
    def _monthly_returns(returns: pd.Series) -> pd.Series:
        """Calculate monthly returns."""
        if len(returns) == 0:
            return pd.Series(dtype=float)

        # Group by year-month
        monthly = (1 + returns).groupby(pd.Grouper(freq="ME")).prod() - 1
        return monthly

    @staticmethod
    def _yearly_returns(returns: pd.Series) -> pd.Series:
        """Calculate yearly returns."""
        if len(returns) == 0:
            return pd.Series(dtype=float)

        # Group by year
        yearly = (1 + returns).groupby(returns.index.year).prod() - 1
        return yearly

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < window:
            return pd.Series(dtype=float)

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return sharpe

    @staticmethod
    def _rolling_drawdown(returns: pd.Series) -> pd.Series:
        """Calculate rolling drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns


def analyze_performance(
    returns: pd.Series,
    prices: pd.Series,
) -> PerformanceAttribution:
    """
    Convenience function for performance attribution.

    Args:
        returns: Daily returns series
        prices: Price series

    Returns:
        PerformanceAttribution

    Example:
        >>> attribution = analyze_performance(returns, prices)
        >>> print(attribution.summary())
    """
    return PerformanceAnalyzer.analyze(returns, prices)
