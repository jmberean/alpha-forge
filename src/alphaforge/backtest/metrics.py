"""
Performance metrics for backtesting.

All metrics are computed on returns series to ensure consistency.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """
    Complete performance metrics for a strategy.

    All annualized metrics assume 252 trading days per year.
    """

    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int  # days
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Distribution metrics
    skewness: float
    kurtosis: float

    # Trade metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Time metrics
    trading_days: int
    years: float

    @classmethod
    def from_returns(
        cls,
        returns: pd.Series,
        positions: pd.Series | None = None,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252,
    ) -> "PerformanceMetrics":
        """
        Compute all metrics from a returns series.

        Args:
            returns: Daily returns series
            positions: Position series (for trade counting)
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year

        Returns:
            PerformanceMetrics instance
        """
        # Clean returns
        returns = returns.dropna()
        if len(returns) == 0:
            raise ValueError("Empty returns series")

        # Time metrics
        trading_days = len(returns)
        years = trading_days / trading_days_per_year

        # Return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = returns.std() * np.sqrt(trading_days_per_year)

        # Sharpe ratio
        excess_returns = returns - (risk_free_rate / trading_days_per_year)
        if volatility > 0:
            sharpe_ratio = (
                excess_returns.mean() / returns.std() * np.sqrt(trading_days_per_year)
            )
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            downside_vol = downside_returns.std() * np.sqrt(trading_days_per_year)
            sortino_ratio = annualized_return / downside_vol
        else:
            sortino_ratio = 0.0

        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Max drawdown duration
        is_underwater = drawdown < 0
        underwater_periods = is_underwater.groupby(
            (~is_underwater).cumsum()
        ).cumsum()
        max_drawdown_duration = int(underwater_periods.max()) if len(underwater_periods) > 0 else 0

        # VaR and CVaR
        var_95 = abs(np.percentile(returns, 5))
        cvar_95 = abs(returns[returns <= -var_95].mean()) if var_95 > 0 else 0

        # Distribution metrics
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))

        # Trade metrics (if positions provided)
        if positions is not None:
            positions = positions.dropna()
            # Detect trade changes
            position_changes = positions.diff().fillna(0)
            entries = (position_changes > 0).sum()
            num_trades = int(entries)

            # Calculate trade returns
            trade_returns = _calculate_trade_returns(returns, positions)

            if len(trade_returns) > 0:
                winning_trades = trade_returns[trade_returns > 0]
                losing_trades = trade_returns[trade_returns < 0]

                win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0

                total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0

                profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
                avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            else:
                win_rate = 0.0
                profit_factor = 0.0
                avg_win = 0.0
                avg_loss = 0.0
        else:
            num_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0

        return cls(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trading_days=trading_days,
            years=years,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "trading_days": self.trading_days,
            "years": self.years,
        }

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  sharpe={self.sharpe_ratio:.2f}, "
            f"  return={self.annualized_return:.1%}, "
            f"  volatility={self.volatility:.1%}, "
            f"  max_dd={self.max_drawdown:.1%}, "
            f"  trades={self.num_trades}, "
            f"  win_rate={self.win_rate:.1%}\n"
            f")"
        )


def _calculate_trade_returns(
    returns: pd.Series, positions: pd.Series
) -> pd.Series:
    """
    Calculate returns for each completed trade.

    A trade starts when position goes from 0 to 1 (or -1)
    and ends when it returns to 0.
    """
    trade_returns = []
    in_trade = False
    trade_cumulative = 0.0

    for i in range(len(positions)):
        pos = positions.iloc[i]
        ret = returns.iloc[i]

        if not in_trade and pos != 0:
            # Start new trade
            in_trade = True
            trade_cumulative = ret

        elif in_trade and pos != 0:
            # Continue trade
            trade_cumulative = (1 + trade_cumulative) * (1 + ret) - 1

        elif in_trade and pos == 0:
            # End trade
            trade_returns.append(trade_cumulative)
            in_trade = False
            trade_cumulative = 0.0

    # If still in trade at end
    if in_trade:
        trade_returns.append(trade_cumulative)

    return pd.Series(trade_returns)


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """
    Calculate maximum drawdown and duration.

    Args:
        returns: Returns series

    Returns:
        Tuple of (max_drawdown, max_drawdown_duration_days)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = abs(drawdown.min())

    # Calculate duration
    is_underwater = drawdown < 0
    underwater_periods = is_underwater.groupby((~is_underwater).cumsum()).cumsum()
    max_duration = int(underwater_periods.max()) if len(underwater_periods) > 0 else 0

    return max_dd, max_duration


def calculate_calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Returns series
        periods_per_year: Periods per year

    Returns:
        Calmar ratio
    """
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    max_dd, _ = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    return ann_return / max_dd


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information Ratio.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Periods per year

    Returns:
        Information Ratio
    """
    # Align series
    returns, benchmark_returns = returns.align(benchmark_returns, join="inner")

    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0

    active_return_ann = active_returns.mean() * periods_per_year
    return active_return_ann / tracking_error
