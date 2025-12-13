"""
Trade analysis and statistics.

Analyzes individual trades from backtests to provide detailed performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Individual trade information."""

    # Entry
    entry_date: datetime
    entry_price: float
    entry_signal: str  # 'long' or 'short'

    # Exit
    exit_date: datetime
    exit_price: float
    exit_signal: str  # 'stop', 'target', 'time', 'signal'

    # Performance
    pnl: float  # Profit/loss in dollars
    pnl_pct: float  # Profit/loss as percentage
    return_: float  # Simple return (pnl_pct)

    # Trade details
    size: float  # Position size (shares/contracts)
    duration: int  # Holding period in days

    # Execution costs
    commission: float = 0.0
    slippage: float = 0.0

    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entry_date": self.entry_date.isoformat(),
            "entry_price": self.entry_price,
            "entry_signal": self.entry_signal,
            "exit_date": self.exit_date.isoformat(),
            "exit_price": self.exit_price,
            "exit_signal": self.exit_signal,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "return": self.return_,
            "size": self.size,
            "duration": self.duration,
            "commission": self.commission,
            "slippage": self.slippage,
        }


@dataclass
class TradeStats:
    """Comprehensive trade statistics."""

    # Basic counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    # Win/loss metrics
    win_rate: float  # Percentage of winning trades
    loss_rate: float  # Percentage of losing trades

    # Profit metrics
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Profit factor (gross profits / gross losses)
    profit_factor: float

    # Expectancy (avg profit per trade)
    expectancy: float

    # Duration metrics
    avg_duration: float  # Average holding period (days)
    avg_win_duration: float
    avg_loss_duration: float

    # Consecutive stats
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Risk/reward
    avg_win_loss_ratio: float  # Average win / average loss

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_duration": self.avg_duration,
            "avg_win_duration": self.avg_win_duration,
            "avg_loss_duration": self.avg_loss_duration,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
        }


class TradeAnalyzer:
    """Analyze trades to compute comprehensive statistics."""

    @staticmethod
    def analyze(trades: list[Trade]) -> TradeStats:
        """
        Analyze a list of trades to compute statistics.

        Args:
            trades: List of Trade objects

        Returns:
            TradeStats with comprehensive metrics
        """
        if not trades:
            return TradeAnalyzer._empty_stats()

        # Separate winners and losers
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]
        breakeven = [t for t in trades if t.pnl == 0]

        # Basic counts
        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        breakeven_trades = len(breakeven)

        # Win/loss rates
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0.0

        # PnL metrics
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        avg_win = sum(t.pnl for t in winners) / winning_trades if winners else 0.0
        avg_loss = sum(t.pnl for t in losers) / losing_trades if losers else 0.0

        largest_win = max((t.pnl for t in trades), default=0.0)
        largest_loss = min((t.pnl for t in trades), default=0.0)

        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = avg_pnl  # Same as average PnL per trade

        # Duration metrics
        avg_duration = (
            sum(t.duration for t in trades) / total_trades if total_trades > 0 else 0.0
        )
        avg_win_duration = (
            sum(t.duration for t in winners) / winning_trades if winners else 0.0
        )
        avg_loss_duration = (
            sum(t.duration for t in losers) / losing_trades if losers else 0.0
        )

        # Consecutive wins/losses
        max_consecutive_wins = TradeAnalyzer._max_consecutive(trades, lambda t: t.pnl > 0)
        max_consecutive_losses = TradeAnalyzer._max_consecutive(
            trades, lambda t: t.pnl < 0
        )

        # Win/loss ratio
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=win_rate,
            loss_rate=loss_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_duration=avg_duration,
            avg_win_duration=avg_win_duration,
            avg_loss_duration=avg_loss_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_win_loss_ratio=avg_win_loss_ratio,
        )

    @staticmethod
    def _empty_stats() -> TradeStats:
        """Return empty stats for when there are no trades."""
        return TradeStats(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            loss_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_duration=0.0,
            avg_win_duration=0.0,
            avg_loss_duration=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            avg_win_loss_ratio=0.0,
        )

    @staticmethod
    def _max_consecutive(trades: list[Trade], condition) -> int:
        """Calculate maximum consecutive trades meeting condition."""
        if not trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if condition(trade):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    @staticmethod
    def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
        """
        Convert trades to DataFrame for analysis.

        Args:
            trades: List of Trade objects

        Returns:
            DataFrame with one row per trade
        """
        if not trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in trades])

    @staticmethod
    def extract_trades_from_positions(
        positions: pd.Series, prices: pd.Series, initial_capital: float = 100000.0
    ) -> list[Trade]:
        """
        Extract individual trades from position series.

        Args:
            positions: Series of position sizes (-1, 0, 1)
            prices: Series of prices
            initial_capital: Initial capital for position sizing

        Returns:
            List of Trade objects
        """
        trades = []
        entry_idx = None
        entry_position = None

        for i in range(len(positions)):
            current_pos = positions.iloc[i]
            prev_pos = positions.iloc[i - 1] if i > 0 else 0

            # Position change detected
            if current_pos != prev_pos:
                # Close existing position
                if entry_idx is not None and prev_pos != 0:
                    exit_date = positions.index[i]
                    exit_price = prices.iloc[i]

                    # Calculate PnL
                    if entry_position > 0:  # Long trade
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:  # Short trade
                        pnl_pct = (entry_price - exit_price) / entry_price

                    # Position size (simplified: use 1 share for now)
                    size = 1.0
                    pnl = pnl_pct * size * entry_price

                    # Duration
                    duration = (exit_date - entry_date).days

                    trade = Trade(
                        entry_date=entry_date,
                        entry_price=entry_price,
                        entry_signal="long" if entry_position > 0 else "short",
                        exit_date=exit_date,
                        exit_price=exit_price,
                        exit_signal="signal",  # Simplified
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        return_=pnl_pct,
                        size=size,
                        duration=max(1, duration),  # At least 1 day
                    )
                    trades.append(trade)

                # Open new position
                if current_pos != 0:
                    entry_idx = i
                    entry_date = positions.index[i]
                    entry_price = prices.iloc[i]
                    entry_position = current_pos
                else:
                    entry_idx = None

        # Close final position if still open
        if entry_idx is not None and entry_position != 0:
            exit_date = positions.index[-1]
            exit_price = prices.iloc[-1]

            if entry_position > 0:  # Long trade
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # Short trade
                pnl_pct = (entry_price - exit_price) / entry_price

            size = 1.0
            pnl = pnl_pct * size * entry_price
            duration = (exit_date - entry_date).days

            trade = Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                entry_signal="long" if entry_position > 0 else "short",
                exit_date=exit_date,
                exit_price=exit_price,
                exit_signal="eod",  # End of data
                pnl=pnl,
                pnl_pct=pnl_pct,
                return_=pnl_pct,
                size=size,
                duration=max(1, duration),
            )
            trades.append(trade)

        return trades


def analyze_trades(trades: list[Trade]) -> TradeStats:
    """
    Convenience function for trade analysis.

    Args:
        trades: List of Trade objects

    Returns:
        TradeStats with comprehensive metrics

    Example:
        >>> stats = analyze_trades(trades)
        >>> print(f"Win Rate: {stats.win_rate:.1%}")
        >>> print(f"Profit Factor: {stats.profit_factor:.2f}")
    """
    return TradeAnalyzer.analyze(trades)
