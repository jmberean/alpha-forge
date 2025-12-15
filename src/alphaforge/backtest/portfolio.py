"""
Portfolio simulation for backtesting.

Tracks positions, cash, and portfolio value over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from alphaforge.strategy.genome import PositionSizing


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    side: str  # 'long' or 'short'
    pnl: float
    return_pct: float
    holding_days: int


@dataclass
class Position:
    """Current open position."""

    symbol: str
    shares: float
    entry_price: float
    entry_date: datetime
    side: str


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""

    date: datetime
    cash: float
    positions_value: float
    total_value: float
    positions: dict[str, Position]
    returns: float


class Portfolio:
    """
    Portfolio simulator with realistic position management.

    Tracks:
    - Cash balance
    - Open positions
    - Trade history
    - Portfolio value over time
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,  # 0.05% slippage
    ) -> None:
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash
            commission_pct: Commission as percentage of trade value
            slippage_pct: Slippage as percentage of price
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # State
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.history: list[PortfolioState] = []

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.history = []

    def get_position_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total value of open positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                total += pos.shares * current_prices[symbol]
        return total

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash + self.get_position_value(current_prices)

    def open_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        target_value: float,
        side: str = "long",
    ) -> Position | None:
        """
        Open a new position.

        Args:
            symbol: Ticker symbol
            price: Current price
            date: Trade date
            target_value: Target position value in dollars
            side: 'long' or 'short'

        Returns:
            Position if opened, None if insufficient funds
        """
        if symbol in self.positions:
            return None  # Already have position

        # Apply slippage (worse price for us)
        if side == "long":
            exec_price = price * (1 + self.slippage_pct)
        else:
            exec_price = price * (1 - self.slippage_pct)

        # Calculate shares
        shares = target_value / exec_price

        # Calculate cost including commission
        trade_value = shares * exec_price
        commission = trade_value * self.commission_pct
        total_cost = trade_value + commission

        # Check sufficient funds
        if total_cost > self.cash:
            # Reduce position to available cash
            available = self.cash / (1 + self.commission_pct)
            shares = available / exec_price
            trade_value = shares * exec_price
            commission = trade_value * self.commission_pct
            total_cost = trade_value + commission

        if shares <= 0:
            return None

        # Execute
        self.cash -= total_cost

        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=exec_price,
            entry_date=date,
            side=side,
        )
        self.positions[symbol] = position

        return position

    def close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
    ) -> Trade | None:
        """
        Close an existing position.

        Args:
            symbol: Ticker symbol
            price: Current price
            date: Trade date

        Returns:
            Trade record if closed, None if no position
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Apply slippage (worse price for us)
        if position.side == "long":
            exec_price = price * (1 - self.slippage_pct)
        else:
            exec_price = price * (1 + self.slippage_pct)

        # Calculate proceeds
        trade_value = position.shares * exec_price
        commission = trade_value * self.commission_pct
        proceeds = trade_value - commission

        # Calculate P&L
        if position.side == "long":
            pnl = (exec_price - position.entry_price) * position.shares - commission * 2
        else:
            pnl = (position.entry_price - exec_price) * position.shares - commission * 2

        return_pct = pnl / (position.shares * position.entry_price)

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=exec_price,
            shares=position.shares,
            side=position.side,
            pnl=pnl,
            return_pct=return_pct,
            holding_days=(date - position.entry_date).days,
        )
        self.trades.append(trade)

        # Update cash
        self.cash += proceeds

        # Remove position
        del self.positions[symbol]

        return trade

    def record_state(
        self,
        date: datetime,
        current_prices: dict[str, float],
        prev_value: float | None = None,
    ) -> PortfolioState:
        """
        Record current portfolio state.

        Args:
            date: Current date
            current_prices: Current prices for all symbols
            prev_value: Previous total value (for return calculation)
        """
        positions_value = self.get_position_value(current_prices)
        total_value = self.cash + positions_value

        if prev_value is not None and prev_value > 0:
            returns = (total_value - prev_value) / prev_value
        else:
            returns = 0.0

        state = PortfolioState(
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            positions=dict(self.positions),
            returns=returns,
        )
        self.history.append(state)

        return state

    def get_returns_series(self) -> pd.Series:
        """Get returns series from history."""
        if not self.history:
            return pd.Series(dtype=float)

        dates = [s.date for s in self.history]
        returns = [s.returns for s in self.history]

        return pd.Series(returns, index=pd.DatetimeIndex(dates))

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from history."""
        if not self.history:
            return pd.Series(dtype=float)

        dates = [s.date for s in self.history]
        values = [s.total_value for s in self.history]

        return pd.Series(values, index=pd.DatetimeIndex(dates))

    def get_trade_summary(self) -> dict:
        """Get summary statistics for trades."""
        if not self.trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "total_pnl": 0.0,
            }

        pnls = [t.pnl for t in self.trades]
        returns = [t.return_pct for t in self.trades]
        winners = [p for p in pnls if p > 0]

        return {
            "num_trades": len(self.trades),
            "win_rate": len(winners) / len(self.trades) if self.trades else 0,
            "avg_return": np.mean(returns),
            "total_pnl": sum(pnls),
            "avg_holding_days": np.mean([t.holding_days for t in self.trades]),
        }


class PositionSizer:
    """
    Calculate position sizes based on different methods.
    """

    def __init__(
        self,
        method: PositionSizing = PositionSizing.FIXED,
        max_position_pct: float = 0.10,
        target_volatility: float = 0.10,
    ) -> None:
        """
        Initialize position sizer.

        Args:
            method: Sizing method
            max_position_pct: Maximum position as % of portfolio
            target_volatility: Target volatility for vol-sizing
        """
        self.method = method
        self.max_position_pct = max_position_pct
        self.target_volatility = target_volatility

    def calculate_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: float | None = None,
    ) -> float:
        """
        Calculate target position value.

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Asset volatility (for vol-sizing)

        Returns:
            Target position value in dollars
        """
        max_value = portfolio_value * self.max_position_pct

        if self.method == PositionSizing.FIXED:
            return max_value

        elif self.method == PositionSizing.VOLATILITY:
            if volatility is None or volatility <= 0:
                return max_value

            # Scale inversely with volatility
            vol_scalar = self.target_volatility / volatility
            target = portfolio_value * self.max_position_pct * vol_scalar
            return min(target, max_value)

        elif self.method == PositionSizing.EQUAL_WEIGHT:
            return max_value

        else:
            return max_value
