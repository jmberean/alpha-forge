"""
Vectorized backtesting engine.

High-performance backtesting using numpy vectorization.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from alphaforge.backtest.metrics import PerformanceMetrics
from alphaforge.backtest.portfolio import Portfolio, PositionSizer
from alphaforge.data.schema import OHLCVData
from alphaforge.features.technical import TechnicalIndicators
from alphaforge.strategy.genome import StrategyGenome
from alphaforge.strategy.signals import PositionTracker, SignalGenerator


@dataclass
class BacktestResult:
    """Complete backtest results."""

    strategy: StrategyGenome
    metrics: PerformanceMetrics

    # Time series
    returns: pd.Series
    positions: pd.Series
    equity_curve: pd.Series

    # Trade details
    num_trades: int
    trades: list

    # Metadata
    start_date: datetime
    end_date: datetime
    data_symbol: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy.id,
            "strategy_name": self.strategy.name,
            "metrics": self.metrics.to_dict(),
            "num_trades": self.num_trades,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "data_symbol": self.data_symbol,
        }

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Backtest: {self.strategy.name}\n"
            f"Period: {self.start_date.date()} to {self.end_date.date()}\n"
            f"Symbol: {self.data_symbol}\n"
            f"Sharpe: {self.metrics.sharpe_ratio:.2f}\n"
            f"Return: {self.metrics.annualized_return:.1%}\n"
            f"Volatility: {self.metrics.volatility:.1%}\n"
            f"Max Drawdown: {self.metrics.max_drawdown:.1%}\n"
            f"Trades: {self.num_trades}\n"
            f"Win Rate: {self.metrics.win_rate:.1%}"
        )


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Uses numpy operations for fast backtesting of multiple strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        strategy: StrategyGenome,
        data: OHLCVData,
        compute_features: bool = True,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy: Strategy specification
            data: OHLCV market data
            compute_features: Whether to compute technical indicators

        Returns:
            BacktestResult with full analysis
        """
        df = data.df.copy()

        # Compute features if needed
        if compute_features:
            df = TechnicalIndicators.compute_all(df)

        # Generate signals
        signal_gen = SignalGenerator(strategy)
        signals_df = signal_gen.generate(df)

        entry_signals = signals_df["entry_signal"]
        exit_signals = signals_df["exit_signal"]

        # Generate positions
        tracker = PositionTracker(
            stop_loss_pct=strategy.stop_loss_pct,
            take_profit_pct=strategy.take_profit_pct,
            max_holding_days=strategy.max_holding_days,
        )
        positions = tracker.compute_positions(df, entry_signals, exit_signals)

        # Calculate returns
        price_returns = df["close"].pct_change().fillna(0)

        # Strategy returns = position * price returns (with lag for signal)
        # Shift positions by 1 to avoid lookahead (trade on next bar)
        lagged_positions = positions.shift(1).fillna(0)
        strategy_returns = lagged_positions * price_returns

        # Apply transaction costs
        position_changes = positions.diff().fillna(0).abs()
        transaction_costs = position_changes * (self.commission_pct + self.slippage_pct)
        strategy_returns = strategy_returns - transaction_costs

        # Calculate equity curve
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        # Calculate metrics
        metrics = PerformanceMetrics.from_returns(
            strategy_returns,
            positions=lagged_positions,
        )

        # Count trades
        position_changes = positions.diff().fillna(0)
        num_trades = int((position_changes != 0).sum() // 2)  # Entry + exit = 1 trade

        return BacktestResult(
            strategy=strategy,
            metrics=metrics,
            returns=strategy_returns,
            positions=positions,
            equity_curve=equity_curve,
            num_trades=num_trades,
            trades=[],  # Detailed trades not tracked in vectorized mode
            start_date=df.index[0].to_pydatetime(),
            end_date=df.index[-1].to_pydatetime(),
            data_symbol=data.symbol,
        )

    def run_detailed(
        self,
        strategy: StrategyGenome,
        data: OHLCVData,
    ) -> BacktestResult:
        """
        Run detailed backtest with trade tracking.

        Slower but provides complete trade history.
        """
        df = data.df.copy()
        df = TechnicalIndicators.compute_all(df)

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )

        # Position sizer
        sizer = PositionSizer(
            method=strategy.sizing_method,
            max_position_pct=strategy.max_position_pct,
            target_volatility=strategy.target_volatility,
        )

        # Generate signals
        signal_gen = SignalGenerator(strategy)
        signals_df = signal_gen.generate(df)

        entry_signals = signals_df["entry_signal"]
        exit_signals = signals_df["exit_signal"]

        # Simulate
        symbol = data.symbol
        positions_series = []
        prev_value = self.initial_capital

        for i in range(1, len(df)):  # Start at 1 to have previous day data
            date = df.index[i].to_pydatetime()
            price = df["close"].iloc[i]
            current_prices = {symbol: price}

            # Check for exit
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]

                # Check stop loss
                if strategy.stop_loss_pct:
                    pct_change = (price - pos.entry_price) / pos.entry_price
                    if pct_change <= -strategy.stop_loss_pct:
                        portfolio.close_position(symbol, price, date)

                # Check take profit
                elif strategy.take_profit_pct:
                    pct_change = (price - pos.entry_price) / pos.entry_price
                    if pct_change >= strategy.take_profit_pct:
                        portfolio.close_position(symbol, price, date)

                # Check max holding
                elif strategy.max_holding_days:
                    holding_days = (date - pos.entry_date).days
                    if holding_days >= strategy.max_holding_days:
                        portfolio.close_position(symbol, price, date)

                # Check exit signal
                elif exit_signals.iloc[i]:
                    portfolio.close_position(symbol, price, date)

            # Check for entry (only if not in position)
            elif entry_signals.iloc[i] and symbol not in portfolio.positions:
                # Calculate volatility for sizing
                if i >= 20:
                    volatility = df["close"].iloc[i - 20 : i].pct_change().std() * np.sqrt(252)
                else:
                    volatility = 0.20  # Default

                target_value = sizer.calculate_size(
                    portfolio.get_total_value(current_prices),
                    price,
                    volatility,
                )

                portfolio.open_position(symbol, price, date, target_value)

            # Record state
            state = portfolio.record_state(date, current_prices, prev_value)
            prev_value = state.total_value
            positions_series.append(1 if symbol in portfolio.positions else 0)

        # Build results
        returns = portfolio.get_returns_series()
        equity_curve = portfolio.get_equity_curve()

        # Pad positions series to match df length
        positions_series = [0] + positions_series  # Add initial 0
        while len(positions_series) < len(df):
            positions_series.append(0)
        positions = pd.Series(positions_series[: len(df)], index=df.index)

        metrics = PerformanceMetrics.from_returns(returns, positions=positions)

        return BacktestResult(
            strategy=strategy,
            metrics=metrics,
            returns=returns,
            positions=positions,
            equity_curve=equity_curve,
            num_trades=len(portfolio.trades),
            trades=portfolio.trades,
            start_date=df.index[0].to_pydatetime(),
            end_date=df.index[-1].to_pydatetime(),
            data_symbol=data.symbol,
        )

    def run_multiple(
        self,
        strategies: list[StrategyGenome],
        data: OHLCVData,
    ) -> list[BacktestResult]:
        """
        Run backtest for multiple strategies.

        Args:
            strategies: List of strategy specifications
            data: OHLCV market data

        Returns:
            List of BacktestResult
        """
        # Compute features once
        df = TechnicalIndicators.compute_all(data.df)
        data_with_features = OHLCVData(
            df=df,
            symbol=data.symbol,
            release_timestamp=data.release_timestamp,
            transaction_timestamp=data.transaction_timestamp,
        )

        results = []
        for strategy in strategies:
            result = self.run(strategy, data_with_features, compute_features=False)
            results.append(result)

        return results


def quick_backtest(
    strategy: StrategyGenome,
    data: OHLCVData,
) -> float:
    """
    Quick backtest returning only Sharpe ratio.

    Optimized for strategy screening.

    Args:
        strategy: Strategy specification
        data: OHLCV data

    Returns:
        Sharpe ratio
    """
    engine = BacktestEngine()
    result = engine.run(strategy, data)
    return result.metrics.sharpe_ratio


def backtest_on_split(
    strategy: StrategyGenome,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    symbol: str = "TEST",
) -> dict:
    """
    Backtest strategy on train/test split.

    Used by CPCV for cross-validation.

    Args:
        strategy: Strategy specification
        train_data: Training data (for fitting if needed)
        test_data: Test data (for evaluation)
        symbol: Symbol name

    Returns:
        Dictionary with sharpe, return, and other metrics
    """
    # Create OHLCVData for test set
    test_ohlcv = OHLCVData(
        df=test_data,
        symbol=symbol,
    )

    engine = BacktestEngine()
    result = engine.run(strategy, test_ohlcv)

    return {
        "sharpe": result.metrics.sharpe_ratio,
        "return": result.metrics.total_return,
        "volatility": result.metrics.volatility,
        "max_drawdown": result.metrics.max_drawdown,
        "num_trades": result.num_trades,
    }
