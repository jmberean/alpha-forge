"""
Signal generation from strategy specifications.

Converts StrategyGenome rules into actual trading signals.
"""

from typing import Optional
import numpy as np
import pandas as pd

from alphaforge.strategy.genome import (
    StrategyGenome,
    Rule,
    RuleGroup,
    Operator,
    LogicalOperator,
)


class SignalGenerator:
    """
    Generate trading signals from StrategyGenome specifications.

    Converts rule-based strategies into vectorized signal arrays.
    """

    def __init__(self, genome: StrategyGenome) -> None:
        """
        Initialize signal generator.

        Args:
            genome: Strategy specification
        """
        self.genome = genome

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry and exit signals for the given data.

        Args:
            df: DataFrame with OHLCV and indicator columns

        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        result = df.copy()

        # Generate entry signals
        result["entry_signal"] = self._evaluate_rule_group(
            df, self.genome.entry_rules
        )

        # Generate exit signals
        result["exit_signal"] = self._evaluate_rule_group(df, self.genome.exit_rules)

        return result

    def _evaluate_rule_group(
        self, df: pd.DataFrame, group: RuleGroup
    ) -> pd.Series:
        """
        Evaluate a group of rules.

        Args:
            df: Data DataFrame
            group: Rule group to evaluate

        Returns:
            Boolean Series indicating where rules are satisfied
        """
        if not group.rules:
            # No rules = always True (for entry) or always False (for exit)
            return pd.Series(False, index=df.index)

        results = []
        for rule in group.rules:
            if isinstance(rule, RuleGroup):
                result = self._evaluate_rule_group(df, rule)
            else:
                result = self._evaluate_rule(df, rule)
            results.append(result)

        # Combine results with logical operator
        if group.operator == LogicalOperator.AND:
            combined = results[0]
            for r in results[1:]:
                combined = combined & r
        else:  # OR
            combined = results[0]
            for r in results[1:]:
                combined = combined | r

        return combined

    def _evaluate_rule(self, df: pd.DataFrame, rule: Rule) -> pd.Series:
        """
        Evaluate a single rule.

        Args:
            df: Data DataFrame
            rule: Rule to evaluate

        Returns:
            Boolean Series
        """
        # Get indicator series
        indicator = self._get_indicator_value(df, rule.indicator)

        # Get comparison value (can be constant or another indicator)
        if isinstance(rule.value, str):
            value = self._get_indicator_value(df, rule.value)
        else:
            value = rule.value

        # Apply operator
        return self._apply_operator(indicator, rule.operator, value)

    def _get_indicator_value(
        self, df: pd.DataFrame, indicator: str
    ) -> pd.Series:
        """
        Get indicator values from DataFrame.

        Handles dynamic indicator names like sma_20, rsi_14.
        """
        # Direct column match
        if indicator in df.columns:
            return df[indicator]

        # Try lowercase
        indicator_lower = indicator.lower()
        if indicator_lower in df.columns:
            return df[indicator_lower]

        # Handle special cases
        if indicator == "close":
            return df["close"]
        if indicator == "open":
            return df["open"]
        if indicator == "high":
            return df["high"]
        if indicator == "low":
            return df["low"]
        if indicator == "volume":
            return df["volume"]

        raise ValueError(
            f"Indicator '{indicator}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    def _apply_operator(
        self,
        indicator: pd.Series,
        operator: Operator,
        value: pd.Series | float,
    ) -> pd.Series:
        """Apply comparison operator."""
        if operator == Operator.GT:
            return indicator > value
        elif operator == Operator.GTE:
            return indicator >= value
        elif operator == Operator.LT:
            return indicator < value
        elif operator == Operator.LTE:
            return indicator <= value
        elif operator == Operator.EQ:
            return indicator == value
        elif operator == Operator.NEQ:
            return indicator != value
        elif operator == Operator.CROSS_ABOVE:
            return self._cross_above(indicator, value)
        elif operator == Operator.CROSS_BELOW:
            return self._cross_below(indicator, value)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def _cross_above(
        self, series1: pd.Series, series2: pd.Series | float
    ) -> pd.Series:
        """
        Detect crossover above.

        True when series1 crosses from below to above series2.
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)

        prev_below = series1.shift(1) <= series2.shift(1)
        curr_above = series1 > series2

        return prev_below & curr_above

    def _cross_below(
        self, series1: pd.Series, series2: pd.Series | float
    ) -> pd.Series:
        """
        Detect crossover below.

        True when series1 crosses from above to below series2.
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)

        prev_above = series1.shift(1) >= series2.shift(1)
        curr_below = series1 < series2

        return prev_above & curr_below


class PositionTracker:
    """
    Track position state based on entry/exit signals.

    Converts signals into actual position holding periods.
    """

    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_holding_days: Optional[int] = None,
    ) -> None:
        """
        Initialize position tracker.

        Args:
            stop_loss_pct: Stop loss as percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit as percentage
            max_holding_days: Maximum holding period
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days

    def compute_positions(
        self,
        df: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
    ) -> pd.Series:
        """
        Compute position series from signals.

        Args:
            df: OHLCV DataFrame
            entry_signals: Boolean entry signals
            exit_signals: Boolean exit signals

        Returns:
            Series with 1 (long), 0 (flat), -1 (short)
        """
        positions = pd.Series(0, index=df.index)
        close_prices = df["close"]

        in_position = False
        entry_price = 0.0
        entry_date_idx = 0

        for i in range(len(df)):
            if not in_position:
                # Check for entry
                if entry_signals.iloc[i]:
                    in_position = True
                    entry_price = close_prices.iloc[i]
                    entry_date_idx = i
                    positions.iloc[i] = 1
            else:
                # Check for exit conditions
                current_price = close_prices.iloc[i]
                holding_days = i - entry_date_idx

                # Calculate return since entry
                pct_return = (current_price - entry_price) / entry_price

                # Check stop loss
                if self.stop_loss_pct and pct_return <= -self.stop_loss_pct:
                    in_position = False
                    positions.iloc[i] = 0
                    continue

                # Check take profit
                if self.take_profit_pct and pct_return >= self.take_profit_pct:
                    in_position = False
                    positions.iloc[i] = 0
                    continue

                # Check max holding days
                if self.max_holding_days and holding_days >= self.max_holding_days:
                    in_position = False
                    positions.iloc[i] = 0
                    continue

                # Check exit signal
                if exit_signals.iloc[i]:
                    in_position = False
                    positions.iloc[i] = 0
                    continue

                # Still in position
                positions.iloc[i] = 1

        return positions


def generate_signals(
    genome: StrategyGenome, df: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convenience function to generate signals and positions.

    Args:
        genome: Strategy specification
        df: OHLCV + indicators DataFrame

    Returns:
        Tuple of (entry_signals, exit_signals, positions)
    """
    # Generate raw signals
    signal_gen = SignalGenerator(genome)
    signals_df = signal_gen.generate(df)

    entry_signals = signals_df["entry_signal"]
    exit_signals = signals_df["exit_signal"]

    # Convert to positions
    tracker = PositionTracker(
        stop_loss_pct=genome.stop_loss_pct,
        take_profit_pct=genome.take_profit_pct,
        max_holding_days=genome.max_holding_days,
    )

    positions = tracker.compute_positions(df, entry_signals, exit_signals)

    return entry_signals, exit_signals, positions
