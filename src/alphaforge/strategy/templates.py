"""
Built-in strategy templates.

These are well-known strategies that serve as:
- Starting points for optimization
- Benchmarks for validation
- Educational examples
"""

from alphaforge.strategy.genome import (
    LogicalOperator,
    Operator,
    PositionSizing,
    Rule,
    RuleGroup,
    StrategyGenome,
)


class StrategyTemplates:
    """Factory for common strategy templates."""

    @staticmethod
    def sma_crossover(
        fast_period: int = 20,
        slow_period: int = 50,
        stop_loss_pct: float = 0.02,
    ) -> StrategyGenome:
        """
        Simple Moving Average Crossover Strategy.

        Entry: Fast SMA crosses above Slow SMA
        Exit: Fast SMA crosses below Slow SMA OR stop loss hit

        Args:
            fast_period: Period for fast SMA
            slow_period: Period for slow SMA
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="SMA Crossover",
            description=f"Buy when SMA({fast_period}) crosses above SMA({slow_period})",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"sma_{fast_period}",
                        operator=Operator.CROSS_ABOVE,
                        value=f"sma_{slow_period}",
                        description="Fast SMA crosses above slow SMA",
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"sma_{fast_period}",
                        operator=Operator.CROSS_BELOW,
                        value=f"sma_{slow_period}",
                        description="Fast SMA crosses below slow SMA",
                    )
                ]
            ),
            stop_loss_pct=stop_loss_pct,
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
            },
            tags=["trend-following", "moving-average"],
        )

    @staticmethod
    def rsi_mean_reversion(
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        stop_loss_pct: float = 0.03,
    ) -> StrategyGenome:
        """
        RSI Mean Reversion Strategy.

        Entry: RSI drops below oversold threshold
        Exit: RSI rises above neutral (50) OR overbought OR stop loss

        Args:
            rsi_period: RSI calculation period
            oversold: Oversold threshold (entry)
            overbought: Overbought threshold (exit)
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="RSI Mean Reversion",
            description=f"Buy when RSI({rsi_period}) < {oversold}, sell when > {overbought}",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"rsi_{rsi_period}",
                        operator=Operator.LT,
                        value=oversold,
                        description="RSI below oversold threshold",
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"rsi_{rsi_period}",
                        operator=Operator.GT,
                        value=overbought,
                        description="RSI above overbought threshold",
                    )
                ],
                operator=LogicalOperator.OR,
            ),
            stop_loss_pct=stop_loss_pct,
            parameters={
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
            },
            tags=["mean-reversion", "oscillator"],
        )

    @staticmethod
    def bollinger_breakout(
        period: int = 20,
        num_std: float = 2.0,
        stop_loss_pct: float = 0.02,
    ) -> StrategyGenome:
        """
        Bollinger Band Breakout Strategy.

        Entry: Close breaks above upper band
        Exit: Close drops below middle band OR stop loss

        Args:
            period: Bollinger band period
            num_std: Number of standard deviations
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="Bollinger Breakout",
            description="Buy on upper band breakout, exit at middle band",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="close",
                        operator=Operator.GT,
                        value="bb_upper",
                        description="Price breaks above upper band",
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="close",
                        operator=Operator.LT,
                        value="bb_middle",
                        description="Price drops below middle band",
                    )
                ]
            ),
            stop_loss_pct=stop_loss_pct,
            parameters={
                "bb_period": period,
                "bb_std": num_std,
            },
            tags=["breakout", "volatility"],
        )

    @staticmethod
    def momentum_rotation(
        lookback: int = 20,
        holding_period: int = 5,
        stop_loss_pct: float = 0.05,
    ) -> StrategyGenome:
        """
        Momentum Rotation Strategy.

        Entry: Positive momentum over lookback period
        Exit: Momentum turns negative OR holding period expires

        Args:
            lookback: Momentum lookback period
            holding_period: Maximum holding period in days
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="Momentum Rotation",
            description=f"Buy on positive {lookback}-day momentum, hold up to {holding_period} days",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"momentum_{lookback}",
                        operator=Operator.GT,
                        value=0,
                        description="Positive momentum",
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator=f"momentum_{lookback}",
                        operator=Operator.LT,
                        value=0,
                        description="Negative momentum",
                    )
                ]
            ),
            stop_loss_pct=stop_loss_pct,
            max_holding_days=holding_period,
            parameters={
                "lookback": lookback,
                "holding_period": holding_period,
            },
            tags=["momentum", "rotation"],
        )

    @staticmethod
    def macd_crossover(
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        stop_loss_pct: float = 0.02,
    ) -> StrategyGenome:
        """
        MACD Crossover Strategy.

        Entry: MACD line crosses above signal line
        Exit: MACD line crosses below signal line

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="MACD Crossover",
            description="Buy when MACD crosses above signal, sell on cross below",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="macd",
                        operator=Operator.CROSS_ABOVE,
                        value="macd_signal",
                        description="MACD crosses above signal",
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="macd",
                        operator=Operator.CROSS_BELOW,
                        value="macd_signal",
                        description="MACD crosses below signal",
                    )
                ]
            ),
            stop_loss_pct=stop_loss_pct,
            parameters={
                "macd_fast": fast,
                "macd_slow": slow,
                "macd_signal": signal,
            },
            tags=["trend-following", "macd"],
        )

    @staticmethod
    def volatility_breakout(
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        stop_loss_pct: float = 0.03,
    ) -> StrategyGenome:
        """
        Volatility Breakout Strategy.

        Entry: Price moves more than ATR * multiplier from previous close
        Exit: Trailing stop based on ATR

        Args:
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for entry threshold
            stop_loss_pct: Initial stop loss percentage
        """
        return StrategyGenome(
            name="Volatility Breakout",
            description=f"Buy on {atr_multiplier}x ATR breakout",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="returns_1d",
                        operator=Operator.GT,
                        value=0.02,  # Simplified: >2% daily move
                        description="Strong positive move",
                    ),
                    Rule(
                        indicator="volume_ratio",
                        operator=Operator.GT,
                        value=1.5,
                        description="Above average volume",
                    ),
                ],
                operator=LogicalOperator.AND,
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="returns_1d",
                        operator=Operator.LT,
                        value=-0.01,
                        description="Negative day",
                    )
                ]
            ),
            stop_loss_pct=stop_loss_pct,
            sizing_method=PositionSizing.VOLATILITY,
            parameters={
                "atr_period": atr_period,
                "atr_multiplier": atr_multiplier,
            },
            tags=["breakout", "volatility"],
        )

    @staticmethod
    def dual_momentum(
        abs_momentum_period: int = 252,
        rel_momentum_period: int = 126,
        stop_loss_pct: float = 0.10,
    ) -> StrategyGenome:
        """
        Dual Momentum Strategy (Gary Antonacci).

        Entry: Both absolute and relative momentum are positive
        Exit: Either momentum turns negative

        Args:
            abs_momentum_period: Absolute momentum lookback
            rel_momentum_period: Relative momentum lookback
            stop_loss_pct: Stop loss percentage
        """
        return StrategyGenome(
            name="Dual Momentum",
            description="Buy when both absolute and relative momentum are positive",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="returns_20d",  # Simplified proxy
                        operator=Operator.GT,
                        value=0,
                        description="Positive absolute momentum",
                    ),
                    Rule(
                        indicator="momentum_10",
                        operator=Operator.GT,
                        value=0,
                        description="Positive relative momentum",
                    ),
                ],
                operator=LogicalOperator.AND,
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="returns_20d",
                        operator=Operator.LT,
                        value=0,
                        description="Negative absolute momentum",
                    ),
                    Rule(
                        indicator="momentum_10",
                        operator=Operator.LT,
                        value=0,
                        description="Negative relative momentum",
                    ),
                ],
                operator=LogicalOperator.OR,
            ),
            stop_loss_pct=stop_loss_pct,
            parameters={
                "abs_momentum_period": abs_momentum_period,
                "rel_momentum_period": rel_momentum_period,
            },
            tags=["momentum", "tactical-allocation"],
        )

    @staticmethod
    def buy_and_hold() -> StrategyGenome:
        """
        Buy and Hold Strategy.

        Entry: First day of the backtest.
        Exit: Last day of the backtest.
        """
        return StrategyGenome(
            name="Buy and Hold",
            description="Benchmark strategy that buys on the first day and holds until the end.",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="close",
                        operator=Operator.GTE,
                        value=0,
                        description="Enter on first day (rule is always true)",
                    )
                ]
            ),
            exit_rules=RuleGroup(rules=[]),  # No exit rules
            stop_loss_pct=None,
            take_profit_pct=None,
            max_holding_days=None,
            tags=["benchmark"],
        )

    @classmethod
    def all_templates(cls) -> list[StrategyGenome]:
        """Get all available strategy templates."""
        return [
            cls.sma_crossover(),
            cls.rsi_mean_reversion(),
            cls.bollinger_breakout(),
            cls.momentum_rotation(),
            cls.macd_crossover(),
            cls.volatility_breakout(),
            cls.dual_momentum(),
            cls.buy_and_hold(),
        ]

    @classmethod
    def get_template(cls, name: str) -> StrategyGenome:
        """Get a specific template by name."""
        templates = {
            "sma_crossover": cls.sma_crossover,
            "rsi_mean_reversion": cls.rsi_mean_reversion,
            "bollinger_breakout": cls.bollinger_breakout,
            "momentum_rotation": cls.momentum_rotation,
            "macd_crossover": cls.macd_crossover,
            "volatility_breakout": cls.volatility_breakout,
            "dual_momentum": cls.dual_momentum,
            "buy_and_hold": cls.buy_and_hold,
        }
        if name not in templates:
            raise ValueError(
                f"Unknown template: {name}. Available: {list(templates.keys())}"
            )
        return templates[name]()
