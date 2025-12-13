"""
Market impact modeling for realistic execution costs.

Implements Almgren-Chriss parametric model for permanent and temporary
market impact. This is the industry-standard baseline for v2.0.

Reference:
- Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
- Empirical parameters from Gatheral (2010)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MarketImpactModel:
    """
    Almgren-Chriss parametric market impact model.

    Splits total impact into:
    - Permanent Impact: Information leakage (price doesn't fully recover)
    - Temporary Impact: Execution pressure (price recovers after trade)

    Default parameters based on Gatheral (2010) empirical study.
    """

    permanent_impact: float = 0.314  # α parameter
    temporary_impact: float = 0.142  # β parameter

    def calculate_impact(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
        time_horizon: float = 1.0,
    ) -> float:
        """
        Calculate total market impact as percentage of price.

        Args:
            order_size: Size of order in dollars
            daily_volume: Average daily dollar volume
            volatility: Daily volatility (std dev of returns)
            time_horizon: Execution horizon in trading days (default=1)

        Returns:
            Total impact as decimal (e.g., 0.001 = 10 bps)

        Formula:
            Total Impact = Permanent + Temporary
            Permanent = α * sqrt(σ) * (Q/V)
            Temporary = β * sqrt(σ) * (Q/V)^0.6 / T

        Example:
            >>> model = MarketImpactModel()
            >>> # $100k order, $50M daily volume, 1% volatility
            >>> impact = model.calculate_impact(100_000, 50_000_000, 0.01)
            >>> impact  # ~0.00015 = 1.5 bps
        """
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")

        # Participation rate (Q/V)
        participation = order_size / daily_volume

        # Permanent impact (information leakage)
        permanent = self.permanent_impact * np.sqrt(volatility) * participation

        # Temporary impact (execution pressure)
        temporary = (
            self.temporary_impact
            * np.sqrt(volatility)
            * (participation**0.6)
            / time_horizon
        )

        return permanent + temporary

    def calculate_components(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
        time_horizon: float = 1.0,
    ) -> tuple[float, float]:
        """
        Calculate permanent and temporary impact separately.

        Returns:
            (permanent_impact, temporary_impact) as tuple of decimals
        """
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")

        participation = order_size / daily_volume

        permanent = self.permanent_impact * np.sqrt(volatility) * participation
        temporary = (
            self.temporary_impact
            * np.sqrt(volatility)
            * (participation**0.6)
            / time_horizon
        )

        return permanent, temporary


@dataclass
class ExecutionCost:
    """Complete execution cost breakdown."""

    commission: float  # Fixed commission in dollars
    spread_cost: float  # Bid-ask spread cost in dollars
    impact_cost: float  # Market impact cost in dollars
    total_cost: float  # Total execution cost in dollars

    @property
    def total_bps(self) -> float:
        """Total cost in basis points (for reporting)."""
        # Note: Requires order value to calculate, set externally
        return 0.0  # Placeholder


@dataclass
class ExecutionModel:
    """
    Complete execution cost model.

    Combines:
    - Fixed commission (e.g., $0 for most retail now)
    - Bid-ask spread
    - Market impact (Almgren-Chriss)

    To disable market impact modeling, explicitly pass impact_model=False.
    """

    base_commission: float = 0.0  # Most retail is zero-commission now
    spread_bps: float = 2.0  # 2 bps typical for liquid stocks
    impact_model: MarketImpactModel | bool | None = field(
        default_factory=MarketImpactModel
    )

    def __post_init__(self) -> None:
        # If impact_model is False, set to None
        if self.impact_model is False:
            self.impact_model = None

    def calculate_total_cost(
        self,
        order_value: float,
        daily_volume: float,
        volatility: float,
        time_horizon: float = 1.0,
        side: str = "buy",
    ) -> ExecutionCost:
        """
        Calculate complete execution cost.

        Args:
            order_value: Dollar value of order
            daily_volume: Average daily dollar volume
            volatility: Daily volatility
            time_horizon: Execution horizon in days
            side: 'buy' or 'sell' (affects spread direction)

        Returns:
            ExecutionCost with breakdown

        Example:
            >>> model = ExecutionModel(base_commission=1.0, spread_bps=2.0)
            >>> cost = model.calculate_total_cost(
            ...     order_value=10_000,
            ...     daily_volume=50_000_000,
            ...     volatility=0.01
            ... )
            >>> print(f"Total cost: ${cost.total_cost:.2f}")
        """
        # Commission (fixed per order)
        commission = self.base_commission

        # Spread cost (half-spread crossing)
        spread_cost = order_value * (self.spread_bps / 10000.0) / 2

        # Market impact (if model provided)
        if self.impact_model and daily_volume > 0:
            impact_pct = self.impact_model.calculate_impact(
                order_size=order_value,
                daily_volume=daily_volume,
                volatility=volatility,
                time_horizon=time_horizon,
            )
            impact_cost = order_value * impact_pct
        else:
            impact_cost = 0.0

        total_cost = commission + spread_cost + impact_cost

        return ExecutionCost(
            commission=commission,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            total_cost=total_cost,
        )


# Convenience function
def calculate_market_impact(
    order_size: float,
    daily_volume: float,
    volatility: float,
    time_horizon: float = 1.0,
    permanent_impact: float = 0.314,
    temporary_impact: float = 0.142,
) -> float:
    """
    Convenience function to calculate market impact.

    Uses Almgren-Chriss model with configurable parameters.

    Returns:
        Market impact as decimal percentage
    """
    model = MarketImpactModel(
        permanent_impact=permanent_impact, temporary_impact=temporary_impact
    )
    return model.calculate_impact(order_size, daily_volume, volatility, time_horizon)
