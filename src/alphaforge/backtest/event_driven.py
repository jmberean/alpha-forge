"""
Event-driven backtesting engine.

Simulates realistic order execution with queue models, latency, and partial fills.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestResult
from alphaforge.backtest.metrics import calculate_metrics
from alphaforge.strategy.genome import StrategyGenome


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float  # Limit price
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0


@dataclass
class Fill:
    """Fill event."""

    order_id: str
    quantity: float
    price: float
    timestamp: datetime
    slippage: float  # Price - order.price


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""

    # Latency
    latency_ms: int = 50  # Order processing latency

    # Queue position
    queue_position_pct: float = 0.5  # Position in queue (0=front, 1=back)

    # Partial fills
    allow_partial_fills: bool = True
    min_fill_pct: float = 0.1  # Minimum 10% fill

    # Market impact (simple model)
    market_impact_coef: float = 0.0001  # Impact per $1M notional


class QueueModel:
    """
    Simulate queue position and fill probability.

    Simple empirical model for MVP7.
    """

    def __init__(self, config: ExecutionConfig):
        """Initialize queue model."""
        self.config = config

    def estimate_fill_probability(
        self,
        order: Order,
        market_price: float,
        volume: float,
    ) -> float:
        """
        Estimate probability of fill.

        Args:
            order: Order to fill
            market_price: Current market price
            volume: Current volume

        Returns:
            Fill probability (0-1)
        """
        # Simple model: Depends on price aggressiveness
        if order.side == OrderSide.BUY:
            price_aggressiveness = (order.price - market_price) / market_price
        else:
            price_aggressiveness = (market_price - order.price) / market_price

        # More aggressive = higher fill probability
        base_prob = 0.5  # Passive order
        if price_aggressiveness > 0.001:  # Crossing spread
            base_prob = 0.95
        elif price_aggressiveness > 0:
            base_prob = 0.7

        # Adjust for queue position
        queue_factor = 1.0 - (self.config.queue_position_pct * 0.5)
        fill_prob = base_prob * queue_factor

        return min(1.0, fill_prob)

    def estimate_fill_quantity(
        self,
        order: Order,
        available_volume: float,
    ) -> float:
        """
        Estimate fill quantity based on available volume.

        Args:
            order: Order
            available_volume: Available volume at price level

        Returns:
            Fill quantity
        """
        max_fill = min(order.quantity - order.filled_quantity, available_volume)

        if self.config.allow_partial_fills:
            # Random partial fill
            fill_pct = np.random.uniform(self.config.min_fill_pct, 1.0)
            return max_fill * fill_pct
        else:
            return max_fill if max_fill == order.quantity else 0.0


class EventDrivenEngine:
    """
    Event-driven backtesting engine.

    Simulates realistic execution with:
    - Order processing latency
    - Queue position
    - Partial fills
    - Market impact
    """

    def __init__(self, config: ExecutionConfig | None = None):
        """
        Initialize engine.

        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        self.queue_model = QueueModel(self.config)

        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.positions = {}
        self.cash = 100000.0  # Starting cash

    def run(
        self,
        strategy: StrategyGenome,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run event-driven backtest.

        Args:
            strategy: Strategy to test
            data: OHLCV data

        Returns:
            BacktestResult with realistic execution

        Note:
            This is a simplified event-driven engine for MVP7.
            In production, use NautilusTrader for full functionality.
        """
        # Reset state
        self.orders = []
        self.fills = []
        self.positions = {data.iloc[0].name: 0.0}  # Start flat
        self.cash = 100000.0

        portfolio_values = []
        returns = []

        # Event loop - process each bar
        for idx in range(len(data)):
            current_bar = data.iloc[idx]
            current_time = current_bar.name

            # 1. Generate signals (simplified)
            signal = self._generate_signal(strategy, data.iloc[:idx + 1])

            # 2. Generate order if signal
            if signal != 0:
                order = self._create_order(
                    signal=signal,
                    price=current_bar["close"],
                    timestamp=current_time,
                )
                self.orders.append(order)

            # 3. Process pending orders (with latency)
            self._process_orders(current_bar)

            # 4. Calculate portfolio value
            position = self.positions.get(current_time, 0.0)
            position_value = position * current_bar["close"]
            portfolio_value = self.cash + position_value

            portfolio_values.append(portfolio_value)

            # 5. Calculate returns
            if idx > 0:
                ret = (portfolio_value - portfolio_values[idx - 1]) / portfolio_values[idx - 1]
                returns.append(ret)
            else:
                returns.append(0.0)

        # Calculate metrics
        returns_series = pd.Series(returns, index=data.index)
        portfolio_series = pd.Series(portfolio_values, index=data.index)

        metrics = calculate_metrics(returns_series, portfolio_series)

        return BacktestResult(
            strategy_id=strategy.id,
            metrics=metrics,
            equity_curve=portfolio_series,
            returns=returns_series,
        )

    def _generate_signal(
        self,
        strategy: StrategyGenome,
        historical_data: pd.DataFrame,
    ) -> int:
        """
        Generate trading signal.

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        # Simplified signal generation (use strategy signals)
        if len(historical_data) < 50:
            return 0

        # Mock: Generate signal based on simple SMA crossover
        if "close" in historical_data.columns:
            close = historical_data["close"]
            sma_fast = close.rolling(20).mean().iloc[-1]
            sma_slow = close.rolling(50).mean().iloc[-1]

            if pd.notna(sma_fast) and pd.notna(sma_slow):
                if sma_fast > sma_slow * 1.01:
                    return 1  # Buy signal
                elif sma_fast < sma_slow * 0.99:
                    return -1  # Sell signal

        return 0

    def _create_order(
        self,
        signal: int,
        price: float,
        timestamp: datetime,
    ) -> Order:
        """Create order from signal."""
        order_id = f"order_{len(self.orders)}"
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL

        # Fixed quantity for simplicity
        quantity = 100.0

        return Order(
            order_id=order_id,
            symbol="STOCK",
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )

    def _process_orders(self, current_bar: pd.Series) -> None:
        """Process pending orders."""
        current_time = current_bar.name
        current_price = current_bar["close"]
        current_volume = current_bar.get("volume", 1000000)

        for order in self.orders:
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                continue

            # Check latency
            time_since_order = (current_time - order.timestamp).total_seconds() * 1000
            if time_since_order < self.config.latency_ms:
                continue  # Not processed yet

            # Check fill probability
            fill_prob = self.queue_model.estimate_fill_probability(
                order, current_price, current_volume
            )

            if np.random.random() < fill_prob:
                # Fill order (or partial fill)
                fill_qty = self.queue_model.estimate_fill_quantity(order, current_volume * 0.01)

                if fill_qty > 0:
                    # Apply market impact
                    notional = fill_qty * current_price
                    impact = self.config.market_impact_coef * (notional / 1_000_000)
                    fill_price = current_price * (1 + impact if order.side == OrderSide.BUY else 1 - impact)

                    # Create fill
                    fill = Fill(
                        order_id=order.order_id,
                        quantity=fill_qty,
                        price=fill_price,
                        timestamp=current_time,
                        slippage=fill_price - order.price,
                    )

                    self.fills.append(fill)

                    # Update order
                    order.filled_quantity += fill_qty
                    order.avg_fill_price = (
                        (order.avg_fill_price * (order.filled_quantity - fill_qty) + fill_price * fill_qty)
                        / order.filled_quantity
                    )

                    if order.filled_quantity >= order.quantity:
                        order.status = OrderStatus.FILLED
                    else:
                        order.status = OrderStatus.PARTIAL

                    # Update position and cash
                    if order.side == OrderSide.BUY:
                        self.positions[current_time] = self.positions.get(current_time, 0.0) + fill_qty
                        self.cash -= fill_qty * fill_price
                    else:
                        self.positions[current_time] = self.positions.get(current_time, 0.0) - fill_qty
                        self.cash += fill_qty * fill_price


def calculate_implementation_shortfall(
    vectorized_result: BacktestResult,
    event_driven_result: BacktestResult,
) -> dict[str, float]:
    """
    Calculate implementation shortfall.

    Compares ideal (vectorized) vs realistic (event-driven) performance.

    Args:
        vectorized_result: Result from vectorized backtest
        event_driven_result: Result from event-driven backtest

    Returns:
        Dict with shortfall metrics

    Example:
        >>> shortfall = calculate_implementation_shortfall(vector_result, event_result)
        >>> print(f"Shortfall: {shortfall['sharpe_shortfall']:.1%}")
    """
    vector_sharpe = vectorized_result.metrics.sharpe_ratio
    event_sharpe = event_driven_result.metrics.sharpe_ratio

    vector_return = vectorized_result.metrics.annualized_return
    event_return = event_driven_result.metrics.annualized_return

    # Shortfall = (Ideal - Realistic) / Ideal
    sharpe_shortfall = (vector_sharpe - event_sharpe) / vector_sharpe if vector_sharpe != 0 else 0.0
    return_shortfall = (vector_return - event_return) / abs(vector_return) if vector_return != 0 else 0.0

    return {
        "vectorized_sharpe": vector_sharpe,
        "event_driven_sharpe": event_sharpe,
        "sharpe_shortfall": sharpe_shortfall,
        "vectorized_return": vector_return,
        "event_driven_return": event_return,
        "return_shortfall": return_shortfall,
        "passed_threshold": sharpe_shortfall < 0.30,  # Less than 30% degradation
    }
