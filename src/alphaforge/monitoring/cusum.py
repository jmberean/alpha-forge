"""
CUSUM (Cumulative Sum) monitoring for strategy degradation detection.

Implements CUSUM control chart to detect when a strategy's performance
degrades from expected levels in real-time.

Algorithm:
    S_t = max(0, S_{t-1} + (z_t - k))

    where:
        z_t = (realized_IR - expected_IR) / stderr(IR)
        k = drift allowance (0.5)

    Alert when S_t > h (threshold)

Reference:
- Page (1954): "Continuous Inspection Schemes"
- Basseville & Nikiforov (1993): "Detection of Abrupt Changes"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AlertLevel(Enum):
    """Alert severity levels for CUSUM monitoring."""

    NORMAL = "normal"
    WARNING = "warning"      # S > 3.0 (log only)
    DEGRADED = "degraded"    # S > 5.0 (reduce allocation 50%)
    ALARM = "alarm"          # S > 7.0 (halt trading)


@dataclass
class CUSUMState:
    """
    Current state of CUSUM monitoring.

    Tracks the cumulative sum statistic and alert level.
    """

    statistic: float  # Current S_t value
    alert_level: AlertLevel  # Current alert level
    periods_since_reset: int  # How long since last reset
    last_z_score: float = 0.0  # Most recent standardized residual

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"CUSUM Status: {self.alert_level.value.upper()}",
            f"",
            f"Statistic (S_t): {self.statistic:.2f}",
            f"Last Z-Score: {self.last_z_score:.2f}",
            f"Periods Since Reset: {self.periods_since_reset}",
            f"",
        ]

        if self.alert_level == AlertLevel.NORMAL:
            lines.append("✓ Strategy performing as expected")
        elif self.alert_level == AlertLevel.WARNING:
            lines.append("⚠ WARNING: Slight degradation detected (monitor closely)")
        elif self.alert_level == AlertLevel.DEGRADED:
            lines.append("⚠ DEGRADED: Significant degradation (reduce allocation)")
        elif self.alert_level == AlertLevel.ALARM:
            lines.append("🚨 ALARM: Critical degradation (halt trading)")

        return "\n".join(lines)


@dataclass
class CUSUMAlert:
    """
    Alert triggered by CUSUM monitoring.
    """

    timestamp: pd.Timestamp
    alert_level: AlertLevel
    statistic: float
    z_score: float
    message: str


@dataclass
class CUSUMResult:
    """
    Complete CUSUM monitoring result.

    Contains full history of CUSUM statistics and alerts.
    """

    statistics: pd.Series  # Time series of S_t values
    z_scores: pd.Series  # Time series of z-scores
    alerts: list[CUSUMAlert] = field(default_factory=list)
    final_state: CUSUMState | None = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"CUSUM Monitoring Result",
            f"",
            f"Periods Monitored: {len(self.statistics)}",
            f"Alerts Triggered: {len(self.alerts)}",
            f"",
            f"Alert Breakdown:",
        ]

        # Count alerts by level
        alert_counts = {}
        for alert in self.alerts:
            level = alert.alert_level.value
            alert_counts[level] = alert_counts.get(level, 0) + 1

        for level in ["warning", "degraded", "alarm"]:
            count = alert_counts.get(level, 0)
            lines.append(f"  {level.upper()}: {count}")

        if self.final_state:
            lines.append(f"")
            lines.append(f"Final Status: {self.final_state.alert_level.value.upper()}")
            lines.append(f"Final Statistic: {self.final_state.statistic:.2f}")

        return "\n".join(lines)


class CUSUMMonitor:
    """
    CUSUM monitor for strategy degradation detection.

    Monitors strategy performance in real-time and alerts when
    performance degrades significantly from expected levels.

    Usage:
        >>> monitor = CUSUMMonitor(expected_sharpe=1.5)
        >>>
        >>> # Update with new returns
        >>> state = monitor.update(new_returns)
        >>>
        >>> if state.alert_level >= AlertLevel.DEGRADED:
        ...     print("Reduce allocation!")
    """

    def __init__(
        self,
        expected_sharpe: float,
        drift_allowance: float = 0.5,
        warning_threshold: float = 3.0,
        degraded_threshold: float = 5.0,
        alarm_threshold: float = 7.0,
        lookback_window: int = 63,  # ~3 months daily
        reset_on_alarm: bool = True,
    ):
        """
        Initialize CUSUM monitor.

        Args:
            expected_sharpe: Expected Sharpe ratio of strategy
            drift_allowance: Allowable drift in z-scores (k parameter)
            warning_threshold: S_t threshold for WARNING alert
            degraded_threshold: S_t threshold for DEGRADED alert
            alarm_threshold: S_t threshold for ALARM alert
            lookback_window: Rolling window for Sharpe calculation
            reset_on_alarm: Whether to reset CUSUM after alarm
        """
        self.expected_sharpe = expected_sharpe
        self.drift_allowance = drift_allowance
        self.warning_threshold = warning_threshold
        self.degraded_threshold = degraded_threshold
        self.alarm_threshold = alarm_threshold
        self.lookback_window = lookback_window
        self.reset_on_alarm = reset_on_alarm

        # Internal state
        self.statistic = 0.0  # S_t
        self.periods_since_reset = 0
        self.returns_history: list[float] = []
        self.statistics_history: list[float] = []
        self.z_scores_history: list[float] = []
        self.timestamps_history: list[pd.Timestamp] = []
        self.alerts: list[CUSUMAlert] = []

    def update(
        self,
        returns: np.ndarray | pd.Series,
        timestamp: pd.Timestamp | None = None,
    ) -> CUSUMState:
        """
        Update CUSUM with new returns.

        Args:
            returns: Array of returns (can be single value or batch)
            timestamp: Optional timestamp for alert tracking

        Returns:
            CUSUMState with current alert level

        Example:
            >>> monitor = CUSUMMonitor(expected_sharpe=1.5)
            >>>
            >>> # Daily update
            >>> state = monitor.update([0.001])  # Small gain
            >>>
            >>> # Batch update
            >>> state = monitor.update(last_week_returns)
        """
        # Convert to numpy array
        if isinstance(returns, pd.Series):
            returns = returns.values
        if isinstance(returns, (int, float)):
            returns = np.array([returns])

        # Default timestamp
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Process each return
        for ret in returns:
            self.returns_history.append(ret)

            # Calculate realized Sharpe over lookback window
            if len(self.returns_history) >= self.lookback_window:
                recent_returns = self.returns_history[-self.lookback_window:]
                realized_sharpe = self._calculate_sharpe(recent_returns)
            elif len(self.returns_history) >= 10:  # Minimum for estimation
                realized_sharpe = self._calculate_sharpe(self.returns_history)
            else:
                # Not enough data yet
                realized_sharpe = self.expected_sharpe

            # Calculate z-score (standardized residual)
            # Estimate stderr of Sharpe using 1/sqrt(n) approximation
            n = min(len(self.returns_history), self.lookback_window)
            stderr_sharpe = 1.0 / np.sqrt(max(n, 10))

            z_score = (realized_sharpe - self.expected_sharpe) / stderr_sharpe

            # Update CUSUM statistic
            # Note: We use negative z_score because we're detecting degradation
            # (lower Sharpe than expected)
            self.statistic = max(0.0, self.statistic + (-z_score - self.drift_allowance))

            self.periods_since_reset += 1
            self.statistics_history.append(self.statistic)
            self.z_scores_history.append(z_score)
            self.timestamps_history.append(timestamp)

        # Determine alert level
        alert_level = self._get_alert_level()

        # Check for new alerts
        self._check_alerts(timestamp, alert_level, z_score)

        # Reset if alarm triggered (optional)
        if self.reset_on_alarm and alert_level == AlertLevel.ALARM:
            self.reset()

        return CUSUMState(
            statistic=self.statistic,
            alert_level=alert_level,
            periods_since_reset=self.periods_since_reset,
            last_z_score=z_score if len(returns) > 0 else 0.0,
        )

    def batch_monitor(
        self,
        returns: pd.Series,
    ) -> CUSUMResult:
        """
        Monitor a batch of historical returns.

        Args:
            returns: Time series of returns with timestamps

        Returns:
            CUSUMResult with full monitoring history

        Example:
            >>> monitor = CUSUMMonitor(expected_sharpe=1.5)
            >>> result = monitor.batch_monitor(historical_returns)
            >>> print(result.summary())
        """
        # Reset monitor
        self.reset()

        # Process each return
        for timestamp, ret in returns.items():
            self.update([ret], timestamp)

        # Build result
        statistics = pd.Series(
            self.statistics_history,
            index=self.timestamps_history,
        )

        z_scores = pd.Series(
            self.z_scores_history,
            index=self.timestamps_history,
        )

        final_state = CUSUMState(
            statistic=self.statistic,
            alert_level=self._get_alert_level(),
            periods_since_reset=self.periods_since_reset,
            last_z_score=self.z_scores_history[-1] if self.z_scores_history else 0.0,
        )

        return CUSUMResult(
            statistics=statistics,
            z_scores=z_scores,
            alerts=self.alerts.copy(),
            final_state=final_state,
        )

    def reset(self) -> None:
        """Reset CUSUM statistic (e.g., after fixing strategy)."""
        self.statistic = 0.0
        self.periods_since_reset = 0

    def _calculate_sharpe(self, returns: list[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        returns_arr = np.array(returns)

        if len(returns_arr) == 0:
            return 0.0

        mean = np.mean(returns_arr)
        std = np.std(returns_arr)

        # Check for near-zero std (numerical precision)
        if std < 1e-10:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = (mean / std) * np.sqrt(252)

        return sharpe

    def _get_alert_level(self) -> AlertLevel:
        """Determine current alert level based on statistic."""
        if self.statistic >= self.alarm_threshold:
            return AlertLevel.ALARM
        elif self.statistic >= self.degraded_threshold:
            return AlertLevel.DEGRADED
        elif self.statistic >= self.warning_threshold:
            return AlertLevel.WARNING
        else:
            return AlertLevel.NORMAL

    def _check_alerts(
        self,
        timestamp: pd.Timestamp,
        alert_level: AlertLevel,
        z_score: float,
    ) -> None:
        """Check if we should trigger a new alert."""
        # Only alert on level transitions (not every period)
        if len(self.alerts) == 0:
            prev_level = AlertLevel.NORMAL
        else:
            prev_level = self.alerts[-1].alert_level

        # Alert on level increase
        if alert_level.value != prev_level.value:
            message = self._format_alert_message(alert_level, z_score)

            alert = CUSUMAlert(
                timestamp=timestamp,
                alert_level=alert_level,
                statistic=self.statistic,
                z_score=z_score,
                message=message,
            )

            self.alerts.append(alert)

    def _format_alert_message(
        self,
        alert_level: AlertLevel,
        z_score: float,
    ) -> str:
        """Format alert message."""
        if alert_level == AlertLevel.WARNING:
            return (
                f"CUSUM WARNING: S_t={self.statistic:.2f} > {self.warning_threshold:.1f}. "
                f"Strategy Sharpe degrading (z={z_score:.2f}). Monitor closely."
            )
        elif alert_level == AlertLevel.DEGRADED:
            return (
                f"CUSUM DEGRADED: S_t={self.statistic:.2f} > {self.degraded_threshold:.1f}. "
                f"Significant degradation (z={z_score:.2f}). Reduce allocation 50%."
            )
        elif alert_level == AlertLevel.ALARM:
            return (
                f"CUSUM ALARM: S_t={self.statistic:.2f} > {self.alarm_threshold:.1f}. "
                f"Critical degradation (z={z_score:.2f}). HALT TRADING."
            )
        else:
            return f"CUSUM back to NORMAL: S_t={self.statistic:.2f}"


# Convenience function
def monitor_strategy_performance(
    returns: pd.Series,
    expected_sharpe: float,
    drift_allowance: float = 0.5,
) -> CUSUMResult:
    """
    Convenience function to monitor strategy performance.

    Args:
        returns: Time series of strategy returns
        expected_sharpe: Expected Sharpe ratio
        drift_allowance: Allowable drift (k parameter)

    Returns:
        CUSUMResult with monitoring history

    Example:
        >>> result = monitor_strategy_performance(
        ...     my_strategy_returns,
        ...     expected_sharpe=1.5,
        ... )
        >>> print(result.summary())
    """
    monitor = CUSUMMonitor(
        expected_sharpe=expected_sharpe,
        drift_allowance=drift_allowance,
    )

    return monitor.batch_monitor(returns)
