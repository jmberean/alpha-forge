"""
Regime detection for adaptive validation thresholds.

Detects market regimes (normal, trending, high_vol, crisis) to adjust
validation thresholds accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class RegimeType(str, Enum):
    """Market regime classification."""

    NORMAL = "normal"
    TRENDING = "trending"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class RegimeThresholds:
    """Validation thresholds for a specific regime."""

    pbo: float
    dsr: float
    sharpe: float
    max_drawdown: float


@dataclass
class RegimeDetection:
    """Results of regime detection."""

    regime: RegimeType
    confidence: float
    features: dict[str, float]
    thresholds: RegimeThresholds


class RegimeDetector:
    """
    Point-in-time market regime detection.

    Detects regimes using only data available at detection time.
    Uses simple, robust heuristics based on volatility and trend.
    """

    # Regime-specific validation thresholds
    THRESHOLDS = {
        RegimeType.NORMAL: RegimeThresholds(
            pbo=0.05, dsr=0.95, sharpe=1.0, max_drawdown=0.20
        ),
        RegimeType.TRENDING: RegimeThresholds(
            pbo=0.07, dsr=0.93, sharpe=0.8, max_drawdown=0.25
        ),
        RegimeType.HIGH_VOL: RegimeThresholds(
            pbo=0.03, dsr=0.97, sharpe=1.2, max_drawdown=0.15
        ),
        RegimeType.CRISIS: RegimeThresholds(
            pbo=0.02, dsr=0.99, sharpe=2.0, max_drawdown=0.10
        ),
    }

    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        crisis_dd_threshold: float = -0.15,
        high_vol_threshold: float = 2.0,
    ):
        """
        Initialize regime detector.

        Args:
            vol_lookback: Lookback period for volatility calculation
            trend_lookback: Lookback period for trend detection
            crisis_dd_threshold: Drawdown threshold to trigger crisis regime
            high_vol_threshold: Volatility multiplier to trigger high_vol regime
        """
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.crisis_dd_threshold = crisis_dd_threshold
        self.high_vol_threshold = high_vol_threshold

    def detect(self, prices: pd.Series, as_of_date: pd.Timestamp | None = None) -> RegimeDetection:
        """
        Detect market regime using only data up to as_of_date.

        Args:
            prices: Price series (close prices)
            as_of_date: Detection date (uses last date if None)

        Returns:
            RegimeDetection with regime classification and features
        """
        if as_of_date is not None:
            prices = prices[prices.index <= as_of_date]

        # Calculate features
        features = self._calculate_features(prices)

        # Classify regime using decision tree
        regime, confidence = self._classify_regime(features)

        return RegimeDetection(
            regime=regime,
            confidence=confidence,
            features=features,
            thresholds=self.THRESHOLDS[regime],
        )

    def _calculate_features(self, prices: pd.Series) -> dict[str, float]:
        """Calculate regime detection features."""
        returns = prices.pct_change().dropna()

        # Realized volatility (annualized)
        realized_vol = returns.iloc[-self.vol_lookback :].std() * np.sqrt(252)

        # Historical average volatility (for normalization)
        if len(returns) > 60:
            hist_vol = returns.iloc[-60:].std() * np.sqrt(252)
        else:
            hist_vol = realized_vol

        # Maximum drawdown over trend lookback
        rolling_max = prices.rolling(window=self.trend_lookback, min_periods=1).max()
        drawdowns = (prices - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trend strength (linear regression slope)
        recent_prices = prices.iloc[-self.trend_lookback :]
        if len(recent_prices) >= 2:
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices.values, 1)
            # Normalize by price level
            trend_strength = slope / recent_prices.iloc[0] * 252  # Annualized
        else:
            trend_strength = 0.0

        # Volatility ratio (current vs historical)
        vol_ratio = realized_vol / hist_vol if hist_vol > 0 else 1.0

        # Recent return (for crisis detection)
        recent_return = (prices.iloc[-1] / prices.iloc[-min(20, len(prices))] - 1) if len(prices) >= 2 else 0.0

        return {
            "realized_vol": realized_vol,
            "vol_ratio": vol_ratio,
            "max_drawdown": max_drawdown,
            "trend_strength": trend_strength,
            "recent_return": recent_return,
        }

    def _classify_regime(
        self, features: dict[str, float]
    ) -> tuple[RegimeType, float]:
        """
        Classify regime using heuristic decision rules.

        Returns:
            Tuple of (regime, confidence)
        """
        # Crisis: Large recent drawdown or extreme recent decline
        if (
            features["max_drawdown"] < self.crisis_dd_threshold
            or features["recent_return"] < -0.15
        ):
            confidence = min(
                1.0,
                abs(features["max_drawdown"]) / abs(self.crisis_dd_threshold) * 0.8
                + 0.2,
            )
            return RegimeType.CRISIS, confidence

        # High volatility: Vol ratio significantly elevated
        if features["vol_ratio"] > self.high_vol_threshold:
            confidence = min(1.0, features["vol_ratio"] / self.high_vol_threshold * 0.7 + 0.3)
            return RegimeType.HIGH_VOL, confidence

        # Trending: Strong trend with moderate volatility
        if abs(features["trend_strength"]) > 0.3 and features["vol_ratio"] < 1.5:
            confidence = min(1.0, abs(features["trend_strength"]) / 0.5 * 0.6 + 0.4)
            return RegimeType.TRENDING, confidence

        # Default: Normal regime
        confidence = 0.7  # Moderate confidence for normal regime
        return RegimeType.NORMAL, confidence


def detect_regime(
    prices: pd.Series, as_of_date: pd.Timestamp | None = None
) -> RegimeDetection:
    """
    Convenience function for regime detection.

    Args:
        prices: Price series (close prices)
        as_of_date: Detection date (uses last date if None)

    Returns:
        RegimeDetection result

    Example:
        >>> detection = detect_regime(df['close'])
        >>> print(f"Regime: {detection.regime}")
        >>> print(f"Thresholds: PBO < {detection.thresholds.pbo}")
    """
    detector = RegimeDetector()
    return detector.detect(prices, as_of_date)
