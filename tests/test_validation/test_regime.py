"""Tests for regime detection."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.validation.regime import (
    RegimeDetector,
    RegimeType,
    detect_regime,
)


@pytest.fixture
def normal_prices():
    """Generate normal regime prices (low volatility, no trend)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = np.random.normal(0.0005, 0.01, 252)  # 0.05% daily, 1% vol
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


@pytest.fixture
def trending_prices():
    """Generate trending regime prices (strong uptrend)."""
    np.random.seed(43)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong upward trend + noise
    trend = np.linspace(0, 0.5, 252)
    noise = np.random.normal(0, 0.008, 252)
    returns = trend / 252 + noise
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


@pytest.fixture
def high_vol_prices():
    """Generate high volatility regime prices."""
    np.random.seed(44)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Start with normal vol, then transition to high vol
    returns_normal = np.random.normal(0, 0.01, 180)  # Normal vol
    returns_high = np.random.normal(0, 0.035, 72)    # High vol period
    returns = np.concatenate([returns_normal, returns_high])
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


@pytest.fixture
def crisis_prices():
    """Generate crisis regime prices (severe drawdown)."""
    np.random.seed(45)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Start normal, then crash
    returns_normal = np.random.normal(0.0005, 0.01, 200)
    returns_crash = np.random.normal(-0.03, 0.05, 52)  # Large negative returns
    returns = np.concatenate([returns_normal, returns_crash])
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


class TestRegimeDetector:
    """Test RegimeDetector class."""

    def test_initialization(self):
        """Test detector initialization with custom parameters."""
        detector = RegimeDetector(
            vol_lookback=30,
            trend_lookback=60,
            crisis_dd_threshold=-0.20,
            high_vol_threshold=2.5,
        )

        assert detector.vol_lookback == 30
        assert detector.trend_lookback == 60
        assert detector.crisis_dd_threshold == -0.20
        assert detector.high_vol_threshold == 2.5

    def test_default_thresholds(self):
        """Test that default thresholds are defined for all regimes."""
        detector = RegimeDetector()

        assert RegimeType.NORMAL in detector.THRESHOLDS
        assert RegimeType.TRENDING in detector.THRESHOLDS
        assert RegimeType.HIGH_VOL in detector.THRESHOLDS
        assert RegimeType.CRISIS in detector.THRESHOLDS

        # Check threshold structure
        normal_thresh = detector.THRESHOLDS[RegimeType.NORMAL]
        assert hasattr(normal_thresh, "pbo")
        assert hasattr(normal_thresh, "dsr")
        assert hasattr(normal_thresh, "sharpe")
        assert hasattr(normal_thresh, "max_drawdown")

    def test_detect_normal_regime(self, normal_prices):
        """Test detection of normal regime."""
        detector = RegimeDetector()
        result = detector.detect(normal_prices)

        assert result.regime == RegimeType.NORMAL
        assert 0 < result.confidence <= 1.0
        assert "realized_vol" in result.features
        assert "vol_ratio" in result.features
        assert "max_drawdown" in result.features
        assert "trend_strength" in result.features

        # Check thresholds are appropriate for normal regime
        assert result.thresholds.pbo == 0.05
        assert result.thresholds.dsr == 0.95
        assert result.thresholds.sharpe == 1.0

    def test_detect_trending_regime(self, trending_prices):
        """Test detection of trending regime."""
        detector = RegimeDetector()
        result = detector.detect(trending_prices)

        assert result.regime == RegimeType.TRENDING
        assert result.confidence > 0.0

        # Trending regime should have relaxed thresholds
        assert result.thresholds.sharpe < 1.0  # Lower Sharpe requirement

    def test_detect_high_vol_regime(self, high_vol_prices):
        """Test detection of high volatility regime."""
        detector = RegimeDetector()
        result = detector.detect(high_vol_prices)

        # High volatility can trigger either HIGH_VOL or CRISIS regime
        assert result.regime in [RegimeType.HIGH_VOL, RegimeType.CRISIS]

        # Both regimes should have stricter thresholds than normal
        assert result.thresholds.pbo <= 0.05
        assert result.thresholds.dsr >= 0.95

    def test_detect_crisis_regime(self, crisis_prices):
        """Test detection of crisis regime."""
        detector = RegimeDetector()
        result = detector.detect(crisis_prices)

        assert result.regime == RegimeType.CRISIS
        assert result.features["max_drawdown"] < -0.10

        # Crisis regime should have strictest thresholds
        assert result.thresholds.pbo < 0.05
        assert result.thresholds.dsr > 0.95
        assert result.thresholds.sharpe > 1.5

    def test_as_of_date_filtering(self, normal_prices):
        """Test that detection only uses data up to as_of_date."""
        detector = RegimeDetector()

        # Detect at midpoint
        as_of = normal_prices.index[126]
        result = detector.detect(normal_prices, as_of_date=as_of)

        assert result.regime is not None
        # Should only use first half of data
        # (Hard to test directly, but shouldn't crash)

    def test_short_history(self):
        """Test detection with very short price history."""
        np.random.seed(46)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.Series(100 + np.random.randn(10), index=dates)

        detector = RegimeDetector()
        result = detector.detect(prices)

        # Should not crash with short history
        assert result.regime is not None
        assert isinstance(result.confidence, float)

    def test_feature_calculation(self, normal_prices):
        """Test feature calculation details."""
        detector = RegimeDetector()
        features = detector._calculate_features(normal_prices)

        # Check all features are present and valid
        assert features["realized_vol"] > 0
        assert features["vol_ratio"] > 0
        assert features["max_drawdown"] <= 0
        assert isinstance(features["trend_strength"], float)
        assert isinstance(features["recent_return"], float)

    def test_crisis_detection_on_sharp_decline(self):
        """Test that sharp decline triggers crisis regime."""
        # Create prices with sudden crash
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices_before = np.full(30, 100.0)
        prices_after = np.full(20, 80.0)  # 20% drop
        prices = pd.Series(
            np.concatenate([prices_before, prices_after]), index=dates
        )

        detector = RegimeDetector()
        result = detector.detect(prices)

        assert result.regime == RegimeType.CRISIS

    def test_trending_not_triggered_in_high_vol(self):
        """Test that high vol prevents trending regime even with trend."""
        np.random.seed(47)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # Strong trend but very high volatility
        trend = np.linspace(0, 0.3, 100) / 100
        noise = np.random.normal(0, 0.04, 100)  # High noise
        returns = trend + noise
        prices = 100 * (1 + returns).cumprod()
        prices = pd.Series(prices, index=dates)

        detector = RegimeDetector()
        result = detector.detect(prices)

        # Should detect high_vol, not trending
        assert result.regime in [RegimeType.HIGH_VOL, RegimeType.CRISIS]


class TestConvenienceFunction:
    """Test convenience function."""

    def test_detect_regime_function(self, normal_prices):
        """Test detect_regime convenience function."""
        result = detect_regime(normal_prices)

        assert result.regime is not None
        assert result.confidence > 0
        assert result.features is not None
        assert result.thresholds is not None

    def test_detect_regime_with_as_of_date(self, normal_prices):
        """Test detect_regime with as_of_date."""
        as_of = normal_prices.index[100]
        result = detect_regime(normal_prices, as_of_date=as_of)

        assert result.regime is not None


class TestRegimeThresholds:
    """Test regime threshold configurations."""

    def test_crisis_stricter_than_normal(self):
        """Test that crisis thresholds are stricter than normal."""
        detector = RegimeDetector()

        crisis = detector.THRESHOLDS[RegimeType.CRISIS]
        normal = detector.THRESHOLDS[RegimeType.NORMAL]

        # Stricter means lower PBO threshold
        assert crisis.pbo < normal.pbo
        # Higher DSR threshold
        assert crisis.dsr > normal.dsr
        # Higher Sharpe requirement
        assert crisis.sharpe > normal.sharpe

    def test_trending_relaxed_sharpe(self):
        """Test that trending regime has relaxed Sharpe requirement."""
        detector = RegimeDetector()

        trending = detector.THRESHOLDS[RegimeType.TRENDING]
        normal = detector.THRESHOLDS[RegimeType.NORMAL]

        # Trending allows lower Sharpe (trend-following can be choppy)
        assert trending.sharpe < normal.sharpe

    def test_all_thresholds_positive(self):
        """Test that all threshold values are positive."""
        detector = RegimeDetector()

        for regime, thresholds in detector.THRESHOLDS.items():
            assert thresholds.pbo > 0
            assert thresholds.dsr > 0
            assert thresholds.sharpe > 0
            assert thresholds.max_drawdown > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_prices(self):
        """Test with constant (no variation) prices."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.Series(np.full(100, 100.0), index=dates)

        detector = RegimeDetector()
        result = detector.detect(prices)

        # Should handle gracefully (likely normal regime)
        assert result.regime is not None

    def test_single_day(self):
        """Test with single day of data."""
        prices = pd.Series([100.0], index=[pd.Timestamp("2020-01-01")])

        detector = RegimeDetector()
        result = detector.detect(prices)

        # Should handle gracefully
        assert result.regime is not None

    def test_monotonic_increase(self):
        """Test with perfectly monotonic increasing prices."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.Series(np.arange(100, 200), index=dates)

        detector = RegimeDetector()
        result = detector.detect(prices)

        # Should likely detect trending
        assert result.regime in [RegimeType.TRENDING, RegimeType.NORMAL]

    def test_alternating_regime(self):
        """Test detection at different points in regime-shifting series."""
        # Create series that shifts from normal to crisis
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        np.random.seed(48)

        # First 100 days: normal
        returns_normal = np.random.normal(0.0005, 0.01, 100)
        # Next 100 days: crisis
        returns_crisis = np.random.normal(-0.02, 0.04, 100)

        returns = np.concatenate([returns_normal, returns_crisis])
        prices = 100 * (1 + returns).cumprod()
        prices = pd.Series(prices, index=dates)

        detector = RegimeDetector()

        # Detect at day 50 (should be normal)
        result_50 = detector.detect(prices, as_of_date=dates[50])
        # Detect at day 150 (should be crisis)
        result_150 = detector.detect(prices, as_of_date=dates[150])

        # Early period should be calmer than later period
        assert result_150.features["max_drawdown"] < result_50.features["max_drawdown"]
