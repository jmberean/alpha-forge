"""Tests for technical indicators."""

import numpy as np

from alphaforge.features.technical import TechnicalIndicators


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators."""

    def test_sma_calculation(self, spy_df):
        """Test SMA calculation."""
        close = spy_df["close"]
        sma_20 = TechnicalIndicators.sma(close, 20)

        # SMA should have NaN for first 19 values
        assert sma_20.isna().sum() == 19

        # Check manual calculation for a point
        idx = 50
        expected = close.iloc[idx - 19 : idx + 1].mean()
        np.testing.assert_almost_equal(sma_20.iloc[idx], expected, decimal=5)

    def test_rsi_bounds(self, spy_df):
        """Test RSI is bounded between 0 and 100."""
        rsi = TechnicalIndicators.rsi(spy_df["close"], 14)

        # Remove NaN values
        rsi_valid = rsi.dropna()

        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_macd_components(self, spy_df):
        """Test MACD returns three components."""
        macd, signal, hist = TechnicalIndicators.macd(spy_df["close"])

        assert len(macd) == len(spy_df)
        assert len(signal) == len(spy_df)
        assert len(hist) == len(spy_df)

        # Histogram should be MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            hist[valid_idx].values,
            (macd[valid_idx] - signal[valid_idx]).values,
            decimal=10,
        )

    def test_bollinger_bands_order(self, spy_df):
        """Test Bollinger bands are properly ordered."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(spy_df["close"])

        # Remove NaN
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())

        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_atr_positive(self, spy_df):
        """Test ATR is always positive."""
        atr = TechnicalIndicators.atr(
            spy_df["high"], spy_df["low"], spy_df["close"], 14
        )

        atr_valid = atr.dropna()
        assert (atr_valid >= 0).all()

    def test_compute_all(self, spy_df):
        """Test compute_all adds all expected indicators."""
        # We need fresh df without indicators
        df = spy_df[["open", "high", "low", "close", "volume"]].copy()
        result = TechnicalIndicators.compute_all(df)

        # Check for expected indicators
        expected_cols = [
            "sma_20",
            "sma_50",
            "sma_200",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "atr_14",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_lookahead_bias(self, spy_df):
        """Test that indicators only use past data (no centered windows)."""
        close = spy_df["close"]

        # SMA at index i should only use data from i-19 to i
        sma_20 = TechnicalIndicators.sma(close, 20)

        # Verify by recalculating at a specific point
        idx = 100
        manual_sma = close.iloc[idx - 19 : idx + 1].mean()
        np.testing.assert_almost_equal(sma_20.iloc[idx], manual_sma, decimal=10)

        # The value should NOT include future data
        # If we change a future value, SMA should not change
        close_modified = close.copy()
        close_modified.iloc[idx + 1] = close_modified.iloc[idx + 1] * 2

        sma_modified = TechnicalIndicators.sma(close_modified, 20)
        np.testing.assert_almost_equal(sma_20.iloc[idx], sma_modified.iloc[idx], decimal=10)
