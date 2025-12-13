"""Tests for data quality validation framework."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.quality import (
    DataQualityChecker,
    QualityReport,
    Severity,
    validate_data_quality,
)


@pytest.fixture
def clean_ohlcv_data():
    """Create clean OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    data = pd.DataFrame({
        "open": np.random.uniform(95, 105, 100),
        "high": np.random.uniform(100, 110, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(95, 105, 100),
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # Ensure OHLC consistency
    data["high"] = data[["open", "close"]].max(axis=1) + np.random.uniform(0, 5, 100)
    data["low"] = data[["open", "close"]].min(axis=1) - np.random.uniform(0, 5, 100)

    return data


class TestDataQualityChecker:
    """Test DataQualityChecker class."""

    def test_check_clean_data(self, clean_ohlcv_data):
        """Test that clean data passes all checks."""
        checker = DataQualityChecker()

        report = checker.check_all(clean_ohlcv_data, "AAPL")

        assert report.checks_run > 0
        assert not report.has_errors
        assert not report.has_critical

    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
        data = pd.DataFrame({
            "close": [100, 101, 102, 103],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_duplicate_timestamps(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.CRITICAL for i in checker.issues)
        assert any("Duplicate timestamp" in i.message for i in checker.issues)

    def test_temporal_ordering(self):
        """Test detection of non-monotonic timestamps."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"])
        data = pd.DataFrame({
            "close": [100, 101, 102, 103],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_temporal_ordering(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.CRITICAL for i in checker.issues)
        assert any("not in ascending order" in i.message for i in checker.issues)

    def test_ohlc_consistency_violation(self):
        """Test detection of OHLC violations."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "open": [100, 100, 100, 100, 100],
            "high": [105, 105, 95, 105, 105],  # Third row: high < low!
            "low": [95, 95, 98, 95, 95],
            "close": [102, 102, 102, 102, 102],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.ERROR for i in checker.issues)
        assert any("OHLC relationship violated" in i.message for i in checker.issues)

    def test_missing_data(self):
        """Test detection of missing data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "open": [100, 101, np.nan, 103, 104],
            "close": [102, 103, 104, 105, 106],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_missing_data(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.WARNING for i in checker.issues)
        assert any("Missing values" in i.message for i in checker.issues)

    def test_price_sanity_large_move(self):
        """Test detection of unrealistic price movements."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, 180, 102, 103],  # 78% jump!
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_price_sanity(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.WARNING for i in checker.issues)
        assert any("Large price move" in i.message for i in checker.issues)
        assert any("stock split" in i.message for i in checker.issues)

    def test_volume_sanity_spike(self, clean_ohlcv_data):
        """Test detection of volume spikes."""
        data = clean_ohlcv_data.copy()

        # Create volume spike (20x median)
        median_vol = data["volume"].median()
        data.loc[data.index[10], "volume"] = median_vol * 20

        checker = DataQualityChecker()
        result = checker.check_volume_sanity(data, "AAPL")

        # This is INFO level, not a failure
        assert result is True
        assert any(i.severity == Severity.INFO for i in checker.issues)
        assert any("Volume spike" in i.message for i in checker.issues)

    def test_large_gaps(self):
        """Test detection of large time gaps."""
        dates = pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-02-15",  # Large gap!
            "2024-02-16",
        ])

        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_large_gaps(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.WARNING for i in checker.issues)
        assert any("Large time gap" in i.message for i in checker.issues)

    def test_zero_prices(self):
        """Test detection of zero prices."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "open": [100, 101, 0, 103, 104],  # Zero price!
            "close": [102, 103, 104, 105, 106],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_zero_prices(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.ERROR for i in checker.issues)
        assert any("Zero price" in i.message for i in checker.issues)

    def test_negative_values(self):
        """Test detection of negative prices/volumes."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, -50, 103, 104],  # Negative price!
            "volume": [1000, 1000, 1000, 1000, 1000],
        }, index=dates)

        checker = DataQualityChecker()
        result = checker.check_negative_values(data, "AAPL")

        assert not result
        assert any(i.severity == Severity.CRITICAL for i in checker.issues)
        assert any("Negative" in i.message for i in checker.issues)


class TestQualityReport:
    """Test QualityReport class."""

    def test_report_properties(self, clean_ohlcv_data):
        """Test report properties."""
        checker = DataQualityChecker()
        report = checker.check_all(clean_ohlcv_data, "AAPL")

        assert hasattr(report, "timestamp")
        assert hasattr(report, "issues")
        assert hasattr(report, "checks_run")
        assert hasattr(report, "checks_passed")
        assert hasattr(report, "checks_failed")

    def test_has_critical(self):
        """Test has_critical property."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, -50],  # Negative = CRITICAL
        }, index=dates)

        checker = DataQualityChecker()
        report = checker.check_all(data, "AAPL")

        assert report.has_critical

    def test_has_errors(self):
        """Test has_errors property."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "open": [100, 101, 0],  # Zero = ERROR
            "close": [102, 103, 104],
        }, index=dates)

        checker = DataQualityChecker()
        report = checker.check_all(data, "AAPL")

        assert report.has_errors

    def test_summary(self, clean_ohlcv_data):
        """Test summary generation."""
        checker = DataQualityChecker()
        report = checker.check_all(clean_ohlcv_data, "AAPL")

        summary = report.summary()

        assert "Data Quality Report" in summary
        assert "Checks:" in summary
        assert str(report.checks_run) in summary


class TestValidateDataQuality:
    """Test validate_data_quality function."""

    def test_validate_clean_data(self, clean_ohlcv_data):
        """Test validation passes for clean data."""
        report = validate_data_quality(clean_ohlcv_data, "AAPL")

        assert report.checks_passed > 0
        assert not report.has_critical

    def test_validate_raises_on_critical(self):
        """Test validation raises on critical issues."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, -50],  # Negative = CRITICAL
        }, index=dates)

        with pytest.raises(ValueError, match="Critical data quality issues"):
            validate_data_quality(data, "AAPL")

    def test_validate_with_warnings_allowed(self):
        """Test validation with warnings allowed."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, np.nan, 103, 104],  # Missing = WARNING
        }, index=dates)

        # Should not raise with allow_warnings=True
        report = validate_data_quality(data, "AAPL", allow_warnings=True)

        assert report.has_errors  # Missing data is an error/warning

    def test_validate_strict_mode(self):
        """Test validation in strict mode (no warnings)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "close": [100, 101, np.nan, 103, 104],  # Missing = WARNING
        }, index=dates)

        # Should raise with allow_warnings=False
        with pytest.raises(ValueError, match="Data quality issues"):
            validate_data_quality(data, "AAPL", allow_warnings=False)


class TestQualityCheckerEdgeCases:
    """Test edge cases for quality checker."""

    def test_empty_dataframe(self):
        """Test checker on empty DataFrame."""
        data = pd.DataFrame()

        checker = DataQualityChecker()
        report = checker.check_all(data, "AAPL")

        # Should handle gracefully
        assert report.checks_run > 0

    def test_single_row(self):
        """Test checker on single row."""
        data = pd.DataFrame({
            "close": [100],
        }, index=pd.to_datetime(["2024-01-01"]))

        checker = DataQualityChecker()
        report = checker.check_all(data, "AAPL")

        # Should handle gracefully
        assert not report.has_critical

    def test_non_ohlc_data(self):
        """Test checker on non-OHLC data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "some_indicator": np.random.randn(10),
        }, index=dates)

        checker = DataQualityChecker()
        report = checker.check_all(data, "INDICATOR")

        # Should skip OHLC checks gracefully
        assert report.checks_run > 0
