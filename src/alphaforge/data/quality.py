"""
Data quality validation framework.

Automated checks to ensure data integrity before backtesting.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List

import pandas as pd


class Severity(Enum):
    """Severity level for quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """A single data quality issue."""

    check_name: str
    severity: Severity
    message: str
    symbol: str | None = None
    date: date | None = None
    value: float | None = None


@dataclass
class QualityReport:
    """Data quality validation report."""

    timestamp: datetime
    issues: List[QualityIssue]
    checks_run: int
    checks_passed: int
    checks_failed: int

    @property
    def has_critical(self) -> bool:
        """Check if report has critical issues."""
        return any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def has_errors(self) -> bool:
        """Check if report has errors."""
        return any(i.severity in [Severity.ERROR, Severity.CRITICAL] for i in self.issues)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Data Quality Report - {self.timestamp}",
            f"Checks: {self.checks_run} total, {self.checks_passed} passed, {self.checks_failed} failed",
            f"Issues: {len(self.issues)} total",
        ]

        if self.issues:
            lines.append("\nIssues by Severity:")
            for severity in Severity:
                count = sum(1 for i in self.issues if i.severity == severity)
                if count > 0:
                    lines.append(f"  {severity.value}: {count}")

        return "\n".join(lines)


class DataQualityChecker:
    """
    Comprehensive data quality validation.

    Runs suite of checks on OHLCV data to detect issues.
    """

    def __init__(self):
        """Initialize checker."""
        self.issues: List[QualityIssue] = []

    def check_all(self, df: pd.DataFrame, symbol: str) -> QualityReport:
        """
        Run all quality checks.

        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Symbol being checked

        Returns:
            QualityReport with all issues found
        """
        self.issues = []
        checks_run = 0
        checks_passed = 0

        # Run all checks
        check_methods = [
            self.check_duplicate_timestamps,
            self.check_temporal_ordering,
            self.check_ohlc_consistency,
            self.check_missing_data,
            self.check_price_sanity,
            self.check_volume_sanity,
            self.check_large_gaps,
            self.check_zero_prices,
            self.check_negative_values,
        ]

        for check_method in check_methods:
            checks_run += 1
            try:
                if check_method(df, symbol):
                    checks_passed += 1
            except Exception as e:
                self.issues.append(QualityIssue(
                    check_name=check_method.__name__,
                    severity=Severity.CRITICAL,
                    message=f"Check failed with exception: {str(e)}",
                    symbol=symbol,
                ))

        checks_failed = checks_run - checks_passed

        return QualityReport(
            timestamp=datetime.now(),
            issues=self.issues,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def check_duplicate_timestamps(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for duplicate timestamps."""
        duplicates = df.index.duplicated()

        if duplicates.any():
            dup_dates = df.index[duplicates].tolist()
            for dup_date in dup_dates[:10]:  # Report first 10
                self.issues.append(QualityIssue(
                    check_name="duplicate_timestamps",
                    severity=Severity.CRITICAL,
                    message=f"Duplicate timestamp detected",
                    symbol=symbol,
                    date=dup_date.date() if hasattr(dup_date, 'date') else dup_date,
                ))
            return False

        return True

    def check_temporal_ordering(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check that timestamps are in order."""
        if not df.index.is_monotonic_increasing:
            self.issues.append(QualityIssue(
                check_name="temporal_ordering",
                severity=Severity.CRITICAL,
                message="Timestamps not in ascending order",
                symbol=symbol,
            ))
            return False

        return True

    def check_ohlc_consistency(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check OHLC relationships: high >= open,close,low."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return True  # Skip if not OHLC data

        # High should be >= all others
        violations_high_low = df['high'] < df['low']
        violations_high_open = df['high'] < df['open']
        violations_high_close = df['high'] < df['close']

        # Low should be <= all others
        violations_low_open = df['low'] > df['open']
        violations_low_close = df['low'] > df['close']

        all_violations = (
            violations_high_low | violations_high_open | violations_high_close |
            violations_low_open | violations_low_close
        )

        if all_violations.any():
            violation_dates = df.index[all_violations].tolist()
            for vio_date in violation_dates[:10]:  # Report first 10
                self.issues.append(QualityIssue(
                    check_name="ohlc_consistency",
                    severity=Severity.ERROR,
                    message="OHLC relationship violated (e.g., high < low)",
                    symbol=symbol,
                    date=vio_date.date() if hasattr(vio_date, 'date') else vio_date,
                ))
            return False

        return True

    def check_missing_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for missing data (NaN values)."""
        missing = df.isnull().any(axis=1)

        if missing.any():
            missing_dates = df.index[missing].tolist()
            for miss_date in missing_dates[:10]:  # Report first 10
                self.issues.append(QualityIssue(
                    check_name="missing_data",
                    severity=Severity.WARNING,
                    message="Missing values (NaN) detected",
                    symbol=symbol,
                    date=miss_date.date() if hasattr(miss_date, 'date') else miss_date,
                ))
            return False

        return True

    def check_price_sanity(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for unrealistic price movements (>50% in one day)."""
        if 'close' not in df.columns:
            return True

        returns = df['close'].pct_change()
        large_moves = abs(returns) > 0.50  # 50% threshold

        if large_moves.any():
            move_dates = df.index[large_moves].tolist()
            for move_date in move_dates[:10]:
                move_pct = returns.loc[move_date] * 100
                self.issues.append(QualityIssue(
                    check_name="price_sanity",
                    severity=Severity.WARNING,
                    message=f"Large price move: {move_pct:.1f}% (check for stock split)",
                    symbol=symbol,
                    date=move_date.date() if hasattr(move_date, 'date') else move_date,
                    value=move_pct,
                ))
            return False

        return True

    def check_volume_sanity(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for unrealistic volume (>10x median)."""
        if 'volume' not in df.columns:
            return True

        median_vol = df['volume'].median()
        if median_vol == 0:
            return True  # Skip if no volume data

        volume_ratio = df['volume'] / median_vol
        large_volume = volume_ratio > 10.0

        if large_volume.any():
            vol_dates = df.index[large_volume].tolist()
            for vol_date in vol_dates[:5]:  # Report first 5
                ratio = volume_ratio.loc[vol_date]
                self.issues.append(QualityIssue(
                    check_name="volume_sanity",
                    severity=Severity.INFO,
                    message=f"Volume spike: {ratio:.1f}x median (may be valid)",
                    symbol=symbol,
                    date=vol_date.date() if hasattr(vol_date, 'date') else vol_date,
                    value=ratio,
                ))

        return True  # Not a failure, just informational

    def check_large_gaps(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for large time gaps in data."""
        if len(df) < 2:
            return True

        time_diffs = df.index.to_series().diff()
        median_diff = time_diffs.median()

        # Flag gaps > 5x median spacing
        large_gaps = time_diffs > (median_diff * 5)

        if large_gaps.any():
            gap_dates = df.index[large_gaps].tolist()
            for gap_date in gap_dates[:5]:
                gap_size = time_diffs.loc[gap_date]
                self.issues.append(QualityIssue(
                    check_name="large_gaps",
                    severity=Severity.WARNING,
                    message=f"Large time gap: {gap_size} (expected ~{median_diff})",
                    symbol=symbol,
                    date=gap_date.date() if hasattr(gap_date, 'date') else gap_date,
                ))
            return False

        return True

    def check_zero_prices(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for zero prices."""
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]

        zero_prices = (df[price_cols] == 0).any(axis=1)

        if zero_prices.any():
            zero_dates = df.index[zero_prices].tolist()
            for zero_date in zero_dates[:10]:
                self.issues.append(QualityIssue(
                    check_name="zero_prices",
                    severity=Severity.ERROR,
                    message="Zero price detected",
                    symbol=symbol,
                    date=zero_date.date() if hasattr(zero_date, 'date') else zero_date,
                ))
            return False

        return True

    def check_negative_values(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for negative prices or volumes."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        negative_values = (df[numeric_cols] < 0).any(axis=1)

        if negative_values.any():
            neg_dates = df.index[negative_values].tolist()
            for neg_date in neg_dates[:10]:
                self.issues.append(QualityIssue(
                    check_name="negative_values",
                    severity=Severity.CRITICAL,
                    message="Negative price or volume detected",
                    symbol=symbol,
                    date=neg_date.date() if hasattr(neg_date, 'date') else neg_date,
                ))
            return False

        return True


def validate_data_quality(
    df: pd.DataFrame,
    symbol: str,
    allow_warnings: bool = True,
) -> QualityReport:
    """
    Validate data quality and return report.

    Args:
        df: OHLCV DataFrame
        symbol: Symbol being validated
        allow_warnings: If False, treat warnings as failures

    Returns:
        QualityReport

    Raises:
        ValueError: If critical issues or errors found
    """
    checker = DataQualityChecker()
    report = checker.check_all(df, symbol)

    # Raise exception for critical issues
    if report.has_critical:
        critical_msgs = [
            i.message for i in report.issues
            if i.severity == Severity.CRITICAL
        ]
        raise ValueError(
            f"Critical data quality issues for {symbol}:\n" + "\n".join(critical_msgs)
        )

    # Optionally raise for errors
    if not allow_warnings and report.has_errors:
        error_msgs = [
            i.message for i in report.issues
            if i.severity in [Severity.ERROR, Severity.WARNING]
        ]
        raise ValueError(
            f"Data quality issues for {symbol}:\n" + "\n".join(error_msgs)
        )

    return report
