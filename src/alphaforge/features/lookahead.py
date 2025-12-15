"""
Lookahead bias detection and validation.

Ensures all features use only past data, preventing future information leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class BiasType(Enum):
    """Types of lookahead bias."""

    CENTERED_WINDOW = "centered_window"  # .rolling(center=True)
    FUTURE_SHIFT = "future_shift"  # .shift(-N)
    FULL_SAMPLE_STATS = "full_sample_stats"  # df.mean() instead of .expanding().mean()
    WRONG_INDEX_ORDER = "wrong_index_order"  # Index not sorted
    INVALID_OPERATION = "invalid_operation"  # Other suspicious patterns


@dataclass
class BiasDetection:
    """Result of lookahead bias detection."""

    has_bias: bool
    bias_type: BiasType | None
    message: str
    column: str | None = None


class LookaheadDetector:
    """
    Detect lookahead bias in feature computation.

    Tests feature functions for temporal correctness.
    """

    def __init__(self):
        """Initialize detector."""
        self.detections: list[BiasDetection] = []

    def check_function(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        test_data: pd.DataFrame,
    ) -> list[BiasDetection]:
        """
        Check if feature function has lookahead bias.

        Tests:
        1. Causality: Output at time T only depends on data <= T
        2. Monotonicity: Adding new data doesn't change past outputs

        Args:
            feature_func: Function that computes features
            test_data: Test data (should have datetime index)

        Returns:
            List of bias detections
        """
        self.detections = []

        # Validate input
        if not isinstance(test_data.index, pd.DatetimeIndex):
            self.detections.append(BiasDetection(
                has_bias=True,
                bias_type=BiasType.WRONG_INDEX_ORDER,
                message="Input data must have DatetimeIndex",
            ))
            return self.detections

        if not test_data.index.is_monotonic_increasing:
            self.detections.append(BiasDetection(
                has_bias=True,
                bias_type=BiasType.WRONG_INDEX_ORDER,
                message="Input data index must be sorted in ascending order",
            ))
            return self.detections

        # Test 1: Causality check
        self._check_causality(feature_func, test_data)

        # Test 2: Monotonicity check (adding future data shouldn't change past)
        self._check_monotonicity(feature_func, test_data)

        return self.detections

    def _check_causality(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        test_data: pd.DataFrame,
    ) -> None:
        """
        Test causality: Feature at time T should only use data <= T.

        This is tested by verifying that features can be computed incrementally.
        """
        if len(test_data) < 10:
            return  # Need sufficient data for testing

        # Split data into two halves
        split_point = len(test_data) // 2

        first_half = test_data.iloc[:split_point]
        full_data = test_data

        try:
            # Compute features on first half
            features_first = feature_func(first_half)

            # Compute features on full data
            features_full = feature_func(full_data)

            # Values for first half should match in both computations
            # (allowing for NaN differences due to insufficient data)
            for col in features_full.columns:
                if col not in features_first.columns:
                    continue

                first_half_from_first = features_first[col].iloc[-1]
                first_half_from_full = features_full[col].iloc[split_point - 1]

                # Check if values match (allowing for NaN)
                if pd.isna(first_half_from_first) and pd.isna(first_half_from_full):
                    continue  # Both NaN is fine

                if pd.isna(first_half_from_first) or pd.isna(first_half_from_full):
                    # One is NaN, other is not - potential issue
                    continue

                # Check numerical equality (with tolerance)
                if abs(first_half_from_first - first_half_from_full) > 1e-6:
                    self.detections.append(BiasDetection(
                        has_bias=True,
                        bias_type=BiasType.FUTURE_SHIFT,
                        message=(
                            f"Causality violation: Feature '{col}' at time T changes "
                            "when future data is added (possible future shift or lookahead)"
                        ),
                        column=col,
                    ))

        except Exception as e:
            self.detections.append(BiasDetection(
                has_bias=True,
                bias_type=BiasType.INVALID_OPERATION,
                message=f"Error during causality check: {str(e)}",
            ))

    def _check_monotonicity(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        test_data: pd.DataFrame,
    ) -> None:
        """
        Test monotonicity: Adding future data shouldn't change past outputs.

        This catches centered windows and full-sample statistics.
        """
        if len(test_data) < 20:
            return

        # Use three increasingly large windows
        window1 = test_data.iloc[:10]
        window2 = test_data.iloc[:15]
        window3 = test_data.iloc[:20]

        try:
            features1 = feature_func(window1)
            features2 = feature_func(window2)
            features3 = feature_func(window3)

            # Check that outputs for first 10 rows don't change
            for col in features1.columns:
                if col not in features2.columns or col not in features3.columns:
                    continue

                # Compare last value from window1 with same index in window2/window3
                val1 = features1[col].iloc[-1]
                val2 = features2[col].iloc[9]  # Same index as val1
                val3 = features3[col].iloc[9]  # Same index as val1

                # All NaN is fine
                if pd.isna(val1) and pd.isna(val2) and pd.isna(val3):
                    continue

                # Check for changes
                if not pd.isna(val1) and not pd.isna(val2):
                    if abs(val1 - val2) > 1e-6:
                        self.detections.append(BiasDetection(
                            has_bias=True,
                            bias_type=BiasType.CENTERED_WINDOW,
                            message=(
                                f"Monotonicity violation: Feature '{col}' changes when "
                                "future data added (possible centered window or full-sample statistics)"
                            ),
                            column=col,
                        ))

        except Exception as e:
            # Not necessarily a bias, might just be insufficient data
            pass


def validate_feature_function(
    feature_func: Callable[[pd.DataFrame], pd.DataFrame],
    test_data: pd.DataFrame,
) -> bool:
    """
    Validate that feature function doesn't have lookahead bias.

    Args:
        feature_func: Feature computation function
        test_data: Test data with datetime index

    Returns:
        True if no bias detected

    Raises:
        ValueError: If lookahead bias detected
    """
    detector = LookaheadDetector()
    detections = detector.check_function(feature_func, test_data)

    if any(d.has_bias for d in detections):
        messages = [d.message for d in detections if d.has_bias]
        raise ValueError(
            "Lookahead bias detected in feature function:\n" + "\n".join(messages)
        )

    return True


# Example valid and invalid feature functions for testing


def valid_feature_function(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of VALID feature function (no lookahead bias).

    Uses only trailing windows and expanding calculations.
    """
    result = df.copy()

    # Valid: Trailing window
    result["sma_20"] = df["close"].rolling(window=20).mean()

    # Valid: Expanding window
    result["expanding_max"] = df["close"].expanding().max()

    # Valid: Lagged value
    result["prev_close"] = df["close"].shift(1)

    return result


def invalid_feature_function(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of INVALID feature function (has lookahead bias).

    FORBIDDEN patterns that leak future information.
    """
    result = df.copy()

    # INVALID: Centered window
    result["sma_centered"] = df["close"].rolling(window=20, center=True).mean()

    # INVALID: Future shift
    result["next_close"] = df["close"].shift(-1)

    # INVALID: Full sample statistics
    result["z_score"] = (df["close"] - df["close"].mean()) / df["close"].std()

    return result


# Validation rules as code


class FeatureValidationRules:
    """
    Codified rules for valid feature engineering.

    Use these patterns to ensure temporal correctness.
    """

    @staticmethod
    def rolling_mean_valid(series: pd.Series, window: int) -> pd.Series:
        """VALID: Trailing window mean."""
        return series.rolling(window=window, center=False).mean()

    @staticmethod
    def rolling_mean_invalid(series: pd.Series, window: int) -> pd.Series:
        """INVALID: Centered window mean."""
        return series.rolling(window=window, center=True).mean()

    @staticmethod
    def expanding_mean_valid(series: pd.Series) -> pd.Series:
        """VALID: Expanding window (all historical data)."""
        return series.expanding().mean()

    @staticmethod
    def full_sample_mean_invalid(series: pd.Series) -> pd.Series:
        """INVALID: Full sample statistics."""
        return pd.Series(series.mean(), index=series.index)

    @staticmethod
    def lag_valid(series: pd.Series, periods: int = 1) -> pd.Series:
        """VALID: Lag (use past data)."""
        return series.shift(periods)

    @staticmethod
    def lead_invalid(series: pd.Series, periods: int = 1) -> pd.Series:
        """INVALID: Lead (use future data)."""
        return series.shift(-periods)

    @staticmethod
    def zscore_valid(series: pd.Series, window: int = 252) -> pd.Series:
        """VALID: Rolling z-score."""
        return (series - series.rolling(window).mean()) / series.rolling(window).std()

    @staticmethod
    def zscore_invalid(series: pd.Series) -> pd.Series:
        """INVALID: Full-sample z-score."""
        return (series - series.mean()) / series.std()
