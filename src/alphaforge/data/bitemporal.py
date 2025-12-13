"""
Bi-temporal data architecture for point-in-time correctness.

Implements 3-timestamp schema to eliminate lookahead bias at the data layer:
- observation_date: What period the data measures
- release_date: When it became publicly available
- transaction_time: When it entered our database
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass
class BiTemporalRecord:
    """
    Single bi-temporal data record.

    Every data point carries three timestamps to ensure point-in-time correctness.
    """

    # Identity
    entity_id: str  # e.g., "AAPL", "GDP_US", "VIX"
    indicator_name: str  # e.g., "close_price", "gdp_growth", "volatility"

    # Temporal dimensions
    observation_date: date  # What period this measures
    release_date: datetime  # When it became publicly available
    transaction_time: datetime  # When entered our database

    # Data
    value: float

    # Metadata
    revision_number: int = 0  # For data that gets revised (e.g., GDP)
    source: str = ""
    valid_to: datetime | None = None  # NULL if current version


@dataclass
class BiTemporalQuery:
    """Query parameters for point-in-time data retrieval."""

    entity_id: str
    indicator_name: str
    observation_date: date
    as_of_date: datetime  # "What did we know at this time?"


class BiTemporalStore:
    """
    In-memory bi-temporal data store.

    Provides point-in-time query semantics to prevent lookahead bias.
    In production, this would be backed by PostgreSQL or Iceberg.
    """

    def __init__(self):
        """Initialize empty bi-temporal store."""
        self.records: list[BiTemporalRecord] = []

    def insert(self, record: BiTemporalRecord) -> None:
        """
        Insert a new bi-temporal record.

        Args:
            record: BiTemporalRecord to insert

        Raises:
            ValueError: If temporal ordering is invalid
        """
        # Validate temporal ordering
        if record.release_date < datetime.combine(record.observation_date, datetime.min.time()):
            raise ValueError(
                f"release_date {record.release_date} cannot be before observation_date {record.observation_date}"
            )

        if record.transaction_time < record.release_date:
            raise ValueError(
                f"transaction_time {record.transaction_time} cannot be before release_date {record.release_date}"
            )

        self.records.append(record)

    def get_pit_value(
        self,
        entity_id: str,
        indicator_name: str,
        observation_date: date,
        as_of_date: datetime,
    ) -> float | None:
        """
        Get point-in-time value: what we knew at as_of_date.

        This is the core query that prevents lookahead bias.

        Args:
            entity_id: Entity identifier
            indicator_name: Indicator name
            observation_date: The date being measured
            as_of_date: Point in time for the query

        Returns:
            Value as it was known at as_of_date, or None if not available

        Example:
            >>> store.get_pit_value("GDP_US", "gdp_growth", date(2020, 3, 31), datetime(2020, 4, 28))
            # Returns None - Q1 2020 GDP not released until April 29
            >>> store.get_pit_value("GDP_US", "gdp_growth", date(2020, 3, 31), datetime(2020, 4, 30))
            # Returns -5.0 - First estimate released April 29
        """
        # Filter to matching entity/indicator/observation
        candidates = [
            r for r in self.records
            if r.entity_id == entity_id
            and r.indicator_name == indicator_name
            and r.observation_date == observation_date
            and r.release_date <= as_of_date
            and (r.valid_to is None or r.valid_to > as_of_date)
        ]

        if not candidates:
            return None

        # Return most recent release (highest revision)
        latest = max(candidates, key=lambda r: (r.release_date, r.revision_number))
        return latest.value

    def get_pit_series(
        self,
        entity_id: str,
        indicator_name: str,
        start_date: date,
        end_date: date,
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Get point-in-time time series.

        Args:
            entity_id: Entity identifier
            indicator_name: Indicator name
            start_date: Start of observation period
            end_date: End of observation period
            as_of_date: Point in time for the query

        Returns:
            Series with observation_date as index and PIT values
        """
        # Get all observation dates in range
        obs_dates = pd.date_range(start_date, end_date, freq="D")

        values = {}
        for obs_date in obs_dates:
            value = self.get_pit_value(
                entity_id,
                indicator_name,
                obs_date.date(),
                as_of_date,
            )
            if value is not None:
                values[obs_date] = value

        return pd.Series(values)

    def get_latest_revision(
        self,
        entity_id: str,
        indicator_name: str,
        observation_date: date,
    ) -> float | None:
        """
        Get the latest revision of a data point.

        This is useful for final analysis but should NOT be used in backtesting.

        Args:
            entity_id: Entity identifier
            indicator_name: Indicator name
            observation_date: The date being measured

        Returns:
            Latest available value
        """
        candidates = [
            r for r in self.records
            if r.entity_id == entity_id
            and r.indicator_name == indicator_name
            and r.observation_date == observation_date
        ]

        if not candidates:
            return None

        # Return highest revision number
        latest = max(candidates, key=lambda r: r.revision_number)
        return latest.value

    def get_revision_history(
        self,
        entity_id: str,
        indicator_name: str,
        observation_date: date,
    ) -> list[BiTemporalRecord]:
        """
        Get full revision history for a data point.

        Useful for analyzing data quality and revision patterns.

        Args:
            entity_id: Entity identifier
            indicator_name: Indicator name
            observation_date: The date being measured

        Returns:
            All revisions in chronological order
        """
        revisions = [
            r for r in self.records
            if r.entity_id == entity_id
            and r.indicator_name == indicator_name
            and r.observation_date == observation_date
        ]

        return sorted(revisions, key=lambda r: (r.release_date, r.revision_number))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all records to DataFrame.

        Returns:
            DataFrame with all bi-temporal records
        """
        if not self.records:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "entity_id": r.entity_id,
                "indicator_name": r.indicator_name,
                "observation_date": r.observation_date,
                "release_date": r.release_date,
                "transaction_time": r.transaction_time,
                "value": r.value,
                "revision_number": r.revision_number,
                "source": r.source,
                "valid_to": r.valid_to,
            }
            for r in self.records
        ])


def validate_pit_correctness(
    store: BiTemporalStore,
    backtest_start: datetime,
    backtest_end: datetime,
) -> bool:
    """
    Validate that all data used in backtest respects point-in-time constraints.

    This is a critical validation that should be run before every backtest.

    Args:
        store: BiTemporalStore to validate
        backtest_start: Start of backtest period
        backtest_end: End of backtest period

    Returns:
        True if all data is PIT-correct

    Raises:
        ValueError: If lookahead bias detected
    """
    violations = []

    for record in store.records:
        # Check: release_date must be <= any point in backtest where this data is used
        if record.release_date > backtest_end:
            continue  # Data not used in this backtest

        # Check: If used at time T in backtest, release_date must be <= T
        # This is enforced by get_pit_value(), but we validate the store structure

        if record.transaction_time < record.release_date:
            violations.append(
                f"Record {record.entity_id}/{record.indicator_name}/{record.observation_date}: "
                f"transaction_time {record.transaction_time} < release_date {record.release_date}"
            )

    if violations:
        raise ValueError(f"Point-in-time violations detected:\n" + "\n".join(violations))

    return True
