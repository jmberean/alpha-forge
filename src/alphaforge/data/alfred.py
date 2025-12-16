"""
ALFRED (Archival Federal Reserve Economic Data) integration.

ALFRED provides vintage data for economic indicators, allowing point-in-time
queries that respect when data was actually released (not just observed).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from alphaforge.data.bitemporal import BiTemporalRecord, BiTemporalStore

# Load .env file
load_dotenv()


@dataclass
class ALFREDSeries:
    """Configuration for an ALFRED data series."""

    series_id: str
    name: str
    frequency: str  # 'D', 'W', 'M', 'Q', 'A'
    units: str


# Key macro series with full vintage history
MACRO_SERIES = {
    "GDP": ALFREDSeries("GDP", "Gross Domestic Product", "Q", "Billions of Dollars"),
    "UNRATE": ALFREDSeries("UNRATE", "Unemployment Rate", "M", "Percent"),
    "CPIAUCSL": ALFREDSeries("CPIAUCSL", "Consumer Price Index", "M", "Index 1982-84=100"),
    "FEDFUNDS": ALFREDSeries("FEDFUNDS", "Federal Funds Rate", "M", "Percent"),
    "DGS10": ALFREDSeries("DGS10", "10-Year Treasury Rate", "D", "Percent"),
    "T10Y2Y": ALFREDSeries("T10Y2Y", "10Y-2Y Yield Spread", "D", "Percent"),
    "UMCSENT": ALFREDSeries("UMCSENT", "Consumer Sentiment", "M", "Index 1966:Q1=100"),
    "INDPRO": ALFREDSeries("INDPRO", "Industrial Production", "M", "Index 2017=100"),
    "PAYEMS": ALFREDSeries("PAYEMS", "Nonfarm Payrolls", "M", "Thousands"),
    "HOUST": ALFREDSeries("HOUST", "Housing Starts", "M", "Thousands of Units"),
    "RETAILSALES": ALFREDSeries("RSXFS", "Retail Sales", "M", "Millions of Dollars"),
    "PCE": ALFREDSeries("PCE", "Personal Consumption Expenditures", "M", "Billions"),
    "M2": ALFREDSeries("M2SL", "M2 Money Supply", "M", "Billions"),
    "VIXCLS": ALFREDSeries("VIXCLS", "VIX Volatility Index", "D", "Index"),
}


class ALFREDClient:
    """
    Client for ALFRED vintage data API.

    Supports both real FRED API calls and mock mode for testing.
    """

    def __init__(self, api_key: str | None = None, use_mock: bool = False):
        """
        Initialize ALFRED client.

        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/)
                    If not provided, reads from FRED_API_KEY environment variable
            use_mock: If True, use mock data instead of real API calls (default: False)
        """
        self.use_mock = use_mock
        self.base_url = "https://api.stlouisfed.org/fred"
        self._last_request_time = 0.0

        if not use_mock:
            self.api_key = api_key or os.environ.get("FRED_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "FRED_API_KEY required. Get one at https://fred.stlouisfed.org/ "
                    "or set FRED_API_KEY environment variable"
                )
        else:
            self.api_key = None

    def _rate_limit(self) -> None:
        """Enforce rate limit of 120 requests/minute (500ms between requests)."""
        elapsed = time.time() - self._last_request_time
        min_interval = 0.5  # 500ms = 120 req/min
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def get_vintage_data(
        self,
        series_id: str,
        vintage_date: date,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> pd.DataFrame:
        """
        Get vintage data: what was known on a specific date.

        Args:
            series_id: FRED series ID (e.g., "GDP")
            vintage_date: Date of the vintage (what was known on this date)
            observation_start: Start of observation period
            observation_end: End of observation period

        Returns:
            DataFrame with columns: observation_date, value, revision_number
        """
        if self.use_mock:
            return self._mock_vintage_data(
                series_id, vintage_date, observation_start, observation_end
            )

        # Real FRED API call
        self._rate_limit()

        params = {
            "api_key": self.api_key,
            "file_type": "json",
            "series_id": series_id,
            "realtime_start": vintage_date.isoformat(),
            "realtime_end": vintage_date.isoformat(),
        }

        if observation_start:
            params["observation_start"] = observation_start.isoformat()
        if observation_end:
            params["observation_end"] = observation_end.isoformat()

        try:
            response = requests.get(
                f"{self.base_url}/series/observations", params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            # Convert to DataFrame
            if not observations:
                return pd.DataFrame(columns=["observation_date", "value", "revision_number"])

            df = pd.DataFrame(observations)
            df = df.rename(columns={"date": "observation_date"})

            # Parse dates and values
            df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["revision_number"] = 0  # FRED doesn't expose revision numbers directly

            return df[["observation_date", "value", "revision_number"]]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"FRED API request failed: {e}") from e

    def _mock_vintage_data(
        self,
        series_id: str,
        vintage_date: date,
        observation_start: date | None,
        observation_end: date | None,
    ) -> pd.DataFrame:
        """
        Mock vintage data for testing.

        Simulates realistic vintage behavior:
        - GDP: Released ~30 days after quarter end
        - Monthly indicators: Released ~15 days after month end
        - Daily indicators: Same-day release
        """
        if series_id not in MACRO_SERIES:
            return pd.DataFrame()

        series = MACRO_SERIES[series_id]

        # Determine release lag
        if series.frequency == "Q":
            release_lag = 30  # days
        elif series.frequency == "M":
            release_lag = 15  # days
        else:
            release_lag = 0  # same day

        # Generate mock observations
        observations = []

        if observation_start and observation_end:
            dates = pd.date_range(observation_start, observation_end, freq=series.frequency)

            for obs_date in dates:
                # Estimate release date
                release_date = obs_date + timedelta(days=release_lag)

                # Only include if released by vintage_date
                if release_date.date() <= vintage_date:
                    # Mock value (in production, comes from API)
                    value = self._generate_mock_value(series_id, obs_date.date())

                    observations.append(
                        {
                            "observation_date": obs_date.date(),
                            "value": value,
                            "revision_number": 0,
                        }
                    )

        return pd.DataFrame(observations)

    def _generate_mock_value(self, series_id: str, obs_date: date) -> float:
        """Generate mock economic data value."""
        # Simple hash-based deterministic values for testing
        import hashlib

        hash_input = f"{series_id}_{obs_date}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Scale to reasonable ranges
        if series_id == "GDP":
            return 20000 + (hash_value % 5000)  # $20-25 trillion
        elif series_id == "UNRATE":
            return 3.0 + (hash_value % 100) / 20.0  # 3-8%
        elif series_id == "CPIAUCSL":
            return 250 + (hash_value % 50)  # 250-300
        elif series_id in ["FEDFUNDS", "DGS10"]:
            return (hash_value % 500) / 100.0  # 0-5%
        elif series_id == "VIXCLS":
            return 10 + (hash_value % 50)  # 10-60
        else:
            return float(hash_value % 1000)

    def get_all_vintages(
        self,
        series_id: str,
        observation_date: date,
    ) -> list[dict[str, Any]]:
        """
        Get all vintages for a specific observation.

        Shows revision history: first release, second release, final.

        Args:
            series_id: FRED series ID
            observation_date: The date being measured

        Returns:
            List of vintages with release_date and value
        """
        if self.use_mock:
            return self._mock_all_vintages(series_id, observation_date)

        # Real FRED API call
        self._rate_limit()

        params = {
            "api_key": self.api_key,
            "file_type": "json",
            "series_id": series_id,
            "observation_start": observation_date.isoformat(),
            "observation_end": observation_date.isoformat(),
        }

        try:
            response = requests.get(
                f"{self.base_url}/series/observations", params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            vintages = []
            for i, obs in enumerate(observations):
                try:
                    vintages.append(
                        {
                            "vintage_date": pd.to_datetime(obs["realtime_start"]).date(),
                            "value": float(obs["value"]),
                            "revision": i,
                        }
                    )
                except (ValueError, KeyError):
                    continue

            return vintages

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"FRED API request failed: {e}") from e

    def _mock_all_vintages(
        self,
        series_id: str,
        observation_date: date,
    ) -> list[dict[str, Any]]:
        """Mock all vintages for testing."""
        if series_id not in MACRO_SERIES:
            return []

        series = MACRO_SERIES[series_id]

        # Determine release schedule
        if series.frequency == "Q":
            # GDP example: 3 releases (advance, preliminary, final)
            first_release = observation_date + timedelta(days=30)
            second_release = observation_date + timedelta(days=60)
            final_release = observation_date + timedelta(days=90)

            base_value = self._generate_mock_value(series_id, observation_date)

            return [
                {"vintage_date": first_release, "value": base_value * 0.98, "revision": 0},
                {"vintage_date": second_release, "value": base_value * 0.99, "revision": 1},
                {"vintage_date": final_release, "value": base_value, "revision": 2},
            ]
        else:
            # Single release for monthly/daily
            release_date = observation_date + timedelta(days=15 if series.frequency == "M" else 0)
            value = self._generate_mock_value(series_id, observation_date)

            return [
                {"vintage_date": release_date, "value": value, "revision": 0},
            ]


class ALFREDSync:
    """
    Synchronize ALFRED data to bi-temporal store.

    Runs daily to fetch latest vintages and update the store.
    """

    def __init__(self, client: ALFREDClient, store: BiTemporalStore):
        """
        Initialize ALFRED sync.

        Args:
            client: ALFRED API client
            store: Bi-temporal store to sync to
        """
        self.client = client
        self.store = store

    def sync_series(
        self,
        series_id: str,
        start_date: date,
        end_date: date,
        vintage_date: date,
    ) -> int:
        """
        Sync a single series to bi-temporal store.

        Args:
            series_id: FRED series ID
            start_date: Start of observation period
            end_date: End of observation period
            vintage_date: Vintage date to sync

        Returns:
            Number of records inserted
        """
        # Get vintage data
        df = self.client.get_vintage_data(
            series_id=series_id,
            vintage_date=vintage_date,
            observation_start=start_date,
            observation_end=end_date,
        )

        count = 0
        for _, row in df.iterrows():
            # Calculate release date (vintage_date for this snapshot)
            release_date = datetime.combine(vintage_date, datetime.min.time())
            transaction_time = datetime.now()

            record = BiTemporalRecord(
                entity_id="US_MACRO",
                indicator_name=series_id,
                observation_date=row["observation_date"],
                release_date=release_date,
                transaction_time=transaction_time,
                value=row["value"],
                revision_number=int(row["revision_number"]),
                source="ALFRED",
                valid_to=None,
            )

            self.store.insert(record)
            count += 1

        return count

    def daily_sync(self, vintage_date: date | None = None) -> dict[str, int]:
        """
        Run daily sync for all tracked series.

        Args:
            vintage_date: Date to sync (defaults to yesterday)

        Returns:
            Dict of series_id -> records inserted
        """
        if vintage_date is None:
            vintage_date = date.today() - timedelta(days=1)

        results = {}

        # Sync last 10 years of data for each series
        end_date = vintage_date
        start_date = end_date - timedelta(days=3650)

        for series_id, series_info in MACRO_SERIES.items():
            count = self.sync_series(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                vintage_date=vintage_date,
            )
            results[series_id] = count

        return results


def get_macro_feature(
    store: BiTemporalStore,
    series_id: str,
    observation_date: date,
    as_of_date: datetime,
) -> float | None:
    """
    Convenience function to get macro feature with PIT semantics.

    Args:
        store: BiTemporalStore
        series_id: FRED series ID (e.g., "GDP")
        observation_date: Date being measured
        as_of_date: Point in time for query

    Returns:
        Value as known at as_of_date, or None if not available

    Example:
        >>> gdp = get_macro_feature(store, "GDP", date(2020, 3, 31), datetime(2020, 4, 28))
        >>> # Returns None - Q1 2020 GDP not yet released
    """
    return store.get_pit_value(
        entity_id="US_MACRO",
        indicator_name=series_id,
        observation_date=observation_date,
        as_of_date=as_of_date,
    )
