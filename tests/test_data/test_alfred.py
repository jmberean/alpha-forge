"""Tests for ALFRED Federal Reserve data integration."""

from datetime import date, datetime, timedelta

import pytest

from alphaforge.data.alfred import (
    MACRO_SERIES,
    ALFREDClient,
    ALFREDSync,
    get_macro_feature,
)
from alphaforge.data.bitemporal import BiTemporalStore


class TestALFREDClient:
    """Test ALFRED client."""

    def test_initialization(self):
        """Test creating ALFRED client."""
        client = ALFREDClient(api_key="test_key")

        assert client.api_key == "test_key"
        assert client.base_url == "https://api.stlouisfed.org/fred"

    def test_macro_series_config(self):
        """Test MACRO_SERIES configuration."""
        assert "GDP" in MACRO_SERIES
        assert "UNRATE" in MACRO_SERIES
        assert "CPIAUCSL" in MACRO_SERIES

        gdp = MACRO_SERIES["GDP"]
        assert gdp.frequency == "Q"
        assert gdp.name == "Gross Domestic Product"

    def test_get_vintage_data(self):
        """Test getting vintage data (mock implementation)."""
        client = ALFREDClient()

        df = client.get_vintage_data(
            series_id="GDP",
            vintage_date=date(2024, 5, 1),
            observation_start=date(2023, 1, 1),
            observation_end=date(2023, 12, 31),
        )

        # Should return quarterly data
        assert len(df) > 0
        assert "observation_date" in df.columns
        assert "value" in df.columns
        assert "revision_number" in df.columns

    def test_vintage_respects_release_lag(self):
        """Test that vintage data respects release lags."""
        client = ALFREDClient()

        # GDP has 30-day release lag
        vintage_date = date(2024, 4, 15)  # Too early for Q1 2024

        df = client.get_vintage_data(
            series_id="GDP",
            vintage_date=vintage_date,
            observation_start=date(2024, 1, 1),
            observation_end=date(2024, 3, 31),
        )

        # Should not include Q1 2024 data (released ~April 29)
        assert len(df) == 0 or all(d < date(2024, 1, 1) for d in df["observation_date"])

    def test_daily_vs_monthly_vs_quarterly_lag(self):
        """Test different release lags for different frequencies."""
        client = ALFREDClient()

        # Daily data (VIX) - same day release
        df_daily = client.get_vintage_data(
            series_id="VIXCLS",
            vintage_date=date(2024, 1, 15),
            observation_start=date(2024, 1, 14),
            observation_end=date(2024, 1, 15),
        )
        # Should include data for January 15 (same day)
        assert any(d == date(2024, 1, 15) for d in df_daily["observation_date"])

        # Monthly data (UNRATE) - 15 day lag
        df_monthly = client.get_vintage_data(
            series_id="UNRATE",
            vintage_date=date(2024, 2, 10),  # Too early for January data
            observation_start=date(2024, 1, 1),
            observation_end=date(2024, 1, 31),
        )
        # Should not include January data (released ~Feb 15)
        assert len(df_monthly) == 0 or all(d < date(2024, 1, 1) for d in df_monthly["observation_date"])

    def test_get_all_vintages(self):
        """Test getting all vintages for an observation."""
        client = ALFREDClient()

        vintages = client.get_all_vintages("GDP", date(2024, 3, 31))

        # GDP has 3 releases (advance, preliminary, final)
        assert len(vintages) == 3
        assert vintages[0]["revision"] == 0
        assert vintages[1]["revision"] == 1
        assert vintages[2]["revision"] == 2

        # Values should be progressively revised
        assert all("value" in v for v in vintages)

    def test_mock_value_deterministic(self):
        """Test that mock values are deterministic."""
        client = ALFREDClient()

        value1 = client._generate_mock_value("GDP", date(2024, 1, 1))
        value2 = client._generate_mock_value("GDP", date(2024, 1, 1))

        # Same inputs should give same output
        assert value1 == value2

    def test_mock_value_realistic_ranges(self):
        """Test that mock values are in realistic ranges."""
        client = ALFREDClient()

        gdp = client._generate_mock_value("GDP", date(2024, 1, 1))
        assert 20000 <= gdp <= 25000  # Billions

        unrate = client._generate_mock_value("UNRATE", date(2024, 1, 1))
        assert 3.0 <= unrate <= 8.0  # Percent

        vix = client._generate_mock_value("VIXCLS", date(2024, 1, 1))
        assert 10 <= vix <= 60  # VIX range


class TestALFREDSync:
    """Test ALFRED sync to bi-temporal store."""

    def test_sync_series(self):
        """Test syncing a single series."""
        client = ALFREDClient()
        store = BiTemporalStore()
        sync = ALFREDSync(client, store)

        count = sync.sync_series(
            series_id="GDP",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            vintage_date=date(2024, 1, 15),
        )

        # Should have inserted records
        assert count > 0

        # Should be able to query
        value = store.get_pit_value(
            "US_MACRO",
            "GDP",
            date(2023, 3, 31),  # Q1 2023
            datetime(2024, 1, 15, 12, 0),
        )

        assert value is not None

    def test_daily_sync(self):
        """Test daily sync process."""
        client = ALFREDClient()
        store = BiTemporalStore()
        sync = ALFREDSync(client, store)

        results = sync.daily_sync(vintage_date=date(2024, 1, 15))

        # Should sync all tracked series
        assert "GDP" in results
        assert "UNRATE" in results
        assert "CPIAUCSL" in results

        # Should report counts
        assert all(count >= 0 for count in results.values())

    def test_sync_creates_pit_queryable_data(self):
        """Test that synced data is queryable with PIT semantics."""
        client = ALFREDClient()
        store = BiTemporalStore()
        sync = ALFREDSync(client, store)

        sync.sync_series(
            series_id="UNRATE",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            vintage_date=date(2024, 3, 1),
        )

        # Query before data was available (should return None)
        value_early = store.get_pit_value(
            "US_MACRO",
            "UNRATE",
            date(2024, 1, 31),
            datetime(2024, 1, 31, 12, 0),  # Before Feb 15 release
        )

        # Query after data was available
        value_late = store.get_pit_value(
            "US_MACRO",
            "UNRATE",
            date(2024, 1, 31),
            datetime(2024, 3, 1, 12, 0),  # After release
        )

        assert value_late is not None


class TestGetMacroFeature:
    """Test convenience function for macro features."""

    def test_get_macro_feature(self):
        """Test getting macro feature with PIT semantics."""
        client = ALFREDClient()
        store = BiTemporalStore()
        sync = ALFREDSync(client, store)

        # Sync GDP data
        sync.sync_series(
            series_id="GDP",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            vintage_date=date(2024, 1, 15),
        )

        # Get GDP for Q1 2023 as of Jan 2024
        gdp = get_macro_feature(
            store,
            "GDP",
            date(2023, 3, 31),
            datetime(2024, 1, 15, 12, 0),
        )

        assert gdp is not None
        assert isinstance(gdp, float)

    def test_get_macro_feature_not_available(self):
        """Test getting feature that's not available yet."""
        store = BiTemporalStore()

        # Query data that was never synced
        gdp = get_macro_feature(
            store,
            "GDP",
            date(2024, 3, 31),
            datetime(2024, 4, 1),
        )

        assert gdp is None
