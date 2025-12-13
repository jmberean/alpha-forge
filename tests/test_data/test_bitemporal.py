"""Tests for bi-temporal data schema and point-in-time queries."""

from datetime import date, datetime, timedelta

import pytest

from alphaforge.data.bitemporal import (
    BiTemporalRecord,
    BiTemporalStore,
    validate_pit_correctness,
)


class TestBiTemporalRecord:
    """Test BiTemporalRecord dataclass."""

    def test_record_creation(self):
        """Test creating a bi-temporal record."""
        record = BiTemporalRecord(
            entity_id="AAPL",
            indicator_name="close_price",
            observation_date=date(2024, 1, 15),
            release_date=datetime(2024, 1, 15, 16, 0),
            transaction_time=datetime(2024, 1, 15, 16, 30),
            value=150.0,
            revision_number=0,
            source="yfinance",
        )

        assert record.entity_id == "AAPL"
        assert record.value == 150.0
        assert record.revision_number == 0


class TestBiTemporalStore:
    """Test BiTemporalStore class."""

    def test_insert_and_retrieve(self):
        """Test inserting and retrieving a record."""
        store = BiTemporalStore()

        record = BiTemporalRecord(
            entity_id="AAPL",
            indicator_name="close_price",
            observation_date=date(2024, 1, 15),
            release_date=datetime(2024, 1, 15, 16, 0),
            transaction_time=datetime(2024, 1, 15, 16, 30),
            value=150.0,
        )

        store.insert(record)

        # Query after release
        value = store.get_pit_value(
            "AAPL",
            "close_price",
            date(2024, 1, 15),
            datetime(2024, 1, 15, 17, 0),
        )

        assert value == 150.0

    def test_pit_query_before_release(self):
        """Test PIT query before data was released."""
        store = BiTemporalStore()

        record = BiTemporalRecord(
            entity_id="GDP_US",
            indicator_name="gdp_growth",
            observation_date=date(2024, 3, 31),  # Q1 2024
            release_date=datetime(2024, 4, 29, 8, 30),  # Released April 29
            transaction_time=datetime(2024, 4, 29, 8, 35),
            value=-5.0,
        )

        store.insert(record)

        # Query before release - should return None
        value_before = store.get_pit_value(
            "GDP_US",
            "gdp_growth",
            date(2024, 3, 31),
            datetime(2024, 4, 28, 23, 59),  # Day before release
        )

        assert value_before is None

        # Query after release - should return value
        value_after = store.get_pit_value(
            "GDP_US",
            "gdp_growth",
            date(2024, 3, 31),
            datetime(2024, 4, 29, 9, 0),  # After release
        )

        assert value_after == -5.0

    def test_data_revisions(self):
        """Test handling of data revisions."""
        store = BiTemporalStore()

        # First release (advance estimate)
        record1 = BiTemporalRecord(
            entity_id="GDP_US",
            indicator_name="gdp_growth",
            observation_date=date(2024, 3, 31),
            release_date=datetime(2024, 4, 29, 8, 30),
            transaction_time=datetime(2024, 4, 29, 8, 35),
            value=-5.0,
            revision_number=0,
        )

        # Second release (preliminary estimate)
        record2 = BiTemporalRecord(
            entity_id="GDP_US",
            indicator_name="gdp_growth",
            observation_date=date(2024, 3, 31),
            release_date=datetime(2024, 5, 30, 8, 30),
            transaction_time=datetime(2024, 5, 30, 8, 35),
            value=-4.8,
            revision_number=1,
        )

        # Final release
        record3 = BiTemporalRecord(
            entity_id="GDP_US",
            indicator_name="gdp_growth",
            observation_date=date(2024, 3, 31),
            release_date=datetime(2024, 6, 28, 8, 30),
            transaction_time=datetime(2024, 6, 28, 8, 35),
            value=-4.5,
            revision_number=2,
        )

        store.insert(record1)
        store.insert(record2)
        store.insert(record3)

        # PIT query in April: only first estimate available
        value_april = store.get_pit_value(
            "GDP_US", "gdp_growth", date(2024, 3, 31), datetime(2024, 4, 30)
        )
        assert value_april == -5.0

        # PIT query in May: preliminary estimate available
        value_may = store.get_pit_value(
            "GDP_US", "gdp_growth", date(2024, 3, 31), datetime(2024, 6, 1)
        )
        assert value_may == -4.8

        # PIT query in July: final estimate available
        value_july = store.get_pit_value(
            "GDP_US", "gdp_growth", date(2024, 3, 31), datetime(2024, 7, 1)
        )
        assert value_july == -4.5

    def test_temporal_ordering_validation(self):
        """Test validation of temporal ordering."""
        store = BiTemporalStore()

        # Invalid: release_date before observation_date
        with pytest.raises(ValueError, match="release_date.*cannot be before observation_date"):
            record = BiTemporalRecord(
                entity_id="AAPL",
                indicator_name="close_price",
                observation_date=date(2024, 1, 15),
                release_date=datetime(2024, 1, 14, 16, 0),  # Before observation!
                transaction_time=datetime(2024, 1, 15, 16, 30),
                value=150.0,
            )
            store.insert(record)

        # Invalid: transaction_time before release_date
        with pytest.raises(ValueError, match="transaction_time.*cannot be before release_date"):
            record = BiTemporalRecord(
                entity_id="AAPL",
                indicator_name="close_price",
                observation_date=date(2024, 1, 15),
                release_date=datetime(2024, 1, 15, 16, 0),
                transaction_time=datetime(2024, 1, 15, 15, 0),  # Before release!
                value=150.0,
            )
            store.insert(record)

    def test_get_pit_series(self):
        """Test retrieving point-in-time time series."""
        store = BiTemporalStore()

        # Insert daily prices for a week
        for i in range(5):
            obs_date = date(2024, 1, 15) + timedelta(days=i)
            record = BiTemporalRecord(
                entity_id="AAPL",
                indicator_name="close_price",
                observation_date=obs_date,
                release_date=datetime.combine(obs_date, datetime.min.time().replace(hour=16)),
                transaction_time=datetime.combine(obs_date, datetime.min.time().replace(hour=17)),
                value=150.0 + i,
            )
            store.insert(record)

        # Get series as of end of week
        series = store.get_pit_series(
            "AAPL",
            "close_price",
            date(2024, 1, 15),
            date(2024, 1, 19),
            datetime(2024, 1, 19, 23, 59),
        )

        assert len(series) == 5
        assert series.iloc[0] == 150.0
        assert series.iloc[-1] == 154.0

    def test_get_latest_revision(self):
        """Test getting latest revision."""
        store = BiTemporalStore()

        # Add multiple revisions
        for rev in range(3):
            record = BiTemporalRecord(
                entity_id="GDP_US",
                indicator_name="gdp_growth",
                observation_date=date(2024, 3, 31),
                release_date=datetime(2024, 4, 29) + timedelta(days=30 * rev),
                transaction_time=datetime(2024, 4, 29) + timedelta(days=30 * rev, hours=1),
                value=-5.0 + rev * 0.2,
                revision_number=rev,
            )
            store.insert(record)

        latest = store.get_latest_revision("GDP_US", "gdp_growth", date(2024, 3, 31))

        assert latest == -4.6  # -5.0 + 2 * 0.2

    def test_get_revision_history(self):
        """Test getting full revision history."""
        store = BiTemporalStore()

        # Add revisions
        for rev in range(3):
            record = BiTemporalRecord(
                entity_id="GDP_US",
                indicator_name="gdp_growth",
                observation_date=date(2024, 3, 31),
                release_date=datetime(2024, 4, 29) + timedelta(days=30 * rev),
                transaction_time=datetime(2024, 4, 29) + timedelta(days=30 * rev, hours=1),
                value=-5.0 + rev * 0.2,
                revision_number=rev,
            )
            store.insert(record)

        history = store.get_revision_history("GDP_US", "gdp_growth", date(2024, 3, 31))

        assert len(history) == 3
        assert history[0].revision_number == 0
        assert history[1].revision_number == 1
        assert history[2].revision_number == 2
        assert history[0].value == -5.0
        assert history[2].value == -4.6

    def test_to_dataframe(self):
        """Test exporting to DataFrame."""
        store = BiTemporalStore()

        # Add some records
        for i in range(3):
            record = BiTemporalRecord(
                entity_id="AAPL",
                indicator_name="close_price",
                observation_date=date(2024, 1, 15) + timedelta(days=i),
                release_date=datetime(2024, 1, 15) + timedelta(days=i, hours=16),
                transaction_time=datetime(2024, 1, 15) + timedelta(days=i, hours=17),
                value=150.0 + i,
            )
            store.insert(record)

        df = store.to_dataframe()

        assert len(df) == 3
        assert "entity_id" in df.columns
        assert "observation_date" in df.columns
        assert "value" in df.columns
        assert df["value"].tolist() == [150.0, 151.0, 152.0]


class TestValidatePITCorrectness:
    """Test PIT correctness validation."""

    def test_valid_store(self):
        """Test validation passes for valid store."""
        store = BiTemporalStore()

        record = BiTemporalRecord(
            entity_id="AAPL",
            indicator_name="close_price",
            observation_date=date(2024, 1, 15),
            release_date=datetime(2024, 1, 15, 16, 0),
            transaction_time=datetime(2024, 1, 15, 16, 30),
            value=150.0,
        )

        store.insert(record)

        # Should pass
        assert validate_pit_correctness(
            store,
            datetime(2024, 1, 15),
            datetime(2024, 1, 31),
        )

    def test_empty_store(self):
        """Test validation on empty store."""
        store = BiTemporalStore()

        # Should pass (no violations)
        assert validate_pit_correctness(
            store,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )
