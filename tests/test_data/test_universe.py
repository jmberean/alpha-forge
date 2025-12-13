"""Tests for universe management and survivorship bias prevention."""

from datetime import date

import pytest

from alphaforge.data.universe import (
    SP500HistoricalConstituents,
    SecurityInfo,
    SurvivorshipBiasError,
    UniverseRegistry,
    populate_registry_from_sp500,
    validate_universe,
)


class TestSecurityInfo:
    """Test SecurityInfo dataclass."""

    def test_security_creation(self):
        """Test creating security info."""
        security = SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            listing_date=date(1980, 12, 12),
            delisting_date=None,
        )

        assert security.symbol == "AAPL"
        assert security.is_active(date(2024, 1, 1))

    def test_is_active_listed_date(self):
        """Test is_active respects listing date."""
        security = SecurityInfo(
            symbol="TSLA",
            name="Tesla Inc.",
            exchange="NASDAQ",
            listing_date=date(2010, 6, 29),
        )

        assert not security.is_active(date(2010, 6, 1))  # Before listing
        assert security.is_active(date(2010, 7, 1))  # After listing
        assert security.is_active(date(2024, 1, 1))  # Currently active

    def test_is_active_delisting_date(self):
        """Test is_active respects delisting date."""
        security = SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            listing_date=date(2015, 1, 1),
            delisting_date=date(2020, 12, 31),
            delisting_reason="merged",
        )

        assert security.is_active(date(2020, 6, 1))  # Active
        assert not security.is_active(date(2021, 1, 1))  # Delisted


class TestUniverseRegistry:
    """Test UniverseRegistry class."""

    def test_add_and_get_security(self):
        """Test adding and retrieving securities."""
        registry = UniverseRegistry()

        security = SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
        )

        registry.add_security(security)

        retrieved = registry.get_security("AAPL")
        assert retrieved == security

    def test_get_nonexistent_security(self):
        """Test getting security that doesn't exist."""
        registry = UniverseRegistry()

        result = registry.get_security("NONEXISTENT")
        assert result is None

    def test_get_universe_include_delisted(self):
        """Test getting universe with delisted securities."""
        registry = UniverseRegistry()

        # Add active security
        registry.add_security(SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            listing_date=date(2000, 1, 1),
        ))

        # Add delisted security
        registry.add_security(SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            listing_date=date(2015, 1, 1),
            delisting_date=date(2020, 12, 31),
        ))

        # Get universe in 2020 (both active)
        universe_2020 = registry.get_universe(date(2020, 6, 1), include_delisted=True)
        assert "AAPL" in universe_2020
        assert "XYZ" in universe_2020

        # Get universe in 2024 (XYZ delisted)
        universe_2024 = registry.get_universe(date(2024, 1, 1), include_delisted=True)
        assert "AAPL" in universe_2024
        assert "XYZ" not in universe_2024

    def test_get_universe_exclude_delisted(self):
        """Test getting universe excluding delisted (survivorship biased)."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
        ))

        registry.add_security(SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            delisting_date=date(2020, 12, 31),
        ))

        # Exclude delisted
        universe = registry.get_universe(date(2024, 1, 1), include_delisted=False)

        # Only currently active
        assert "AAPL" in universe
        assert "XYZ" not in universe

    def test_get_delisted_securities(self):
        """Test getting delisted securities in a period."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            delisting_date=date(2020, 6, 15),
            delisting_reason="bankruptcy",
        ))

        registry.add_security(SecurityInfo(
            symbol="ABC",
            name="ABC Inc",
            exchange="NASDAQ",
            delisting_date=date(2020, 9, 30),
            delisting_reason="merged",
        ))

        # Get delistings in 2020
        delisted = registry.get_delisted_securities(
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert len(delisted) == 2
        assert delisted[0].symbol == "XYZ"  # Sorted by date
        assert delisted[1].symbol == "ABC"

    def test_check_survivorship_bias_no_bias(self):
        """Test survivorship bias check with complete universe."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(symbol="AAPL", name="Apple", exchange="NASDAQ"))
        registry.add_security(SecurityInfo(symbol="MSFT", name="Microsoft", exchange="NASDAQ"))

        analysis = registry.check_survivorship_bias(
            backtest_symbols=["AAPL", "MSFT"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )

        assert not analysis["has_survivorship_bias"]
        assert analysis["missing_total"] == 0
        assert len(analysis["missing_delisted"]) == 0

    def test_check_survivorship_bias_with_bias(self):
        """Test survivorship bias detection."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(symbol="AAPL", name="Apple", exchange="NASDAQ"))
        registry.add_security(SecurityInfo(symbol="MSFT", name="Microsoft", exchange="NASDAQ"))

        # Delisted security that should be included
        registry.add_security(SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            listing_date=date(2015, 1, 1),
            delisting_date=date(2022, 6, 30),
        ))

        # Backtest only includes survivors
        analysis = registry.check_survivorship_bias(
            backtest_symbols=["AAPL", "MSFT"],  # Missing XYZ!
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )

        assert analysis["has_survivorship_bias"]
        assert "XYZ" in analysis["missing_delisted"]
        assert analysis["bias_severity"] > 0.0

    def test_check_survivorship_bias_severity(self):
        """Test bias severity calculation."""
        registry = UniverseRegistry()

        # Add 10 securities, 3 delisted
        for i in range(7):
            registry.add_security(SecurityInfo(symbol=f"ACTIVE{i}", name=f"Active {i}", exchange="NYSE"))

        for i in range(3):
            registry.add_security(SecurityInfo(
                symbol=f"DELISTED{i}",
                name=f"Delisted {i}",
                exchange="NYSE",
                delisting_date=date(2022, 1, 1),
            ))

        # Backtest missing all delisted
        backtest_symbols = [f"ACTIVE{i}" for i in range(7)]

        analysis = registry.check_survivorship_bias(
            backtest_symbols=backtest_symbols,
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )

        # 3 out of 10 = 30% bias
        assert abs(analysis["bias_severity"] - 0.3) < 0.01


class TestValidateUniverse:
    """Test universe validation function."""

    def test_validate_universe_passes(self):
        """Test validation passes with complete universe."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(symbol="AAPL", name="Apple", exchange="NASDAQ"))
        registry.add_security(SecurityInfo(symbol="MSFT", name="Microsoft", exchange="NASDAQ"))

        # Should not raise
        result = validate_universe(
            registry,
            backtest_symbols=["AAPL", "MSFT"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            max_bias=0.0,
        )

        assert result is True

    def test_validate_universe_raises_on_bias(self):
        """Test validation raises on survivorship bias."""
        registry = UniverseRegistry()

        registry.add_security(SecurityInfo(symbol="AAPL", name="Apple", exchange="NASDAQ"))

        registry.add_security(SecurityInfo(
            symbol="XYZ",
            name="XYZ Corp",
            exchange="NYSE",
            delisting_date=date(2022, 1, 1),
        ))

        # Should raise
        with pytest.raises(SurvivorshipBiasError, match="Survivorship bias detected"):
            validate_universe(
                registry,
                backtest_symbols=["AAPL"],  # Missing XYZ
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                max_bias=0.0,
            )

    def test_validate_universe_with_tolerance(self):
        """Test validation with bias tolerance."""
        registry = UniverseRegistry()

        # 9 active, 1 delisted = 10% bias
        for i in range(9):
            registry.add_security(SecurityInfo(symbol=f"ACTIVE{i}", name=f"Active {i}", exchange="NYSE"))

        registry.add_security(SecurityInfo(
            symbol="DELISTED",
            name="Delisted Corp",
            exchange="NYSE",
            delisting_date=date(2022, 1, 1),
        ))

        backtest_symbols = [f"ACTIVE{i}" for i in range(9)]

        # Should raise with 0% tolerance
        with pytest.raises(SurvivorshipBiasError):
            validate_universe(
                registry,
                backtest_symbols=backtest_symbols,
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                max_bias=0.0,
            )

        # Should pass with 15% tolerance
        result = validate_universe(
            registry,
            backtest_symbols=backtest_symbols,
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            max_bias=0.15,
        )

        assert result is True


class TestSP500HistoricalConstituents:
    """Test S&P 500 historical constituents tracking."""

    def test_initialization(self):
        """Test SP500 historical data initialization."""
        sp500 = SP500HistoricalConstituents()

        assert len(sp500.constituents) > 0

    def test_get_constituents(self):
        """Test getting constituents as of date."""
        sp500 = SP500HistoricalConstituents()

        constituents_2020 = sp500.get_constituents(date(2020, 6, 1))

        assert "AAPL" in constituents_2020
        assert "MSFT" in constituents_2020
        assert len(constituents_2020) > 0

    def test_delistings_tracked(self):
        """Test that delistings are tracked."""
        sp500 = SP500HistoricalConstituents()

        constituents_2020 = sp500.get_constituents(date(2020, 6, 1))
        constituents_2023 = sp500.get_constituents(date(2023, 6, 1))

        # Should have delisted securities in 2020 that are gone by 2023
        assert "XYZ_DELISTED_2021" in constituents_2020
        assert "XYZ_DELISTED_2021" not in constituents_2023

    def test_get_additions_and_deletions(self):
        """Test getting additions and deletions."""
        sp500 = SP500HistoricalConstituents()

        changes = sp500.get_additions_and_deletions(
            date(2020, 1, 1),
            date(2023, 12, 31),
        )

        assert "additions" in changes
        assert "deletions" in changes
        assert isinstance(changes["additions"], list)
        assert isinstance(changes["deletions"], list)


class TestPopulateRegistryFromSP500:
    """Test populating registry from S&P 500 data."""

    def test_populate_registry(self):
        """Test populating registry from SP500 data."""
        registry = UniverseRegistry()
        sp500 = SP500HistoricalConstituents()

        count = populate_registry_from_sp500(registry, sp500)

        # Should have added securities
        assert count > 0

        # Should be able to query
        aapl = registry.get_security("AAPL")
        assert aapl is not None
        assert aapl.symbol == "AAPL"

    def test_populated_registry_tracks_delistings(self):
        """Test that populated registry has delisting info."""
        registry = UniverseRegistry()
        sp500 = SP500HistoricalConstituents()

        populate_registry_from_sp500(registry, sp500)

        # Get delisted security
        delisted = registry.get_security("XYZ_DELISTED_2021")
        assert delisted is not None
        assert delisted.delisting_date is not None
        assert delisted.delisting_reason == "merged"
