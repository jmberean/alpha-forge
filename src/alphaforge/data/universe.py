"""
Universe management and survivorship bias prevention.

Ensures backtests include delisted securities to avoid survivorship bias.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Set

import pandas as pd


@dataclass
class SecurityInfo:
    """Information about a security."""

    symbol: str
    name: str
    exchange: str
    sector: str | None = None
    industry: str | None = None

    # Lifecycle
    listing_date: date | None = None
    delisting_date: date | None = None
    delisting_reason: str | None = None  # 'merged', 'bankruptcy', 'voluntary', etc.

    # Identifiers
    cusip: str | None = None
    isin: str | None = None

    def is_active(self, as_of_date: date) -> bool:
        """Check if security was active on a given date."""
        if self.listing_date and as_of_date < self.listing_date:
            return False
        if self.delisting_date and as_of_date >= self.delisting_date:
            return False
        return True


class UniverseRegistry:
    """
    Registry of all securities including delisted.

    Critical for survivorship bias prevention.
    """

    def __init__(self):
        """Initialize empty registry."""
        self.securities: dict[str, SecurityInfo] = {}

    def add_security(self, security: SecurityInfo) -> None:
        """
        Add security to registry.

        Args:
            security: SecurityInfo to add
        """
        self.securities[security.symbol] = security

    def get_security(self, symbol: str) -> SecurityInfo | None:
        """
        Get security info by symbol.

        Args:
            symbol: Security symbol

        Returns:
            SecurityInfo or None if not found
        """
        return self.securities.get(symbol)

    def get_universe(
        self,
        as_of_date: date,
        include_delisted: bool = True,
    ) -> list[str]:
        """
        Get list of securities in universe on a specific date.

        Args:
            as_of_date: Date for universe composition
            include_delisted: If False, only currently active securities

        Returns:
            List of symbols
        """
        symbols = []

        for symbol, info in self.securities.items():
            if include_delisted:
                # Include if active on as_of_date
                if info.is_active(as_of_date):
                    symbols.append(symbol)
            else:
                # Only currently active (survivorship biased - use with caution!)
                if info.delisting_date is None:
                    symbols.append(symbol)

        return sorted(symbols)

    def get_delisted_securities(
        self,
        start_date: date,
        end_date: date,
    ) -> list[SecurityInfo]:
        """
        Get securities delisted during a period.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            List of delisted securities
        """
        delisted = []

        for info in self.securities.values():
            if info.delisting_date:
                if start_date <= info.delisting_date <= end_date:
                    delisted.append(info)

        return sorted(delisted, key=lambda x: x.delisting_date)  # type: ignore

    def check_survivorship_bias(
        self,
        backtest_symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, any]:
        """
        Check for survivorship bias in backtest universe.

        Args:
            backtest_symbols: Symbols used in backtest
            start_date: Backtest start
            end_date: Backtest end

        Returns:
            Dict with survivorship analysis
        """
        # Get all securities that should be included (active at any point)
        required = set()
        for symbol, info in self.securities.items():
            # Include if active at any point during backtest
            if info.listing_date and info.listing_date > end_date:
                continue  # Listed after backtest
            if info.delisting_date and info.delisting_date < start_date:
                continue  # Delisted before backtest

            required.add(symbol)

        backtest_set = set(backtest_symbols)
        missing = required - backtest_set

        # Categorize missing securities
        missing_delisted = []
        missing_active = []

        for symbol in missing:
            info = self.securities[symbol]
            if info.delisting_date and info.delisting_date <= end_date:
                missing_delisted.append(symbol)
            else:
                missing_active.append(symbol)

        has_bias = len(missing_delisted) > 0

        return {
            "has_survivorship_bias": has_bias,
            "total_required": len(required),
            "total_in_backtest": len(backtest_set),
            "missing_total": len(missing),
            "missing_delisted": sorted(missing_delisted),
            "missing_active": sorted(missing_active),
            "bias_severity": len(missing_delisted) / len(required) if required else 0.0,
        }


class SurvivorshipBiasError(Exception):
    """Raised when survivorship bias is detected."""

    pass


def validate_universe(
    registry: UniverseRegistry,
    backtest_symbols: list[str],
    start_date: date,
    end_date: date,
    max_bias: float = 0.0,
) -> bool:
    """
    Validate backtest universe for survivorship bias.

    Args:
        registry: Universe registry
        backtest_symbols: Symbols in backtest
        start_date: Backtest start
        end_date: Backtest end
        max_bias: Maximum acceptable bias ratio (0.0 = strict)

    Returns:
        True if validation passes

    Raises:
        SurvivorshipBiasError: If bias exceeds threshold
    """
    analysis = registry.check_survivorship_bias(
        backtest_symbols,
        start_date,
        end_date,
    )

    if analysis["has_survivorship_bias"] and analysis["bias_severity"] > max_bias:
        raise SurvivorshipBiasError(
            f"Survivorship bias detected: {len(analysis['missing_delisted'])} delisted "
            f"securities missing from backtest universe. "
            f"Bias severity: {analysis['bias_severity']:.1%}. "
            f"Missing symbols: {analysis['missing_delisted'][:10]}"
        )

    return True


class SP500HistoricalConstituents:
    """
    Track S&P 500 historical constituents.

    In production, this would be sourced from:
    - CRSP (academic)
    - Sharadar Core US Equities
    - Norgate Data
    """

    def __init__(self):
        """Initialize with mock data."""
        self.constituents: dict[date, Set[str]] = {}
        self._initialize_mock_data()

    def _initialize_mock_data(self):
        """Initialize mock S&P 500 constituent data."""
        # Mock: Some companies that were in S&P 500 but are now delisted
        base_2020 = {
            "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK.B", "JNJ", "V", "PG",
            # Add some that got delisted
            "XYZ_DELISTED_2021", "ABC_DELISTED_2022",
        }

        # Simulate quarterly rebalancing
        dates = pd.date_range("2020-01-01", "2024-01-01", freq="QE")

        for i, d in enumerate(dates):
            constituents = base_2020.copy()

            # Simulate delistings
            if d.year >= 2021:
                constituents.discard("XYZ_DELISTED_2021")
            if d.year >= 2022:
                constituents.discard("ABC_DELISTED_2022")

            self.constituents[d.date()] = constituents

    def get_constituents(self, as_of_date: date) -> Set[str]:
        """
        Get S&P 500 constituents as of a specific date.

        Args:
            as_of_date: Date for constituent list

        Returns:
            Set of symbols
        """
        # Find closest date <= as_of_date
        valid_dates = [d for d in self.constituents.keys() if d <= as_of_date]

        if not valid_dates:
            return set()

        closest_date = max(valid_dates)
        return self.constituents[closest_date]

    def get_additions_and_deletions(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, list[tuple[date, str]]]:
        """
        Get additions and deletions during period.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            Dict with 'additions' and 'deletions' lists
        """
        dates = sorted([d for d in self.constituents.keys() if start_date <= d <= end_date])

        additions = []
        deletions = []

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            prev_set = self.constituents[prev_date]
            curr_set = self.constituents[curr_date]

            added = curr_set - prev_set
            removed = prev_set - curr_set

            for symbol in added:
                additions.append((curr_date, symbol))

            for symbol in removed:
                deletions.append((curr_date, symbol))

        return {
            "additions": additions,
            "deletions": deletions,
        }


def populate_registry_from_sp500(
    registry: UniverseRegistry,
    sp500: SP500HistoricalConstituents,
) -> int:
    """
    Populate registry from S&P 500 historical data.

    Args:
        registry: UniverseRegistry to populate
        sp500: S&P 500 historical data

    Returns:
        Number of securities added
    """
    all_symbols = set()

    for constituents in sp500.constituents.values():
        all_symbols.update(constituents)

    count = 0
    for symbol in all_symbols:
        # Determine listing/delisting dates from constituent changes
        listing_date = None
        delisting_date = None

        for d in sorted(sp500.constituents.keys()):
            if symbol in sp500.constituents[d]:
                if listing_date is None:
                    listing_date = d
            else:
                if listing_date is not None and delisting_date is None:
                    delisting_date = d
                    break

        security = SecurityInfo(
            symbol=symbol,
            name=f"{symbol} Inc.",
            exchange="NYSE",
            sector="Technology",  # Mock
            listing_date=listing_date,
            delisting_date=delisting_date,
            delisting_reason="merged" if delisting_date else None,
        )

        registry.add_security(security)
        count += 1

    return count
