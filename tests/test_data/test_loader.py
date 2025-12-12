"""Tests for data loader."""

import pytest
from datetime import date
from pathlib import Path

from alphaforge.data.loader import MarketDataLoader
from alphaforge.data.schema import OHLCVData


class TestMarketDataLoader:
    """Tests for MarketDataLoader."""

    def test_load_spy_data(self):
        """Test loading real SPY data."""
        loader = MarketDataLoader()
        data = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 6, 30))

        assert isinstance(data, OHLCVData)
        assert data.symbol == "SPY"
        assert len(data) > 100  # Should have ~125 trading days

    def test_data_has_required_columns(self):
        """Test that loaded data has all required OHLCV columns."""
        loader = MarketDataLoader()
        data = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 3, 31))

        required = {"open", "high", "low", "close", "volume"}
        assert required.issubset(set(data.df.columns))

    def test_ohlc_consistency(self):
        """Test that OHLC data is consistent (high >= low, etc.)."""
        loader = MarketDataLoader()
        data = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 3, 31))

        df = data.df
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_caching(self, tmp_path):
        """Test that data is cached and reused."""
        cache_dir = tmp_path / "cache"
        loader = MarketDataLoader(cache_dir=cache_dir)

        # First load - should fetch from yfinance
        data1 = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 1, 31))

        # Check cache file exists
        cache_files = list(cache_dir.glob("SPY_*.parquet"))
        assert len(cache_files) >= 1

        # Second load - should use cache
        data2 = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 1, 31))

        # Data should be identical
        assert len(data1) == len(data2)

    def test_returns_calculation(self):
        """Test returns calculation on loaded data."""
        loader = MarketDataLoader()
        data = loader.load("SPY", start=date(2023, 1, 1), end=date(2023, 3, 31))

        returns = data.returns
        assert len(returns) == len(data)
        assert returns.isna().sum() == 1  # First value is NaN

    def test_invalid_symbol(self):
        """Test that invalid symbol raises error."""
        loader = MarketDataLoader()

        with pytest.raises(ValueError):
            loader.load("INVALID_SYMBOL_12345", start=date(2023, 1, 1), end=date(2023, 1, 31))

    def test_load_multiple_symbols(self):
        """Test loading multiple symbols."""
        loader = MarketDataLoader()
        data = loader.load_multiple(
            ["SPY", "QQQ"],
            start=date(2023, 1, 1),
            end=date(2023, 3, 31),
        )

        assert "SPY" in data
        assert "QQQ" in data
        assert len(data["SPY"]) > 0
        assert len(data["QQQ"]) > 0
