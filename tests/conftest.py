"""
Pytest fixtures for AlphaForge tests.

Uses real market data from yfinance (cached for reproducibility).
NO SYNTHETIC OR FAKE DATA.
"""

import pytest
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> Path:
    """Create a temporary cache directory for tests."""
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="session")
def spy_data():
    """
    Load real SPY data for testing.

    Uses a fixed date range for reproducibility.
    Data is cached after first fetch.
    """
    from alphaforge.data.loader import MarketDataLoader

    loader = MarketDataLoader()

    # Use a fixed historical range for reproducible tests
    data = loader.load(
        "SPY",
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
        use_cache=True,
    )

    return data


@pytest.fixture(scope="session")
def spy_df(spy_data) -> pd.DataFrame:
    """Get SPY DataFrame with features computed."""
    from alphaforge.features.technical import TechnicalIndicators

    return TechnicalIndicators.compute_all(spy_data.df)


@pytest.fixture
def sample_strategy():
    """Get a sample SMA crossover strategy."""
    from alphaforge.strategy.templates import StrategyTemplates

    return StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)


@pytest.fixture
def sample_returns() -> pd.Series:
    """
    Get sample returns from real SPY data.

    Uses cached data to avoid API calls during tests.
    """
    from alphaforge.data.loader import MarketDataLoader

    loader = MarketDataLoader()
    data = loader.load("SPY", start=date(2022, 1, 1), end=date(2022, 12, 31))

    return data.df["close"].pct_change().dropna()


@pytest.fixture
def backtest_engine():
    """Get a backtest engine with default settings."""
    from alphaforge.backtest.engine import BacktestEngine

    return BacktestEngine(initial_capital=100000.0)


@pytest.fixture
def validation_pipeline():
    """Get a validation pipeline with default settings."""
    from alphaforge.validation.pipeline import ValidationPipeline

    return ValidationPipeline()
