"""
Data layer: Bi-temporal data storage and point-in-time queries.

This module provides:
- MarketDataLoader: yfinance-based data loading with caching
- BiTemporalSchema: Three-timestamp schema (observation, release, transaction)
- PointInTimeQuery: Query data as it was known at a specific time
"""

from alphaforge.data.loader import MarketDataLoader
from alphaforge.data.schema import MarketData, BiTemporalRecord
from alphaforge.data.pit import PointInTimeQuery

__all__ = [
    "MarketDataLoader",
    "MarketData",
    "BiTemporalRecord",
    "PointInTimeQuery",
]
