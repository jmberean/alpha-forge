"""
Data layer: Bi-temporal data storage and point-in-time queries.

This module provides:
- MarketDataLoader: yfinance-based data loading with caching
- BiTemporalSchema: Three-timestamp schema (observation, release, transaction)
- PointInTimeQuery: Query data as it was known at a specific time
- BiTemporalStore: In-memory bi-temporal data store with PIT queries
- ALFREDClient: Federal Reserve vintage data integration
- UniverseRegistry: Survivorship bias prevention
- DataQualityChecker: Automated data validation
"""

from alphaforge.data.alfred import ALFREDClient, ALFREDSync, get_macro_feature
from alphaforge.data.bitemporal import BiTemporalRecord as BiTemporalRecordNew
from alphaforge.data.bitemporal import BiTemporalStore, validate_pit_correctness
from alphaforge.data.loader import MarketDataLoader
from alphaforge.data.pit import PointInTimeQuery
from alphaforge.data.quality import DataQualityChecker, QualityReport, validate_data_quality
from alphaforge.data.schema import BiTemporalRecord, MarketData
from alphaforge.data.universe import (
    SP500HistoricalConstituents,
    SecurityInfo,
    SurvivorshipBiasError,
    UniverseRegistry,
    validate_universe,
)

__all__ = [
    # Original exports
    "MarketDataLoader",
    "MarketData",
    "BiTemporalRecord",
    "PointInTimeQuery",
    # MVP4 exports
    "BiTemporalStore",
    "BiTemporalRecordNew",
    "validate_pit_correctness",
    "ALFREDClient",
    "ALFREDSync",
    "get_macro_feature",
    "UniverseRegistry",
    "SecurityInfo",
    "SP500HistoricalConstituents",
    "validate_universe",
    "SurvivorshipBiasError",
    "DataQualityChecker",
    "QualityReport",
    "validate_data_quality",
]
