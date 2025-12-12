"""
Bi-temporal data schema definitions.

Every data point carries three timestamps:
- observation_date: What period the data measures (e.g., Q1 2020)
- release_date: When it became publicly available
- transaction_time: When entered our database

This prevents lookahead bias at the data infrastructure level.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator


class MarketData(BaseModel):
    """OHLCV market data with bi-temporal timestamps."""

    symbol: str = Field(..., description="Ticker symbol")
    observation_date: date = Field(..., description="Trading date")
    release_date: datetime = Field(..., description="When data became available")
    transaction_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When entered database",
    )

    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: Optional[float] = Field(None, ge=0, description="Adjusted close")

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        """High must be >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info) -> float:
        """High must be >= open and close."""
        data = info.data
        if "open" in data and v < data["open"]:
            raise ValueError("high must be >= open")
        if "close" in data and v < data["close"]:
            raise ValueError("high must be >= close")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_close(cls, v: float, info) -> float:
        """Low must be <= open and close."""
        data = info.data
        if "open" in data and v > data["open"]:
            raise ValueError("low must be <= open")
        if "close" in data and v > data["close"]:
            raise ValueError("low must be <= close")
        return v


class BiTemporalRecord(BaseModel):
    """Generic bi-temporal record for any time series data."""

    entity_id: str = Field(..., description="Entity identifier (symbol, indicator name)")
    observation_date: date = Field(..., description="What period the data measures")
    release_date: datetime = Field(..., description="When it became publicly available")
    transaction_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When entered database",
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="End of validity (NULL if current version)",
    )
    value: float = Field(..., description="The data value")
    revision_number: int = Field(default=0, ge=0, description="Revision number")
    source: str = Field(default="unknown", description="Data source")


@dataclass
class OHLCVData:
    """
    DataFrame wrapper for OHLCV data with bi-temporal awareness.

    Attributes:
        df: DataFrame with columns [open, high, low, close, volume, adjusted_close]
        symbol: Ticker symbol
        release_timestamp: When this data snapshot was released
        transaction_timestamp: When data entered the system
    """

    df: pd.DataFrame
    symbol: str
    release_timestamp: datetime = field(default_factory=datetime.utcnow)
    transaction_timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate DataFrame structure."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        self._validate_ohlc()

    def _validate_ohlc(self) -> None:
        """Validate OHLC data integrity."""
        df = self.df

        # High >= Low
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            bad_dates = df.index[invalid_hl].tolist()
            raise ValueError(f"High < Low on dates: {bad_dates[:5]}")

        # High >= Open and Close
        invalid_ho = df["high"] < df["open"]
        invalid_hc = df["high"] < df["close"]
        if invalid_ho.any() or invalid_hc.any():
            raise ValueError("High must be >= Open and Close")

        # Low <= Open and Close
        invalid_lo = df["low"] > df["open"]
        invalid_lc = df["low"] > df["close"]
        if invalid_lo.any() or invalid_lc.any():
            raise ValueError("Low must be <= Open and Close")

        # Positive values
        for col in ["open", "high", "low", "close", "volume"]:
            if (df[col] < 0).any():
                raise ValueError(f"Negative values in {col}")

    @property
    def returns(self) -> pd.Series:
        """Calculate simple returns."""
        return self.df["close"].pct_change()

    @property
    def log_returns(self) -> pd.Series:
        """Calculate log returns."""
        return np.log(self.df["close"] / self.df["close"].shift(1))

    def as_of(self, timestamp: datetime) -> "OHLCVData":
        """
        Return data as it was known at the given timestamp.

        This is the core point-in-time query for preventing lookahead bias.
        """
        # Filter to data that was released before the timestamp
        mask = self.df.index <= pd.Timestamp(timestamp)
        filtered_df = self.df[mask].copy()

        return OHLCVData(
            df=filtered_df,
            symbol=self.symbol,
            release_timestamp=self.release_timestamp,
            transaction_timestamp=self.transaction_timestamp,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"OHLCVData(symbol={self.symbol}, "
            f"start={self.df.index[0].date()}, "
            f"end={self.df.index[-1].date()}, "
            f"rows={len(self.df)})"
        )
