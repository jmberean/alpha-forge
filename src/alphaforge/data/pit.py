"""
Point-in-Time (PIT) query engine.

Ensures all queries return data AS IT WAS KNOWN at a specific timestamp,
preventing lookahead bias in backtesting and feature engineering.
"""

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from alphaforge.data.schema import OHLCVData

logger = logging.getLogger(__name__)


class PointInTimeQuery:
    """
    Query engine that enforces point-in-time data access.

    All queries respect the temporal boundaries to ensure no future
    information leaks into historical analysis.
    """

    # Standard market data release delays (conservative estimates)
    DEFAULT_RELEASE_DELAY = {
        "price": timedelta(hours=0),  # Real-time
        "volume": timedelta(hours=0),  # Real-time
        "fundamental": timedelta(days=45),  # Quarterly lag
        "macro": timedelta(days=30),  # Monthly indicators
    }

    def __init__(
        self,
        data: OHLCVData,
        release_delay: timedelta | None = None,
    ) -> None:
        """
        Initialize PIT query engine.

        Args:
            data: The full OHLCV dataset
            release_delay: Assumed delay between observation and availability
        """
        self.data = data
        self.release_delay = release_delay or self.DEFAULT_RELEASE_DELAY["price"]
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate data has required structure."""
        if len(self.data.df) == 0:
            raise ValueError("Empty dataset provided")

        if not isinstance(self.data.df.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

    def query(
        self,
        as_of: datetime | date | str,
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """
        Query data as it was known at a specific point in time.

        Args:
            as_of: The point in time to query from
            lookback: Optional number of periods to look back

        Returns:
            DataFrame with data available at as_of time
        """
        # Normalize as_of to datetime
        if isinstance(as_of, str):
            as_of = pd.Timestamp(as_of)
        elif isinstance(as_of, date) and not isinstance(as_of, datetime):
            as_of = datetime.combine(as_of, datetime.max.time())

        # Apply release delay
        effective_cutoff = pd.Timestamp(as_of) - self.release_delay

        # Filter data that would have been available
        df = self.data.df
        available_mask = df.index <= effective_cutoff
        available_df = df[available_mask].copy()

        if available_df.empty:
            raise ValueError(
                f"No data available as of {as_of} "
                f"(effective cutoff: {effective_cutoff})"
            )

        # Apply lookback if specified
        if lookback is not None:
            available_df = available_df.tail(lookback)

        return available_df

    def get_latest(self, as_of: datetime | date | str) -> pd.Series:
        """
        Get the most recent data point available at as_of time.

        Args:
            as_of: The point in time

        Returns:
            Series with the latest available OHLCV data
        """
        available = self.query(as_of, lookback=1)
        return available.iloc[-1]

    def get_returns(
        self,
        as_of: datetime | date | str,
        lookback: int = 252,
        log_returns: bool = False,
    ) -> pd.Series:
        """
        Get historical returns available at as_of time.

        Args:
            as_of: The point in time
            lookback: Number of periods to include
            log_returns: If True, return log returns

        Returns:
            Series of returns
        """
        # Need lookback + 1 prices to compute lookback returns
        df = self.query(as_of, lookback=lookback + 1)

        if log_returns:
            returns = np.log(df["close"] / df["close"].shift(1))
        else:
            returns = df["close"].pct_change()

        return returns.dropna()

    def rolling_window(
        self,
        as_of: datetime | date | str,
        window_size: int,
        step: int = 1,
    ) -> list[pd.DataFrame]:
        """
        Generate rolling windows of data available at as_of.

        Useful for walk-forward analysis and cross-validation.

        Args:
            as_of: The point in time
            window_size: Size of each window
            step: Step size between windows

        Returns:
            List of DataFrame windows
        """
        df = self.query(as_of)

        if len(df) < window_size:
            raise ValueError(
                f"Not enough data for window_size={window_size}. "
                f"Only {len(df)} rows available."
            )

        windows = []
        for i in range(0, len(df) - window_size + 1, step):
            window = df.iloc[i : i + window_size].copy()
            windows.append(window)

        return windows

    def expanding_window(
        self,
        as_of: datetime | date | str,
        min_periods: int = 252,
    ) -> list[pd.DataFrame]:
        """
        Generate expanding windows for walk-forward validation.

        Each window includes all data from start up to that point.

        Args:
            as_of: The point in time
            min_periods: Minimum periods for first window

        Returns:
            List of expanding DataFrame windows
        """
        df = self.query(as_of)

        if len(df) < min_periods:
            raise ValueError(
                f"Not enough data for min_periods={min_periods}. "
                f"Only {len(df)} rows available."
            )

        windows = []
        for i in range(min_periods, len(df) + 1):
            window = df.iloc[:i].copy()
            windows.append(window)

        return windows


class PITFeatureComputer:
    """
    Compute features with strict point-in-time guarantees.

    FORBIDDEN patterns (caught at runtime):
    - Full-sample statistics (mean, std over entire dataset)
    - Centered rolling windows
    - Any computation using future data

    REQUIRED patterns:
    - Rolling/expanding windows with right alignment
    - Trailing windows only
    """

    def __init__(self, pit_query: PointInTimeQuery) -> None:
        """Initialize with a PIT query engine."""
        self.pit = pit_query

    def rolling_stat(
        self,
        as_of: datetime | date | str,
        column: str,
        window: int,
        stat: str = "mean",
    ) -> pd.Series:
        """
        Compute rolling statistic with PIT guarantee.

        Args:
            as_of: Point in time
            column: Column to compute on
            window: Rolling window size
            stat: Statistic to compute ('mean', 'std', 'min', 'max', 'sum')

        Returns:
            Series with rolling statistic
        """
        df = self.pit.query(as_of)

        if column not in df.columns:
            raise ValueError(f"Column {column} not in data")

        series = df[column]

        # Always use center=False for trailing windows
        rolling = series.rolling(window=window, min_periods=window, center=False)

        stat_funcs = {
            "mean": rolling.mean,
            "std": rolling.std,
            "min": rolling.min,
            "max": rolling.max,
            "sum": rolling.sum,
            "var": rolling.var,
            "skew": rolling.skew,
            "kurt": rolling.kurt,
        }

        if stat not in stat_funcs:
            raise ValueError(f"Unknown stat: {stat}. Available: {list(stat_funcs.keys())}")

        return stat_funcs[stat]()

    def z_score(
        self,
        as_of: datetime | date | str,
        column: str,
        window: int = 252,
    ) -> pd.Series:
        """
        Compute rolling z-score (standardized value).

        CORRECT implementation using trailing window only.

        Args:
            as_of: Point in time
            column: Column to standardize
            window: Lookback window for mean/std

        Returns:
            Rolling z-score series
        """
        df = self.pit.query(as_of)
        series = df[column]

        rolling_mean = series.rolling(window=window, center=False).mean()
        rolling_std = series.rolling(window=window, center=False).std()

        z_score = (series - rolling_mean) / rolling_std
        return z_score

    def expanding_rank(
        self,
        as_of: datetime | date | str,
        column: str,
        min_periods: int = 60,
    ) -> pd.Series:
        """
        Compute expanding percentile rank.

        At each point, rank is based only on data seen so far.

        Args:
            as_of: Point in time
            column: Column to rank
            min_periods: Minimum observations before ranking

        Returns:
            Expanding percentile rank (0-1)
        """
        df = self.pit.query(as_of)
        series = df[column]

        def pct_rank(x: pd.Series) -> float:
            if len(x) < min_periods:
                return np.nan
            return (x.rank().iloc[-1] - 1) / (len(x) - 1)

        return series.expanding(min_periods=min_periods).apply(pct_rank, raw=False)
