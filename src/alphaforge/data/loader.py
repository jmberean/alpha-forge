"""
Market data loader using yfinance with local caching.

IMPORTANT: This module uses REAL market data only. No synthetic or fake data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from alphaforge.data.schema import OHLCVData

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Load market data from yfinance with local Parquet caching.

    Uses bi-temporal awareness: tracks when data was fetched (release_timestamp)
    and when it entered the local system (transaction_timestamp).
    """

    DEFAULT_CACHE_DIR = Path("data/cache")

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_expiry_days: int = 1,
    ) -> None:
        """
        Initialize the market data loader.

        Args:
            cache_dir: Directory for caching Parquet files
            cache_expiry_days: Days before cache is considered stale
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_days = cache_expiry_days

    def _get_cache_path(self, symbol: str, start: date, end: date) -> Path:
        """Generate cache file path for a query."""
        cache_key = f"{symbol}_{start}_{end}"
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{symbol}_{hash_key}.parquet"

    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path for a cache file."""
        return cache_path.with_suffix(".json")

    def _is_cache_valid(self, cache_path: Path, end_date: date) -> bool:
        """Check if cache is valid and not stale."""
        if not cache_path.exists():
            return False

        metadata_path = self._get_metadata_path(cache_path)
        if not metadata_path.exists():
            return False

        with open(metadata_path) as f:
            metadata = json.load(f)

        fetch_time = datetime.fromisoformat(metadata["fetch_timestamp"])
        cache_end = date.fromisoformat(metadata["end_date"])
        now = datetime.now(UTC).replace(tzinfo=None)

        # Cache is invalid if:
        # 1. It's older than expiry days AND end_date is recent (data might have updated)
        # 2. The requested end_date is after the cached end_date
        is_stale = (now - fetch_time).days > self.cache_expiry_days
        is_incomplete = end_date > cache_end

        # For historical data (end_date far in past), cache doesn't expire
        is_historical = (date.today() - end_date).days > 30

        if is_historical:
            return not is_incomplete

        return not (is_stale or is_incomplete)

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        cache_path: Path,
        symbol: str,
        start: date,
        end: date,
    ) -> None:
        """Save data to Parquet cache with metadata."""
        # Save Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, cache_path)

        # Save metadata
        metadata = {
            "symbol": symbol,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "fetch_timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "rows": len(df),
        }
        with open(self._get_metadata_path(cache_path), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cached {len(df)} rows for {symbol} to {cache_path}")

    def _load_from_cache(self, cache_path: Path) -> tuple[pd.DataFrame, datetime]:
        """Load data from cache and return with fetch timestamp."""
        df = pq.read_table(cache_path).to_pandas()

        with open(self._get_metadata_path(cache_path)) as f:
            metadata = json.load(f)

        fetch_timestamp = datetime.fromisoformat(metadata["fetch_timestamp"])
        return df, fetch_timestamp

    def load(
        self,
        symbol: str,
        start: str | date,
        end: str | date | None = None,
        use_cache: bool = True,
    ) -> OHLCVData:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'AAPL')
            start: Start date
            end: End date (defaults to today)
            use_cache: Whether to use local cache

        Returns:
            OHLCVData with bi-temporal timestamps
        """
        # Normalize dates
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if end is None:
            end = date.today()
        elif isinstance(end, str):
            end = date.fromisoformat(end)

        cache_path = self._get_cache_path(symbol, start, end)
        release_timestamp = datetime.now(UTC).replace(tzinfo=None)

        # Try cache first
        if use_cache and self._is_cache_valid(cache_path, end):
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            df, release_timestamp = self._load_from_cache(cache_path)
        else:
            logger.info(f"Fetching {symbol} from yfinance: {start} to {end}")
            df = self._fetch_from_yfinance(symbol, start, end)

            if use_cache:
                self._save_to_cache(df, cache_path, symbol, start, end)

            release_timestamp = datetime.now(UTC).replace(tzinfo=None)

        return OHLCVData(
            df=df,
            symbol=symbol,
            release_timestamp=release_timestamp,
            transaction_timestamp=datetime.now(UTC).replace(tzinfo=None),
        )

    def _fetch_from_yfinance(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch data from yfinance."""
        ticker = yf.Ticker(symbol)

        # Add one day to end for inclusive range
        end_inclusive = end + timedelta(days=1)

        df = ticker.history(
            start=start.isoformat(),
            end=end_inclusive.isoformat(),
            auto_adjust=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol} from {start} to {end}")

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns from yfinance: {missing}")

        # Add adjusted close if available
        if "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "adjusted_close"})
        elif "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        # Select and order columns
        df = df[["open", "high", "low", "close", "volume", "adjusted_close"]]

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    def load_multiple(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        use_cache: bool = True,
    ) -> dict[str, OHLCVData]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            use_cache: Whether to use cache

        Returns:
            Dictionary mapping symbol to OHLCVData
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.load(symbol, start, end, use_cache)
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                raise

        return results

    def clear_cache(self, symbol: str | None = None) -> int:
        """
        Clear cached data.

        Args:
            symbol: If provided, clear only this symbol's cache.
                   If None, clear all cache.

        Returns:
            Number of files deleted
        """
        deleted = 0

        if symbol:
            pattern = f"{symbol}_*.parquet"
        else:
            pattern = "*.parquet"

        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            metadata_file = self._get_metadata_path(cache_file)
            if metadata_file.exists():
                metadata_file.unlink()
            deleted += 1

        logger.info(f"Cleared {deleted} cache files")
        return deleted
