"""
Feature store with point-in-time guarantees.

Manages feature computation and retrieval ensuring no lookahead bias.
"""

from datetime import datetime, date
from typing import Optional, Union
from pathlib import Path
import logging
import hashlib
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alphaforge.data.schema import OHLCVData
from alphaforge.data.pit import PointInTimeQuery
from alphaforge.features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Point-in-time feature store for technical indicators.

    Features are computed and cached with temporal awareness.
    All queries respect the as-of timestamp to prevent lookahead bias.
    """

    DEFAULT_CACHE_DIR = Path("data/features")

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the feature store.

        Args:
            cache_dir: Directory for caching computed features
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._feature_registry: dict[str, callable] = {}
        self._register_default_features()

    def _register_default_features(self) -> None:
        """Register built-in technical indicators as features."""
        self._feature_registry = {
            "sma_20": lambda df: TechnicalIndicators.sma(df["close"], 20),
            "sma_50": lambda df: TechnicalIndicators.sma(df["close"], 50),
            "sma_200": lambda df: TechnicalIndicators.sma(df["close"], 200),
            "ema_12": lambda df: TechnicalIndicators.ema(df["close"], 12),
            "ema_26": lambda df: TechnicalIndicators.ema(df["close"], 26),
            "rsi_14": lambda df: TechnicalIndicators.rsi(df["close"], 14),
            "rsi_7": lambda df: TechnicalIndicators.rsi(df["close"], 7),
            "atr_14": lambda df: TechnicalIndicators.atr(
                df["high"], df["low"], df["close"], 14
            ),
            "volatility_20": lambda df: TechnicalIndicators.volatility(df["close"], 20),
            "momentum_10": lambda df: TechnicalIndicators.momentum(df["close"], 10),
            "returns_1d": lambda df: df["close"].pct_change(1),
            "returns_5d": lambda df: df["close"].pct_change(5),
            "returns_20d": lambda df: df["close"].pct_change(20),
            "volume_ratio": lambda df: df["volume"]
            / TechnicalIndicators.sma(df["volume"], 20),
        }

    def register_feature(self, name: str, compute_fn: callable) -> None:
        """
        Register a custom feature computation function.

        Args:
            name: Feature name
            compute_fn: Function that takes DataFrame and returns Series
        """
        if name in self._feature_registry:
            logger.warning(f"Overwriting existing feature: {name}")
        self._feature_registry[name] = compute_fn

    def _get_cache_path(self, symbol: str, feature_set: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{symbol}_{feature_set}.parquet"

    def compute_features(
        self,
        data: OHLCVData,
        feature_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute features for the given data.

        Args:
            data: OHLCV data
            feature_names: List of feature names to compute.
                          If None, compute all registered features.

        Returns:
            DataFrame with computed features
        """
        df = data.df.copy()

        if feature_names is None:
            feature_names = list(self._feature_registry.keys())

        for name in feature_names:
            if name not in self._feature_registry:
                raise ValueError(f"Unknown feature: {name}")

            try:
                df[name] = self._feature_registry[name](data.df)
            except Exception as e:
                logger.error(f"Failed to compute feature {name}: {e}")
                raise

        return df

    def compute_all_features(self, data: OHLCVData) -> pd.DataFrame:
        """
        Compute all standard technical indicators.

        Args:
            data: OHLCV data

        Returns:
            DataFrame with all indicators
        """
        return TechnicalIndicators.compute_all(data.df)

    def get_features(
        self,
        data: OHLCVData,
        as_of: Union[datetime, date, str],
        feature_names: Optional[list[str]] = None,
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get features with point-in-time guarantee.

        Args:
            data: OHLCV data
            as_of: Point in time for the query
            feature_names: Features to retrieve
            lookback: Optional lookback period

        Returns:
            DataFrame with features available at as_of
        """
        # Create PIT query engine
        pit = PointInTimeQuery(data)

        # Get available data as of timestamp
        available_df = pit.query(as_of, lookback=lookback)

        # Create OHLCVData for available period
        available_data = OHLCVData(
            df=available_df,
            symbol=data.symbol,
            release_timestamp=data.release_timestamp,
            transaction_timestamp=data.transaction_timestamp,
        )

        # Compute features on available data only
        features = self.compute_features(available_data, feature_names)

        return features

    def get_feature_matrix(
        self,
        data: OHLCVData,
        as_of: Union[datetime, date, str],
        feature_names: list[str],
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Get a feature matrix suitable for ML models.

        Args:
            data: OHLCV data
            as_of: Point in time
            feature_names: Features to include
            dropna: Drop rows with NaN values

        Returns:
            DataFrame with only the specified feature columns
        """
        features = self.get_features(data, as_of, feature_names)
        matrix = features[feature_names]

        if dropna:
            matrix = matrix.dropna()

        return matrix

    def save_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        feature_set: str = "default",
    ) -> Path:
        """
        Save computed features to Parquet.

        Args:
            features: Feature DataFrame
            symbol: Ticker symbol
            feature_set: Name for this feature set

        Returns:
            Path to saved file
        """
        cache_path = self._get_cache_path(symbol, feature_set)

        table = pa.Table.from_pandas(features)
        pq.write_table(table, cache_path)

        # Save metadata
        metadata = {
            "symbol": symbol,
            "feature_set": feature_set,
            "features": list(features.columns),
            "rows": len(features),
            "date_range": {
                "start": features.index[0].isoformat(),
                "end": features.index[-1].isoformat(),
            },
            "computed_at": datetime.utcnow().isoformat(),
        }

        metadata_path = cache_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved features to {cache_path}")
        return cache_path

    def load_features(
        self,
        symbol: str,
        feature_set: str = "default",
    ) -> Optional[pd.DataFrame]:
        """
        Load cached features.

        Args:
            symbol: Ticker symbol
            feature_set: Feature set name

        Returns:
            Feature DataFrame or None if not cached
        """
        cache_path = self._get_cache_path(symbol, feature_set)

        if not cache_path.exists():
            return None

        df = pq.read_table(cache_path).to_pandas()
        logger.info(f"Loaded features from {cache_path}")
        return df

    @property
    def available_features(self) -> list[str]:
        """List available feature names."""
        return list(self._feature_registry.keys())


class FeatureView:
    """
    A named view of features with specific configuration.

    Useful for defining feature sets for different strategy types.
    """

    def __init__(
        self,
        name: str,
        features: list[str],
        description: str = "",
    ) -> None:
        """
        Create a feature view.

        Args:
            name: View name
            features: List of feature names
            description: Description of this view's purpose
        """
        self.name = name
        self.features = features
        self.description = description

    def get_features(
        self,
        store: FeatureStore,
        data: OHLCVData,
        as_of: Union[datetime, date, str],
    ) -> pd.DataFrame:
        """
        Get features for this view.

        Args:
            store: Feature store instance
            data: OHLCV data
            as_of: Point in time

        Returns:
            Feature DataFrame
        """
        return store.get_feature_matrix(data, as_of, self.features)


# Pre-defined feature views
MOMENTUM_VIEW = FeatureView(
    name="momentum",
    features=["rsi_14", "momentum_10", "returns_1d", "returns_5d", "returns_20d"],
    description="Momentum indicators for trend-following strategies",
)

VOLATILITY_VIEW = FeatureView(
    name="volatility",
    features=["atr_14", "volatility_20", "volume_ratio"],
    description="Volatility indicators for risk management",
)

TREND_VIEW = FeatureView(
    name="trend",
    features=["sma_20", "sma_50", "sma_200", "ema_12", "ema_26"],
    description="Trend indicators for identifying market direction",
)

FULL_VIEW = FeatureView(
    name="full",
    features=[
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "rsi_14",
        "atr_14",
        "volatility_20",
        "momentum_10",
        "returns_1d",
        "returns_5d",
        "volume_ratio",
    ],
    description="Complete feature set for comprehensive analysis",
)
