"""
Technical indicators with point-in-time guarantees.

All indicators use TRAILING windows only (center=False) to prevent lookahead bias.
"""

from typing import Optional
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Technical indicator calculations with strict PIT compliance.

    All methods return Series aligned with the input index.
    NaN values at the start indicate insufficient lookback data.
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            SMA series
        """
        return series.rolling(window=period, min_periods=period, center=False).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            series: Price series
            period: Lookback period (used for span)

        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = TechnicalIndicators.ema(series, fast_period)
        slow_ema = TechnicalIndicators.ema(series, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(
            span=signal_period, adjust=False, min_periods=signal_period
        ).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            series: Price series
            period: Moving average period
            num_std: Number of standard deviations

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period, min_periods=period, center=False).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR series
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D smoothing period

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period, min_periods=k_period, center=False).min()
        highest_high = high.rolling(
            window=k_period, min_periods=k_period, center=False
        ).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period, min_periods=d_period, center=False).mean()

        return k, d

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Williams %R.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period

        Returns:
            Williams %R (-100 to 0)
        """
        highest_high = high.rolling(window=period, min_periods=period, center=False).max()
        lowest_low = low.rolling(window=period, min_periods=period, center=False).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average Directional Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            ADX series
        """
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Calculate ATR
        atr = TechnicalIndicators.atr(high, low, close, period)

        # Calculate +DI and -DI
        plus_di = 100 * (
            plus_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
        )
        minus_di = 100 * (
            minus_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr
        )

        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()

        return adx

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume Weighted Average Price (cumulative for intraday).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()

        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Price Momentum (rate of change).

        Args:
            series: Price series
            period: Lookback period

        Returns:
            Momentum as percentage change
        """
        return series.pct_change(periods=period) * 100

    @staticmethod
    def volatility(
        series: pd.Series,
        period: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Rolling volatility (standard deviation of returns).

        Args:
            series: Price series
            period: Lookback period
            annualize: If True, annualize the volatility
            trading_days: Trading days per year for annualization

        Returns:
            Volatility series
        """
        returns = series.pct_change()
        vol = returns.rolling(window=period, min_periods=period, center=False).std()

        if annualize:
            vol = vol * np.sqrt(trading_days)

        return vol

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all standard technical indicators for OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with original columns plus all indicators
        """
        result = df.copy()

        # Trend indicators
        result["sma_20"] = TechnicalIndicators.sma(df["close"], 20)
        result["sma_50"] = TechnicalIndicators.sma(df["close"], 50)
        result["sma_200"] = TechnicalIndicators.sma(df["close"], 200)
        result["ema_12"] = TechnicalIndicators.ema(df["close"], 12)
        result["ema_26"] = TechnicalIndicators.ema(df["close"], 26)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(df["close"])
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist

        # Momentum indicators
        result["rsi_14"] = TechnicalIndicators.rsi(df["close"], 14)
        result["momentum_10"] = TechnicalIndicators.momentum(df["close"], 10)

        # Volatility indicators
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"])
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower
        result["atr_14"] = TechnicalIndicators.atr(
            df["high"], df["low"], df["close"], 14
        )
        result["volatility_20"] = TechnicalIndicators.volatility(df["close"], 20)

        # Oscillators
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            df["high"], df["low"], df["close"]
        )
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        result["williams_r"] = TechnicalIndicators.williams_r(
            df["high"], df["low"], df["close"]
        )

        # Trend strength
        result["adx_14"] = TechnicalIndicators.adx(
            df["high"], df["low"], df["close"], 14
        )

        # Volume indicators
        result["obv"] = TechnicalIndicators.obv(df["close"], df["volume"])
        result["volume_sma_20"] = TechnicalIndicators.sma(df["volume"], 20)
        result["volume_ratio"] = df["volume"] / result["volume_sma_20"]

        return result
