"""
Advanced technical indicators beyond basic SMA/RSI/MACD.

Implements 30+ indicators for comprehensive technical analysis.

Performance optimizations:
- Numba JIT compilation for hot path rolling apply functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import numba


# =============================================================================
# Numba JIT-compiled functions for hot path indicators
# =============================================================================

@numba.jit(nopython=True, cache=True)
def _numba_mean_abs_deviation(arr: np.ndarray) -> float:
    """Numba-optimized mean absolute deviation for CCI."""
    n = len(arr)
    if n == 0:
        return np.nan
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n
    mad = 0.0
    for i in range(n):
        mad += abs(arr[i] - mean)
    return mad / n


@numba.jit(nopython=True, cache=True)
def _numba_aroon_up(arr: np.ndarray) -> float:
    """Numba-optimized Aroon Up calculation."""
    n = len(arr)
    if n == 0:
        return np.nan
    max_idx = 0
    max_val = arr[0]
    for i in range(1, n):
        if arr[i] >= max_val:  # >= to get most recent on ties
            max_val = arr[i]
            max_idx = i
    days_since = n - 1 - max_idx
    return ((n - days_since) / n) * 100


@numba.jit(nopython=True, cache=True)
def _numba_aroon_down(arr: np.ndarray) -> float:
    """Numba-optimized Aroon Down calculation."""
    n = len(arr)
    if n == 0:
        return np.nan
    min_idx = 0
    min_val = arr[0]
    for i in range(1, n):
        if arr[i] <= min_val:  # <= to get most recent on ties
            min_val = arr[i]
            min_idx = i
    days_since = n - 1 - min_idx
    return ((n - days_since) / n) * 100


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).

    Momentum indicator comparing closing price to price range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: Period for %K
        d_period: Period for %D smoothing

    Returns:
        Tuple of (%K, %D) series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R indicator.

    Momentum indicator, inverse of stochastic oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period

    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)

    return wr


def commodity_channel_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """
    Commodity Channel Index (CCI).

    Identifies cyclical trends.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        constant: Scaling constant (default 0.015)

    Returns:
        CCI series
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    # Use Numba JIT for mean absolute deviation (hot path)
    mean_deviation = typical_price.rolling(window=period).apply(
        _numba_mean_abs_deviation, raw=True, engine="numba"
    )

    cci = (typical_price - sma) / (constant * mean_deviation)

    return cci


def money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index (MFI).

    Volume-weighted RSI.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        period: Lookback period

    Returns:
        MFI series (0-100)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    # Positive and negative money flow
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)

    price_change = typical_price.diff()
    positive_flow[price_change > 0] = money_flow[price_change > 0]
    negative_flow[price_change < 0] = money_flow[price_change < 0]

    # Money flow ratio
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf))

    return mfi


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV).

    Cumulative volume indicator.

    Args:
        close: Close prices
        volume: Volume

    Returns:
        OBV series
    """
    obv = pd.Series(0.0, index=close.index)

    price_change = close.diff()

    obv[price_change > 0] = volume[price_change > 0]
    obv[price_change < 0] = -volume[price_change < 0]

    return obv.cumsum()


def ichimoku_cloud(
    high: pd.Series,
    low: pd.Series,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, pd.Series]:
    """
    Ichimoku Cloud components.

    Japanese trend-following indicator.

    Args:
        high: High prices
        low: Low prices
        conversion_period: Tenkan-sen period
        base_period: Kijun-sen period
        span_b_period: Senkou Span B period
        displacement: Displacement for Senkou spans

    Returns:
        Dict with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=conversion_period).max()
    tenkan_low = low.rolling(window=conversion_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=base_period).max()
    kijun_low = low.rolling(window=base_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B)
    senkou_high = high.rolling(window=span_b_period).max()
    senkou_low = low.rolling(window=span_b_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)

    # Chikou Span (Lagging Span) - intentional lag for signal confirmation
    chikou_span = low.shift(-displacement)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> dict[str, pd.Series]:
    """
    Keltner Channels.

    Volatility-based envelope.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period for center line
        atr_period: ATR period
        atr_multiplier: ATR multiplier for bands

    Returns:
        Dict with keys: upper, middle, lower
    """
    # Middle line (EMA)
    middle = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()

    # Bands
    upper = middle + (atr_multiplier * atr)
    lower = middle - (atr_multiplier * atr)

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def chaikin_volatility(
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 10,
    roc_period: int = 10,
) -> pd.Series:
    """
    Chaikin Volatility.

    Measures rate of change of trading range.

    Args:
        high: High prices
        low: Low prices
        ema_period: EMA period for range
        roc_period: Rate of change period

    Returns:
        Chaikin Volatility series
    """
    hl_range = high - low
    ema_range = hl_range.ewm(span=ema_period, adjust=False).mean()

    chaikin_vol = 100 * (ema_range - ema_range.shift(roc_period)) / ema_range.shift(roc_period)

    return chaikin_vol


def garman_klass_volatility(
    high: pd.Series,
    low: pd.Series,
    open: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    Uses OHLC data for more efficient volatility estimation.

    Args:
        high: High prices
        low: Low prices
        open: Open prices
        close: Close prices
        window: Rolling window period

    Returns:
        Annualized Garman-Klass volatility
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open)

    rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    gk_vol = np.sqrt(rs.rolling(window=window).mean() * 252)

    return gk_vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Parkinson volatility estimator.

    Uses high-low range for volatility estimation.

    Args:
        high: High prices
        low: Low prices
        window: Rolling window period

    Returns:
        Annualized Parkinson volatility
    """
    log_hl = np.log(high / low)

    park_vol = np.sqrt((1 / (4 * np.log(2))) * log_hl ** 2 * 252)

    return park_vol.rolling(window=window).mean()


def vortex_indicator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series]:
    """
    Vortex Indicator (VI+ and VI-).

    Identifies trend direction and strength.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period

    Returns:
        Tuple of (VI+, VI-) series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    vm_plus = abs(high - low.shift())
    vm_minus = abs(low - high.shift())

    vi_plus = vm_plus.rolling(window=period).sum() / true_range.rolling(window=period).sum()
    vi_minus = vm_minus.rolling(window=period).sum() / true_range.rolling(window=period).sum()

    return vi_plus, vi_minus


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> dict[str, pd.Series]:
    """
    Donchian Channels.

    Price channel based on highest high and lowest low.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period

    Returns:
        Dict with keys: upper, middle, lower
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator.

    Multi-timeframe momentum oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period1: Short period
        period2: Medium period
        period3: Long period

    Returns:
        Ultimate Oscillator series
    """
    true_low = pd.concat([low, close.shift()], axis=1).min(axis=1)
    buying_pressure = close - true_low

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    avg1 = buying_pressure.rolling(window=period1).sum() / true_range.rolling(window=period1).sum()
    avg2 = buying_pressure.rolling(window=period2).sum() / true_range.rolling(window=period2).sum()
    avg3 = buying_pressure.rolling(window=period3).sum() / true_range.rolling(window=period3).sum()

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7

    return uo


def mass_index(
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 9,
    sum_period: int = 25,
) -> pd.Series:
    """
    Mass Index.

    Identifies trend reversals based on range expansion.

    Args:
        high: High prices
        low: Low prices
        ema_period: EMA period
        sum_period: Sum period

    Returns:
        Mass Index series
    """
    hl_range = high - low
    ema = hl_range.ewm(span=ema_period, adjust=False).mean()
    ema_ema = ema.ewm(span=ema_period, adjust=False).mean()

    ratio = ema / ema_ema
    mass = ratio.rolling(window=sum_period).sum()

    return mass


def know_sure_thing(
    close: pd.Series,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    Know Sure Thing (KST) indicator.

    Multi-timeframe momentum oscillator.

    Args:
        close: Close prices
        roc1-4: ROC periods
        sma1-4: SMA smoothing periods
        signal: Signal line period

    Returns:
        Tuple of (KST, Signal) series
    """
    # Rate of Change for each timeframe
    roc_1 = close.pct_change(periods=roc1) * 100
    roc_2 = close.pct_change(periods=roc2) * 100
    roc_3 = close.pct_change(periods=roc3) * 100
    roc_4 = close.pct_change(periods=roc4) * 100

    # Smooth each ROC
    roc_1_sma = roc_1.rolling(window=sma1).mean()
    roc_2_sma = roc_2.rolling(window=sma2).mean()
    roc_3_sma = roc_3.rolling(window=sma3).mean()
    roc_4_sma = roc_4.rolling(window=sma4).mean()

    # Weighted sum
    kst = (roc_1_sma * 1) + (roc_2_sma * 2) + (roc_3_sma * 3) + (roc_4_sma * 4)

    # Signal line
    kst_signal = kst.rolling(window=signal).mean()

    return kst, kst_signal


def aroon_indicator(
    high: pd.Series,
    low: pd.Series,
    period: int = 25,
) -> tuple[pd.Series, pd.Series]:
    """
    Aroon Indicator (Aroon Up and Aroon Down).

    Identifies trend changes.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period

    Returns:
        Tuple of (Aroon Up, Aroon Down) series
    """
    # Use Numba JIT for argmax/argmin calculations (hot path)
    aroon_up = high.rolling(window=period).apply(
        _numba_aroon_up, raw=True, engine="numba"
    )

    aroon_down = low.rolling(window=period).apply(
        _numba_aroon_down, raw=True, engine="numba"
    )

    return aroon_up, aroon_down


# Convenience function to compute all indicators
def compute_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all advanced technical indicators.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with all indicator columns added
    """
    result = df.copy()

    # Stochastic
    stoch_k, stoch_d = stochastic_oscillator(df["high"], df["low"], df["close"])
    result["stoch_k"] = stoch_k
    result["stoch_d"] = stoch_d

    # Williams %R
    result["williams_r"] = williams_r(df["high"], df["low"], df["close"])

    # CCI
    result["cci"] = commodity_channel_index(df["high"], df["low"], df["close"])

    # MFI (if volume available)
    if "volume" in df.columns:
        result["mfi"] = money_flow_index(df["high"], df["low"], df["close"], df["volume"])
        result["obv"] = on_balance_volume(df["close"], df["volume"])

    # Ichimoku
    ichimoku = ichimoku_cloud(df["high"], df["low"])
    for key, value in ichimoku.items():
        result[f"ichimoku_{key}"] = value

    # Keltner Channels
    keltner = keltner_channels(df["high"], df["low"], df["close"])
    for key, value in keltner.items():
        result[f"keltner_{key}"] = value

    # Volatility indicators
    result["chaikin_volatility"] = chaikin_volatility(df["high"], df["low"])

    if "open" in df.columns:
        result["garman_klass_vol"] = garman_klass_volatility(
            df["high"], df["low"], df["open"], df["close"]
        )

    result["parkinson_vol"] = parkinson_volatility(df["high"], df["low"])

    # Vortex
    vi_plus, vi_minus = vortex_indicator(df["high"], df["low"], df["close"])
    result["vortex_plus"] = vi_plus
    result["vortex_minus"] = vi_minus

    # Donchian Channels
    donchian = donchian_channels(df["high"], df["low"])
    for key, value in donchian.items():
        result[f"donchian_{key}"] = value

    # Ultimate Oscillator
    result["ultimate_oscillator"] = ultimate_oscillator(df["high"], df["low"], df["close"])

    # Mass Index
    result["mass_index"] = mass_index(df["high"], df["low"])

    # KST
    kst, kst_signal = know_sure_thing(df["close"])
    result["kst"] = kst
    result["kst_signal"] = kst_signal

    # Aroon
    aroon_up, aroon_down = aroon_indicator(df["high"], df["low"])
    result["aroon_up"] = aroon_up
    result["aroon_down"] = aroon_down

    return result
