"""Tests for advanced technical indicators, lookahead detection, and LLM safety."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from alphaforge.features.advanced_technical import (
    aroon_indicator,
    commodity_channel_index,
    compute_all_advanced_indicators,
    donchian_channels,
    ichimoku_cloud,
    keltner_channels,
    money_flow_index,
    on_balance_volume,
    stochastic_oscillator,
    williams_r,
)
from alphaforge.features.llm_safety import (
    CANARY_QUESTIONS,
    LLMTemporalSafetyTester,
    MockLLM,
    require_temporal_safety,
)
from alphaforge.features.lookahead import (
    BiasType,
    LookaheadDetector,
    invalid_feature_function,
    valid_feature_function,
    validate_feature_function,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = pd.DataFrame({
        "open": np.random.uniform(95, 105, 100),
        "high": np.random.uniform(100, 110, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(95, 105, 100),
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # Ensure OHLC consistency
    data["high"] = data[["open", "close"]].max(axis=1) + 2
    data["low"] = data[["open", "close"]].min(axis=1) - 2

    return data


class TestAdvancedIndicators:
    """Test advanced technical indicators."""

    def test_stochastic_oscillator(self, sample_ohlcv):
        """Test Stochastic Oscillator."""
        k, d = stochastic_oscillator(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(k) == len(sample_ohlcv)
        assert len(d) == len(sample_ohlcv)
        # Values should be 0-100
        assert k[k.notna()].between(0, 100).all()
        assert d[d.notna()].between(0, 100).all()

    def test_williams_r(self, sample_ohlcv):
        """Test Williams %R."""
        wr = williams_r(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(wr) == len(sample_ohlcv)
        # Values should be -100 to 0
        assert wr[wr.notna()].between(-100, 0).all()

    def test_cci(self, sample_ohlcv):
        """Test Commodity Channel Index."""
        cci = commodity_channel_index(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(cci) == len(sample_ohlcv)
        assert not cci.isna().all()

    def test_mfi(self, sample_ohlcv):
        """Test Money Flow Index."""
        mfi = money_flow_index(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        assert len(mfi) == len(sample_ohlcv)
        # Values should be 0-100
        assert mfi[mfi.notna()].between(0, 100).all()

    def test_obv(self, sample_ohlcv):
        """Test On-Balance Volume."""
        obv = on_balance_volume(
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        assert len(obv) == len(sample_ohlcv)
        # OBV should be cumulative
        assert not obv.isna().all()

    def test_ichimoku_cloud(self, sample_ohlcv):
        """Test Ichimoku Cloud."""
        ichimoku = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
        )

        assert "tenkan_sen" in ichimoku
        assert "kijun_sen" in ichimoku
        assert "senkou_span_a" in ichimoku
        assert "senkou_span_b" in ichimoku
        assert "chikou_span" in ichimoku

        for series in ichimoku.values():
            assert len(series) == len(sample_ohlcv)

    def test_keltner_channels(self, sample_ohlcv):
        """Test Keltner Channels."""
        keltner = keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert "upper" in keltner
        assert "middle" in keltner
        assert "lower" in keltner

        # Upper > Middle > Lower
        valid_idx = keltner["upper"].notna() & keltner["middle"].notna() & keltner["lower"].notna()
        assert (keltner["upper"][valid_idx] >= keltner["middle"][valid_idx]).all()
        assert (keltner["middle"][valid_idx] >= keltner["lower"][valid_idx]).all()

    def test_donchian_channels(self, sample_ohlcv):
        """Test Donchian Channels."""
        donchian = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
        )

        assert "upper" in donchian
        assert "middle" in donchian
        assert "lower" in donchian

        # Upper should be max high
        assert donchian["upper"].notna().any()
        assert donchian["lower"].notna().any()

    def test_aroon_indicator(self, sample_ohlcv):
        """Test Aroon Indicator."""
        aroon_up, aroon_down = aroon_indicator(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
        )

        assert len(aroon_up) == len(sample_ohlcv)
        assert len(aroon_down) == len(sample_ohlcv)

        # Values should be 0-100
        assert aroon_up[aroon_up.notna()].between(0, 100).all()
        assert aroon_down[aroon_down.notna()].between(0, 100).all()

    def test_compute_all_advanced_indicators(self, sample_ohlcv):
        """Test computing all indicators at once."""
        result = compute_all_advanced_indicators(sample_ohlcv)

        # Should have all original columns
        for col in sample_ohlcv.columns:
            assert col in result.columns

        # Should have added many new columns
        assert len(result.columns) > len(sample_ohlcv.columns)

        # Check some expected columns
        assert "stoch_k" in result.columns
        assert "williams_r" in result.columns
        assert "cci" in result.columns
        assert "mfi" in result.columns
        assert "obv" in result.columns


class TestLookaheadDetection:
    """Test lookahead bias detection."""

    def test_valid_function_passes(self):
        """Test that valid function passes detection."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "close": np.random.uniform(95, 105, 50),
        }, index=dates)

        detector = LookaheadDetector()
        detections = detector.check_function(valid_feature_function, data)

        # Should have no bias
        assert not any(d.has_bias for d in detections)

    def test_invalid_function_detected(self):
        """Test that invalid function is detected."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "close": np.random.uniform(95, 105, 50),
        }, index=dates)

        detector = LookaheadDetector()
        detections = detector.check_function(invalid_feature_function, data)

        # Should detect bias
        assert any(d.has_bias for d in detections)

        # Should detect specific types
        bias_types = [d.bias_type for d in detections if d.has_bias]
        assert any(bt in [BiasType.CENTERED_WINDOW, BiasType.FUTURE_SHIFT, BiasType.FULL_SAMPLE_STATS]
                   for bt in bias_types if bt is not None)

    def test_validate_feature_function_raises(self):
        """Test that validation raises on invalid function."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "close": np.random.uniform(95, 105, 50),
        }, index=dates)

        with pytest.raises(ValueError, match="Lookahead bias detected"):
            validate_feature_function(invalid_feature_function, data)

    def test_validate_feature_function_passes(self):
        """Test that validation passes on valid function."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "close": np.random.uniform(95, 105, 50),
        }, index=dates)

        # Should not raise
        result = validate_feature_function(valid_feature_function, data)
        assert result is True


class TestLLMTemporalSafety:
    """Test LLM temporal safety framework."""

    def test_canary_questions(self):
        """Test canary questions are configured."""
        assert len(CANARY_QUESTIONS) > 0

        for canary in CANARY_QUESTIONS:
            assert canary.question
            assert canary.earliest_knowable
            assert canary.correct_answer
            assert canary.category

    def test_mock_llm_respects_cutoff(self):
        """Test MockLLM respects knowledge cutoff."""
        llm = MockLLM(knowledge_cutoff=datetime(2020, 1, 1))

        # Ask about 2020 election (after cutoff) - should not know
        response = llm.query("Who won the 2020 US presidential election?")
        assert "don't have information" in response.lower() or "not enough information" in response.lower()

        # Mock LLM with later cutoff
        llm2 = MockLLM(knowledge_cutoff=datetime(2021, 1, 1))

        # Now should know
        response2 = llm2.query("Who won the 2020 US presidential election?")
        assert "biden" in response2.lower()

    def test_temporal_safety_tester(self):
        """Test temporal safety testing."""
        # LLM with early cutoff
        llm = MockLLM(knowledge_cutoff=datetime(2020, 1, 1))
        tester = LLMTemporalSafetyTester(llm)

        # Test with cutoff before all canaries
        result = tester.test_temporal_isolation(datetime(2019, 12, 31))

        # Should pass (LLM doesn't have future knowledge)
        assert result.passed
        assert len(result.violations) == 0

    def test_temporal_safety_tester_detects_leakage(self):
        """Test detection of temporal leakage."""
        # LLM with late cutoff (knows everything)
        llm = MockLLM(knowledge_cutoff=datetime(2023, 1, 1))
        tester = LLMTemporalSafetyTester(llm)

        # Test with earlier cutoff
        result = tester.test_temporal_isolation(datetime(2020, 1, 1))

        # Should fail (LLM has future knowledge)
        assert not result.passed
        assert len(result.violations) > 0

    def test_require_temporal_safety(self):
        """Test require_temporal_safety function."""
        # Safe LLM
        llm_safe = MockLLM(knowledge_cutoff=datetime(2020, 1, 1))

        # Should not raise
        require_temporal_safety(llm_safe, datetime(2020, 1, 1))

        # Unsafe LLM
        llm_unsafe = MockLLM(knowledge_cutoff=datetime(2023, 1, 1))

        # Should raise
        with pytest.raises(ValueError, match="failed temporal safety test"):
            require_temporal_safety(llm_unsafe, datetime(2020, 1, 1))
