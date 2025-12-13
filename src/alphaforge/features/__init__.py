"""
Feature store: Point-in-time technical indicators and advanced feature engineering.

This module provides:
- TechnicalIndicators: RSI, MACD, SMA, Bollinger Bands, ATR, etc.
- AdvancedIndicators: 30+ advanced technical indicators
- FeatureStore: Point-in-time feature computation and storage
- LookaheadDetector: Detect and prevent lookahead bias
- LLMTemporalSafetyTester: Ensure LLM features respect temporal constraints
"""

from alphaforge.features.advanced_technical import compute_all_advanced_indicators
from alphaforge.features.llm_safety import (
    CANARY_QUESTIONS,
    LLMTemporalSafetyTester,
    MockLLM,
    SafeLLMFeatures,
    require_temporal_safety,
)
from alphaforge.features.lookahead import (
    BiasType,
    FeatureValidationRules,
    LookaheadDetector,
    invalid_feature_function,
    valid_feature_function,
    validate_feature_function,
)
from alphaforge.features.store import FeatureStore
from alphaforge.features.technical import TechnicalIndicators

__all__ = [
    # Original
    "TechnicalIndicators",
    "FeatureStore",
    # MVP5 - Advanced Indicators
    "compute_all_advanced_indicators",
    # MVP5 - Lookahead Detection
    "LookaheadDetector",
    "BiasType",
    "validate_feature_function",
    "FeatureValidationRules",
    "valid_feature_function",
    "invalid_feature_function",
    # MVP5 - LLM Safety
    "LLMTemporalSafetyTester",
    "MockLLM",
    "SafeLLMFeatures",
    "require_temporal_safety",
    "CANARY_QUESTIONS",
]
