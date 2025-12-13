"""
Validation pipeline: Statistical validation to eliminate false discoveries.

This module provides:
- DeflatedSharpeRatio: DSR accounting for multiple testing
- CombinatorialPurgedCV: CPCV with 12,870 combinations
- ProbabilityOfOverfitting: PBO calculator
- ValidationPipeline: Full validation funnel
"""

from alphaforge.validation.cpcv import CombinatorialPurgedCV, CPCVResult
from alphaforge.validation.dsr import DeflatedSharpeRatio
from alphaforge.validation.pbo import ProbabilityOfOverfitting
from alphaforge.validation.pipeline import ValidationPipeline, ValidationResult

__all__ = [
    "DeflatedSharpeRatio",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "ProbabilityOfOverfitting",
    "ValidationPipeline",
    "ValidationResult",
]
