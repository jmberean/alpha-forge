"""Advanced Strategy Discovery System.

This module implements a state-of-the-art strategy discovery engine featuring:
- Expression Tree Genetic Programming (STGP)
- Multi-objective optimization (NSGA-III)
- Surrogate fitness prediction
- Factor Zoo with diversity maintenance
- Dynamic ensemble combination

Based on research from:
- AAAI 2025 AlphaForge paper (arxiv.org/html/2406.18394v1)
- Eurocast 2024 (arxiv.org/html/2504.05418v1)
- Multi-objective GP literature (arxiv.org/html/2412.00896v1)
"""

from alphaforge.discovery.expression.tree import ExpressionTree
from alphaforge.discovery.expression.types import DataType, NodeType
from alphaforge.discovery.evolution.nsga3 import NSGA3Optimizer, NSGA3Config
from alphaforge.discovery.orchestrator import DiscoveryOrchestrator, DiscoveryConfig

__all__ = [
    "ExpressionTree",
    "DataType",
    "NodeType",
    "NSGA3Optimizer",
    "NSGA3Config",
    "DiscoveryOrchestrator",
    "DiscoveryConfig",
]
