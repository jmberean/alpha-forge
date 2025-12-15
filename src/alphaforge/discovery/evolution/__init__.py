"""Evolution engine for multi-objective genetic programming.

Implements NSGA-III for evolving expression trees with multiple objectives:
- Sharpe ratio (maximize)
- Maximum drawdown (minimize -> negate)
- Turnover (minimize -> negate)
- Complexity (minimize -> negate)
"""

from alphaforge.discovery.evolution.nsga3 import (
    NSGA3Config,
    NSGA3Optimizer,
    NSGA3Result,
)
from alphaforge.discovery.evolution.population import (
    Population,
    PopulationStats,
    create_initial_population,
    inject_diversity,
)

__all__ = [
    "NSGA3Config",
    "NSGA3Optimizer",
    "NSGA3Result",
    "Population",
    "PopulationStats",
    "create_initial_population",
    "inject_diversity",
]
