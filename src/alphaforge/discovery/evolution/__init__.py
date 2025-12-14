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
    ObjectiveValues,
)
from alphaforge.discovery.evolution.population import (
    Population,
    create_initial_population,
)

__all__ = [
    "NSGA3Config",
    "NSGA3Optimizer",
    "ObjectiveValues",
    "Population",
    "create_initial_population",
]
