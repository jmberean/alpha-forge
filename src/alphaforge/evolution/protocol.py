
from typing import Protocol, Any, runtime_checkable
from alphaforge.strategy.genome import StrategyGenome

@runtime_checkable
class Evolvable(Protocol):
    """Protocol for evolvable genomes (Strategies, Trees, etc)."""
    
    @property
    def hash(self) -> str:
        """Unique hash of the genome."""
        ...

    def mutate(self, rng: Any) -> "Evolvable":
        """Produce a mutated copy."""
        ...

    def crossover(self, other: "Evolvable", rng: Any) -> tuple["Evolvable", "Evolvable"]:
        """Produce two children via crossover."""
        ...

    def to_strategy_genome(self) -> StrategyGenome:
        """Convert to standard StrategyGenome for backtesting."""
        ...

    def complexity_score(self) -> float:
        """Get complexity score (0.0 to 1.0)."""
        ...
