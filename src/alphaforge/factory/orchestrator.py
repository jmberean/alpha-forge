"""
Strategy factory orchestrator.

Coordinates strategy generation from multiple sources.
"""

from dataclasses import dataclass, field
from typing import Callable

from alphaforge.factory.genetic import EvolutionConfig, GeneticStrategyEvolver
from alphaforge.strategy.genome import StrategyGenome


@dataclass
class FactoryConfig:
    """Configuration for strategy factory."""

    # Genetic evolution
    use_genetic: bool = True
    genetic_population: int = 100
    genetic_generations: int = 50

    # Target metrics
    target_strategies: int = 100  # Total strategies to generate
    min_fitness: float = 1.0  # Minimum fitness to accept


@dataclass
class StrategyPool:
    """Pool of generated strategies."""

    strategies: list[StrategyGenome] = field(default_factory=list)
    fitness_scores: dict[str, float] = field(default_factory=dict)

    def add_strategy(self, strategy: StrategyGenome, fitness: float) -> None:
        """Add strategy to pool."""
        self.strategies.append(strategy)
        self.fitness_scores[strategy.id] = fitness

    def get_best(self, n: int = 10) -> list[StrategyGenome]:
        """Get top N strategies by fitness."""
        sorted_strategies = sorted(
            self.strategies,
            key=lambda s: self.fitness_scores.get(s.id, -999.0),
            reverse=True,
        )
        return sorted_strategies[:n]

    def filter_by_fitness(self, min_fitness: float) -> list[StrategyGenome]:
        """Filter strategies by minimum fitness."""
        return [
            s for s in self.strategies
            if self.fitness_scores.get(s.id, -999.0) >= min_fitness
        ]

    def deduplicate(self) -> None:
        """Remove duplicate strategies."""
        seen = set()
        unique_strategies = []
        unique_fitness = {}

        for strategy in self.strategies:
            if strategy.id not in seen:
                seen.add(strategy.id)
                unique_strategies.append(strategy)
                if strategy.id in self.fitness_scores:
                    unique_fitness[strategy.id] = self.fitness_scores[strategy.id]

        self.strategies = unique_strategies
        self.fitness_scores = unique_fitness


class StrategyFactory:
    """
    Orchestrate strategy generation from multiple sources.

    Coordinates:
    - Genetic evolution
    - Template variations
    - Parameter optimization
    - Deduplication
    """

    def __init__(
        self,
        fitness_function: Callable[[StrategyGenome], float],
        config: FactoryConfig | None = None,
    ):
        """
        Initialize strategy factory.

        Args:
            fitness_function: Function to evaluate strategy fitness
            config: Factory configuration
        """
        self.fitness_function = fitness_function
        self.config = config or FactoryConfig()
        self.pool = StrategyPool()

    def generate(self) -> StrategyPool:
        """
        Generate strategy candidates.

        Returns:
            StrategyPool with generated strategies
        """
        # Reset pool
        self.pool = StrategyPool()

        # 1. Genetic evolution (if enabled)
        if self.config.use_genetic:
            self._generate_genetic()

        # 2. Template variations
        self._generate_template_variations()

        # 3. Deduplicate
        self.pool.deduplicate()

        # 4. Filter by minimum fitness
        qualified = self.pool.filter_by_fitness(self.config.min_fitness)

        return self.pool

    def _generate_genetic(self) -> None:
        """Generate strategies using genetic evolution."""
        evolver = GeneticStrategyEvolver(
            fitness_function=self.fitness_function,
            config=EvolutionConfig(
                population_size=self.config.genetic_population,
                n_generations=self.config.genetic_generations,
            ),
        )

        result = evolver.evolve()

        # Add all population to pool
        for strategy in result.population:
            fitness = strategy.fitness if hasattr(strategy, 'fitness') else self.fitness_function(strategy)
            self.pool.add_strategy(strategy, fitness)

    def _generate_template_variations(self) -> None:
        """Generate variations of strategy templates."""
        from alphaforge.strategy.templates import StrategyTemplates
        import random

        # Generate some template variations
        n_variations = max(0, self.config.target_strategies - len(self.pool.strategies))
        n_variations = min(n_variations, 50)  # Cap at 50

        for _ in range(n_variations):
            # Random template
            template_choice = random.choice([
                StrategyTemplates.sma_crossover,
                StrategyTemplates.rsi_mean_reversion,
                StrategyTemplates.macd_crossover,
                StrategyTemplates.bollinger_breakout,
                StrategyTemplates.dual_momentum,
            ])

            # Random parameters matching actual template signatures
            if template_choice == StrategyTemplates.sma_crossover:
                strategy = template_choice(
                    fast_period=random.randint(10, 30),
                    slow_period=random.randint(40, 100),
                )
            elif template_choice == StrategyTemplates.rsi_mean_reversion:
                strategy = template_choice(
                    rsi_period=random.randint(10, 20),
                    oversold=float(random.randint(20, 35)),
                    overbought=float(random.randint(65, 80)),
                )
            elif template_choice == StrategyTemplates.macd_crossover:
                strategy = template_choice(
                    fast=random.randint(8, 15),
                    slow=random.randint(20, 30),
                    signal=random.randint(6, 12),
                )
            elif template_choice == StrategyTemplates.bollinger_breakout:
                strategy = template_choice(
                    period=random.randint(15, 25),
                    num_std=random.uniform(1.5, 2.5),
                )
            elif template_choice == StrategyTemplates.dual_momentum:
                strategy = template_choice(
                    abs_momentum_period=random.randint(180, 300),
                    rel_momentum_period=random.randint(90, 160),
                )
            else:
                strategy = template_choice()

            # Evaluate
            fitness = self.fitness_function(strategy)
            self.pool.add_strategy(strategy, fitness)

    def get_top_strategies(self, n: int = 10) -> list[tuple[StrategyGenome, float]]:
        """
        Get top N strategies with their fitness scores.

        Args:
            n: Number of top strategies to return

        Returns:
            List of (strategy, fitness) tuples
        """
        top_strategies = self.pool.get_best(n)
        return [
            (strategy, self.pool.fitness_scores.get(strategy.id, 0.0))
            for strategy in top_strategies
        ]


def generate_strategy_candidates(
    fitness_function: Callable[[StrategyGenome], float],
    n_candidates: int = 100,
) -> list[StrategyGenome]:
    """
    Convenience function to generate strategy candidates.

    Args:
        fitness_function: Function to evaluate fitness
        n_candidates: Target number of candidates

    Returns:
        List of strategy candidates

    Example:
        >>> def my_fitness(strategy):
        ...     # Run backtest and return Sharpe
        ...     return 1.5
        >>> strategies = generate_strategy_candidates(my_fitness, n_candidates=50)
    """
    factory = StrategyFactory(
        fitness_function=fitness_function,
        config=FactoryConfig(target_strategies=n_candidates),
    )

    pool = factory.generate()

    return pool.strategies
