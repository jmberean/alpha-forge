"""
Genetic programming for strategy evolution using DEAP.

Evolves trading strategies through mutation, crossover, and selection.
"""

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from deap import algorithms, base, creator, tools

from alphaforge.strategy.genome import StrategyGenome


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""

    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elitism: int = 5  # Number of best individuals to keep
    random_seed: int | None = None


@dataclass
class EvolutionResult:
    """Result from genetic evolution."""

    best_strategy: StrategyGenome
    best_fitness: float
    population: list[StrategyGenome]
    fitness_history: list[list[float]]  # Fitness per generation
    generation_count: int


class GeneticStrategyEvolver:
    """
    Evolve trading strategies using genetic algorithms.

    Uses DEAP framework for genetic programming operations.
    """

    def __init__(
        self,
        fitness_function: Callable[[StrategyGenome], float],
        config: EvolutionConfig | None = None,
    ):
        """
        Initialize genetic evolver.

        Args:
            fitness_function: Function to evaluate strategy fitness (higher is better)
            config: Evolution configuration
        """
        self.fitness_function = fitness_function
        self.config = config or EvolutionConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP framework."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        self.toolbox.register("evaluate", self._evaluate)

    def evolve(
        self,
        initial_population: list[StrategyGenome] | None = None,
    ) -> EvolutionResult:
        """
        Run genetic evolution.

        Args:
            initial_population: Starting population (random if None)

        Returns:
            EvolutionResult with best strategy and evolution history
        """
        # Initialize population
        if initial_population is None:
            population = self._create_random_population()
        else:
            population = initial_population

        # Track fitness history
        fitness_history = []

        # Evaluate initial population
        fitnesses = [self._evaluate_strategy(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness = fit

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Record fitness
            current_fitness = [ind.fitness for ind in population]
            fitness_history.append(current_fitness)

            # Select next generation
            offspring = self.toolbox.select(population, len(population) - self.config.elitism)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    if hasattr(child1, 'fitness'):
                        delattr(child1, 'fitness')
                    if hasattr(child2, 'fitness'):
                        delattr(child2, 'fitness')

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.config.mutation_prob:
                    self.toolbox.mutate(mutant)
                    if hasattr(mutant, 'fitness'):
                        delattr(mutant, 'fitness')

            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not hasattr(ind, 'fitness')]
            fitnesses = [self._evaluate_strategy(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness = fit

            # Elitism: Keep best individuals
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite = population[:self.config.elitism]

            # Replace population
            population[:] = elite + offspring

        # Final fitness
        final_fitness = [ind.fitness for ind in population]
        fitness_history.append(final_fitness)

        # Get best strategy
        best_strategy = max(population, key=lambda x: x.fitness)
        best_fitness = best_strategy.fitness

        return EvolutionResult(
            best_strategy=best_strategy,
            best_fitness=best_fitness,
            population=population,
            fitness_history=fitness_history,
            generation_count=self.config.n_generations,
        )

    def _create_random_population(self) -> list[StrategyGenome]:
        """Create random initial population."""
        from alphaforge.strategy.templates import StrategyTemplates

        population = []

        # Create variations of basic strategies
        templates = [
            StrategyTemplates.sma_crossover,
            StrategyTemplates.rsi_mean_reversion,
            StrategyTemplates.macd_crossover,
            StrategyTemplates.bollinger_breakout,
            StrategyTemplates.dual_momentum,
        ]

        for _ in range(self.config.population_size):
            # Pick random template
            template_func = random.choice(templates)

            # Random parameters matching actual template signatures
            if template_func == StrategyTemplates.sma_crossover:
                strategy = template_func(
                    fast_period=random.randint(10, 30),
                    slow_period=random.randint(40, 100),
                )
            elif template_func == StrategyTemplates.rsi_mean_reversion:
                strategy = template_func(
                    rsi_period=random.randint(10, 20),
                    oversold=float(random.randint(20, 35)),
                    overbought=float(random.randint(65, 80)),
                )
            elif template_func == StrategyTemplates.macd_crossover:
                strategy = template_func(
                    fast=random.randint(8, 15),
                    slow=random.randint(20, 30),
                    signal=random.randint(6, 12),
                )
            elif template_func == StrategyTemplates.bollinger_breakout:
                strategy = template_func(
                    period=random.randint(15, 25),
                    num_std=random.uniform(1.5, 2.5),
                )
            elif template_func == StrategyTemplates.dual_momentum:
                strategy = template_func(
                    abs_momentum_period=random.randint(180, 300),
                    rel_momentum_period=random.randint(90, 160),
                )
            else:
                strategy = template_func()

            # Store fitness placeholder
            strategy.fitness = 0.0

            population.append(strategy)

        return population

    def _evaluate_strategy(self, strategy: StrategyGenome) -> float:
        """Evaluate strategy fitness."""
        try:
            fitness = self.fitness_function(strategy)
            return fitness
        except Exception:
            # Failed strategies get very low fitness
            return -999.0

    def _evaluate(self, individual: StrategyGenome) -> tuple[float]:
        """DEAP evaluation wrapper."""
        fitness = self._evaluate_strategy(individual)
        return (fitness,)

    def _crossover(self, ind1: StrategyGenome, ind2: StrategyGenome) -> tuple[StrategyGenome, StrategyGenome]:
        """
        Crossover two strategies.

        Simple parameter-level crossover for now.
        """
        # For this MVP, we do simple parameter swapping
        # In production, this would be more sophisticated

        # Swap some parameters (simplified)
        if hasattr(ind1, 'signals') and hasattr(ind2, 'signals'):
            if len(ind1.signals) > 0 and len(ind2.signals) > 0:
                # Swap random signals
                if random.random() < 0.5:
                    if random.random() < 0.5 and len(ind1.signals) > 1:
                        idx = random.randint(0, len(ind1.signals) - 1)
                        ind1.signals[idx], ind2.signals[idx] = ind2.signals[idx], ind1.signals[idx]

        return ind1, ind2

    def _mutate(self, individual: StrategyGenome) -> tuple[StrategyGenome]:
        """
        Mutate a strategy.

        Randomly adjust parameters.
        """
        # Simple parameter mutation
        if hasattr(individual, 'signals') and len(individual.signals) > 0:
            signal_idx = random.randint(0, len(individual.signals) - 1)
            signal = individual.signals[signal_idx]

            # Mutate indicator parameters
            if hasattr(signal, 'indicator') and signal.indicator:
                if hasattr(signal.indicator, 'period'):
                    # Mutate period +/- 20%
                    current = signal.indicator.period
                    delta = int(current * random.uniform(-0.2, 0.2))
                    signal.indicator.period = max(5, current + delta)

        return (individual,)

    def clone(self, individual: StrategyGenome) -> StrategyGenome:
        """Clone an individual (deep copy)."""
        import copy
        return copy.deepcopy(individual)


def simple_fitness_function(strategy: StrategyGenome) -> float:
    """
    Simple mock fitness function for testing.

    In production, this would run a backtest and return Sharpe ratio or other metric.

    Args:
        strategy: Strategy to evaluate

    Returns:
        Fitness score (higher is better)
    """
    # Mock: Return random fitness based on strategy hash
    import hashlib
    strategy_str = f"{strategy.name}_{strategy.id}"
    hash_value = int(hashlib.md5(strategy_str.encode()).hexdigest(), 16)

    # Normalize to 0-5 range (like Sharpe ratios)
    fitness = (hash_value % 1000) / 200.0

    return fitness
