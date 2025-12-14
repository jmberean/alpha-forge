"""Population management for genetic programming.

Handles population initialization, diversity maintenance, and statistics.
"""

from dataclasses import dataclass, field
from typing import Iterator
import random

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.operators.selection import Individual


@dataclass
class PopulationStats:
    """Statistics about a population."""

    size: int
    unique_formulas: int
    avg_size: float
    avg_depth: float
    min_fitness: dict[str, float]
    max_fitness: dict[str, float]
    avg_fitness: dict[str, float]
    pareto_front_size: int


@dataclass
class Population:
    """Population of individuals for evolution.

    Maintains diversity through hash-based deduplication.
    """

    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0
    _formula_hashes: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]

    def add(self, individual: Individual) -> bool:
        """Add individual if unique.

        Args:
            individual: Individual to add

        Returns:
            True if added (was unique), False if duplicate
        """
        formula_hash = individual.tree.hash

        if formula_hash in self._formula_hashes:
            return False

        self._formula_hashes.add(formula_hash)
        self.individuals.append(individual)
        return True

    def add_tree(self, tree: ExpressionTree) -> bool:
        """Add a tree as a new individual with empty fitness.

        Args:
            tree: Expression tree to add

        Returns:
            True if added, False if duplicate
        """
        individual = Individual(
            tree=tree,
            fitness={},
            rank=0,
            crowding_distance=0.0,
        )
        return self.add(individual)

    def remove(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        if individual in self.individuals:
            self.individuals.remove(individual)
            self._formula_hashes.discard(individual.tree.hash)

    def clear(self) -> None:
        """Clear all individuals."""
        self.individuals.clear()
        self._formula_hashes.clear()

    def get_pareto_front(self) -> list[Individual]:
        """Get individuals on the Pareto front (rank 0)."""
        return [ind for ind in self.individuals if ind.rank == 0]

    def get_best_by_objective(self, objective: str) -> Individual | None:
        """Get individual with best value for given objective."""
        if not self.individuals:
            return None

        return max(
            self.individuals,
            key=lambda ind: ind.fitness.get(objective, float("-inf")),
        )

    def compute_stats(self) -> PopulationStats:
        """Compute population statistics."""
        if not self.individuals:
            return PopulationStats(
                size=0,
                unique_formulas=0,
                avg_size=0.0,
                avg_depth=0.0,
                min_fitness={},
                max_fitness={},
                avg_fitness={},
                pareto_front_size=0,
            )

        sizes = [ind.tree.size for ind in self.individuals]
        depths = [ind.tree.depth for ind in self.individuals]

        # Aggregate fitness stats
        objectives = set()
        for ind in self.individuals:
            objectives.update(ind.fitness.keys())

        min_fitness = {}
        max_fitness = {}
        avg_fitness = {}

        for obj in objectives:
            values = [
                ind.fitness.get(obj, 0)
                for ind in self.individuals
                if obj in ind.fitness
            ]
            if values:
                min_fitness[obj] = min(values)
                max_fitness[obj] = max(values)
                avg_fitness[obj] = sum(values) / len(values)

        return PopulationStats(
            size=len(self.individuals),
            unique_formulas=len(self._formula_hashes),
            avg_size=sum(sizes) / len(sizes),
            avg_depth=sum(depths) / len(depths),
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            avg_fitness=avg_fitness,
            pareto_front_size=len(self.get_pareto_front()),
        )

    def to_trees(self) -> list[ExpressionTree]:
        """Extract all trees from population."""
        return [ind.tree for ind in self.individuals]


def create_initial_population(
    size: int,
    generator: TreeGenerator | None = None,
    seed: int | None = None,
    warm_start_trees: list[ExpressionTree] | None = None,
) -> Population:
    """Create initial population with optional warm start.

    Args:
        size: Target population size
        generator: Tree generator (created if not provided)
        seed: Random seed
        warm_start_trees: Optional trees to include in initial population

    Returns:
        Population with unique individuals
    """
    rng = random.Random(seed)
    generator = generator or TreeGenerator(seed=seed)

    population = Population()

    # Add warm start trees first
    if warm_start_trees:
        for tree in warm_start_trees:
            if len(population) >= size:
                break
            try:
                population.add_tree(tree.clone())
            except ValueError:
                continue

    # Fill remaining with random trees
    attempts = 0
    max_attempts = size * 10

    while len(population) < size and attempts < max_attempts:
        attempts += 1
        try:
            tree = generator.generate(method="ramped")
            population.add_tree(tree)
        except (ValueError, RecursionError):
            continue

    return population


def inject_diversity(
    population: Population,
    n_inject: int,
    generator: TreeGenerator,
    rng: random.Random | None = None,
) -> int:
    """Inject random individuals to maintain diversity.

    Replaces worst individuals (highest rank, lowest crowding distance).

    Args:
        population: Population to modify
        n_inject: Number of individuals to inject
        generator: Tree generator
        rng: Random number generator

    Returns:
        Number of individuals actually injected
    """
    rng = rng or random.Random()

    # Sort by rank (descending) and crowding distance (ascending)
    # to identify worst individuals
    sorted_individuals = sorted(
        population.individuals,
        key=lambda ind: (-ind.rank, ind.crowding_distance),
    )

    # Remove worst individuals
    n_to_remove = min(n_inject, len(population) // 4)  # Max 25% replacement
    for i in range(n_to_remove):
        if sorted_individuals:
            population.remove(sorted_individuals.pop(0))

    # Inject new random individuals
    injected = 0
    attempts = 0
    max_attempts = n_inject * 10

    while injected < n_inject and attempts < max_attempts:
        attempts += 1
        try:
            tree = generator.generate(method="ramped")
            if population.add_tree(tree):
                injected += 1
        except (ValueError, RecursionError):
            continue

    return injected
