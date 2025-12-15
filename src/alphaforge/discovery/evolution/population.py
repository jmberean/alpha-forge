"""Population management for genetic programming.

Handles population initialization, diversity maintenance, and statistics.
"""

from dataclasses import dataclass, field
from typing import Iterator, Callable
import random

from alphaforge.evolution.protocol import Evolvable
from alphaforge.discovery.operators.selection import Individual
from alphaforge.evolution.genomes import ExpressionGenome


@dataclass
class PopulationStats:
    """Statistics about a population."""

    size: int
    unique_formulas: int
    avg_complexity: float
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
    _genome_hashes: set[str] = field(default_factory=set)

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
        genome_hash = individual.genome.hash

        if genome_hash in self._genome_hashes:
            return False

        self._genome_hashes.add(genome_hash)
        self.individuals.append(individual)
        return True

    def add_genome(self, genome: Evolvable) -> bool:
        """Add a genome as a new individual with empty fitness.

        Args:
            genome: Evolvable genome to add

        Returns:
            True if added, False if duplicate
        """
        individual = Individual(
            genome=genome,
            fitness={},
            rank=0,
            crowding_distance=0.0,
        )
        return self.add(individual)

    def remove(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        if individual in self.individuals:
            self.individuals.remove(individual)
            self._genome_hashes.discard(individual.genome.hash)

    def clear(self) -> None:
        """Clear all individuals."""
        self.individuals.clear()
        self._genome_hashes.clear()

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
                avg_complexity=0.0,
                min_fitness={},
                max_fitness={},
                avg_fitness={},
                pareto_front_size=0,
            )

        complexities = [ind.genome.complexity_score() for ind in self.individuals]

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
            unique_formulas=len(self._genome_hashes),
            avg_complexity=sum(complexities) / len(complexities),
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            avg_fitness=avg_fitness,
            pareto_front_size=len(self.get_pareto_front()),
        )

    def to_genomes(self) -> list[Evolvable]:
        """Extract all genomes from population."""
        return [ind.genome for ind in self.individuals]


def create_initial_population(
    size: int,
    generator: Callable[[], Evolvable],
    seed: int | None = None,
    warm_start_genomes: list[Evolvable] | None = None,
) -> Population:
    """Create initial population with optional warm start.

    Args:
        size: Target population size
        generator: Function to generate new genomes
        seed: Random seed
        warm_start_genomes: Optional genomes to include in initial population

    Returns:
        Population with unique individuals
    """
    # rng = random.Random(seed) # Not needed if generator handles randomness internally or we assume externally managed
    
    population = Population()

    # Add warm start genomes first
    if warm_start_genomes:
        for genome in warm_start_genomes:
            if len(population) >= size:
                break
            try:
                # Assuming clone logic if needed, but genomes should be immutable/safe
                # Actually, warm start items should be fresh copies?
                # Evolvable doesn't mandate clone(). Assume caller provides fresh instances or we rely on immutability.
                # ExpressionTree is immutable-ish. TemplateGenome creates copy on mutation.
                population.add_genome(genome)
            except ValueError:
                continue

    # Fill remaining with random genomes
    attempts = 0
    max_attempts = size * 10

    while len(population) < size and attempts < max_attempts:
        attempts += 1
        try:
            genome = generator()
            population.add_genome(genome)
        except (ValueError, RecursionError):
            continue

    return population


def inject_diversity(
    population: Population,
    n_inject: int,
    generator: Callable[[], Evolvable],
    rng: random.Random | None = None,
) -> int:
    """Inject random individuals to maintain diversity.

    Replaces worst individuals (highest rank, lowest crowding distance).

    Args:
        population: Population to modify
        n_inject: Number of individuals to inject
        generator: Function to generate new genomes
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
            genome = generator()
            if population.add_genome(genome):
                injected += 1
        except (ValueError, RecursionError):
            continue

    return injected
