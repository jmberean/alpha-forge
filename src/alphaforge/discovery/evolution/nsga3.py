"""NSGA-III: Non-dominated Sorting Genetic Algorithm III.

Implements many-objective optimization with reference point-based niching.
Handles 3+ objectives better than NSGA-II.

Based on:
- Deb & Jain (2014) "An Evolutionary Many-Objective Optimization Algorithm..."
- IEEE Transactions on Evolutionary Computation, 18(4), 577-601
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor
import os
import random
import math
import numpy as np


# Optimal thread count for parallel evaluation
N_EVAL_THREADS = min(os.cpu_count() or 4, 8)  # Cap at 8 for evaluation

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.expression.compiler import compile_tree
from alphaforge.discovery.operators.crossover import crossover
from alphaforge.discovery.operators.mutation import mutate
from alphaforge.discovery.operators.selection import (
    Individual,
    compute_pareto_ranks,
    compute_crowding_distance,
    dominates,
)
from alphaforge.discovery.evolution.population import (
    Population,
    create_initial_population,
    inject_diversity,
)


@dataclass
class NSGA3Config:
    """Configuration for NSGA-III optimizer.

    Attributes:
        population_size: Population size (should match reference points)
        n_generations: Number of generations
        n_objectives: Number of objectives to optimize
        crossover_prob: Probability of crossover
        mutation_prob: Probability of mutation
        diversity_injection: Inject random individuals every N generations (0 to disable)
        n_reference_points: Number of reference points (None = auto-calculate)
        seed: Random seed
    """

    population_size: int = 200
    n_generations: int = 100
    n_objectives: int = 4  # Sharpe, MaxDD, Turnover, Complexity
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    diversity_injection: int = 20  # Every 20 generations
    n_reference_points: int | None = None
    seed: int | None = None


@dataclass
class NSGA3Result:
    """Result from NSGA-III optimization.

    Attributes:
        pareto_front: Individuals on final Pareto front
        population: Final population
        best_by_objective: Best individual for each objective
        generation_stats: Statistics per generation
        n_generations: Number of generations completed
    """

    pareto_front: list[Individual]
    population: Population
    best_by_objective: dict[str, Individual]
    generation_stats: list[dict[str, Any]]
    n_generations: int


class NSGA3Optimizer:
    """NSGA-III optimizer for many-objective genetic programming.

    Optimizes expression trees across multiple objectives:
    - Sharpe ratio (maximize)
    - Maximum drawdown (minimize -> negate for maximization)
    - Turnover (minimize -> negate)
    - Complexity (minimize -> negate)
    - Correlation to existing strategies (minimize -> negate)
    """

    def __init__(
        self,
        fitness_functions: dict[str, Callable[[ExpressionTree], float]],
        config: NSGA3Config | None = None,
    ):
        """Initialize NSGA-III optimizer.

        Args:
            fitness_functions: Dict of objective_name -> fitness_function
                               All should be MAXIMIZATION (negate if minimizing)
            config: Configuration
        """
        self.fitness_functions = fitness_functions
        self.config = config or NSGA3Config()
        self.rng = random.Random(self.config.seed)

        # Generate reference points
        self.reference_points = self._generate_reference_points()

        # Adjust population size to match reference points
        if len(self.reference_points) != self.config.population_size:
            self.config.population_size = len(self.reference_points)

        # Tree generator
        self.tree_generator = TreeGenerator(seed=self.config.seed)

    def _generate_reference_points(self) -> np.ndarray:
        """Generate uniformly distributed reference points on unit simplex.

        Uses Das & Dennis (1998) method for structured reference points.

        Returns:
            Array of shape (n_points, n_objectives)
        """
        n_obj = self.config.n_objectives

        if self.config.n_reference_points:
            # Use specified number
            n_partitions = self._calculate_partitions(
                n_obj, self.config.n_reference_points
            )
        else:
            # Auto-calculate based on objectives
            if n_obj <= 3:
                n_partitions = 12
            elif n_obj <= 6:
                n_partitions = 4
            else:
                n_partitions = 3

        return self._das_dennis_reference_points(n_obj, n_partitions)

    def _calculate_partitions(self, n_obj: int, target_points: int) -> int:
        """Calculate number of partitions to get close to target points."""
        for p in range(1, 50):
            n_points = math.comb(n_obj + p - 1, p)
            if n_points >= target_points:
                return p
        return 12  # Fallback

    def _das_dennis_reference_points(
        self, n_obj: int, n_partitions: int
    ) -> np.ndarray:
        """Generate Das-Dennis reference points on unit simplex."""
        ref_points = []

        def generate_recursive(
            obj_idx: int, remaining: int, current_point: list[float]
        ) -> None:
            if obj_idx == n_obj - 1:
                current_point.append(remaining / n_partitions)
                ref_points.append(current_point.copy())
                current_point.pop()
            else:
                for i in range(remaining + 1):
                    current_point.append(i / n_partitions)
                    generate_recursive(obj_idx + 1, remaining - i, current_point)
                    current_point.pop()

        generate_recursive(0, n_partitions, [])
        return np.array(ref_points)

    def optimize(
        self,
        initial_population: list[ExpressionTree] | None = None,
        on_generation: Callable[[int, dict], None] | None = None,
    ) -> NSGA3Result:
        """Run NSGA-III optimization.

        Args:
            initial_population: Optional warm-start trees
            on_generation: Optional callback called after each generation
                           with (generation_number, stats_dict)

        Returns:
            NSGA3Result with Pareto front and statistics
        """
        # Initialize population
        population = create_initial_population(
            size=self.config.population_size,
            generator=self.tree_generator,
            seed=self.config.seed,
            warm_start_trees=initial_population,
        )

        # Evaluate initial population
        self._evaluate_population(population)

        # Compute initial ranks
        compute_pareto_ranks(population.individuals)
        compute_crowding_distance(population.individuals)

        generation_stats = []

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Create offspring
            offspring = self._create_offspring(population)

            # Evaluate offspring
            self._evaluate_population(offspring)

            # Combine parent and offspring
            combined = Population()
            for ind in population.individuals + offspring.individuals:
                combined.add(ind)

            # Environmental selection
            population = self._environmental_selection(combined)

            # Diversity injection
            if (self.config.diversity_injection > 0 and
                gen > 0 and gen % self.config.diversity_injection == 0):
                n_inject = max(1, self.config.population_size // 20)
                inject_diversity(population, n_inject, self.tree_generator, self.rng)
                self._evaluate_population(population)
                compute_pareto_ranks(population.individuals)
                compute_crowding_distance(population.individuals)

            # Track statistics
            stats = self._compute_generation_stats(population, gen)
            generation_stats.append(stats)

            # Call callback if provided
            if on_generation is not None:
                on_generation(gen + 1, stats)

            population.generation = gen + 1

        # Final ranking
        compute_pareto_ranks(population.individuals)
        compute_crowding_distance(population.individuals)

        # Extract results
        pareto_front = population.get_pareto_front()

        best_by_objective = {}
        for obj_name in self.fitness_functions.keys():
            best = population.get_best_by_objective(obj_name)
            if best:
                best_by_objective[obj_name] = best

        return NSGA3Result(
            pareto_front=pareto_front,
            population=population,
            best_by_objective=best_by_objective,
            generation_stats=generation_stats,
            n_generations=self.config.n_generations,
        )

    def _evaluate_population(self, population: Population) -> None:
        """Evaluate all individuals without fitness.

        Uses parallel evaluation for significant speedup on multi-core machines.
        """
        # Filter to only unevaluated individuals
        unevaluated = [ind for ind in population.individuals if not ind.fitness]

        if not unevaluated:
            return

        # For small batches, sequential is faster (avoid thread overhead)
        if len(unevaluated) < 4:
            for ind in unevaluated:
                self._evaluate_individual(ind)
            return

        # Parallel evaluation using thread pool
        # Threading works well here because numpy/pandas release the GIL
        with ThreadPoolExecutor(max_workers=N_EVAL_THREADS) as executor:
            # Submit all evaluations
            futures = {executor.submit(self._evaluate_individual, ind): ind
                       for ind in unevaluated}

            # Wait for all to complete (results are stored in individuals)
            for future in futures:
                future.result()  # Raises exception if evaluation failed

    def _evaluate_individual(self, individual: Individual) -> None:
        """Evaluate single individual across all objectives."""
        fitness = {}

        for obj_name, fitness_func in self.fitness_functions.items():
            try:
                value = fitness_func(individual.tree)
                fitness[obj_name] = value
            except Exception:
                # Failed evaluation gets very low fitness
                fitness[obj_name] = -999.0

        individual.fitness = fitness

    def _create_offspring(self, population: Population) -> Population:
        """Create offspring through crossover and mutation."""
        offspring = Population()

        # Create population_size offspring
        n_offspring = 0
        max_attempts = self.config.population_size * 3
        attempts = 0

        while n_offspring < self.config.population_size and attempts < max_attempts:
            attempts += 1

            # Select two parents (binary tournament)
            parent1 = self._binary_tournament_select(population)
            parent2 = self._binary_tournament_select(population)

            # Crossover
            if self.rng.random() < self.config.crossover_prob:
                try:
                    child1, child2 = crossover(
                        parent1.tree,
                        parent2.tree,
                        rng=self.rng,
                        method="subtree",
                    )
                except Exception:
                    child1 = parent1.tree.clone()
                    child2 = parent2.tree.clone()
            else:
                child1 = parent1.tree.clone()
                child2 = parent2.tree.clone()

            # Mutation
            if self.rng.random() < self.config.mutation_prob:
                try:
                    child1 = mutate(child1, rng=self.rng)
                except Exception:
                    pass

            if self.rng.random() < self.config.mutation_prob:
                try:
                    child2 = mutate(child2, rng=self.rng)
                except Exception:
                    pass

            # Add to offspring
            for child in [child1, child2]:
                if n_offspring < self.config.population_size:
                    if offspring.add_tree(child):
                        n_offspring += 1

        return offspring

    def _binary_tournament_select(self, population: Population) -> Individual:
        """Select individual using binary tournament."""
        i1 = self.rng.choice(population.individuals)
        i2 = self.rng.choice(population.individuals)

        # Prefer lower rank
        if i1.rank < i2.rank:
            return i1
        elif i2.rank < i1.rank:
            return i2

        # Same rank, prefer higher crowding distance
        if i1.crowding_distance > i2.crowding_distance:
            return i1
        else:
            return i2

    def _environmental_selection(self, combined: Population) -> Population:
        """Select next generation using NSGA-III reference point niching.

        Args:
            combined: Combined parent + offspring population

        Returns:
            Selected population of target size
        """
        # Compute Pareto ranks
        compute_pareto_ranks(combined.individuals)

        # Select by fronts until reaching target size
        selected = Population()
        front_idx = 0

        while True:
            # Get current front
            current_front = [
                ind for ind in combined.individuals if ind.rank == front_idx
            ]

            if not current_front:
                break

            # If adding entire front doesn't exceed limit, add all
            if len(selected) + len(current_front) <= self.config.population_size:
                for ind in current_front:
                    selected.add(ind)
                front_idx += 1
            else:
                # Need to select subset of this front using niching
                n_needed = self.config.population_size - len(selected)
                niched = self._niching_selection(current_front, n_needed)
                for ind in niched:
                    selected.add(ind)
                break

        # Recompute crowding distance for selected population
        compute_crowding_distance(selected.individuals)

        return selected

    def _niching_selection(
        self, front: list[Individual], n_select: int
    ) -> list[Individual]:
        """Select from front using reference point niching.

        Args:
            front: Individuals on same Pareto front
            n_select: Number to select

        Returns:
            Selected individuals
        """
        if len(front) <= n_select:
            return front

        # Normalize objectives
        normalized = self._normalize_objectives(front)

        # Associate each individual with closest reference point
        associations = self._associate_to_reference_points(normalized)

        # Select using niche preservation
        selected = []
        niche_counts = {i: 0 for i in range(len(self.reference_points))}

        # Track which individuals are available
        available = set(range(len(front)))

        while len(selected) < n_select and available:
            # Find least populated niche
            min_niche_count = min(niche_counts.values())
            min_niches = [
                i for i, count in niche_counts.items() if count == min_niche_count
            ]

            # Randomly select one of the least populated niches
            niche_idx = self.rng.choice(min_niches)

            # Find individuals in this niche
            niche_members = [
                i for i in available if associations[i] == niche_idx
            ]

            if niche_members:
                # Select individual with best perpendicular distance
                distances = [
                    np.linalg.norm(
                        normalized[i] - self.reference_points[niche_idx]
                    )
                    for i in niche_members
                ]
                best_idx = niche_members[np.argmin(distances)]

                selected.append(front[best_idx])
                available.remove(best_idx)
                niche_counts[niche_idx] += 1
            else:
                # No more individuals in this niche, remove it
                niche_counts[niche_idx] = float('inf')

        return selected

    def _normalize_objectives(self, individuals: list[Individual]) -> np.ndarray:
        """Normalize objective values to [0, 1] range.

        Args:
            individuals: Individuals to normalize

        Returns:
            Normalized fitness array of shape (n_individuals, n_objectives)
        """
        objectives = list(self.fitness_functions.keys())
        n_ind = len(individuals)
        n_obj = len(objectives)

        fitness_array = np.zeros((n_ind, n_obj))

        for i, ind in enumerate(individuals):
            for j, obj in enumerate(objectives):
                fitness_array[i, j] = ind.fitness.get(obj, 0.0)

        # Normalize each objective
        for j in range(n_obj):
            min_val = fitness_array[:, j].min()
            max_val = fitness_array[:, j].max()

            if max_val > min_val:
                fitness_array[:, j] = (fitness_array[:, j] - min_val) / (max_val - min_val)
            else:
                fitness_array[:, j] = 0.5

        return fitness_array

    def _associate_to_reference_points(
        self, normalized: np.ndarray
    ) -> list[int]:
        """Associate each individual to nearest reference point.

        Vectorized implementation using NumPy broadcasting.

        Args:
            normalized: Normalized fitness array of shape (n_individuals, n_objectives)

        Returns:
            List of reference point indices (one per individual)
        """
        # Vectorized distance calculation using broadcasting
        # normalized: (n_individuals, n_objectives)
        # reference_points: (n_ref_points, n_objectives)
        # Result: distances of shape (n_individuals, n_ref_points)
        distances = np.linalg.norm(
            normalized[:, None, :] - self.reference_points[None, :, :],
            axis=2
        )

        # Find closest reference point for each individual
        closest_indices = np.argmin(distances, axis=1)

        return closest_indices.tolist()

    def _compute_generation_stats(
        self, population: Population, generation: int
    ) -> dict[str, Any]:
        """Compute statistics for current generation."""
        stats = population.compute_stats()

        return {
            "generation": generation,
            "population_size": stats.size,
            "unique_formulas": stats.unique_formulas,
            "avg_size": stats.avg_size,
            "avg_depth": stats.avg_depth,
            "pareto_front_size": stats.pareto_front_size,
            "fitness": {
                "min": stats.min_fitness,
                "max": stats.max_fitness,
                "avg": stats.avg_fitness,
            },
        }
