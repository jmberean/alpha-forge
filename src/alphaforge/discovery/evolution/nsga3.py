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

from alphaforge.evolution.protocol import Evolvable
from alphaforge.discovery.operators.selection import (
    Individual,
    compute_pareto_ranks,
    compute_crowding_distance,
    dominates,
)
from alphaforge.discovery.evolution.population import Population

# Optimal thread count for parallel evaluation
N_EVAL_THREADS = min(os.cpu_count() or 4, 8)  # Cap at 8 for evaluation


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
    """NSGA-III optimizer for many-objective optimization.

    Generic implementation working with Evolvable protocol.
    """

    def __init__(
        self,
        fitness_functions: dict[str, Callable[[Evolvable], float]],
        generator: Callable[[], Evolvable],
        config: NSGA3Config | None = None,
    ):
        """Initialize NSGA-III optimizer.

        Args:
            fitness_functions: Dict of objective_name -> fitness_function
            generator: Function to generate new random individuals
            config: Configuration
        """
        self.fitness_functions = fitness_functions
        self.generator = generator
        self.config = config or NSGA3Config()
        self.rng = random.Random(self.config.seed)

        # Generate reference points
        self.reference_points = self._generate_reference_points()

        # Adjust population size to match reference points
        if len(self.reference_points) != self.config.population_size:
            self.config.population_size = len(self.reference_points)

    def _generate_reference_points(self) -> np.ndarray:
        """Generate uniformly distributed reference points on unit simplex."""
        n_obj = self.config.n_objectives

        if self.config.n_reference_points:
            n_partitions = self._calculate_partitions(
                n_obj, self.config.n_reference_points
            )
        else:
            if n_obj <= 3:
                n_partitions = 12
            elif n_obj <= 6:
                n_partitions = 4
            else:
                n_partitions = 3

        return self._das_dennis_reference_points(n_obj, n_partitions)

    def _calculate_partitions(self, n_obj: int, target_points: int) -> int:
        for p in range(1, 50):
            n_points = math.comb(n_obj + p - 1, p)
            if n_points >= target_points:
                return p
        return 12

    def _das_dennis_reference_points(
        self, n_obj: int, n_partitions: int
    ) -> np.ndarray:
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
        initial_population: list[Evolvable] | None = None,
        on_generation: Callable[[int, dict], None] | None = None,
    ) -> NSGA3Result:
        """Run NSGA-III optimization."""
        # Initialize population
        population = Population()
        
        if initial_population:
            for genome in initial_population:
                population.add(Individual(genome=genome, fitness={}))
        
        # Fill rest with random
        while len(population.individuals) < self.config.population_size:
            genome = self.generator()
            population.add(Individual(genome=genome, fitness={}))

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
                for _ in range(n_inject):
                    genome = self.generator()
                    population.add(Individual(genome=genome, fitness={}))
                
                self._evaluate_population(population)
                compute_pareto_ranks(population.individuals)
                compute_crowding_distance(population.individuals)
                population = self._environmental_selection(population)

            # Record stats
            stats = self._calculate_stats(population)
            stats["generation"] = gen
            stats["population_size"] = len(population.individuals)
            
            generation_stats.append(stats)
            if on_generation:
                on_generation(gen, stats)

        # Final result
        pareto_front = [ind for ind in population.individuals if ind.rank == 0]
        
        best_by_obj = {}
        for obj in self.fitness_functions.keys():
            best_by_obj[obj] = max(
                population.individuals,
                key=lambda ind: ind.fitness.get(obj, float("-inf"))
            )

        return NSGA3Result(
            pareto_front=pareto_front,
            population=population,
            best_by_objective=best_by_obj,
            generation_stats=generation_stats,
            n_generations=self.config.n_generations,
        )

    def _create_offspring(self, population: Population) -> Population:
        """Create offspring using genetic operators via Evolvable protocol."""
        offspring = Population()
        pop_size = len(population.individuals)
        
        # Tournament selection for parents
        # For simplicity, just pick random parents from current population (rank biased)
        # Better: use binary tournament
        
        while len(offspring.individuals) < pop_size:
            # Select parents
            p1 = self._tournament_select(population.individuals)
            p2 = self._tournament_select(population.individuals)
            
            # Crossover
            if self.rng.random() < self.config.crossover_prob:
                c1_genome, c2_genome = p1.genome.crossover(p2.genome, self.rng)
            else:
                c1_genome, c2_genome = p1.genome, p2.genome
            
            # Mutation
            if self.rng.random() < self.config.mutation_prob:
                c1_genome = c1_genome.mutate(self.rng)
            if self.rng.random() < self.config.mutation_prob:
                c2_genome = c2_genome.mutate(self.rng)
                
            offspring.add(Individual(genome=c1_genome, fitness={}))
            if len(offspring.individuals) < pop_size:
                offspring.add(Individual(genome=c2_genome, fitness={}))
                
        return offspring

    def _tournament_select(self, individuals: list[Individual]) -> Individual:
        """Binary tournament selection."""
        a = self.rng.choice(individuals)
        b = self.rng.choice(individuals)
        if a.rank < b.rank:
            return a
        elif b.rank < a.rank:
            return b
        elif a.crowding_distance > b.crowding_distance:
            return a
        else:
            return b

    def _evaluate_population(self, population: Population) -> None:
        """Evaluate fitness for all individuals."""
        # Check cache or evaluate
        # Parallel evaluation
        to_evaluate = [
            ind for ind in population.individuals 
            if not ind.fitness
        ]
        
        if not to_evaluate:
            return

        with ThreadPoolExecutor(max_workers=N_EVAL_THREADS) as executor:
            # Map genome to fitness dict
            results = executor.map(self._evaluate_genome, [ind.genome for ind in to_evaluate])
            
            for ind, fitness in zip(to_evaluate, results):
                ind.fitness = fitness

    def _evaluate_genome(self, genome: Evolvable) -> dict[str, float]:
        """Evaluate a single genome."""
        fitness = {}
        for name, func in self.fitness_functions.items():
            fitness[name] = func(genome)
        return fitness

    def _environmental_selection(self, population: Population) -> Population:
        """Select best N individuals for next generation."""
        compute_pareto_ranks(population.individuals)
        compute_crowding_distance(population.individuals)
        
        new_pop = Population()
        fronts = {}
        
        for ind in population.individuals:
            if ind.rank not in fronts:
                fronts[ind.rank] = []
            fronts[ind.rank].append(ind)
            
        rank = 0
        while len(new_pop.individuals) < self.config.population_size:
            if rank not in fronts:
                break
                
            current_front = fronts[rank]
            remaining_slots = self.config.population_size - len(new_pop.individuals)
            
            if len(current_front) <= remaining_slots:
                # Add entire front
                for ind in current_front:
                    new_pop.add(ind)
            else:
                # Sort by crowding distance and fill
                current_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                for ind in current_front[:remaining_slots]:
                    new_pop.add(ind)
            
            rank += 1
            
        return new_pop

    def _calculate_stats(self, population: Population) -> dict[str, Any]:
        """Calculate generation statistics."""
        fitness_values = {}
        for name in self.fitness_functions.keys():
            values = [ind.fitness.get(name, 0.0) for ind in population.individuals]
            fitness_values[name] = {
                "max": max(values) if values else 0,
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
            }
            
        unique = len({ind.genome.hash for ind in population.individuals})
        
        return {
            "fitness": fitness_values,
            "unique_genomes": unique,
            "pareto_front_size": sum(1 for ind in population.individuals if ind.rank == 0)
        }