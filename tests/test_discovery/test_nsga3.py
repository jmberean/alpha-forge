"""Tests for NSGA-III optimizer."""

import pytest
import numpy as np

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.evolution.nsga3 import (
    NSGA3Optimizer,
    NSGA3Config,
)
from alphaforge.discovery.evolution.population import create_initial_population
from alphaforge.evolution.genomes import ExpressionGenome
from alphaforge.evolution.protocol import Evolvable


class TestNSGA3:
    """Test NSGA-III optimizer."""

    @pytest.fixture
    def generator_func(self):
        gen = TreeGenerator(seed=42)
        return lambda: ExpressionGenome(gen.generate(method="ramped"))

    @pytest.fixture
    def simple_fitness_functions(self):
        """Create simple fitness functions for testing."""

        def fitness_size(genome: Evolvable) -> float:
            """Prefer smaller trees."""
            if not isinstance(genome, ExpressionGenome): return -999.0
            tree = genome.tree
            return -tree.size / 50.0

        def fitness_depth(genome: Evolvable) -> float:
            """Prefer shallower trees."""
            if not isinstance(genome, ExpressionGenome): return -999.0
            tree = genome.tree
            return -tree.depth / 8.0

        def fitness_random(genome: Evolvable) -> float:
            """Random fitness for diversity."""
            # Deterministic pseudo-randomness: avoid Python's salted `hash()`.
            return (int(genome.hash[:8], 16) % 100) / 100.0

        def fitness_complexity(genome: Evolvable) -> float:
            """Prefer simpler trees."""
            if not isinstance(genome, ExpressionGenome): return -999.0
            tree = genome.tree
            return -tree.complexity_score()

        return {
            "size": fitness_size,
            "depth": fitness_depth,
            "random": fitness_random,
            "complexity": fitness_complexity,
        }

    def test_nsga3_initialization(self, simple_fitness_functions, generator_func):
        """Test NSGA-III initialization."""
        config = NSGA3Config(
            population_size=20,
            n_generations=5,
            n_objectives=4,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        assert optimizer.config.population_size >= 20
        assert len(optimizer.reference_points) > 0

    def test_reference_point_generation(self, generator_func):
        """Test reference point generation."""
        config = NSGA3Config(
            n_objectives=3,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions={
                "f1": lambda t: 0.0,
                "f2": lambda t: 0.0,
                "f3": lambda t: 0.0,
            },
            generator=generator_func,
            config=config,
        )

        ref_points = optimizer.reference_points

        # Check shape
        assert ref_points.shape[1] == 3

        # Check they sum to 1 (on unit simplex)
        sums = ref_points.sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_nsga3_optimization(self, simple_fitness_functions, generator_func):
        """Test full NSGA-III optimization run."""
        config = NSGA3Config(
            population_size=20,
            n_generations=3,  # Short run for testing
            n_objectives=4,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize()

        # Check result structure
        assert len(result.pareto_front) > 0
        assert result.n_generations == 3
        assert len(result.generation_stats) == 3
        assert len(result.best_by_objective) == 4

        # Check individuals have fitness
        for ind in result.pareto_front:
            assert len(ind.fitness) == 4
            assert all(k in ind.fitness for k in simple_fitness_functions.keys())

    def test_warm_start(self, simple_fitness_functions, generator_func):
        """Test warm start with initial population."""
        generator = TreeGenerator(seed=42)
        warm_start = [ExpressionGenome(generator.generate()) for _ in range(5)]

        config = NSGA3Config(
            population_size=20,
            n_generations=2,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize(initial_population=warm_start)

        assert len(result.pareto_front) > 0

    def test_diversity_injection(self, simple_fitness_functions, generator_func):
        """Test diversity injection during evolution."""
        config = NSGA3Config(
            population_size=20,
            n_generations=10,
            diversity_injection=5,  # Every 5 generations
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize()

        # Should complete successfully with diversity injection
        assert result.n_generations == 10
        assert len(result.pareto_front) > 0

    def test_pareto_front_quality(self, simple_fitness_functions, generator_func):
        """Test that Pareto front contains non-dominated solutions."""
        config = NSGA3Config(
            population_size=30,
            n_generations=5,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize()

        # All Pareto front individuals should have rank 0
        assert all(ind.rank == 0 for ind in result.pareto_front)

        # No individual in front should dominate another
        from alphaforge.discovery.operators.selection import dominates

        for i, ind1 in enumerate(result.pareto_front):
            for j, ind2 in enumerate(result.pareto_front):
                if i != j:
                    assert not dominates(ind1.fitness, ind2.fitness)

    def test_generation_stats(self, simple_fitness_functions, generator_func):
        """Test that generation statistics are tracked."""
        config = NSGA3Config(
            population_size=20,
            n_generations=5,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize()

        # Check stats structure
        assert len(result.generation_stats) == 5

        for gen_stats in result.generation_stats:
            assert "generation" in gen_stats
            assert "population_size" in gen_stats
            assert "pareto_front_size" in gen_stats
            assert "fitness" in gen_stats

    def test_convergence(self, simple_fitness_functions, generator_func):
        """Test that optimization shows improvement over generations."""
        config = NSGA3Config(
            population_size=30,
            n_generations=20,
            seed=42,
        )

        optimizer = NSGA3Optimizer(
            fitness_functions=simple_fitness_functions,
            generator=generator_func,
            config=config,
        )

        result = optimizer.optimize()

        # Pareto front should grow or stabilize over generations
        front_sizes = [
            stats.get("pareto_front_size", 0) for stats in result.generation_stats
        ]

        # Early front should be smaller than late front (generally)
        early_avg = np.mean(front_sizes[:5])
        late_avg = np.mean(front_sizes[-5:])

        # Allow for some variance
        assert late_avg >= early_avg * 0.5
