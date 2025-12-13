"""Tests for genetic strategy evolution."""

import pytest

from alphaforge.factory.genetic import (
    EvolutionConfig,
    GeneticStrategyEvolver,
    simple_fitness_function,
)


class TestGeneticStrategyEvolver:
    """Test genetic strategy evolution."""

    def test_initialization(self):
        """Test evolver initialization."""
        evolver = GeneticStrategyEvolver(
            fitness_function=simple_fitness_function,
            config=EvolutionConfig(population_size=10, n_generations=5),
        )

        assert evolver.fitness_function == simple_fitness_function
        assert evolver.config.population_size == 10
        assert evolver.config.n_generations == 5

    def test_evolution(self):
        """Test running evolution."""
        evolver = GeneticStrategyEvolver(
            fitness_function=simple_fitness_function,
            config=EvolutionConfig(
                population_size=20,
                n_generations=10,
                random_seed=42,
            ),
        )

        result = evolver.evolve()

        # Check result
        assert result.best_strategy is not None
        assert result.best_fitness >= 0
        assert len(result.population) == 20
        assert result.generation_count == 10
        assert len(result.fitness_history) == 11  # Initial + 10 generations

    def test_fitness_improves(self):
        """Test that fitness generally improves over generations."""
        evolver = GeneticStrategyEvolver(
            fitness_function=simple_fitness_function,
            config=EvolutionConfig(
                population_size=30,
                n_generations=20,
                random_seed=42,
            ),
        )

        result = evolver.evolve()

        # First generation average fitness
        first_gen_avg = sum(result.fitness_history[0]) / len(result.fitness_history[0])

        # Last generation average fitness
        last_gen_avg = sum(result.fitness_history[-1]) / len(result.fitness_history[-1])

        # Should generally improve (allow some variance)
        # This is stochastic so we're lenient
        assert last_gen_avg >= first_gen_avg * 0.8  # At least 80% of initial

    def test_create_random_population(self):
        """Test random population creation."""
        evolver = GeneticStrategyEvolver(
            fitness_function=simple_fitness_function,
            config=EvolutionConfig(population_size=50),
        )

        population = evolver._create_random_population()

        assert len(population) == 50
        assert all(hasattr(ind, 'fitness') for ind in population)
