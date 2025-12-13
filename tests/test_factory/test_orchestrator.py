"""Tests for strategy factory orchestrator."""

import pytest

from alphaforge.factory.genetic import simple_fitness_function
from alphaforge.factory.orchestrator import (
    FactoryConfig,
    StrategyFactory,
    StrategyPool,
    generate_strategy_candidates,
)


class TestStrategyPool:
    """Test strategy pool."""

    def test_add_strategy(self):
        """Test adding strategies to pool."""
        from alphaforge.strategy.templates import StrategyTemplates

        pool = StrategyPool()

        strategy = StrategyTemplates.sma_crossover()
        pool.add_strategy(strategy, fitness=1.5)

        assert len(pool.strategies) == 1
        assert strategy.id in pool.fitness_scores

    def test_get_best(self):
        """Test getting best strategies."""
        from alphaforge.strategy.templates import StrategyTemplates

        pool = StrategyPool()

        for i in range(10):
            strategy = StrategyTemplates.sma_crossover(fast_period=10 + i, slow_period=50 + i)
            pool.add_strategy(strategy, fitness=float(i))

        best = pool.get_best(n=3)

        assert len(best) == 3
        # Should be sorted by fitness (descending)
        assert pool.fitness_scores[best[0].id] >= pool.fitness_scores[best[1].id]

    def test_filter_by_fitness(self):
        """Test filtering by minimum fitness."""
        from alphaforge.strategy.templates import StrategyTemplates

        pool = StrategyPool()

        for i in range(10):
            strategy = StrategyTemplates.sma_crossover(fast_period=10 + i, slow_period=50 + i)
            pool.add_strategy(strategy, fitness=float(i))

        filtered = pool.filter_by_fitness(min_fitness=5.0)

        assert len(filtered) == 5  # 5, 6, 7, 8, 9
        assert all(pool.fitness_scores[s.id] >= 5.0 for s in filtered)


class TestStrategyFactory:
    """Test strategy factory."""

    def test_initialization(self):
        """Test factory initialization."""
        factory = StrategyFactory(
            fitness_function=simple_fitness_function,
            config=FactoryConfig(target_strategies=50),
        )

        assert factory.fitness_function == simple_fitness_function
        assert factory.config.target_strategies == 50

    def test_generate(self):
        """Test strategy generation."""
        factory = StrategyFactory(
            fitness_function=simple_fitness_function,
            config=FactoryConfig(
                target_strategies=30,
                use_genetic=True,
                genetic_population=20,
                genetic_generations=5,
            ),
        )

        pool = factory.generate()

        # Should have generated strategies
        assert len(pool.strategies) > 0
        # Should have at least some from genetic + templates
        assert len(pool.strategies) >= 20

    def test_get_top_strategies(self):
        """Test getting top strategies."""
        factory = StrategyFactory(
            fitness_function=simple_fitness_function,
            config=FactoryConfig(
                target_strategies=30,
                genetic_population=20,
                genetic_generations=3,
            ),
        )

        factory.generate()
        top = factory.get_top_strategies(n=5)

        assert len(top) <= 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)


def test_generate_strategy_candidates():
    """Test convenience function."""
    candidates = generate_strategy_candidates(
        fitness_function=simple_fitness_function,
        n_candidates=25,
    )

    assert len(candidates) > 0
    assert len(candidates) >= 20  # Should generate at least close to target
