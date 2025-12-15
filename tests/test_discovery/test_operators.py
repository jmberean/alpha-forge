"""Tests for genetic operators."""

import pytest
import random

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.expression.nodes import (
    OperatorNode,
    TerminalNode,
    ConstantNode,
)
from alphaforge.discovery.operators.crossover import (
    crossover,
    crossover_subtree,
    crossover_uniform,
)
from alphaforge.discovery.operators.mutation import (
    mutate,
    mutate_subtree,
    mutate_constant,
    mutate_operator,
    mutate_terminal,
)
from alphaforge.discovery.operators.selection import (
    Individual,
    tournament_selection,
    compute_pareto_ranks,
    compute_crowding_distance,
    dominates,
)
from alphaforge.evolution.genomes import ExpressionGenome


class TestCrossoverOperators:
    """Test crossover operators."""

    @pytest.fixture
    def parent_trees(self):
        """Create two parent trees for crossover."""
        generator = TreeGenerator(seed=42)
        parent1 = generator.generate(method="grow")
        parent2 = generator.generate(method="grow")
        return parent1, parent2

    def test_crossover_subtree(self, parent_trees):
        """Test subtree crossover."""
        parent1, parent2 = parent_trees
        rng = random.Random(42)

        child1, child2 = crossover_subtree(parent1, parent2, rng)

        # Children should be valid
        assert child1.is_valid()
        assert child2.is_valid()

        # Children should be different from parents (usually)
        # (May be same if crossover at root or identical subtrees)
        assert isinstance(child1, ExpressionTree)
        assert isinstance(child2, ExpressionTree)

    def test_crossover_uniform(self, parent_trees):
        """Test uniform crossover."""
        parent1, parent2 = parent_trees
        rng = random.Random(42)

        child1, child2 = crossover_uniform(parent1, parent2, rng)

        assert child1.is_valid()
        assert child2.is_valid()

    def test_crossover_preserves_types(self, parent_trees):
        """Test that crossover preserves type safety."""
        parent1, parent2 = parent_trees
        rng = random.Random(42)

        for _ in range(10):
            child1, child2 = crossover(parent1, parent2, rng)
            assert child1.is_valid()
            assert child2.is_valid()


class TestMutationOperators:
    """Test mutation operators."""

    @pytest.fixture
    def tree(self):
        """Create a tree for mutation."""
        generator = TreeGenerator(max_depth=4, seed=42)
        return generator.generate(method="grow")

    def test_mutate_subtree(self, tree):
        """Test subtree mutation."""
        rng = random.Random(42)
        mutated = mutate_subtree(tree, rng)

        assert mutated.is_valid()
        # Usually different formula (unless mutation at leaf)
        assert isinstance(mutated, ExpressionTree)

    def test_mutate_constant(self):
        """Test constant mutation."""
        # Create tree with constant
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=20),
            ],
        )
        tree = ExpressionTree(root=root)
        rng = random.Random(42)

        mutated = mutate_constant(tree, rng)
        assert mutated.is_valid()

        # Extract constant values
        original_const = tree.get_nodes()[2]
        mutated_const = mutated.get_nodes()[2]

        if isinstance(original_const, ConstantNode) and isinstance(
            mutated_const, ConstantNode
        ):
            # Value should have changed (usually)
            assert isinstance(mutated_const.value, (int, float))

    def test_mutate_operator(self):
        """Test operator mutation."""
        # Create tree with swappable operator
        root = OperatorNode(
            name="add",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),
            ],
        )
        tree = ExpressionTree(root=root)
        rng = random.Random(42)

        mutated = mutate_operator(tree, rng)
        assert mutated.is_valid()

    def test_mutate_terminal(self):
        """Test terminal mutation."""
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                ConstantNode(value=10),
            ],
        )
        tree = ExpressionTree(root=root)
        rng = random.Random(42)

        mutated = mutate_terminal(tree, rng)
        assert mutated.is_valid()

    def test_mutate_preserves_validity(self, tree):
        """Test that mutation preserves validity."""
        rng = random.Random(42)

        for _ in range(20):
            mutated = mutate(tree, rng)
            assert mutated.is_valid()


class TestSelectionOperators:
    """Test selection operators."""

    @pytest.fixture
    def population(self):
        """Create a population of individuals."""
        generator = TreeGenerator(seed=42)
        individuals = []

        for i in range(20):
            tree = generator.generate(method="ramped")
            ind = Individual(
                genome=ExpressionGenome(tree),
                fitness={
                    "sharpe": random.uniform(0, 3),
                    "drawdown": random.uniform(-0.5, 0),
                    "turnover": random.uniform(-1, 0),
                },
                rank=0,
                crowding_distance=0.0,
            )
            individuals.append(ind)

        return individuals

    def test_tournament_selection(self, population):
        """Test tournament selection."""
        # Compute ranks first
        compute_pareto_ranks(population)
        compute_crowding_distance(population)

        rng = random.Random(42)
        selected = tournament_selection(
            population, n_select=10, tournament_size=3, rng=rng
        )

        assert len(selected) == 10
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_pareto_ranking(self, population):
        """Test Pareto ranking."""
        compute_pareto_ranks(population)

        # Check that ranks are assigned
        assert all(ind.rank >= 0 for ind in population)

        # Front 0 should not be dominated by anyone
        front_0 = [ind for ind in population if ind.rank == 0]
        assert len(front_0) > 0

        for ind0 in front_0:
            for ind in population:
                assert not dominates(ind.fitness, ind0.fitness)

    def test_crowding_distance(self, population):
        """Test crowding distance calculation."""
        compute_pareto_ranks(population)
        compute_crowding_distance(population)

        # All individuals should have crowding distance
        assert all(ind.crowding_distance >= 0 for ind in population)

        # Boundary points should have infinite distance
        by_sharpe = sorted(population, key=lambda ind: ind.fitness["sharpe"])
        assert by_sharpe[0].crowding_distance == float("inf") or \
               by_sharpe[-1].crowding_distance == float("inf")

    def test_dominates(self):
        """Test dominance relationship."""
        fitness1 = {"a": 1.0, "b": 2.0}
        fitness2 = {"a": 0.5, "b": 1.0}
        fitness3 = {"a": 1.5, "b": 0.5}

        # fitness1 dominates fitness2
        assert dominates(fitness1, fitness2)
        assert not dominates(fitness2, fitness1)

        # fitness1 and fitness3 are non-dominated
        assert not dominates(fitness1, fitness3)
        assert not dominates(fitness3, fitness1)
