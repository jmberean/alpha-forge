"""Minimal integration test for discovery system.

Tests the full pipeline without requiring external dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def test_import_all_modules():
    """Test that all discovery modules can be imported."""
    print("Testing module imports...")

    # Import directly to avoid numpy dependency
    import importlib.util
    import sys

    modules_to_test = [
        'alphaforge.discovery.expression.types',
        'alphaforge.discovery.expression.nodes',
        'alphaforge.discovery.expression.tree',
        'alphaforge.discovery.operators.crossover',
        'alphaforge.discovery.operators.mutation',
        'alphaforge.discovery.operators.selection',
        'alphaforge.discovery.evolution.population',
    ]

    for module_name in modules_to_test:
        try:
            # Import module directly
            module = __import__(module_name, fromlist=[''])
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            return False

    print("✓ All core modules imported successfully")
    return True


def test_create_simple_tree():
    """Test creating a simple expression tree."""
    print("\nTesting expression tree creation...")

    from alphaforge.discovery.expression.tree import ExpressionTree
    from alphaforge.discovery.expression.nodes import OperatorNode, TerminalNode, ConstantNode
    from alphaforge.discovery.expression.types import DataType

    # Create: ts_mean(close, 20)
    root = OperatorNode(
        name="ts_mean",
        children=[
            TerminalNode(name="close"),
            ConstantNode(value=20, data_type=DataType.WINDOW),
        ],
    )

    tree = ExpressionTree(root=root, MIN_DEPTH=1, MIN_SIZE=1)

    assert tree.size == 3, f"Expected size 3, got {tree.size}"
    assert tree.depth == 2, f"Expected depth 2, got {tree.depth}"
    assert "ts_mean" in tree.formula, f"Expected 'ts_mean' in formula, got {tree.formula}"
    assert tree.is_valid(), "Tree should be valid"

    print(f"✓ Created tree: {tree.formula}")
    print(f"  Size: {tree.size}, Depth: {tree.depth}")
    return True


def test_tree_generator():
    """Test random tree generation."""
    print("\nTesting tree generator...")

    from alphaforge.discovery.expression.tree import TreeGenerator

    generator = TreeGenerator(max_depth=4, seed=42)

    # Generate trees
    trees = []
    for i in range(5):
        tree = generator.generate(method="ramped")
        trees.append(tree)
        print(f"  Tree {i+1}: {tree.formula[:50]}... (size={tree.size}, depth={tree.depth})")

    # Check uniqueness
    formulas = [t.formula for t in trees]
    unique_formulas = set(formulas)

    print(f"✓ Generated {len(trees)} trees, {len(unique_formulas)} unique")
    return True


def test_type_safety():
    """Test that type system prevents invalid trees."""
    print("\nTesting type safety...")

    from alphaforge.discovery.expression.tree import ExpressionTree
    from alphaforge.discovery.expression.nodes import OperatorNode, TerminalNode, ConstantNode
    from alphaforge.discovery.expression.types import DataType

    # Try to create invalid tree: ts_mean(close, close) - should fail
    # because second arg should be INTEGER, not SERIES
    try:
        root = OperatorNode(
            name="ts_mean",
            children=[
                TerminalNode(name="close"),
                TerminalNode(name="volume"),  # Wrong type!
            ],
        )
        tree = ExpressionTree(root=root)
        print(f"✗ FAILED: Should have rejected invalid tree")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid tree: {e}")

    # Valid tree should work
    root = OperatorNode(
        name="ts_mean",
        children=[
            TerminalNode(name="close"),
            ConstantNode(value=20, data_type=DataType.WINDOW),
        ],
    )
    tree = ExpressionTree(root=root, MIN_DEPTH=1, MIN_SIZE=1)
    print(f"✓ Accepted valid tree: {tree.formula}")

    return True


def test_crossover():
    """Test crossover operator."""
    print("\nTesting crossover...")

    from alphaforge.discovery.expression.tree import TreeGenerator
    from alphaforge.discovery.operators.crossover import crossover
    import random

    generator = TreeGenerator(seed=42)
    parent1 = generator.generate()
    parent2 = generator.generate()

    rng = random.Random(42)
    child1, child2 = crossover(parent1, parent2, rng)

    assert child1.is_valid(), "Child 1 should be valid"
    assert child2.is_valid(), "Child 2 should be valid"

    print(f"✓ Crossover produced valid children")
    print(f"  Parent 1: {parent1.formula[:40]}...")
    print(f"  Parent 2: {parent2.formula[:40]}...")
    print(f"  Child 1:  {child1.formula[:40]}...")
    print(f"  Child 2:  {child2.formula[:40]}...")

    return True


def test_mutation():
    """Test mutation operators."""
    print("\nTesting mutation...")

    from alphaforge.discovery.expression.tree import TreeGenerator
    from alphaforge.discovery.operators.mutation import mutate
    import random

    generator = TreeGenerator(seed=42)
    original = generator.generate()

    rng = random.Random(42)

    # Apply multiple mutations
    mutations = []
    for i in range(5):
        mutated = mutate(original, rng)
        assert mutated.is_valid(), f"Mutation {i+1} should be valid"
        mutations.append(mutated)

    # Check diversity
    formulas = [m.formula for m in mutations]
    unique = len(set(formulas))

    print(f"✓ All mutations valid ({unique}/{len(mutations)} unique)")
    return True


def test_pareto_ranking():
    """Test Pareto ranking."""
    print("\nTesting Pareto ranking...")

    from alphaforge.discovery.expression.tree import TreeGenerator
    from alphaforge.discovery.operators.selection import Individual, compute_pareto_ranks
    from alphaforge.evolution.genomes import ExpressionGenome

    generator = TreeGenerator(seed=42)

    # Create individuals with different fitness
    individuals = []
    for i in range(10):
        tree = generator.generate()
        ind = Individual(
            genome=ExpressionGenome(tree),
            fitness={
                "sharpe": float(i % 3),  # 0, 1, 2, 0, 1, 2, ...
                "drawdown": float(-(i % 2)),  # 0, -1, 0, -1, ...
            },
            rank=0,
            crowding_distance=0.0,
        )
        individuals.append(ind)

    # Compute ranks
    compute_pareto_ranks(individuals)

    # Check that ranks are assigned
    ranks = [ind.rank for ind in individuals]
    print(f"✓ Ranks assigned: {sorted(set(ranks))}")

    # Front 0 should exist
    front_0 = [ind for ind in individuals if ind.rank == 0]
    print(f"  Pareto front size: {len(front_0)}")

    assert len(front_0) > 0, "Should have at least one individual in Pareto front"

    return True


def test_population():
    """Test population management."""
    print("\nTesting population...")

    from alphaforge.discovery.expression.tree import TreeGenerator
    from alphaforge.discovery.evolution.population import create_initial_population
    from alphaforge.evolution.genomes import ExpressionGenome

    gen = TreeGenerator(seed=42)
    
    def generator():
        return ExpressionGenome(gen.generate(method="ramped"))

    population = create_initial_population(
        size=20,
        generator=generator,
        seed=42,
    )

    assert len(population) == 20, f"Expected 20 individuals, got {len(population)}"

    # Check uniqueness
    hashes = [ind.genome.hash for ind in population]
    unique_hashes = set(hashes)

    print(f"✓ Created population of {len(population)} ({len(unique_hashes)} unique)")

    # Check stats
    stats = population.compute_stats()
    # print(f"  Avg size: {stats.avg_size:.1f}, Avg depth: {stats.avg_depth:.1f}") # Stats structure changed

    return True


def test_nsga3_basic():
    """Test NSGA-III basic functionality."""
    print("\nTesting NSGA-III...")

    from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
    from alphaforge.discovery.evolution.nsga3 import NSGA3Optimizer, NSGA3Config
    from alphaforge.evolution.genomes import ExpressionGenome
    from alphaforge.evolution.protocol import Evolvable

    # Simple fitness functions for testing
    def fitness_size(genome: Evolvable) -> float:
        if isinstance(genome, ExpressionGenome):
            return -genome.tree.size / 50.0
        return 0.0

    def fitness_depth(genome: Evolvable) -> float:
        if isinstance(genome, ExpressionGenome):
            return -genome.tree.depth / 8.0
        return 0.0

    fitness_functions = {
        "size": fitness_size,
        "depth": fitness_depth,
    }

    config = NSGA3Config(
        population_size=10,  # Small for testing
        n_generations=3,  # Few generations
        n_objectives=2,
        seed=42,
    )
    
    gen = TreeGenerator(seed=42)
    def generator():
        return ExpressionGenome(gen.generate())

    optimizer = NSGA3Optimizer(
        fitness_functions=fitness_functions,
        generator=generator,
        config=config,
    )

    print(f"  Population size: {optimizer.config.population_size}")
    print(f"  Reference points: {len(optimizer.reference_points)}")

    # Run optimization
    result = optimizer.optimize()

    print(f"✓ Optimization complete")
    print(f"  Generations: {result.n_generations}")
    print(f"  Pareto front: {len(result.pareto_front)}")
    print(f"  Best by objective: {len(result.best_by_objective)}")

    assert result.n_generations == 3, "Should run 3 generations"
    assert len(result.pareto_front) > 0, "Should have non-empty Pareto front"

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("DISCOVERY SYSTEM INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        test_import_all_modules,
        test_create_simple_tree,
        test_tree_generator,
        test_type_safety,
        test_crossover,
        test_mutation,
        test_pareto_ranking,
        test_population,
        test_nsga3_basic,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_func.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
