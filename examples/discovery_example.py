"""Example: Strategy Discovery using Multi-Objective Genetic Programming.

This example demonstrates the advanced strategy discovery system featuring:
- Expression tree genetic programming
- Multi-objective optimization (NSGA-III)
- Automatic validation on held-out data
- Factor zoo construction
- Ensemble portfolio creation

Based on research from AAAI 2025 AlphaForge paper.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Core discovery components
from alphaforge.discovery import (
    DiscoveryOrchestrator,
    DiscoveryConfig,
    ExpressionTree,
)
from alphaforge.discovery.expression.tree import TreeGenerator
from alphaforge.discovery.expression.compiler import compile_tree

# Data loading
from alphaforge.data.loader import MarketDataLoader


def main():
    """Run strategy discovery example."""
    print("=" * 80)
    print("AlphaForge Strategy Discovery System")
    print("=" * 80)

    # =========================================================================
    # Step 1: Load Market Data
    # =========================================================================
    print("\n[Step 1] Loading market data...")

    loader = MarketDataLoader()
    data = loader.load(
        symbol="SPY",
        start="2020-01-01",
        end="2023-12-31",
    )

    print(f"Loaded {len(data)} days of data for SPY")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # =========================================================================
    # Step 2: Configure Discovery System
    # =========================================================================
    print("\n[Step 2] Configuring discovery system...")

    config = DiscoveryConfig(
        population_size=100,  # Smaller for quick example
        n_generations=20,  # Fewer generations for speed
        n_objectives=4,  # Sharpe, MaxDD, Turnover, Complexity
        crossover_prob=0.9,
        mutation_prob=0.3,
        diversity_injection=10,  # Inject diversity every 10 generations
        min_sharpe=0.5,  # Minimum acceptable Sharpe ratio
        max_turnover=0.2,  # Maximum 20% turnover per day
        max_complexity=0.7,  # Prefer simpler strategies
        validation_split=0.3,  # 30% for validation
        seed=42,
    )

    print(f"Population size: {config.population_size}")
    print(f"Generations: {config.n_generations}")
    print(f"Objectives: {config.n_objectives}")
    print(f"Validation split: {config.validation_split:.0%}")

    # =========================================================================
    # Step 3: (Optional) Create Warm Start Population
    # =========================================================================
    print("\n[Step 3] Creating warm start formulas...")

    # Generate some seed formulas based on known patterns
    generator = TreeGenerator(max_depth=4, seed=42)

    warm_start_formulas = []
    for _ in range(10):
        tree = generator.generate(method="ramped")
        warm_start_formulas.append(tree)
        print(f"  - {tree.formula[:60]}...")

    # =========================================================================
    # Step 4: Run Discovery
    # =========================================================================
    print("\n[Step 4] Running strategy discovery...")
    print("This will take a few minutes...\n")

    orchestrator = DiscoveryOrchestrator(data, config)

    result = orchestrator.discover(
        warm_start_formulas=warm_start_formulas
    )

    print(f"\n✓ Discovery complete!")
    print(f"  - Generations: {result.n_generations}")
    print(f"  - Pareto front size: {len(result.pareto_front)}")
    print(f"  - Factor zoo size: {len(result.factor_zoo)}")

    # =========================================================================
    # Step 5: Analyze Results
    # =========================================================================
    print("\n[Step 5] Analyzing discovered strategies...")

    # Show Pareto front
    print(f"\n{'='*80}")
    print("PARETO FRONT - Non-Dominated Strategies")
    print(f"{'='*80}")

    for i, ind in enumerate(result.pareto_front[:5], 1):
        print(f"\nStrategy {i}:")
        print(f"  Formula: {ind.tree.formula}")
        print(f"  Size: {ind.tree.size} nodes, Depth: {ind.tree.depth}")
        print(f"  Complexity: {ind.tree.complexity_score():.3f}")
        print(f"  Fitness:")
        for obj_name, value in ind.fitness.items():
            print(f"    {obj_name:12s}: {value:8.4f}")

    # Show best by each objective
    print(f"\n{'='*80}")
    print("BEST BY OBJECTIVE")
    print(f"{'='*80}")

    for obj_name, ind in result.best_by_objective.items():
        print(f"\nBest {obj_name}:")
        print(f"  Formula: {ind.tree.formula[:70]}...")
        print(f"  Value: {ind.fitness[obj_name]:.4f}")

    # Show factor zoo
    if result.factor_zoo:
        print(f"\n{'='*80}")
        print("FACTOR ZOO - Validated High-Quality Formulas")
        print(f"{'='*80}")

        for i, tree in enumerate(result.factor_zoo[:10], 1):
            print(f"\n{i}. {tree.formula[:70]}...")
            print(f"   Size: {tree.size}, Depth: {tree.depth}, "
                  f"Complexity: {tree.complexity_score():.3f}")

    # Show ensemble weights
    print(f"\n{'='*80}")
    print("ENSEMBLE PORTFOLIO")
    print(f"{'='*80}")

    if result.ensemble_weights:
        print(f"\nNumber of strategies in ensemble: {len(result.ensemble_weights)}")
        print(f"Weight per strategy: {1.0/len(result.ensemble_weights):.4f}")
    else:
        print("\nNo ensemble created (no strategies passed validation)")

    # =========================================================================
    # Step 6: Evolution Statistics
    # =========================================================================
    print(f"\n{'='*80}")
    print("EVOLUTION STATISTICS")
    print(f"{'='*80}")

    # Extract metrics over generations
    generations = [s["generation"] for s in result.generation_stats]
    pareto_sizes = [s["pareto_front_size"] for s in result.generation_stats]

    print(f"\nPareto front growth:")
    print(f"  Gen 0:    {pareto_sizes[0]}")
    print(f"  Gen {result.n_generations//2:2d}:   {pareto_sizes[result.n_generations//2]}")
    print(f"  Gen {result.n_generations-1:2d}:   {pareto_sizes[-1]}")

    # Average fitness trends
    if result.generation_stats[0]["fitness"]["avg"]:
        print(f"\nAverage fitness (first objective):")
        first_obj = list(result.generation_stats[0]["fitness"]["avg"].keys())[0]
        avg_fitness = [
            s["fitness"]["avg"][first_obj]
            for s in result.generation_stats
        ]
        print(f"  Gen 0:    {avg_fitness[0]:.4f}")
        print(f"  Gen {result.n_generations//2:2d}:   {avg_fitness[result.n_generations//2]:.4f}")
        print(f"  Gen {result.n_generations-1:2d}:   {avg_fitness[-1]:.4f}")

    # =========================================================================
    # Step 7: Test Expression Evaluation
    # =========================================================================
    print(f"\n{'='*80}")
    print("EXPRESSION EVALUATION EXAMPLE")
    print(f"{'='*80}")

    if result.pareto_front:
        # Pick a strategy and evaluate it
        example_strategy = result.pareto_front[0]
        example_tree = example_strategy.tree

        print(f"\nEvaluating: {example_tree.formula[:70]}...")

        # Compile and evaluate
        compiled = compile_tree(example_tree)
        signal = compiled(data)

        print(f"\nSignal statistics:")
        print(f"  Mean:   {signal.mean():8.4f}")
        print(f"  Std:    {signal.std():8.4f}")
        print(f"  Min:    {signal.min():8.4f}")
        print(f"  Max:    {signal.max():8.4f}")
        print(f"  NaN%:   {signal.isna().mean()*100:6.2f}%")

    # =========================================================================
    # Step 8: Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"""
Discovery completed successfully!

Key Results:
  ✓ Evolved {config.population_size} strategies over {result.n_generations} generations
  ✓ Pareto front: {len(result.pareto_front)} non-dominated strategies
  ✓ Factor zoo: {len(result.factor_zoo)} validated formulas
  ✓ Ensemble: {len(result.ensemble_weights)} strategies

Multi-Objective Optimization:
  - Sharpe ratio (maximize returns per risk)
  - Maximum drawdown (minimize losses)
  - Turnover (minimize trading costs)
  - Complexity (prefer simpler strategies)

Next Steps:
  1. Run full validation pipeline on best strategies
  2. Test on additional symbols/time periods
  3. Deploy ensemble for live trading
  4. Continue evolution with more generations
    """)

    print("=" * 80)


def analyze_single_formula():
    """Example: Analyze a single expression tree formula."""
    print("\n" + "=" * 80)
    print("SINGLE FORMULA ANALYSIS")
    print("=" * 80)

    # Create a simple formula manually
    from alphaforge.discovery.expression.nodes import (
        OperatorNode,
        TerminalNode,
        ConstantNode,
    )

    # Formula: rank(ts_mean(close, 20))
    formula = ExpressionTree(
        root=OperatorNode(
            name="rank",
            children=[
                OperatorNode(
                    name="ts_mean",
                    children=[
                        TerminalNode(name="close"),
                        ConstantNode(value=20),
                    ],
                ),
            ],
        )
    )

    print(f"\nFormula: {formula.formula}")
    print(f"Size: {formula.size} nodes")
    print(f"Depth: {formula.depth}")
    print(f"Complexity: {formula.complexity_score():.3f}")
    print(f"Hash: {formula.hash}")

    # Load data and evaluate
    loader = MarketDataLoader()
    data = loader.load("SPY", start="2023-01-01", end="2023-12-31")

    compiled = compile_tree(formula)
    signal = compiled(data)

    print(f"\nSignal (last 5 days):")
    print(signal.tail())


if __name__ == "__main__":
    # Run main discovery example
    main()

    # Optionally analyze a single formula
    # analyze_single_formula()
