"""Discovery Orchestrator - Main interface for strategy discovery system.

Coordinates:
1. Expression tree evolution (NSGA-III)
2. Fitness evaluation (backtesting)
3. Factor zoo management (validated formulas)
4. Ensemble construction (multi-strategy portfolios)

Usage:
    orchestrator = DiscoveryOrchestrator(market_data)
    result = orchestrator.discover(n_generations=100)
    strategies = result.pareto_front
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import numpy as np

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.expression.compiler import compile_tree
from alphaforge.discovery.evolution.nsga3 import (
    NSGA3Optimizer,
    NSGA3Config,
    NSGA3Result,
)
from alphaforge.discovery.operators.selection import Individual
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.metrics import PerformanceMetrics
from alphaforge.strategy.genome import StrategyGenome
from alphaforge.evolution.genomes import ExpressionGenome
from alphaforge.evolution.protocol import Evolvable


# Get optimal worker count for this machine
N_WORKERS = min(os.cpu_count() or 4, 12)  # Cap at 12 to avoid overhead


@dataclass
class DiscoveryConfig:
    """Configuration for strategy discovery.

    Attributes:
        population_size: Population size for evolution
        n_generations: Number of generations to evolve
        n_objectives: Number of objectives (default 4: Sharpe, MaxDD, Turnover, Complexity)
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        diversity_injection: Inject diversity every N generations
        min_sharpe: Minimum Sharpe ratio threshold
        max_turnover: Maximum turnover threshold (trades per day)
        max_complexity: Maximum complexity score threshold
        validation_split: Fraction of data for validation (0.3 = 30% validation)
        min_trades: Minimum number of trades required (default 10)
        min_volatility: Minimum annualized volatility required (default 0.05)
        seed: Random seed
    """

    population_size: int = 200
    n_generations: int = 100
    n_objectives: int = 4
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    diversity_injection: int = 20
    min_sharpe: float = 0.5
    max_turnover: float = 0.2
    max_complexity: float = 0.7
    validation_split: float = 0.3
    min_trades: int = 10
    min_volatility: float = 0.05
    seed: int | None = None


@dataclass
class DiscoveryResult:
    """Result from strategy discovery.

    Attributes:
        pareto_front: Strategies on Pareto front
        best_by_objective: Best strategy for each objective
        factor_zoo: All validated formulas
        ensemble_weights: Weights for ensemble portfolio
        generation_stats: Statistics per generation
        n_generations: Number of generations completed
    """

    pareto_front: list[Individual]
    best_by_objective: dict[str, Individual]
    factor_zoo: list[ExpressionTree]
    ensemble_weights: dict[str, float]
    generation_stats: list[dict[str, Any]]
    n_generations: int


class DiscoveryOrchestrator:
    """Orchestrates strategy discovery using multi-objective GP.

    This is the main entry point for the discovery system.
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: DiscoveryConfig | None = None,
    ):
        """Initialize discovery orchestrator.

        Args:
            market_data: OHLCV data with DatetimeIndex
            config: Discovery configuration
        """
        self.market_data = market_data
        self.config = config or DiscoveryConfig()

        # Split data for training/validation
        self._split_data()

        # Initialize components
        self.backtest_engine = BacktestEngine()
        self.factor_zoo: list[ExpressionTree] = []
        
        # Tree generator
        self.tree_generator = TreeGenerator(seed=self.config.seed)

        # Cache for compiled expressions
        self._compiled_cache: dict[str, Any] = {}

        # Cache for fitness evaluations (key optimization - avoids redundant backtests)
        self._fitness_cache: dict[str, dict[str, float]] = {}

    def _split_data(self) -> None:
        """Split data into training and validation sets."""
        split_idx = int(len(self.market_data) * (1 - self.config.validation_split))

        self.train_data = self.market_data.iloc[:split_idx]
        self.validation_data = self.market_data.iloc[split_idx:]

    def discover(
        self,
        warm_start_formulas: list[ExpressionTree] | None = None,
        on_generation: Callable[[int, dict], None] | None = None,
    ) -> DiscoveryResult:
        """Run strategy discovery process.

        Args:
            warm_start_formulas: Optional formulas to seed population
            on_generation: Optional callback called after each generation
                           with (generation_number, stats_dict)

        Returns:
            DiscoveryResult with Pareto front and validated strategies
        """
        # Create fitness functions
        fitness_functions = self._create_fitness_functions()

        # Configure NSGA-III
        nsga3_config = NSGA3Config(
            population_size=self.config.population_size,
            n_generations=self.config.n_generations,
            n_objectives=self.config.n_objectives,
            crossover_prob=self.config.crossover_prob,
            mutation_prob=self.config.mutation_prob,
            diversity_injection=self.config.diversity_injection,
            seed=self.config.seed,
        )
        
        # Generator for new individuals
        def generator() -> Evolvable:
            return ExpressionGenome(self.tree_generator.generate(method="ramped"))

        # Run evolution
        optimizer = NSGA3Optimizer(
            fitness_functions=fitness_functions,
            generator=generator,
            config=nsga3_config,
        )
        
        # Prepare initial population
        initial_population = []
        if warm_start_formulas:
            initial_population = [ExpressionGenome(tree) for tree in warm_start_formulas]

        nsga3_result = optimizer.optimize(
            initial_population=initial_population,
            on_generation=on_generation,
        )

        # Validate strategies on held-out data
        validated_pareto = self._validate_strategies(nsga3_result.pareto_front)

        # Build factor zoo from validated strategies
        self._build_factor_zoo(validated_pareto)

        # Create ensemble
        ensemble_weights = self._create_ensemble(validated_pareto)

        # Return ALL Pareto strategies (not just validated) so user sees results
        # Validated ones have "validation_sharpe" in their fitness dict
        return DiscoveryResult(
            pareto_front=nsga3_result.pareto_front,  # All Pareto strategies
            best_by_objective=nsga3_result.best_by_objective,
            factor_zoo=self.factor_zoo,
            ensemble_weights=ensemble_weights,
            generation_stats=nsga3_result.generation_stats,
            n_generations=nsga3_result.n_generations,
        )

    def _create_fitness_functions(self) -> dict[str, Callable[[Evolvable], float]]:
        """Create fitness functions for each objective.

        OPTIMIZED: Uses a single combined evaluation function that runs ONE backtest
        and extracts all metrics, instead of running 3-4 separate backtests per individual.

        Returns:
            Dict of objective_name -> fitness_function
            All functions return higher = better
        """
        # Use combined fitness that caches results
        return {
            "sharpe": lambda genome: self._get_cached_fitness(genome).get("sharpe", -999.0),
            "drawdown": lambda genome: self._get_cached_fitness(genome).get("drawdown", -999.0),
            "turnover": lambda genome: self._get_cached_fitness(genome).get("turnover", -999.0),
            "complexity": lambda genome: self._get_cached_fitness(genome).get("complexity", -999.0),
        }

    def _get_cached_fitness(self, genome: Evolvable) -> dict[str, float]:
        """Get all fitness values for a tree, using cache to avoid redundant backtests.

        This is the key optimization - ONE backtest extracts ALL metrics.
        """
        # Only support ExpressionGenome for now in Discovery
        if not isinstance(genome, ExpressionGenome):
            return {
                "sharpe": -999.0,
                "drawdown": -999.0,
                "turnover": -999.0,
                "complexity": -999.0,
            }
            
        tree = genome.tree
        cache_key = tree.hash

        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        # Compute all fitness values in ONE backtest
        fitness = self._compute_all_fitness(tree)
        self._fitness_cache[cache_key] = fitness
        return fitness

    def _compute_all_fitness(self, tree: ExpressionTree) -> dict[str, float]:
        """Compute all fitness objectives in a single backtest pass.

        Returns dict with: sharpe, drawdown, turnover, complexity
        All values are oriented so higher = better.
        """
        try:
            # Compile and evaluate expression ONCE
            signal = self._evaluate_tree(tree, self.train_data)

            # Check for valid signal
            if signal.isna().all():
                raise ValueError("Signal is all NaN")

            # Generate positions (-1, 0, 1)
            positions = self._signals_to_positions(signal)

            # Simple vectorized backtest
            prices = self.train_data["close"]
            price_returns = prices.pct_change().fillna(0)

            # Strategy returns = position * price returns (with lag for signal)
            lagged_positions = positions.shift(1).fillna(0)
            strategy_returns = lagged_positions * price_returns

            # Apply simple transaction costs (0.1% per trade)
            position_changes = positions.diff().fillna(0).abs()
            transaction_costs = position_changes * 0.001
            strategy_returns = strategy_returns - transaction_costs

            # Extract metrics
            metrics = PerformanceMetrics.from_returns(strategy_returns)

            # Check activity constraints (Fix #4.1: Zero-Trade Loophole)
            if (metrics.num_trades < self.config.min_trades or 
                metrics.volatility < self.config.min_volatility):
                return {
                    "sharpe": -999.0,
                    "drawdown": -999.0,
                    "turnover": -999.0,
                    "complexity": -999.0,
                }

            # Calculate turnover
            avg_turnover = position_changes.mean()

            return {
                "sharpe": metrics.sharpe_ratio,
                "drawdown": -metrics.max_drawdown,  # Negated (higher = better)
                "turnover": -avg_turnover * 10,     # Negated (higher = better)
                "complexity": -tree.complexity_score(),  # Negated (higher = better)
            }

        except Exception:
            return {
                "sharpe": -999.0,
                "drawdown": -999.0,
                "turnover": -999.0,
                "complexity": -999.0,
            }

    def _evaluate_tree(self, tree: ExpressionTree, data: pd.DataFrame) -> pd.Series:
        """Evaluate expression tree on data.

        Uses compilation cache for efficiency.

        Args:
            tree: Expression tree to evaluate
            data: Market data

        Returns:
            Signal series (numeric values)
        """
        tree_hash = tree.hash

        if tree_hash not in self._compiled_cache:
            self._compiled_cache[tree_hash] = compile_tree(tree)

        compiled = self._compiled_cache[tree_hash]
        return compiled(data)

    def _signals_to_positions(self, signal: pd.Series) -> pd.Series:
        """Convert continuous signal to discrete positions.

        Uses expanding window normalization to prevent lookahead bias (Fix #4.2).

        Args:
            signal: Continuous signal values

        Returns:
            Positions series (-1, 0, 1)
        """
        # Drop NaN for normalization calculation
        # valid_signal = signal.dropna() # Cannot dropna, indexes must align
        
        # Use expanding window z-score
        # Requires at least 20 periods to establish a baseline
        expanding_mean = signal.expanding(min_periods=20).mean()
        expanding_std = signal.expanding(min_periods=20).std()
        
        # Avoid division by zero
        expanding_std = expanding_std.replace(0, np.nan)
        
        signal_norm = (signal - expanding_mean) / expanding_std

        # Threshold at ±0.5 std
        positions = pd.Series(0, index=signal.index, dtype=float)
        positions[signal_norm > 0.5] = 1.0
        positions[signal_norm < -0.5] = -1.0

        # Fill NaN positions (start of series) with 0
        positions = positions.fillna(0)

        return positions

    def _validate_strategies(
        self, strategies: list[Individual]
    ) -> list[Individual]:
        """Validate strategies on held-out validation data.

        Filters out strategies that don't meet minimum thresholds.

        Args:
            strategies: Strategies to validate

        Returns:
            List of validated strategies
        """
        validated = []

        for ind in strategies:
            # Need to unwrap ExpressionGenome
            if not isinstance(ind.genome, ExpressionGenome):
                continue
            tree = ind.genome.tree
            
            try:
                # Evaluate on validation data
                signal = self._evaluate_tree(tree, self.validation_data)

                # Skip if all NaN
                if signal.isna().all():
                    continue

                positions = self._signals_to_positions(signal)

                # Simple vectorized backtest on validation data
                prices = self.validation_data["close"]
                price_returns = prices.pct_change().fillna(0)
                lagged_positions = positions.shift(1).fillna(0)
                strategy_returns = lagged_positions * price_returns

                # Apply transaction costs
                position_changes = positions.diff().fillna(0).abs()
                transaction_costs = position_changes * 0.001
                strategy_returns = strategy_returns - transaction_costs

                metrics = PerformanceMetrics.from_returns(strategy_returns)

                # Check thresholds
                if (metrics.sharpe_ratio >= self.config.min_sharpe and
                    metrics.max_drawdown <= abs(1.0 / self.config.min_sharpe) and
                    tree.complexity_score() <= self.config.max_complexity):

                    # Update fitness with validation metrics
                    ind.fitness["validation_sharpe"] = metrics.sharpe_ratio
                    validated.append(ind)

            except Exception:
                continue

        return validated

    def _build_factor_zoo(self, validated_strategies: list[Individual]) -> None:
        """Build factor zoo from validated strategies.

        Factor zoo contains unique, high-quality formulas.

        Args:
            validated_strategies: Validated strategies to add to zoo
        """
        # Add formulas meeting quality criteria
        for ind in validated_strategies:
            if not isinstance(ind.genome, ExpressionGenome):
                continue
            tree = ind.genome.tree
            
            sharpe = ind.fitness.get("sharpe", 0)
            complexity = tree.complexity_score()

            # Quality criteria: Sharpe > 1.0 and complexity < 0.5
            if sharpe > 1.0 and complexity < 0.5:
                # Check uniqueness
                if not any(tree.hash == f.hash for f in self.factor_zoo):
                    self.factor_zoo.append(tree.clone())

        # Sort by Sharpe ratio (descending)
        # Note: _get_cached_fitness needs ExpressionGenome
        # But factor_zoo stores ExpressionTree
        # We need to wrap it temporarily or update sort key
        
        # Simplified sort key that re-computes or trusts metadata if available
        # But we don't have metadata on tree objects easily accessible here
        # Let's rebuild wrappers
        zoo_genomes = [ExpressionGenome(t) for t in self.factor_zoo]
        
        zoo_genomes.sort(
            key=lambda g: self._get_cached_fitness(g).get("sharpe", -999.0),
            reverse=True,
        )

        # Keep top 100
        self.factor_zoo = [g.tree for g in zoo_genomes[:100]]

    def _create_ensemble(
        self, strategies: list[Individual]
    ) -> dict[str, float]:
        """Create ensemble portfolio from strategies.

        Uses equal weighting for strategies on Pareto front.

        Args:
            strategies: Strategies to combine

        Returns:
            Dict of tree_hash -> weight
        """
        if not strategies:
            return {}

        # Equal weight ensemble
        weight = 1.0 / len(strategies)

        ensemble_weights = {
            ind.genome.hash: weight
            for ind in strategies
        }

        return ensemble_weights

    def to_strategy_genomes(
        self, individuals: list[Individual]
    ) -> list[StrategyGenome]:
        """Convert individuals to StrategyGenome format.

        This allows discovered strategies to use the existing
        validation pipeline.

        Args:
            individuals: Individuals to convert

        Returns:
            List of StrategyGenome objects
        """
        genomes = []

        for i, ind in enumerate(individuals):
            # Create genome with expression tree as signal
            if not isinstance(ind.genome, ExpressionGenome):
                continue
            
            # Delegate to ExpressionGenome.to_strategy_genome
            genome = ind.genome.to_strategy_genome()
            # Update metadata with fitness
            genome.metadata["fitness"] = ind.fitness
            
            genomes.append(genome)

        return genomes
