"""Selection operators for multi-objective genetic programming.

Provides selection strategies:
- Tournament selection: Compare random individuals
- Lexicase selection: Filter by random objective ordering

Performance optimizations:
- Vectorized Pareto ranking using NumPy (O(n²) vs O(n³))
- Vectorized crowding distance calculation
"""

from __future__ import annotations

import random
from typing import TypeVar, Callable
from dataclasses import dataclass
import numpy as np

from alphaforge.evolution.protocol import Evolvable


@dataclass
class Individual:
    """Individual with fitness values for selection."""

    genome: Evolvable
    fitness: dict[str, float]  # Objective name -> value
    rank: int = 0  # Pareto rank (lower is better)
    crowding_distance: float = 0.0


def tournament_selection(
    population: list[Individual],
    n_select: int,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> list[Individual]:
    """Select individuals using tournament selection.

    For multi-objective: Compare by Pareto rank, then crowding distance.

    Args:
        population: Population to select from
        n_select: Number of individuals to select
        tournament_size: Number of individuals per tournament
        rng: Random number generator

    Returns:
        List of selected individuals
    """
    rng = rng or random.Random()
    selected = []

    for _ in range(n_select):
        # Random tournament
        tournament = rng.sample(population, min(tournament_size, len(population)))

        # Sort by rank (lower is better), then crowding distance (higher is better)
        tournament.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        selected.append(tournament[0])

    return selected


def lexicase_selection(
    population: list[Individual],
    n_select: int,
    rng: random.Random | None = None,
    epsilon: float = 1e-6,
) -> list[Individual]:
    """Select individuals using epsilon-lexicase selection.

    Filters population by random ordering of objectives until one remains.
    Good for maintaining diverse specialists.

    Args:
        population: Population to select from
        n_select: Number of individuals to select
        rng: Random number generator
        epsilon: Tolerance for "equally good" comparisons

    Returns:
        List of selected individuals
    """
    rng = rng or random.Random()
    selected = []

    if not population:
        return selected

    # Get objective names
    objectives = list(population[0].fitness.keys())

    for _ in range(n_select):
        # Start with full population
        candidates = list(population)

        # Random ordering of objectives
        obj_order = rng.sample(objectives, len(objectives))

        for obj in obj_order:
            if len(candidates) <= 1:
                break

            # Find best value for this objective
            # Assuming higher is better for all objectives (negate for minimization)
            values = [ind.fitness.get(obj, float("-inf")) for ind in candidates]
            best_value = max(values)

            # Keep individuals within epsilon of best
            candidates = [
                ind for ind, val in zip(candidates, values)
                if val >= best_value - epsilon
            ]

        # Select random from remaining (usually just one)
        if candidates:
            selected.append(rng.choice(candidates))
        elif population:
            selected.append(rng.choice(population))

    return selected


def binary_tournament(
    ind1: Individual,
    ind2: Individual,
) -> Individual:
    """Binary tournament between two individuals.

    Uses Pareto dominance: if one dominates the other, it wins.
    Otherwise, prefer higher crowding distance.

    Args:
        ind1: First individual
        ind2: Second individual

    Returns:
        Winner of the tournament
    """
    # Compare by rank first
    if ind1.rank < ind2.rank:
        return ind1
    elif ind2.rank < ind1.rank:
        return ind2

    # Same rank, compare crowding distance
    if ind1.crowding_distance > ind2.crowding_distance:
        return ind1
    else:
        return ind2


def dominates(fitness1: dict[str, float], fitness2: dict[str, float]) -> bool:
    """Check if fitness1 dominates fitness2.

    Dominates means: at least as good in all objectives,
    and strictly better in at least one.

    Assumes higher is better for all objectives.

    Args:
        fitness1: First fitness dict
        fitness2: Second fitness dict

    Returns:
        True if fitness1 dominates fitness2
    """
    dominated_in_any = False
    better_in_any = False

    for obj in fitness1.keys():
        v1 = fitness1.get(obj, float("-inf"))
        v2 = fitness2.get(obj, float("-inf"))

        if v1 < v2:
            dominated_in_any = True
        elif v1 > v2:
            better_in_any = True

    return better_in_any and not dominated_in_any


def _build_fitness_matrix(
    population: list[Individual], objectives: list[str]
) -> np.ndarray:
    """Build fitness matrix from population for vectorized operations.

    Args:
        population: List of individuals
        objectives: List of objective names

    Returns:
        Array of shape (n_individuals, n_objectives)
    """
    n = len(population)
    m = len(objectives)
    fitness_matrix = np.zeros((n, m))

    for i, ind in enumerate(population):
        for j, obj in enumerate(objectives):
            fitness_matrix[i, j] = ind.fitness.get(obj, float("-inf"))

    return fitness_matrix


def _vectorized_dominates(fitness_matrix: np.ndarray) -> np.ndarray:
    """Compute dominance matrix using vectorized NumPy operations.

    Args:
        fitness_matrix: Array of shape (n, m) where n=individuals, m=objectives

    Returns:
        Boolean matrix of shape (n, n) where [i,j]=True means i dominates j
    """
    n = len(fitness_matrix)

    # Broadcasting: compare all pairs
    # fitness_matrix[:, None, :] shape: (n, 1, m)
    # fitness_matrix[None, :, :] shape: (1, n, m)

    # i >= j for all objectives
    geq = fitness_matrix[:, None, :] >= fitness_matrix[None, :, :]  # (n, n, m)
    all_geq = np.all(geq, axis=2)  # (n, n)

    # i > j for at least one objective
    gt = fitness_matrix[:, None, :] > fitness_matrix[None, :, :]  # (n, n, m)
    any_gt = np.any(gt, axis=2)  # (n, n)

    # i dominates j if all_geq[i,j] AND any_gt[i,j]
    dominates_matrix = all_geq & any_gt

    return dominates_matrix


def compute_pareto_ranks(population: list[Individual]) -> None:
    """Compute Pareto ranks for all individuals in place.

    Rank 0 = Pareto front (non-dominated)
    Rank 1 = Dominated only by rank 0
    etc.

    Vectorized implementation using NumPy for O(n²) performance.

    Args:
        population: Population to rank (modified in place)
    """
    if not population:
        return

    n = len(population)
    objectives = list(population[0].fitness.keys())

    # Build fitness matrix and compute dominance
    fitness_matrix = _build_fitness_matrix(population, objectives)
    dominates_matrix = _vectorized_dominates(fitness_matrix)

    # Track remaining individuals and ranks
    remaining = np.ones(n, dtype=bool)
    ranks = np.zeros(n, dtype=int)
    current_rank = 0

    while remaining.any():
        # For each remaining individual i, check if any remaining j dominates it
        # dominates_matrix[j, i] = True means j dominates i
        # So we check column i for any True values from remaining individuals
        # Mask the dominance matrix to only consider remaining dominators
        masked_dom = dominates_matrix & remaining[:, None]  # (n, n) - row j only counts if j is remaining
        # For individual i, check if any j dominates it: sum along axis=0
        is_dominated = masked_dom.any(axis=0) & remaining

        # Non-dominated in remaining = remaining AND NOT dominated
        non_dominated = remaining & ~is_dominated

        # Assign rank to non-dominated
        ranks[non_dominated] = current_rank

        # Remove from remaining
        remaining[non_dominated] = False

        current_rank += 1

    # Apply ranks to individuals
    for i, ind in enumerate(population):
        ind.rank = int(ranks[i])


def compute_crowding_distance(population: list[Individual]) -> None:
    """Compute crowding distance for all individuals in place.

    Crowding distance measures how isolated an individual is in objective space.
    Higher distance = more isolated = more valuable for diversity.

    Vectorized implementation using NumPy argsort.

    Args:
        population: Population (modified in place)
    """
    n = len(population)
    if n < 2:
        for ind in population:
            ind.crowding_distance = float("inf")
        return

    # Reset distances
    distances = np.zeros(n)

    # Get objective names and build matrix
    objectives = list(population[0].fitness.keys())
    fitness_matrix = _build_fitness_matrix(population, objectives)

    for j in range(len(objectives)):
        # Sort indices by this objective
        sorted_idx = np.argsort(fitness_matrix[:, j])

        # Boundary points get infinite distance
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        # Range for normalization
        obj_range = fitness_matrix[sorted_idx[-1], j] - fitness_matrix[sorted_idx[0], j]

        if obj_range == 0:
            continue

        # Interior points: distance += (f[i+1] - f[i-1]) / range
        for i in range(1, n - 1):
            idx = sorted_idx[i]
            if distances[idx] < float("inf"):
                idx_prev = sorted_idx[i - 1]
                idx_next = sorted_idx[i + 1]
                distances[idx] += (
                    fitness_matrix[idx_next, j] - fitness_matrix[idx_prev, j]
                ) / obj_range

    # Apply distances to individuals
    for i, ind in enumerate(population):
        ind.crowding_distance = float(distances[i])
