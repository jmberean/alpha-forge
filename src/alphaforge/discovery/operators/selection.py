"""Selection operators for multi-objective genetic programming.

Provides selection strategies:
- Tournament selection: Compare random individuals
- Lexicase selection: Filter by random objective ordering
"""

import random
from typing import TypeVar, Callable
from dataclasses import dataclass

from alphaforge.discovery.expression.tree import ExpressionTree


@dataclass
class Individual:
    """Individual with fitness values for selection."""

    tree: ExpressionTree
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


def compute_pareto_ranks(population: list[Individual]) -> None:
    """Compute Pareto ranks for all individuals in place.

    Rank 0 = Pareto front (non-dominated)
    Rank 1 = Dominated only by rank 0
    etc.

    Args:
        population: Population to rank (modified in place)
    """
    remaining = set(range(len(population)))
    current_rank = 0

    while remaining:
        # Find non-dominated in remaining
        non_dominated = []

        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(population[j].fitness, population[i].fitness):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)

        # Assign rank
        for i in non_dominated:
            population[i].rank = current_rank
            remaining.remove(i)

        current_rank += 1


def compute_crowding_distance(population: list[Individual]) -> None:
    """Compute crowding distance for all individuals in place.

    Crowding distance measures how isolated an individual is in objective space.
    Higher distance = more isolated = more valuable for diversity.

    Args:
        population: Population (modified in place)
    """
    if len(population) < 2:
        for ind in population:
            ind.crowding_distance = float("inf")
        return

    # Reset distances
    for ind in population:
        ind.crowding_distance = 0.0

    # Get objective names
    objectives = list(population[0].fitness.keys())

    for obj in objectives:
        # Sort by this objective
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.get(obj, 0))

        # Boundary points get infinite distance
        sorted_pop[0].crowding_distance = float("inf")
        sorted_pop[-1].crowding_distance = float("inf")

        # Range for normalization
        obj_range = (
            sorted_pop[-1].fitness.get(obj, 0) - sorted_pop[0].fitness.get(obj, 0)
        )

        if obj_range == 0:
            continue

        # Interior points
        for i in range(1, len(sorted_pop) - 1):
            distance = (
                sorted_pop[i + 1].fitness.get(obj, 0)
                - sorted_pop[i - 1].fitness.get(obj, 0)
            ) / obj_range
            sorted_pop[i].crowding_distance += distance
