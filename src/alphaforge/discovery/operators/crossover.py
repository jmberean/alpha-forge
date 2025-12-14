"""Crossover operators for expression tree genetic programming.

Provides crossover strategies:
- Subtree crossover: Swap random subtrees between parents
- Uniform crossover: Mix operators/terminals at each position
"""

import random
from typing import Tuple

from alphaforge.discovery.expression.tree import ExpressionTree
from alphaforge.discovery.expression.nodes import (
    Node,
    OperatorNode,
    TerminalNode,
    ConstantNode,
    collect_nodes,
)


def crossover_subtree(
    parent1: ExpressionTree,
    parent2: ExpressionTree,
    rng: random.Random | None = None,
) -> Tuple[ExpressionTree, ExpressionTree]:
    """Swap random subtrees between two parents.

    Selects crossover points with matching return types to maintain validity.

    Args:
        parent1: First parent tree
        parent2: Second parent tree
        rng: Random number generator

    Returns:
        Tuple of two offspring trees
    """
    rng = rng or random.Random()

    nodes1 = parent1.get_nodes()
    nodes2 = parent2.get_nodes()

    # Find compatible crossover points (same return type)
    compatible_pairs: list[Tuple[int, int]] = []

    for i, n1 in enumerate(nodes1):
        for j, n2 in enumerate(nodes2):
            if n1.return_type == n2.return_type:
                compatible_pairs.append((i, j))

    if not compatible_pairs:
        # No compatible points, return clones
        return parent1.clone(), parent2.clone()

    # Select random compatible pair
    idx1, idx2 = rng.choice(compatible_pairs)

    # Get subtrees
    subtree1 = nodes1[idx1].clone()
    subtree2 = nodes2[idx2].clone()

    # Create offspring by swapping
    try:
        offspring1 = parent1.replace_subtree(idx1, subtree2)
        offspring2 = parent2.replace_subtree(idx2, subtree1)

        # Validate offspring
        if offspring1.is_valid() and offspring2.is_valid():
            return offspring1, offspring2
        elif offspring1.is_valid():
            return offspring1, parent2.clone()
        elif offspring2.is_valid():
            return parent1.clone(), offspring2
        else:
            return parent1.clone(), parent2.clone()

    except (ValueError, IndexError):
        return parent1.clone(), parent2.clone()


def crossover_uniform(
    parent1: ExpressionTree,
    parent2: ExpressionTree,
    rng: random.Random | None = None,
    swap_prob: float = 0.5,
) -> Tuple[ExpressionTree, ExpressionTree]:
    """Mix elements from both parents at each compatible position.

    At each position where both trees have nodes of same type,
    randomly choose which parent contributes.

    This is a less disruptive crossover than subtree crossover.

    Args:
        parent1: First parent tree
        parent2: Second parent tree
        rng: Random number generator
        swap_prob: Probability of swapping at each position

    Returns:
        Tuple of two offspring trees
    """
    rng = rng or random.Random()

    # For simplicity, use subtree crossover with lower probability
    # of selecting deep nodes (favoring shallow crossover)

    nodes1 = parent1.get_nodes()
    nodes2 = parent2.get_nodes()

    # Weight positions by inverse depth (prefer shallow crossovers)
    def get_depth(tree: ExpressionTree, idx: int) -> int:
        """Get depth of node at index."""
        # Approximate by position in list (pre-order traversal)
        return min(idx, tree.depth)

    # Find compatible positions with depth weighting
    compatible_shallow: list[Tuple[int, int]] = []
    compatible_deep: list[Tuple[int, int]] = []

    for i, n1 in enumerate(nodes1):
        for j, n2 in enumerate(nodes2):
            if n1.return_type == n2.return_type:
                depth1 = get_depth(parent1, i)
                depth2 = get_depth(parent2, j)
                avg_depth = (depth1 + depth2) / 2

                if avg_depth < 2:
                    compatible_shallow.append((i, j))
                else:
                    compatible_deep.append((i, j))

    # Prefer shallow crossovers
    if compatible_shallow and rng.random() < 0.7:
        compatible_pairs = compatible_shallow
    elif compatible_deep:
        compatible_pairs = compatible_deep
    elif compatible_shallow:
        compatible_pairs = compatible_shallow
    else:
        return parent1.clone(), parent2.clone()

    # Multiple crossover points with probability
    offspring1 = parent1.clone()
    offspring2 = parent2.clone()

    # Try a few swaps
    n_swaps = rng.randint(1, min(3, len(compatible_pairs)))
    pairs_to_swap = rng.sample(compatible_pairs, n_swaps)

    for idx1, idx2 in pairs_to_swap:
        if rng.random() < swap_prob:
            try:
                nodes1_current = offspring1.get_nodes()
                nodes2_current = offspring2.get_nodes()

                if idx1 < len(nodes1_current) and idx2 < len(nodes2_current):
                    subtree1 = nodes1_current[idx1].clone()
                    subtree2 = nodes2_current[idx2].clone()

                    offspring1 = offspring1.replace_subtree(idx1, subtree2)
                    offspring2 = offspring2.replace_subtree(idx2, subtree1)
            except (ValueError, IndexError):
                continue

    # Validate and return
    try:
        if offspring1.is_valid() and offspring2.is_valid():
            return offspring1, offspring2
    except:
        pass

    return parent1.clone(), parent2.clone()


def crossover(
    parent1: ExpressionTree,
    parent2: ExpressionTree,
    rng: random.Random | None = None,
    method: str = "subtree",
) -> Tuple[ExpressionTree, ExpressionTree]:
    """Apply crossover using specified method.

    Args:
        parent1: First parent tree
        parent2: Second parent tree
        rng: Random number generator
        method: Crossover method ("subtree" or "uniform")

    Returns:
        Tuple of two offspring trees
    """
    rng = rng or random.Random()

    if method == "subtree":
        return crossover_subtree(parent1, parent2, rng)
    elif method == "uniform":
        return crossover_uniform(parent1, parent2, rng)
    else:
        raise ValueError(f"Unknown crossover method: {method}")
