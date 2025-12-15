"""Mutation operators for expression tree genetic programming.

Provides various mutation strategies:
- Subtree mutation: Replace a random subtree with a new random tree
- Constant mutation: Perturb numeric constants
- Operator mutation: Swap operator with type-compatible alternative
- Terminal mutation: Swap terminal with different data source
"""

import random
from typing import Any

from alphaforge.discovery.expression.tree import ExpressionTree, TreeGenerator
from alphaforge.discovery.expression.nodes import (
    Node,
    OperatorNode,
    TerminalNode,
    ConstantNode,
    collect_nodes,
)
from alphaforge.discovery.expression.types import (
    DataType,
    OPERATOR_SIGNATURES,
    get_operators_returning,
)


def mutate_subtree(
    tree: ExpressionTree,
    rng: random.Random | None = None,
    max_depth: int = 4,
) -> ExpressionTree:
    """Replace a random subtree with a new random subtree.

    This is the most disruptive mutation, introducing new structure.

    Args:
        tree: Tree to mutate
        rng: Random number generator
        max_depth: Maximum depth of new subtree

    Returns:
        New mutated tree
    """
    rng = rng or random.Random()
    nodes = tree.get_nodes()

    if len(nodes) <= 1:
        return tree.clone()

    # Select random node (excluding root for safety)
    idx = rng.randint(1, len(nodes) - 1)
    target_node = nodes[idx]

    # Generate replacement with same return type
    generator = TreeGenerator(max_depth=max_depth, seed=rng.randint(0, 2**31))
    new_subtree = generator._generate_node(
        target_type=target_node.return_type,
        current_depth=0,
        max_depth=max_depth,
        method="grow",
    )

    try:
        return tree.replace_subtree(idx, new_subtree)
    except (ValueError, IndexError):
        return tree.clone()


def mutate_constant(
    tree: ExpressionTree,
    rng: random.Random | None = None,
    scale: float = 0.2,
) -> ExpressionTree:
    """Perturb a random constant node.

    For integers (window sizes): ±20% or ±1 minimum
    For scalars: Gaussian noise with given scale

    Args:
        tree: Tree to mutate
        rng: Random number generator
        scale: Mutation scale (fraction of value)

    Returns:
        New mutated tree
    """
    rng = rng or random.Random()

    # Find constant nodes
    nodes = tree.get_nodes()
    constant_indices = [
        i for i, n in enumerate(nodes)
        if isinstance(n, ConstantNode)
    ]

    if not constant_indices:
        return tree.clone()

    # Select random constant
    idx = rng.choice(constant_indices)
    node = nodes[idx]

    if not isinstance(node, ConstantNode):
        return tree.clone()

    # Mutate value
    if node.data_type == DataType.WINDOW:
        # Mutate window size
        current = int(node.value)
        # Small perturbation (+/- 1 to 5)
        delta = rng.randint(-5, 5)
        new_value = max(2, min(252, current + delta))
        new_node = ConstantNode(value=new_value, data_type=DataType.WINDOW)
    else:
        delta = abs(node.value) * scale if node.value != 0 else scale
        new_value = node.value + rng.gauss(0, delta)
        new_value = max(ConstantNode.SCALAR_RANGE[0],
                       min(ConstantNode.SCALAR_RANGE[1], new_value))
        new_node = ConstantNode(value=new_value, data_type=DataType.SCALAR)

    try:
        return tree.replace_subtree(idx, new_node)
    except (ValueError, IndexError):
        return tree.clone()


def mutate_operator(
    tree: ExpressionTree,
    rng: random.Random | None = None,
) -> ExpressionTree:
    """Swap an operator with a type-compatible alternative.

    Preserves tree structure, only changes operator function.

    Args:
        tree: Tree to mutate
        rng: Random number generator

    Returns:
        New mutated tree
    """
    rng = rng or random.Random()

    # Find operator nodes
    nodes = tree.get_nodes()
    operator_indices = [
        i for i, n in enumerate(nodes)
        if isinstance(n, OperatorNode)
    ]

    if not operator_indices:
        return tree.clone()

    # Select random operator
    idx = rng.choice(operator_indices)
    node = nodes[idx]

    if not isinstance(node, OperatorNode):
        return tree.clone()

    # Find compatible operators (same arity and compatible types)
    current_sig = OPERATOR_SIGNATURES[node.name]
    compatible = [
        name for name, sig in OPERATOR_SIGNATURES.items()
        if (sig.arity == current_sig.arity
            and sig.return_type == current_sig.return_type
            and sig.input_types == current_sig.input_types
            and name != node.name)
    ]

    if not compatible:
        return tree.clone()

    # Create new operator with same children
    new_name = rng.choice(compatible)
    new_node = OperatorNode(name=new_name, children=[c.clone() for c in node.children])

    try:
        return tree.replace_subtree(idx, new_node)
    except (ValueError, IndexError):
        return tree.clone()


def mutate_terminal(
    tree: ExpressionTree,
    rng: random.Random | None = None,
) -> ExpressionTree:
    """Swap a terminal with a different data source.

    Args:
        tree: Tree to mutate
        rng: Random number generator

    Returns:
        New mutated tree
    """
    rng = rng or random.Random()

    # Find terminal nodes
    nodes = tree.get_nodes()
    terminal_indices = [
        i for i, n in enumerate(nodes)
        if isinstance(n, TerminalNode)
    ]

    if not terminal_indices:
        return tree.clone()

    # Select random terminal
    idx = rng.choice(terminal_indices)
    node = nodes[idx]

    if not isinstance(node, TerminalNode):
        return tree.clone()

    # Select different terminal
    alternatives = [t for t in TerminalNode.VALID_TERMINALS if t != node.name]
    if not alternatives:
        return tree.clone()

    new_name = rng.choice(list(alternatives))
    new_node = TerminalNode(name=new_name)

    try:
        return tree.replace_subtree(idx, new_node)
    except (ValueError, IndexError):
        return tree.clone()


def mutate(
    tree: ExpressionTree,
    rng: random.Random | None = None,
    subtree_prob: float = 0.3,
    constant_prob: float = 0.3,
    operator_prob: float = 0.2,
    terminal_prob: float = 0.2,
) -> ExpressionTree:
    """Apply a random mutation to the tree.

    Args:
        tree: Tree to mutate
        rng: Random number generator
        subtree_prob: Probability of subtree mutation
        constant_prob: Probability of constant mutation
        operator_prob: Probability of operator mutation
        terminal_prob: Probability of terminal mutation

    Returns:
        New mutated tree
    """
    rng = rng or random.Random()

    # Normalize probabilities
    total = subtree_prob + constant_prob + operator_prob + terminal_prob
    r = rng.random() * total

    if r < subtree_prob:
        return mutate_subtree(tree, rng)
    elif r < subtree_prob + constant_prob:
        return mutate_constant(tree, rng)
    elif r < subtree_prob + constant_prob + operator_prob:
        return mutate_operator(tree, rng)
    else:
        return mutate_terminal(tree, rng)
