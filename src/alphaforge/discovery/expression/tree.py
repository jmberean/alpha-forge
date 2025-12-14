"""Expression tree for factor formula representation.

An ExpressionTree represents a complete factor formula like:
    rank(ts_corr(close, volume, 20)) * delay(returns, 5)

Trees are immutable after construction for safe evolution operations.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator
import random
import hashlib

from alphaforge.discovery.expression.types import (
    DataType,
    NodeType,
    OPERATOR_SIGNATURES,
    get_operators_returning,
)
from alphaforge.discovery.expression.nodes import (
    Node,
    OperatorNode,
    TerminalNode,
    ConstantNode,
    count_nodes,
    get_depth,
    collect_nodes,
)


@dataclass
class ExpressionTree:
    """Expression tree representing a factor formula.

    Attributes:
        root: The root node of the tree
        metadata: Optional metadata (e.g., origin, generation)
    """

    root: Node
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tree constraints
    MAX_DEPTH: int = 8
    MIN_DEPTH: int = 2
    MAX_SIZE: int = 50
    MIN_SIZE: int = 3

    def __post_init__(self) -> None:
        """Validate tree structure."""
        if not self.is_valid():
            raise ValueError("Invalid expression tree structure")

    @property
    def size(self) -> int:
        """Get total number of nodes."""
        return count_nodes(self.root)

    @property
    def depth(self) -> int:
        """Get tree depth."""
        return get_depth(self.root)

    @property
    def return_type(self) -> DataType:
        """Get the return type of the tree."""
        return self.root.return_type

    @property
    def formula(self) -> str:
        """Get string representation of the formula."""
        return self.root.to_string()

    @property
    def hash(self) -> str:
        """Get a hash of the formula for deduplication."""
        return hashlib.md5(self.formula.encode()).hexdigest()[:12]

    def is_valid(self) -> bool:
        """Check if tree is structurally valid."""
        # Check depth constraints
        depth = self.depth
        if depth < self.MIN_DEPTH or depth > self.MAX_DEPTH:
            return False

        # Check size constraints
        size = self.size
        if size < self.MIN_SIZE or size > self.MAX_SIZE:
            return False

        # Check type consistency
        return self._validate_types(self.root)

    def _validate_types(self, node: Node) -> bool:
        """Recursively validate type consistency."""
        if isinstance(node, OperatorNode):
            if not node.is_complete():
                return False
            if not node.validate_types():
                return False
            return all(self._validate_types(c) for c in node.children)
        return True

    def clone(self) -> "ExpressionTree":
        """Create a deep copy of the tree."""
        return ExpressionTree(
            root=self.root.clone(),
            metadata=dict(self.metadata),
        )

    def get_nodes(self) -> list[Node]:
        """Get all nodes in the tree."""
        return collect_nodes(self.root)

    def get_node_at_index(self, index: int) -> Node | None:
        """Get node at given index (pre-order traversal)."""
        nodes = self.get_nodes()
        if 0 <= index < len(nodes):
            return nodes[index]
        return None

    def get_subtree_at_index(self, index: int) -> Node | None:
        """Get subtree rooted at given index."""
        return self.get_node_at_index(index)

    def replace_subtree(self, index: int, new_subtree: Node) -> "ExpressionTree":
        """Create new tree with subtree at index replaced.

        Returns a new ExpressionTree (immutable operation).
        """
        if index == 0:
            # Replace root
            return ExpressionTree(root=new_subtree.clone(), metadata=self.metadata)

        # Clone tree and find parent of target
        new_root = self.root.clone()
        nodes = collect_nodes(new_root)

        if index >= len(nodes):
            raise IndexError(f"Index {index} out of range (tree size: {len(nodes)})")

        # Find parent and replace child
        target = nodes[index]
        for node in nodes:
            if isinstance(node, OperatorNode):
                for i, child in enumerate(node.children):
                    if child.node_id == target.node_id:
                        node.children[i] = new_subtree.clone()
                        return ExpressionTree(root=new_root, metadata=self.metadata)

        raise ValueError(f"Could not find parent of node at index {index}")

    def get_terminals(self) -> list[str]:
        """Get list of terminal names used in tree."""
        terminals = []
        for node in self.get_nodes():
            if isinstance(node, TerminalNode):
                terminals.append(node.name)
        return list(set(terminals))

    def get_operators(self) -> list[str]:
        """Get list of operator names used in tree."""
        operators = []
        for node in self.get_nodes():
            if isinstance(node, OperatorNode):
                operators.append(node.name)
        return operators

    def complexity_score(self) -> float:
        """Calculate complexity score for parsimony pressure.

        Higher score = more complex = penalized.
        """
        size_penalty = self.size / self.MAX_SIZE
        depth_penalty = self.depth / self.MAX_DEPTH

        # Count expensive operators
        expensive_ops = {"ts_corr", "ts_cov", "ts_rank"}
        op_count = sum(1 for op in self.get_operators() if op in expensive_ops)
        op_penalty = op_count * 0.1

        return size_penalty + depth_penalty + op_penalty

    def __str__(self) -> str:
        return self.formula

    def __repr__(self) -> str:
        return f"ExpressionTree({self.formula}, size={self.size}, depth={self.depth})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpressionTree):
            return False
        return self.formula == other.formula

    def __hash__(self) -> int:
        return hash(self.formula)


class TreeGenerator:
    """Factory for generating random expression trees.

    Supports multiple generation methods:
    - grow: Random depth (tends toward smaller trees)
    - full: Maximum depth at all branches
    - ramped: 50/50 mix of grow and full
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_depth: int = 2,
        terminals: list[str] | None = None,
        seed: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.terminals = terminals or ["close", "volume", "returns", "high", "low"]
        self.rng = random.Random(seed)

        # Categorize operators by return type
        self.series_ops = get_operators_returning(DataType.SERIES)
        self.boolean_ops = get_operators_returning(DataType.BOOLEAN)

    def generate(
        self,
        method: str = "ramped",
        return_type: DataType = DataType.SERIES,
    ) -> ExpressionTree:
        """Generate a random expression tree.

        Args:
            method: Generation method ("grow", "full", or "ramped")
            return_type: Required return type of tree

        Returns:
            A valid ExpressionTree
        """
        if method == "ramped":
            method = self.rng.choice(["grow", "full"])

        root = self._generate_node(
            target_type=return_type,
            current_depth=0,
            max_depth=self.max_depth,
            method=method,
        )

        try:
            return ExpressionTree(root=root)
        except ValueError:
            # If invalid, try again with simpler tree
            return self.generate(method="grow", return_type=return_type)

    def _generate_node(
        self,
        target_type: DataType,
        current_depth: int,
        max_depth: int,
        method: str,
    ) -> Node:
        """Recursively generate a node of the target type."""
        # Force terminal at max depth or probabilistically for "grow"
        at_max = current_depth >= max_depth
        force_terminal = at_max or (
            method == "grow"
            and current_depth >= self.min_depth
            and self.rng.random() < 0.3
        )

        if force_terminal:
            return self._generate_terminal(target_type)

        # Generate operator
        return self._generate_operator(target_type, current_depth, max_depth, method)

    def _generate_terminal(self, target_type: DataType) -> Node:
        """Generate a terminal node of the target type."""
        if target_type == DataType.SERIES:
            return TerminalNode(name=self.rng.choice(self.terminals))
        elif target_type == DataType.INTEGER:
            return ConstantNode.random_integer(self.rng)
        elif target_type == DataType.SCALAR:
            return ConstantNode.random_scalar(self.rng)
        elif target_type == DataType.BOOLEAN:
            # Generate a comparison for boolean
            return self._generate_operator(
                target_type, self.max_depth - 1, self.max_depth, "grow"
            )
        else:
            raise ValueError(f"Cannot generate terminal of type {target_type}")

    def _generate_operator(
        self,
        target_type: DataType,
        current_depth: int,
        max_depth: int,
        method: str,
    ) -> OperatorNode:
        """Generate an operator node returning the target type."""
        # Select operators that return target type
        if target_type == DataType.SERIES:
            valid_ops = self.series_ops
        elif target_type == DataType.BOOLEAN:
            valid_ops = self.boolean_ops
        else:
            # For other types, use series ops (most common)
            valid_ops = self.series_ops

        if not valid_ops:
            raise ValueError(f"No operators return type {target_type}")

        # Select random operator
        op_name = self.rng.choice(valid_ops)
        sig = OPERATOR_SIGNATURES[op_name]

        # Generate children
        children = []
        for input_type in sig.input_types:
            child = self._generate_node(
                target_type=input_type,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                method=method,
            )
            children.append(child)

        return OperatorNode(name=op_name, children=children)

    def generate_population(
        self,
        size: int,
        return_type: DataType = DataType.SERIES,
    ) -> list[ExpressionTree]:
        """Generate a population of unique trees."""
        population = []
        seen_hashes: set[str] = set()

        attempts = 0
        max_attempts = size * 10

        while len(population) < size and attempts < max_attempts:
            attempts += 1
            try:
                tree = self.generate(method="ramped", return_type=return_type)
                if tree.hash not in seen_hashes:
                    seen_hashes.add(tree.hash)
                    population.append(tree)
            except (ValueError, RecursionError):
                continue

        return population
