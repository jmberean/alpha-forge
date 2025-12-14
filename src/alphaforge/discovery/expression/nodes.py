"""Expression tree nodes for genetic programming.

Implements AST nodes for factor formula representation:
- OperatorNode: Functions with children (e.g., ts_mean, add)
- TerminalNode: Data references (e.g., close, volume)
- ConstantNode: Literal values (e.g., 20, 0.5)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import copy
import uuid

from alphaforge.discovery.expression.types import (
    DataType,
    NodeType,
    OPERATOR_SIGNATURES,
    OperatorSignature,
)


@dataclass
class Node(ABC):
    """Abstract base class for expression tree nodes."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Get the type of this node."""
        pass

    @property
    @abstractmethod
    def return_type(self) -> DataType:
        """Get the return type of this node."""
        pass

    @property
    @abstractmethod
    def arity(self) -> int:
        """Get the number of children this node expects."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert node to string representation."""
        pass

    @abstractmethod
    def clone(self) -> "Node":
        """Create a deep copy of this node."""
        pass

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass
class OperatorNode(Node):
    """Operator node with children.

    Represents functions like ts_mean(close, 20), add(x, y), etc.
    """

    name: str = ""
    children: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name and self.name not in OPERATOR_SIGNATURES:
            raise ValueError(f"Unknown operator: {self.name}")

    @property
    def node_type(self) -> NodeType:
        return NodeType.OPERATOR

    @property
    def signature(self) -> OperatorSignature:
        """Get the operator signature."""
        return OPERATOR_SIGNATURES[self.name]

    @property
    def return_type(self) -> DataType:
        return self.signature.return_type

    @property
    def arity(self) -> int:
        return self.signature.arity

    @property
    def input_types(self) -> tuple[DataType, ...]:
        """Get expected input types."""
        return self.signature.input_types

    def is_complete(self) -> bool:
        """Check if all children are filled."""
        return len(self.children) == self.arity

    def validate_types(self) -> bool:
        """Validate that children types match signature."""
        if not self.is_complete():
            return False
        child_types = tuple(c.return_type for c in self.children)
        return self.signature.accepts(child_types)

    def to_string(self) -> str:
        if not self.children:
            return f"{self.name}()"
        child_strs = [c.to_string() for c in self.children]
        return f"{self.name}({', '.join(child_strs)})"

    def clone(self) -> "OperatorNode":
        return OperatorNode(
            node_id=str(uuid.uuid4())[:8],
            name=self.name,
            children=[c.clone() for c in self.children],
        )


@dataclass
class TerminalNode(Node):
    """Terminal node representing data reference.

    Represents price/volume data like close, open, high, low, volume, returns.
    """

    name: str = ""
    data_type: DataType = DataType.SERIES

    # Standard terminal names
    VALID_TERMINALS: frozenset[str] = frozenset({
        "open", "high", "low", "close", "volume",
        "returns", "vwap", "adj_close",
    })

    def __post_init__(self) -> None:
        if self.name and self.name not in self.VALID_TERMINALS:
            raise ValueError(f"Unknown terminal: {self.name}. Valid: {self.VALID_TERMINALS}")

    @property
    def node_type(self) -> NodeType:
        return NodeType.TERMINAL

    @property
    def return_type(self) -> DataType:
        return self.data_type

    @property
    def arity(self) -> int:
        return 0

    def to_string(self) -> str:
        return self.name

    def clone(self) -> "TerminalNode":
        return TerminalNode(
            node_id=str(uuid.uuid4())[:8],
            name=self.name,
            data_type=self.data_type,
        )


@dataclass
class ConstantNode(Node):
    """Constant node representing a literal value.

    Used for window sizes (integers) or numeric constants (scalars).
    """

    value: float | int = 0
    data_type: DataType = DataType.INTEGER

    # Valid ranges for constants
    INTEGER_RANGE: tuple[int, int] = (2, 252)  # Window sizes: 2 days to 1 year
    SCALAR_RANGE: tuple[float, float] = (-10.0, 10.0)

    def __post_init__(self) -> None:
        if self.data_type == DataType.INTEGER:
            self.value = int(self.value)

    @property
    def node_type(self) -> NodeType:
        return NodeType.CONSTANT

    @property
    def return_type(self) -> DataType:
        return self.data_type

    @property
    def arity(self) -> int:
        return 0

    def to_string(self) -> str:
        if self.data_type == DataType.INTEGER:
            return str(int(self.value))
        return f"{self.value:.4f}"

    def clone(self) -> "ConstantNode":
        return ConstantNode(
            node_id=str(uuid.uuid4())[:8],
            value=self.value,
            data_type=self.data_type,
        )

    @classmethod
    def random_integer(cls, rng: Any = None) -> "ConstantNode":
        """Create a random integer constant (window size)."""
        import random
        r = rng if rng else random
        # Bias toward common window sizes
        common_windows = [5, 10, 20, 50, 100, 200]
        if r.random() < 0.5:
            value = r.choice(common_windows)
        else:
            value = r.randint(cls.INTEGER_RANGE[0], cls.INTEGER_RANGE[1])
        return cls(value=value, data_type=DataType.INTEGER)

    @classmethod
    def random_scalar(cls, rng: Any = None) -> "ConstantNode":
        """Create a random scalar constant."""
        import random
        r = rng if rng else random
        value = r.uniform(cls.SCALAR_RANGE[0], cls.SCALAR_RANGE[1])
        return cls(value=value, data_type=DataType.SCALAR)


def count_nodes(node: Node) -> int:
    """Count total nodes in a subtree."""
    if isinstance(node, OperatorNode):
        return 1 + sum(count_nodes(c) for c in node.children)
    return 1


def get_depth(node: Node) -> int:
    """Get the depth of a subtree."""
    if isinstance(node, OperatorNode) and node.children:
        return 1 + max(get_depth(c) for c in node.children)
    return 1


def collect_nodes(node: Node) -> list[Node]:
    """Collect all nodes in a subtree (pre-order traversal)."""
    result = [node]
    if isinstance(node, OperatorNode):
        for child in node.children:
            result.extend(collect_nodes(child))
    return result


def collect_nodes_by_type(node: Node, target_type: DataType) -> list[Node]:
    """Collect all nodes returning the specified type."""
    return [n for n in collect_nodes(node) if n.return_type == target_type]
