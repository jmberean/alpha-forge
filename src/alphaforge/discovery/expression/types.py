"""Type system for strongly-typed genetic programming (STGP).

Ensures type-safe operator combinations during evolution.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import FrozenSet


class DataType(Enum):
    """Data types for expression tree nodes."""

    SCALAR = auto()      # Single numeric value
    SERIES = auto()      # Time series (pd.Series)
    BOOLEAN = auto()     # Boolean value or series
    INTEGER = auto()     # Integer (for window sizes, periods)

    def is_numeric(self) -> bool:
        """Check if type is numeric."""
        return self in (DataType.SCALAR, DataType.SERIES, DataType.INTEGER)

    def is_compatible_with(self, other: "DataType") -> bool:
        """Check type compatibility for operations."""
        if self == other:
            return True
        # Scalar can promote to Series
        if self == DataType.SCALAR and other == DataType.SERIES:
            return True
        if self == DataType.SERIES and other == DataType.SCALAR:
            return True
        # Integer is compatible with numeric types
        if self == DataType.INTEGER and other.is_numeric():
            return True
        if other == DataType.INTEGER and self.is_numeric():
            return True
        return False


class NodeType(Enum):
    """Types of nodes in expression tree."""

    OPERATOR = auto()    # Function with children (arity >= 1)
    TERMINAL = auto()    # Leaf node (data reference)
    CONSTANT = auto()    # Literal value


@dataclass(frozen=True)
class OperatorSignature:
    """Type signature for an operator.

    Defines input types and return type for type checking.
    """

    name: str
    input_types: tuple[DataType, ...]
    return_type: DataType
    arity: int

    def __post_init__(self) -> None:
        if len(self.input_types) != self.arity:
            raise ValueError(
                f"Input types length ({len(self.input_types)}) must match arity ({self.arity})"
            )

    def accepts(self, arg_types: tuple[DataType, ...]) -> bool:
        """Check if given argument types are acceptable."""
        if len(arg_types) != self.arity:
            return False
        return all(
            given.is_compatible_with(expected)
            for given, expected in zip(arg_types, self.input_types)
        )


# Standard operator signatures
OPERATOR_SIGNATURES: dict[str, OperatorSignature] = {
    # Arithmetic (Series -> Series)
    "add": OperatorSignature("add", (DataType.SERIES, DataType.SERIES), DataType.SERIES, 2),
    "sub": OperatorSignature("sub", (DataType.SERIES, DataType.SERIES), DataType.SERIES, 2),
    "mul": OperatorSignature("mul", (DataType.SERIES, DataType.SERIES), DataType.SERIES, 2),
    "div": OperatorSignature("div", (DataType.SERIES, DataType.SERIES), DataType.SERIES, 2),
    "abs": OperatorSignature("abs", (DataType.SERIES,), DataType.SERIES, 1),
    "neg": OperatorSignature("neg", (DataType.SERIES,), DataType.SERIES, 1),
    "log": OperatorSignature("log", (DataType.SERIES,), DataType.SERIES, 1),
    "sqrt": OperatorSignature("sqrt", (DataType.SERIES,), DataType.SERIES, 1),
    "sign": OperatorSignature("sign", (DataType.SERIES,), DataType.SERIES, 1),
    "power": OperatorSignature("power", (DataType.SERIES, DataType.SCALAR), DataType.SERIES, 2),

    # Temporal (Series, Integer -> Series)
    "delay": OperatorSignature("delay", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "delta": OperatorSignature("delta", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_mean": OperatorSignature("ts_mean", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_std": OperatorSignature("ts_std", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_min": OperatorSignature("ts_min", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_max": OperatorSignature("ts_max", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_sum": OperatorSignature("ts_sum", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_rank": OperatorSignature("ts_rank", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_argmax": OperatorSignature("ts_argmax", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_argmin": OperatorSignature("ts_argmin", (DataType.SERIES, DataType.INTEGER), DataType.SERIES, 2),
    "ts_corr": OperatorSignature("ts_corr", (DataType.SERIES, DataType.SERIES, DataType.INTEGER), DataType.SERIES, 3),
    "ts_cov": OperatorSignature("ts_cov", (DataType.SERIES, DataType.SERIES, DataType.INTEGER), DataType.SERIES, 3),

    # Cross-sectional (Series -> Series)
    "rank": OperatorSignature("rank", (DataType.SERIES,), DataType.SERIES, 1),
    "scale": OperatorSignature("scale", (DataType.SERIES,), DataType.SERIES, 1),
    "zscore": OperatorSignature("zscore", (DataType.SERIES,), DataType.SERIES, 1),

    # Comparison (Series, Series -> Boolean)
    "gt": OperatorSignature("gt", (DataType.SERIES, DataType.SERIES), DataType.BOOLEAN, 2),
    "lt": OperatorSignature("lt", (DataType.SERIES, DataType.SERIES), DataType.BOOLEAN, 2),
    "gte": OperatorSignature("gte", (DataType.SERIES, DataType.SERIES), DataType.BOOLEAN, 2),
    "lte": OperatorSignature("lte", (DataType.SERIES, DataType.SERIES), DataType.BOOLEAN, 2),
    "eq": OperatorSignature("eq", (DataType.SERIES, DataType.SERIES), DataType.BOOLEAN, 2),

    # Logical (Boolean -> Boolean)
    "and_": OperatorSignature("and_", (DataType.BOOLEAN, DataType.BOOLEAN), DataType.BOOLEAN, 2),
    "or_": OperatorSignature("or_", (DataType.BOOLEAN, DataType.BOOLEAN), DataType.BOOLEAN, 2),
    "not_": OperatorSignature("not_", (DataType.BOOLEAN,), DataType.BOOLEAN, 1),

    # Conditional
    "if_else": OperatorSignature("if_else", (DataType.BOOLEAN, DataType.SERIES, DataType.SERIES), DataType.SERIES, 3),
}


def get_operators_returning(return_type: DataType) -> list[str]:
    """Get all operators that return the specified type."""
    return [
        name for name, sig in OPERATOR_SIGNATURES.items()
        if sig.return_type == return_type
    ]


def get_operators_accepting(input_type: DataType) -> list[str]:
    """Get all operators that can accept the specified type as first argument."""
    return [
        name for name, sig in OPERATOR_SIGNATURES.items()
        if sig.input_types and sig.input_types[0].is_compatible_with(input_type)
    ]
