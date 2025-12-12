"""
Strategy Genome: Universal strategy representation.

All strategies are serialized to a standard format for:
- Consistent backtesting across different engines
- Genetic programming and optimization
- Reproducibility and audit trails
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import json
import uuid
import hashlib


class Operator(str, Enum):
    """Comparison operators for rules."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


class LogicalOperator(str, Enum):
    """Logical operators for combining rules."""

    AND = "and"
    OR = "or"


class OrderType(str, Enum):
    """Order types for execution."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSizing(str, Enum):
    """Position sizing methods."""

    FIXED = "fixed"
    VOLATILITY = "volatility"
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"


class Urgency(str, Enum):
    """Execution urgency levels."""

    PASSIVE = "passive"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"


@dataclass
class Rule:
    """
    A single trading rule condition.

    Examples:
        RSI > 70 (overbought)
        SMA_20 crosses above SMA_50
        Close > Bollinger Upper Band
    """

    indicator: str
    operator: Operator
    value: float | str  # Can be a number or another indicator name
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "operator": self.operator.value,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Rule":
        return cls(
            indicator=data["indicator"],
            operator=Operator(data["operator"]),
            value=data["value"],
            description=data.get("description", ""),
        )


@dataclass
class RuleGroup:
    """
    A group of rules combined with a logical operator.

    Supports nested groups for complex conditions:
        (RSI > 70 AND MACD < 0) OR (Price > BB_Upper)
    """

    rules: list[Rule | "RuleGroup"]
    operator: LogicalOperator = LogicalOperator.AND

    def to_dict(self) -> dict:
        return {
            "rules": [
                r.to_dict() if isinstance(r, Rule) else r.to_dict() for r in self.rules
            ],
            "operator": self.operator.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleGroup":
        rules = []
        for r in data["rules"]:
            if "rules" in r:  # Nested group
                rules.append(cls.from_dict(r))
            else:
                rules.append(Rule.from_dict(r))
        return cls(rules=rules, operator=LogicalOperator(data["operator"]))


@dataclass
class Filter:
    """
    A filter for universe or regime selection.

    Examples:
        Universe: S&P 500 constituents
        Regime: VIX < 20 (low volatility)
    """

    name: str
    conditions: list[Rule]
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "conditions": [c.to_dict() for c in self.conditions],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Filter":
        return cls(
            name=data["name"],
            conditions=[Rule.from_dict(c) for c in data["conditions"]],
            description=data.get("description", ""),
        )


@dataclass
class StrategyGenome:
    """
    Universal strategy representation.

    This is the standard format for all strategies in AlphaForge.
    It captures the complete specification needed to:
    - Execute the strategy in backtesting
    - Reproduce results exactly
    - Optimize parameters
    - Track in MLflow
    """

    # Identity
    name: str
    version: str = "1.0.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Signals
    entry_rules: RuleGroup = field(default_factory=lambda: RuleGroup(rules=[]))
    exit_rules: RuleGroup = field(default_factory=lambda: RuleGroup(rules=[]))

    # Filters
    universe_filter: Optional[Filter] = None
    regime_filter: Optional[Filter] = None

    # Position sizing
    sizing_method: PositionSizing = PositionSizing.FIXED
    max_position_pct: float = 0.10  # 10% max position
    target_volatility: float = 0.10  # For volatility sizing

    # Risk management
    stop_loss_pct: Optional[float] = None  # e.g., 0.02 for 2%
    take_profit_pct: Optional[float] = None
    max_holding_days: Optional[int] = None

    # Execution
    order_type: OrderType = OrderType.MARKET
    urgency: Urgency = Urgency.NEUTRAL

    # Metadata
    description: str = ""
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: list[str] = field(default_factory=list)

    # Parameters (for optimization)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "entry_rules": self.entry_rules.to_dict(),
            "exit_rules": self.exit_rules.to_dict(),
            "universe_filter": self.universe_filter.to_dict()
            if self.universe_filter
            else None,
            "regime_filter": self.regime_filter.to_dict()
            if self.regime_filter
            else None,
            "sizing_method": self.sizing_method.value,
            "max_position_pct": self.max_position_pct,
            "target_volatility": self.target_volatility,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_holding_days": self.max_holding_days,
            "order_type": self.order_type.value,
            "urgency": self.urgency.value,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at,
            "tags": self.tags,
            "parameters": self.parameters,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyGenome":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            version=data.get("version", "1.0.0"),
            entry_rules=RuleGroup.from_dict(data["entry_rules"]),
            exit_rules=RuleGroup.from_dict(data["exit_rules"]),
            universe_filter=Filter.from_dict(data["universe_filter"])
            if data.get("universe_filter")
            else None,
            regime_filter=Filter.from_dict(data["regime_filter"])
            if data.get("regime_filter")
            else None,
            sizing_method=PositionSizing(data.get("sizing_method", "fixed")),
            max_position_pct=data.get("max_position_pct", 0.10),
            target_volatility=data.get("target_volatility", 0.10),
            stop_loss_pct=data.get("stop_loss_pct"),
            take_profit_pct=data.get("take_profit_pct"),
            max_holding_days=data.get("max_holding_days"),
            order_type=OrderType(data.get("order_type", "market")),
            urgency=Urgency(data.get("urgency", "neutral")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyGenome":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this strategy configuration.

        Used for caching and deduplication.
        """
        # Exclude volatile fields
        stable_dict = self.to_dict()
        del stable_dict["id"]
        del stable_dict["created_at"]

        json_str = json.dumps(stable_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def clone(self, **overrides) -> "StrategyGenome":
        """
        Create a copy with optional field overrides.

        Useful for parameter optimization.
        """
        data = self.to_dict()
        data.update(overrides)
        data["id"] = str(uuid.uuid4())[:8]  # New ID
        data["created_at"] = datetime.utcnow().isoformat()
        return self.from_dict(data)

    def with_parameters(self, **params) -> "StrategyGenome":
        """
        Create a copy with updated parameters.

        Args:
            **params: Parameter values to update

        Returns:
            New StrategyGenome with updated parameters
        """
        new_params = {**self.parameters, **params}
        return self.clone(parameters=new_params)

    def __repr__(self) -> str:
        return f"StrategyGenome(id={self.id}, name={self.name}, v{self.version})"
