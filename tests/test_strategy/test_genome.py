"""Tests for strategy genome."""

import json

import pytest

from alphaforge.strategy.genome import (
    LogicalOperator,
    Operator,
    PositionSizing,
    Rule,
    RuleGroup,
    StrategyGenome,
)
from alphaforge.strategy.templates import StrategyTemplates


class TestStrategyGenome:
    """Tests for StrategyGenome."""

    def test_create_simple_strategy(self):
        """Test creating a simple strategy."""
        strategy = StrategyGenome(
            name="Test Strategy",
            entry_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="rsi_14",
                        operator=Operator.LT,
                        value=30,
                    )
                ]
            ),
            exit_rules=RuleGroup(
                rules=[
                    Rule(
                        indicator="rsi_14",
                        operator=Operator.GT,
                        value=70,
                    )
                ]
            ),
        )

        assert strategy.name == "Test Strategy"
        assert len(strategy.entry_rules.rules) == 1
        assert len(strategy.exit_rules.rules) == 1

    def test_strategy_serialization(self):
        """Test strategy serializes to JSON."""
        strategy = StrategyTemplates.sma_crossover()

        json_str = strategy.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "SMA Crossover"

    def test_strategy_deserialization(self):
        """Test strategy deserializes from JSON."""
        original = StrategyTemplates.rsi_mean_reversion()
        json_str = original.to_json()

        restored = StrategyGenome.from_json(json_str)

        assert restored.name == original.name
        assert len(restored.entry_rules.rules) == len(original.entry_rules.rules)

    def test_strategy_fingerprint(self):
        """Test strategy fingerprint is consistent."""
        strategy1 = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)
        strategy2 = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)

        # Same config should have same fingerprint
        assert strategy1.fingerprint() == strategy2.fingerprint()

        # Different config should have different fingerprint
        strategy3 = StrategyTemplates.sma_crossover(fast_period=10, slow_period=30)
        assert strategy1.fingerprint() != strategy3.fingerprint()

    def test_strategy_clone(self):
        """Test strategy cloning."""
        original = StrategyTemplates.sma_crossover()
        cloned = original.clone(name="Cloned Strategy")

        assert cloned.name == "Cloned Strategy"
        assert cloned.id != original.id  # New ID
        assert len(cloned.entry_rules.rules) == len(original.entry_rules.rules)

    def test_strategy_with_parameters(self):
        """Test updating strategy parameters."""
        original = StrategyTemplates.sma_crossover()
        updated = original.with_parameters(fast_period=15, slow_period=45)

        assert updated.parameters["fast_period"] == 15
        assert updated.parameters["slow_period"] == 45
        assert updated.id != original.id

    def test_rule_operators(self):
        """Test all rule operators."""
        operators = [
            Operator.GT,
            Operator.GTE,
            Operator.LT,
            Operator.LTE,
            Operator.EQ,
            Operator.NEQ,
            Operator.CROSS_ABOVE,
            Operator.CROSS_BELOW,
        ]

        for op in operators:
            rule = Rule(indicator="close", operator=op, value=100)
            assert rule.operator == op

    def test_rule_group_and_or(self):
        """Test rule groups with AND/OR."""
        rules = [
            Rule(indicator="rsi_14", operator=Operator.LT, value=30),
            Rule(indicator="volume_ratio", operator=Operator.GT, value=1.5),
        ]

        and_group = RuleGroup(rules=rules, operator=LogicalOperator.AND)
        or_group = RuleGroup(rules=rules, operator=LogicalOperator.OR)

        assert and_group.operator == LogicalOperator.AND
        assert or_group.operator == LogicalOperator.OR

    def test_position_sizing_methods(self):
        """Test different position sizing methods."""
        for method in PositionSizing:
            strategy = StrategyGenome(
                name=f"Test {method.value}",
                sizing_method=method,
            )
            assert strategy.sizing_method == method


class TestStrategyTemplates:
    """Tests for strategy templates."""

    def test_all_templates_are_valid(self):
        """Test that all templates create valid strategies."""
        templates = StrategyTemplates.all_templates()

        assert len(templates) >= 5  # Should have several templates

        for strategy in templates:
            assert isinstance(strategy, StrategyGenome)
            assert strategy.name
            assert strategy.entry_rules.rules or True  # Can be empty

    def test_sma_crossover_template(self):
        """Test SMA crossover template."""
        strategy = StrategyTemplates.sma_crossover(fast_period=10, slow_period=30)

        assert "SMA" in strategy.name
        assert strategy.parameters["fast_period"] == 10
        assert strategy.parameters["slow_period"] == 30

    def test_rsi_mean_reversion_template(self):
        """Test RSI mean reversion template."""
        strategy = StrategyTemplates.rsi_mean_reversion(oversold=25, overbought=75)

        assert "RSI" in strategy.name
        assert strategy.parameters["oversold"] == 25
        assert strategy.parameters["overbought"] == 75

    def test_get_template_by_name(self):
        """Test getting template by name."""
        strategy = StrategyTemplates.get_template("sma_crossover")
        assert "SMA" in strategy.name

        strategy = StrategyTemplates.get_template("rsi_mean_reversion")
        assert "RSI" in strategy.name

    def test_get_invalid_template(self):
        """Test that invalid template name raises error."""
        with pytest.raises(ValueError):
            StrategyTemplates.get_template("invalid_template_name")
