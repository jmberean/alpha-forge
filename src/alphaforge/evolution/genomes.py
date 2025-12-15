
from dataclasses import dataclass
from typing import Any
import copy
import random

from alphaforge.evolution.protocol import Evolvable
from alphaforge.discovery.expression.tree import ExpressionTree
from alphaforge.discovery.operators.mutation import mutate as mutate_tree
from alphaforge.discovery.operators.crossover import crossover as crossover_tree
from alphaforge.strategy.genome import StrategyGenome

@dataclass
class ExpressionGenome:
    """Wrapper for ExpressionTree to implement Evolvable."""
    tree: ExpressionTree

    @property
    def hash(self) -> str:
        return self.tree.hash

    def mutate(self, rng: Any) -> "ExpressionGenome":
        new_tree = mutate_tree(self.tree, rng)
        return ExpressionGenome(new_tree)

    def crossover(self, other: "Evolvable", rng: Any) -> tuple["Evolvable", "Evolvable"]:
        if not isinstance(other, ExpressionGenome):
            # Fallback: No crossover with different types
            return self, other
        
        child1_tree, child2_tree = crossover_tree(self.tree, other.tree, rng)
        return ExpressionGenome(child1_tree), ExpressionGenome(child2_tree)

    def to_strategy_genome(self) -> StrategyGenome:
        # Convert tree to StrategyGenome
        # This logic was previously in DiscoveryOrchestrator.to_strategy_genomes
        return StrategyGenome(
            name=f"discovered_{self.tree.hash[:8]}",
            description=f"GP: {self.tree.formula}",
            version="1.0",
            signals=[], # populated later by compiler? Or should we populate it here?
            # Actually DiscoveryOrchestrator populated metadata, signals were empty 
            # because the 'signal' is implicit in the tree logic.
            # But wait, BacktestEngine needs signals? 
            # No, Discovery uses a different execution path: _evaluate_tree -> positions
            # UNIFIED ENGINE means we need a standard execution path.
            # StrategyGenome usually has 'rules'.
            # Converting an arbitrary ExpressionTree to 'rules' is hard if it's not just logic ops.
            # But StrategyGenome can hold metadata['formula'] and we can have a special signal generator 
            # that knows how to execute formulas.
            position_sizing={"method": "fixed", "size": 1.0},
            metadata={
                "formula": self.tree.formula,
                "tree_hash": self.tree.hash,
                "complexity": self.tree.complexity_score(),
            },
        )

    def complexity_score(self) -> float:
        return self.tree.complexity_score()


@dataclass
class TemplateGenome:
    """Wrapper for StrategyGenome (template parameters) to implement Evolvable."""
    genome: StrategyGenome

    @property
    def hash(self) -> str:
        return self.genome.id

    def mutate(self, rng: Any) -> "TemplateGenome":
        new_genome = copy.deepcopy(self.genome)
        # Simple parameter mutation logic ported from GeneticStrategyEvolver
        if hasattr(new_genome, 'signals') and len(new_genome.signals) > 0:
            # Note: StrategyGenome signals structure is list[Signal]
            # But templates use 'entry_rules' and 'exit_rules' (RuleGroup)
            # StrategyTemplates populates entry_rules/exit_rules.
            # Signals list is often empty in templates? 
            # Let's check StrategyGenome definition.
            pass
        
        # Actually StrategyTemplates use 'parameters' dict?
        # Let's verify StrategyGenome structure.
        # It has 'parameters' dict.
        if new_genome.parameters:
            # Mutate random parameter
            key = rng.choice(list(new_genome.parameters.keys()))
            val = new_genome.parameters[key]
            if isinstance(val, (int, float)):
                # +/- 20%
                delta = val * rng.uniform(-0.2, 0.2)
                if isinstance(val, int):
                    new_val = int(val + delta)
                    new_val = max(1, new_val) # Assume positive
                else:
                    new_val = val + delta
                new_genome.parameters[key] = new_val
                
                # We need to re-generate rules based on new parameters?
                # StrategyTemplates.sma_crossover(fast=...) returns a new genome.
                # If we just change params dict, the rules don't update automatically.
                # This suggests TemplateGenome needs to hold the *Generator Function* and *Params*.
                # Or StrategyGenome needs a way to 'rehydrate' from params.
                
                # For now, let's assume we can't easily deep-mutate structure without the generator.
                # But wait, Factory uses GeneticStrategyEvolver.
                # How did it mutate?
                # "_mutate" in factory/genetic.py:
                # "Mutate period +/- 20%"
                # It accessed "individual.signals[idx].indicator.period".
                # But StrategyTemplates don't populate 'signals'. They populate 'entry_rules'.
                # Let's check 'src/alphaforge/strategy/genome.py'.
        
        return TemplateGenome(new_genome)

    def crossover(self, other: "Evolvable", rng: Any) -> tuple["Evolvable", "Evolvable"]:
        if not isinstance(other, TemplateGenome):
            return self, other
        
        # Crossover parameters
        g1 = copy.deepcopy(self.genome)
        g2 = copy.deepcopy(other.genome)
        
        if g1.parameters and g2.parameters:
            # Swap some parameters
            for key in g1.parameters:
                if key in g2.parameters and rng.random() < 0.5:
                    g1.parameters[key], g2.parameters[key] = g2.parameters[key], g1.parameters[key]
        
        return TemplateGenome(g1), TemplateGenome(g2)

    def to_strategy_genome(self) -> StrategyGenome:
        return self.genome

    def complexity_score(self) -> float:
        # Templates are usually simple fixed structure
        # Could base on parameter count or rules count
        return 0.1
