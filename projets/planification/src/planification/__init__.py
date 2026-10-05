"""Branche planification : planification de tâches et IA embarquée.

Première pierre : planificateur symbolique STRIPS (A*), domaine de manipulation
« monde des blocs », et chaîne instruction en langage naturel -> objectif -> plan,
avec un analyseur à règles (hors ligne) ou un analyseur fondé sur un LLM (Claude).

Principe de sûreté : le modèle de langage ne produit qu'un *objectif* ; il est validé
puis confié à un planificateur symbolique dont chaque action est vérifiable.
"""

from planification.blocks import blocks_world
from planification.language import (
    ClaudeGoalParser,
    GoalParser,
    GoalParsingError,
    KeywordGoalParser,
    plan_from_instruction,
    validate_goal,
)
from planification.strips import Action, Fact, Problem, plan, validate_plan

__all__ = [
    "Action",
    "ClaudeGoalParser",
    "Fact",
    "GoalParser",
    "GoalParsingError",
    "KeywordGoalParser",
    "Problem",
    "blocks_world",
    "plan",
    "plan_from_instruction",
    "validate_goal",
    "validate_plan",
]
