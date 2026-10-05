"""Du langage naturel à l'objectif symbolique, puis au plan.

Deux analyseurs interchangeables implémentent :class:`GoalParser` :

- :class:`KeywordGoalParser` : règles en français, sans dépendance ni réseau ;
- :class:`ClaudeGoalParser` : modèle de langage Claude avec sortie JSON contrainte par
  un schéma (dépendance optionnelle ``anthropic``).

Dans les deux cas l'objectif est validé avant d'être confié au planificateur.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from planification.blocks import TABLE, blocks_world
from planification.strips import Action, Fact, plan, validate_plan

PREDICATES = {"sur": 2, "sur_table": 1}


class GoalParsingError(ValueError):
    """L'instruction n'a pas pu être traduite en un objectif valide."""


class GoalParser(Protocol):
    def parse(self, instruction: str, objects: Iterable[str]) -> frozenset[Fact]: ...


def _fact(block: str, support: str) -> Fact:
    return ("sur_table", block) if support == TABLE else ("sur", block, support)


def validate_goal(goal: Iterable[Fact], objects: Iterable[str]) -> None:
    """Vérifie prédicats, objets, auto-support et supports contradictoires."""
    objects = set(objects)
    supports: dict[str, str] = {}
    occupied: set[str] = set()
    for fact in goal:
        name, *args = fact
        if PREDICATES.get(name) != len(args):
            raise GoalParsingError(f"prédicat inconnu ou mal formé : {fact}")
        unknown = [a for a in args if a not in objects]
        if unknown:
            raise GoalParsingError(
                f"objet(s) inconnu(s) {unknown} ; disponibles : {sorted(objects)}"
            )
        block, support = args[0], (args[1] if name == "sur" else TABLE)
        if block == support:
            raise GoalParsingError(f"le cube {block!r} ne peut pas être posé sur lui-même")
        if supports.setdefault(block, support) != support:
            raise GoalParsingError(f"objectif contradictoire pour le cube {block!r}")
        if support != TABLE:
            if support in occupied:
                raise GoalParsingError(f"deux cubes ne peuvent pas reposer sur {support!r}")
            occupied.add(support)


class KeywordGoalParser:
    """Analyseur à règles : « mets le cube rouge sur le cube bleu et le vert sur la table »."""

    _ARTICLE = r"(?:(?:le|la|les|l')\s*)?"
    _NOUN = r"(?:(?:cube|bloc)\s+)?"
    _PATTERN = re.compile(rf"{_ARTICLE}{_NOUN}(\w+)\s+sur\s+{_ARTICLE}{_NOUN}(\w+)", re.IGNORECASE)

    def parse(self, instruction: str, objects: Iterable[str]) -> frozenset[Fact]:
        objects = list(objects)
        pairs = self._PATTERN.findall(instruction.lower())
        if not pairs:
            raise GoalParsingError(f"aucune consigne « X sur Y » reconnue dans : {instruction!r}")
        goal = frozenset(_fact(block, support) for block, support in pairs)
        validate_goal(goal, [*objects, TABLE])
        return goal


DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = (
    "Tu traduis les instructions données à un bras robotique qui manipule des cubes en un "
    "objectif symbolique. Chaque fait indique sur quoi un cube doit reposer à la fin de la "
    "tâche : un autre cube ou la table. N'inclus que les contraintes demandées explicitement "
    "ou nécessairement impliquées par l'instruction. Si rien de faisable n'est demandé, "
    "renvoie une liste vide."
)


def _goal_schema(objects: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "faits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cube": {"type": "string", "enum": objects},
                        "support": {"type": "string", "enum": [*objects, TABLE]},
                    },
                    "required": ["cube", "support"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["faits"],
        "additionalProperties": False,
    }


class ClaudeGoalParser:
    """Analyseur fondé sur Claude (API Messages, sortie structurée par schéma JSON).

    Le schéma restreint les noms aux cubes réellement présents ; l'objectif reste
    ensuite validé localement. ``client`` peut être injecté (tests, proxy, autre plateforme) ;
    par défaut ``anthropic.Anthropic()`` lit ``ANTHROPIC_API_KEY`` ou le profil ``ant auth login``.
    En cas de refus du modèle, l'API rejoue la requête sur un modèle de repli (``fallbacks``).
    """

    def __init__(
        self, client: Any = None, *, model: str = DEFAULT_MODEL, effort: str = "low"
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dépend de l'environnement
                raise ImportError(
                    "ClaudeGoalParser nécessite le SDK : pip install 'anthropic>=1.0'"
                ) from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = (
            effort  # tâche d'extraction simple : effort faible = plus rapide et moins cher
        )

    def parse(self, instruction: str, objects: Iterable[str]) -> frozenset[Fact]:
        objects = sorted(objects)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _goal_schema(objects)},
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Cubes présents : {', '.join(objects)}\nInstruction : {instruction}",
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise GoalParsingError("le modèle a refusé de traiter cette instruction")
        if response.stop_reason == "max_tokens":
            raise GoalParsingError("réponse du modèle tronquée (max_tokens atteint)")
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise GoalParsingError("réponse du modèle sans contenu texte")
        try:
            facts = json.loads(text)["faits"]
            goal = frozenset(_fact(item["cube"], item["support"]) for item in facts)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise GoalParsingError(f"réponse du modèle illisible : {text!r}") from exc
        validate_goal(goal, [*objects, TABLE])
        return goal


def plan_from_instruction(
    instruction: str,
    configuration: Mapping[str, str],
    parser: GoalParser | None = None,
) -> tuple[frozenset[Fact], list[Action] | None]:
    """Instruction -> objectif validé -> plan optimal vérifié (``None`` si irréalisable)."""
    parser = parser if parser is not None else KeywordGoalParser()
    goal = parser.parse(instruction, configuration.keys())
    problem = blocks_world(configuration, goal)
    steps = plan(problem)
    if steps is not None and not validate_plan(problem, steps):  # pragma: no cover - garde-fou
        raise RuntimeError("le planificateur a produit un plan invalide")
    return goal, steps
