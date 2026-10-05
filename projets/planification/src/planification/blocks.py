"""Domaine « monde des blocs » : un bras avec pince empile des cubes sur une table."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from planification.strips import Action, Fact, Problem

TABLE = "table"


def _initial_state(configuration: Mapping[str, str]) -> frozenset[Fact]:
    facts: set[Fact] = {("main_vide",)}
    supports = set(configuration.values())
    for block, support in configuration.items():
        facts.add(("sur_table", block) if support == TABLE else ("sur", block, support))
        if block not in supports:
            facts.add(("libre", block))
    return frozenset(facts)


def _actions(blocks: list[str]) -> tuple[Action, ...]:
    actions = []
    for x in blocks:
        actions.append(
            Action(
                f"prendre({x})",
                frozenset({("sur_table", x), ("libre", x), ("main_vide",)}),
                frozenset({("tient", x)}),
                frozenset({("sur_table", x), ("libre", x), ("main_vide",)}),
            )
        )
        actions.append(
            Action(
                f"poser({x})",
                frozenset({("tient", x)}),
                frozenset({("sur_table", x), ("libre", x), ("main_vide",)}),
                frozenset({("tient", x)}),
            )
        )
        for y in blocks:
            if x == y:
                continue
            actions.append(
                Action(
                    f"empiler({x}, {y})",
                    frozenset({("tient", x), ("libre", y)}),
                    frozenset({("sur", x, y), ("libre", x), ("main_vide",)}),
                    frozenset({("tient", x), ("libre", y)}),
                )
            )
            actions.append(
                Action(
                    f"depiler({x}, {y})",
                    frozenset({("sur", x, y), ("libre", x), ("main_vide",)}),
                    frozenset({("tient", x), ("libre", y)}),
                    frozenset({("sur", x, y), ("libre", x), ("main_vide",)}),
                )
            )
    return tuple(actions)


def blocks_world(configuration: Mapping[str, str], goal: Iterable[Fact]) -> Problem:
    """Construit un problème à partir de ``{cube: support}`` (support = autre cube ou ``"table"``)."""
    blocks = sorted(configuration)
    for block, support in configuration.items():
        if support != TABLE and support not in configuration:
            raise ValueError(f"le cube {block!r} repose sur un support inconnu {support!r}")
    return Problem(_initial_state(configuration), frozenset(goal), _actions(blocks))
