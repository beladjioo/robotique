"""Planification symbolique de type STRIPS.

Un état est un ensemble de faits ; une action a des préconditions, des faits ajoutés et
des faits retirés. La recherche A* avec coût unitaire et heuristique nulle retourne un
plan de longueur minimale.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Fact = tuple[str, ...]
State = frozenset[Fact]


@dataclass(frozen=True)
class Action:
    name: str
    preconditions: frozenset[Fact]
    add: frozenset[Fact]
    delete: frozenset[Fact]

    def applicable(self, state: State) -> bool:
        return self.preconditions <= state

    def apply(self, state: State) -> State:
        return (state - self.delete) | self.add

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Problem:
    initial: State
    goal: frozenset[Fact]
    actions: tuple[Action, ...]

    def is_goal(self, state: State) -> bool:
        return self.goal <= state


def plan(
    problem: Problem,
    heuristic: Callable[[State, frozenset[Fact]], float] | None = None,
    max_expansions: int = 200_000,
) -> list[Action] | None:
    """Recherche A* dans l'espace d'états ; ``None`` si l'objectif est inatteignable.

    Sans heuristique, la recherche est une recherche à coût uniforme (plan optimal).
    """
    h = heuristic or (lambda _state, _goal: 0.0)
    counter = itertools.count()
    frontier = [(h(problem.initial, problem.goal), next(counter), 0, problem.initial)]
    parents: dict[State, tuple[State, Action] | None] = {problem.initial: None}
    cost = {problem.initial: 0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _, _, g, state = heapq.heappop(frontier)
        if g > cost[state]:
            continue
        if problem.is_goal(state):
            steps = []
            while parents[state] is not None:
                state, action = parents[state]
                steps.append(action)
            return steps[::-1]
        expansions += 1
        for action in problem.actions:
            if action.applicable(state):
                successor = action.apply(state)
                if g + 1 < cost.get(successor, 1 << 30):
                    cost[successor] = g + 1
                    parents[successor] = (state, action)
                    heapq.heappush(
                        frontier,
                        (g + 1 + h(successor, problem.goal), next(counter), g + 1, successor),
                    )
    return None


def goal_count(state: State, goal: frozenset[Fact]) -> float:
    """Heuristique « faits manquants » (rapide, non admissible en général)."""
    return float(len(goal - state))


def validate_plan(problem: Problem, steps: Sequence[Action] | Iterable[Action]) -> bool:
    """Rejoue un plan et vérifie préconditions puis objectif — garde-fou avant exécution réelle."""
    state = problem.initial
    for action in steps:
        if not action.applicable(state):
            return False
        state = action.apply(state)
    return problem.is_goal(state)
