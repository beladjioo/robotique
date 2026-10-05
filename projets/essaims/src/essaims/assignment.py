"""Allocation de tâches multi-robots par enchères (algorithme de Bertsekas).

Chaque robot enchérit sur la tâche qui lui rapporte le plus au prix courant ; le
mécanisme se distribue naturellement (seuls les prix sont partagés) et le résultat est
optimal à ``n * epsilon`` près.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def auction_assignment(benefit: ArrayLike, epsilon: float | None = None) -> np.ndarray:
    """Affecte chaque robot (ligne) à une tâche distincte (colonne) en maximisant le bénéfice total.

    Retourne ``assignment`` tel que le robot ``i`` réalise la tâche ``assignment[i]``.
    Requiert au moins autant de tâches que de robots. Pour des bénéfices entiers,
    ``epsilon < 1 / n`` garantit l'optimalité exacte.
    """
    benefit = np.asarray(benefit, dtype=float)
    n_agents, n_tasks = benefit.shape
    if n_agents > n_tasks:
        raise ValueError("il faut au moins autant de tâches que de robots")
    if epsilon is None:
        spread = np.ptp(benefit) if benefit.size else 1.0
        epsilon = max(spread, 1e-9) / (10.0 * (n_agents + 1))
    prices = np.zeros(n_tasks)
    owner = np.full(n_tasks, -1)
    assignment = np.full(n_agents, -1)
    unassigned = list(range(n_agents))
    while unassigned:
        agent = unassigned.pop()
        values = benefit[agent] - prices
        best = int(np.argmax(values))
        if n_tasks > 1:
            second_value = np.max(np.delete(values, best))
        else:
            second_value = values[best]
        prices[best] += values[best] - second_value + epsilon
        if owner[best] >= 0:
            assignment[owner[best]] = -1
            unassigned.append(int(owner[best]))
        owner[best] = agent
        assignment[agent] = best
    return assignment
