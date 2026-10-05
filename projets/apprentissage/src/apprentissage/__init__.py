"""Branche apprentissage : apprentissage automatique pour la robotique.

Première pierre : environnements à l'API Gymnasium (grille, atteinte de cible avec
randomisation de domaine), Q-learning tabulaire et recherche directe de politique par
méthode de l'entropie croisée (CEM).
"""

from apprentissage.agents import (
    LinearPolicy,
    cross_entropy_method,
    evaluate_policy,
    greedy_rollout,
    q_learning,
)
from apprentissage.envs import GridWorld, PointMassReach

__all__ = [
    "GridWorld",
    "LinearPolicy",
    "PointMassReach",
    "cross_entropy_method",
    "evaluate_policy",
    "greedy_rollout",
    "q_learning",
]
