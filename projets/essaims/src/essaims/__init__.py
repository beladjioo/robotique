"""Branche essaims : systèmes multi-robots.

Première pierre : théorie des graphes (laplacien, connectivité algébrique), consensus
et commande de formation distribués, flocking de Reynolds et allocation de tâches
par enchères (algorithme de Bertsekas).
"""

from essaims.assignment import auction_assignment
from essaims.boids import BoidsParams, boids_step, polarization
from essaims.consensus import consensus_step, formation_step
from essaims.graphs import (
    algebraic_connectivity,
    complete_graph,
    disk_graph,
    is_connected,
    laplacian,
    ring_graph,
)

__all__ = [
    "BoidsParams",
    "algebraic_connectivity",
    "auction_assignment",
    "boids_step",
    "complete_graph",
    "consensus_step",
    "disk_graph",
    "formation_step",
    "is_connected",
    "laplacian",
    "polarization",
    "ring_graph",
]
