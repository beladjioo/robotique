"""Démo : allocation de cibles par enchères puis formation en hexagone et flocking.

python projets/essaims/examples/demo_essaim.py
"""

import numpy as np

from essaims import (
    BoidsParams,
    algebraic_connectivity,
    auction_assignment,
    boids_step,
    disk_graph,
    formation_step,
    polarization,
    ring_graph,
)

rng = np.random.default_rng(0)

# 1) Allocation : 6 robots, 6 zones à inspecter, on minimise la distance totale.
robots = rng.uniform(0, 20, size=(6, 2))
zones = rng.uniform(0, 20, size=(6, 2))
distance = np.linalg.norm(robots[:, None] - zones[None], axis=-1)
assignment = auction_assignment(-distance)
greedy = distance.argmin(axis=1)
print(
    f"Enchères : distance totale {distance[np.arange(6), assignment].sum():.1f} m "
    f"(affectation {assignment.tolist()}) ; le plus-proche-voisin glouton donnerait "
    f"{len(set(greedy.tolist()))} zones distinctes seulement"
)

# 2) Formation hexagonale sur graphe en anneau (communication locale).
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
hexagon = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
p = robots.copy()
graph = ring_graph(6)
for _ in range(2000):
    p = formation_step(p, graph, hexagon, dt=0.05)
error = np.abs((p - p.mean(axis=0)) - hexagon).max()
print(
    f"Formation : erreur de forme {error:.2e} m (connectivité algébrique {algebraic_connectivity(graph):.2f})"
)

# 3) Flocking : 40 agents à caps aléatoires s'alignent sans coordinateur central.
p = rng.uniform(0, 6, size=(40, 2))
v = rng.normal(size=(40, 2))
print(f"Flocking : polarisation initiale {polarization(v):.2f}", end="")
for _ in range(2000):
    p, v = boids_step(p, v, BoidsParams(), dt=0.05)
print(
    f" -> finale {polarization(v):.2f} ; essaim connexe : "
    f"{algebraic_connectivity(disk_graph(p, 3.0)) > 0}"
)
