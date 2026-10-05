"""Démo : planification A* dans un entrepôt puis suivi du chemin en pure pursuit.

python projets/navigation/examples/demo_navigation.py [--plot]
"""

import argparse

import numpy as np

from navigation import (
    OccupancyGrid,
    PurePursuit,
    astar,
    demo_maps,
    path_length,
    shortcut_path,
    unicycle_step,
)

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="enregistre outputs/navigation.png")
args = parser.parse_args()

grid = OccupancyGrid.from_ascii(demo_maps.WAREHOUSE, demo_maps.WAREHOUSE_RESOLUTION)
planning_grid = grid.inflate(0.2)  # rayon du robot + marge
start = planning_grid.world_to_cell(*demo_maps.WAREHOUSE_START)
goal = planning_grid.world_to_cell(*demo_maps.WAREHOUSE_GOAL)

raw = astar(planning_grid, start, goal)
if raw is None:
    raise SystemExit("aucun chemin trouvé")
cells = shortcut_path(planning_grid, raw)
waypoints = np.array([grid.cell_to_world(c) for c in cells])
print(
    f"A* : {len(raw)} cellules -> {len(cells)} points après raccourcissement, "
    f"longueur {path_length(waypoints):.2f} m"
)

controller = PurePursuit(waypoints, lookahead=0.4, speed=0.6)
pose = np.array([*waypoints[0], 0.0])
trace = [pose]
dt = 0.02
for step in range(5000):
    v, w, reached = controller.compute(pose)
    if reached:
        print(
            f"But atteint en {step * dt:.1f} s, erreur finale "
            f"{np.linalg.norm(pose[:2] - waypoints[-1]) * 100:.1f} cm"
        )
        break
    pose = unicycle_step(pose, v, w, dt)
    trace.append(pose)
else:
    print("But non atteint")

if args.plot:
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = np.array(trace)
    H, W = grid.shape
    extent = (0, W * grid.resolution, 0, H * grid.resolution)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(grid.occupied, origin="lower", extent=extent, cmap="Greys")
    ax.plot(waypoints[:, 0], waypoints[:, 1], "o--", label="chemin planifié")
    ax.plot(trace[:, 0], trace[:, 1], label="trajectoire suivie")
    ax.set_aspect("equal")
    ax.legend()
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/navigation.png", dpi=120, bbox_inches="tight")
    print("Figure enregistrée dans outputs/navigation.png")
