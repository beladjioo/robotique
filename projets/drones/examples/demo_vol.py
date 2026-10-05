"""Démo : un quadrirotor plan suit une série de points de passage.

python projets/drones/examples/demo_vol.py [--plot]
"""

import argparse

import numpy as np

from drones import CascadedController, PlanarQuadrotor, WaypointTrajectory
from robocore import rk4_step

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="enregistre outputs/drone.png")
args = parser.parse_args()

quad = PlanarQuadrotor()
controller = CascadedController(quad)
trajectory = WaypointTrajectory([[0, 0], [0, 1.5], [2, 2], [3, 0.5], [3, 0]], segment_duration=2.5)

dt, x, log = 0.002, np.zeros(6), []
for k in range(int((trajectory.duration + 2.0) / dt)):
    ref = trajectory.sample(k * dt)
    u = controller.compute(x, ref)
    log.append([*x[:3], *ref.position])
    x = rk4_step(quad.dynamics, x, np.asarray(u), dt)
log = np.array(log)
error = np.linalg.norm(log[:, :2] - log[:, 3:], axis=1)
print(
    f"Erreur de suivi : moyenne {error.mean() * 100:.1f} cm, max {error.max() * 100:.1f} cm ; "
    f"inclinaison max {np.rad2deg(np.abs(log[:, 2]).max()):.1f}°"
)

if args.plot:
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log[:, 3], log[:, 4], "--", label="référence")
    ax.plot(log[:, 0], log[:, 1], label="vol")
    ax.scatter(*trajectory.waypoints.T, c="k", zorder=3, label="points de passage")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.legend()
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/drone.png", dpi=120, bbox_inches="tight")
    print("Figure enregistrée dans outputs/drone.png")
