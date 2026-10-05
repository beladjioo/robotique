"""Démo : stabilisation d'un pendule inversé sur chariot par LQR (modèle non linéaire).

python projets/commande/examples/demo_pendule_inverse.py [--plot]
"""

import argparse

import numpy as np

from commande import CartPole, lqr
from robocore import rk4_step

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="enregistre outputs/pendule.png")
args = parser.parse_args()

pole = CartPole()
A, B = pole.linearized()
K, _ = lqr(A, B, Q=np.diag([1.0, 1.0, 10.0, 1.0]), R=[[0.1]])
print("Gain LQR K =", np.round(K, 2))

x = np.array([0.0, 0.0, np.deg2rad(20.0), 0.0])  # pendule incliné de 20°
dt, history = 0.01, []
for k in range(800):
    u = float(np.clip(-(K @ x)[0], -20.0, 20.0))  # saturation de l'actionneur
    history.append([k * dt, *x, u])
    x = rk4_step(pole.dynamics, x, [u], dt)
history = np.array(history)
print(
    f"Angle final {np.rad2deg(x[2]):.4f}°, position {x[0]:.4f} m, "
    f"effort max {np.abs(history[:, 5]).max():.1f} N"
)

if args.plot:
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    axes[0].plot(history[:, 0], np.rad2deg(history[:, 3]))
    axes[0].set_ylabel("angle (°)")
    axes[1].plot(history[:, 0], history[:, 1])
    axes[1].set_ylabel("chariot (m)")
    axes[2].plot(history[:, 0], history[:, 5])
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("temps (s)")
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/pendule.png", dpi=120, bbox_inches="tight")
    print("Figure enregistrée dans outputs/pendule.png")
