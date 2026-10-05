"""Démo : un UR5e va chercher un objet puis le dépose (IK + trajectoires quintiques).

Lancer depuis la racine du dépôt :  python projets/manipulation/examples/demo_pick_place.py
"""

import numpy as np

from manipulation import QuinticTrajectory, solve_ik, ur5e
from robocore import homogeneous, rotx

robot = ur5e()
home = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
gripper_down = rotx(np.pi)  # pince orientée vers le bas

waypoints = {
    "approche": homogeneous(gripper_down, [0.45, 0.20, 0.30]),
    "saisie": homogeneous(gripper_down, [0.45, 0.20, 0.12]),
    "transfert": homogeneous(gripper_down, [0.10, 0.50, 0.30]),
    "dépose": homogeneous(gripper_down, [0.10, 0.50, 0.12]),
}

q = home
for name, pose in waypoints.items():
    result = solve_ik(robot, pose, q)
    if not result.success:
        raise SystemExit(f"IK impossible pour l'étape « {name} »")
    traj = QuinticTrajectory(q, result.q, duration=2.0)
    worst = 0.0
    for t in np.linspace(0.0, 2.0, 50):
        qt, qd, _ = traj.sample(t)
        worst = max(worst, float(np.max(np.abs(qd))))
    p = robot.forward_kinematics(result.q)[:3, 3]
    print(
        f"{name:>10s} : {result.iterations:3d} itérations IK, "
        f"pince en {np.round(p, 3)}, vitesse articulaire max {worst:.2f} rad/s, "
        f"manipulabilité {robot.manipulability(result.q):.3f}"
    )
    q = result.q
