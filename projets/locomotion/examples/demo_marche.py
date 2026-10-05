"""Démo : marche bipède par DCM avec poussée, puis CPG quadrupède trot -> bond.

python projets/locomotion/examples/demo_marche.py
"""

import numpy as np

from locomotion import (
    HopfCPG,
    LinearInvertedPendulum,
    dcm_walk,
    foot_trajectory,
    leg_inverse_kinematics,
)

lipm = LinearInvertedPendulum(com_height=0.8)
walk = dcm_walk(lipm, step_length=0.3, step_duration=0.5, n_steps=12, push=(2.2, 0.4))
print("Bipède — longueurs de pas (m) :", np.round(np.diff(walk.footsteps), 3))
print(
    f"           vitesse moyenne {walk.com[-1] / walk.times[-1]:.2f} m/s, poussée de 0,4 m/s à t = 2,2 s"
)

cpg = HopfCPG("trot", frequency=2.0, rng=np.random.default_rng(0))
for gait in ("trot", "bond"):
    cpg.set_gait(gait)
    for _ in range(2000):  # 10 s de simulation
        cpg.step(0.005)
    phases_deg = np.round(np.rad2deg(cpg.relative_phases())).astype(int) % 360
    print(
        f"Quadrupède — allure {gait:>5s}, phases relatives (°) : {phases_deg}, "
        f"erreur {np.rad2deg(cpg.phase_error()):.2f}°"
    )

hip, knee = leg_inverse_kinematics(
    *foot_trajectory(cpg.phases()[0], step_length=0.12, step_height=0.06, stance_height=0.3),
    0.2,
    0.2,
)
print(
    f"Consignes articulaires patte avant-gauche : hanche {np.rad2deg(hip):.1f}°, genou {np.rad2deg(knee):.1f}°"
)
