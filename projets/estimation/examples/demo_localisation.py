"""Démo : odométrie seule vs EKF vs filtre particulaire (localisation globale).

python projets/estimation/examples/demo_localisation.py [--plot]
"""

import argparse

import numpy as np

from estimation import EKFLocalizer, ParticleFilter, motion_model, simulate_run

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="enregistre outputs/localisation.png")
args = parser.parse_args()

landmarks = np.array([[2.0, 1.0], [-1.5, 3.5], [4.0, 5.0], [0.5, 6.5], [-3.0, 0.5]])
rng = np.random.default_rng(7)
run = simulate_run(landmarks, steps=600, rng=rng)

ekf = EKFLocalizer(
    run.true_poses[0], np.eye(3) * 1e-4, motion_noise=(0.05, 0.03), measurement_noise=(0.1, 0.03)
)
pf = ParticleFilter.uniform(
    2000,
    ((-4.0, 5.0), (-1.0, 7.0)),
    motion_noise=(0.1, 0.06),
    measurement_noise=(0.3, 0.1),
    rng=rng,
)
odom = [run.true_poses[0]]
ekf_track, pf_track = [ekf.x.copy()], [pf.estimate()]
for (v, w), seen in zip(run.odometry, run.measurements, strict=True):
    odom.append(motion_model(odom[-1], v, w, run.dt))
    ekf.predict(v, w, run.dt)
    pf.predict(v, w, run.dt)
    for i, z in seen:
        ekf.update(z, landmarks[i])
        pf.update(z, landmarks[i])
    if pf.effective_sample_size() < 0.5 * len(pf.particles):
        pf.resample()
    ekf_track.append(ekf.x.copy())
    pf_track.append(pf.estimate())

truth = run.true_poses
for name, track in [("odométrie", odom), ("EKF", ekf_track), ("filtre particulaire", pf_track)]:
    track = np.array(track)
    err = np.linalg.norm(track[:, :2] - truth[:, :2], axis=1)
    print(
        f"{name:>20s} : erreur finale {err[-1]:.3f} m, erreur moyenne (2e moitié) "
        f"{err[len(err) // 2 :].mean():.3f} m"
    )

if args.plot:
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(truth[:, 0], truth[:, 1], "k", label="vérité")
    ax.plot(*np.array(odom)[:, :2].T, "--", label="odométrie")
    ax.plot(*np.array(ekf_track)[:, :2].T, label="EKF")
    ax.plot(*np.array(pf_track)[:, :2].T, ":", label="filtre particulaire")
    ax.scatter(*landmarks.T, marker="*", s=150, c="gold", edgecolors="k", label="amers")
    ax.set_aspect("equal")
    ax.legend()
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/localisation.png", dpi=120, bbox_inches="tight")
    print("Figure enregistrée dans outputs/localisation.png")
