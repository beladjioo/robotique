"""Simulateur de trajectoire, d'odométrie bruitée et de mesures sur amers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from estimation.models import motion_model, range_bearing


@dataclass
class SimulatedRun:
    dt: float
    landmarks: np.ndarray
    true_poses: np.ndarray  # (T+1, 3)
    odometry: np.ndarray  # (T, 2) : (v, w) mesurés
    # mesures[k] = liste de (indice_amer, (distance, gisement)) reçues après le pas k
    measurements: list[list[tuple[int, np.ndarray]]] = field(default_factory=list)


def simulate_run(
    landmarks: ArrayLike,
    *,
    steps: int = 400,
    dt: float = 0.1,
    initial_pose: ArrayLike = (0.0, 0.0, 0.0),
    speed: float = 0.5,
    turn_rate: float = 0.15,
    motion_noise: tuple[float, float] = (0.05, 0.03),
    measurement_noise: tuple[float, float] = (0.1, 0.03),
    sensor_range: float = 6.0,
    rng: np.random.Generator | None = None,
) -> SimulatedRun:
    """Simule un robot qui décrit une boucle en percevant les amers à portée."""
    rng = rng if rng is not None else np.random.default_rng()
    landmarks = np.asarray(landmarks, dtype=float)
    poses = [np.asarray(initial_pose, dtype=float)]
    odometry, measurements = [], []
    for k in range(steps):
        v, w = speed, turn_rate * np.sin(0.02 * k) + turn_rate
        poses.append(motion_model(poses[-1], v, w, dt))
        odometry.append([v + rng.normal(0, motion_noise[0]), w + rng.normal(0, motion_noise[1])])
        seen = []
        for i, landmark in enumerate(landmarks):
            z = range_bearing(poses[-1], landmark)
            if z[0] <= sensor_range:
                seen.append((i, z + rng.normal(0.0, measurement_noise)))
        measurements.append(seen)
    return SimulatedRun(dt, landmarks, np.array(poses), np.array(odometry), measurements)
