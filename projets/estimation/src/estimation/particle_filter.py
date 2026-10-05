"""Filtre particulaire (localisation de Monte-Carlo, MCL)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from estimation.models import motion_model, range_bearing
from robocore import wrap_angle


class ParticleFilter:
    """Localisation de Monte-Carlo : gère les distributions multimodales et la localisation globale."""

    def __init__(
        self,
        particles: ArrayLike,
        *,
        motion_noise: tuple[float, float],
        measurement_noise: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.particles = np.asarray(particles, dtype=float).copy()
        self.log_weights = np.full(len(self.particles), -np.log(len(self.particles)))
        self.motion_noise = np.asarray(motion_noise, dtype=float)
        self.measurement_noise = np.asarray(measurement_noise, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def uniform(
        cls,
        n: int,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        **kwargs,
    ) -> ParticleFilter:
        """Particules uniformes dans ``((xmin, xmax), (ymin, ymax))`` : robot « kidnappé »."""
        rng = kwargs.get("rng") or np.random.default_rng()
        kwargs["rng"] = rng
        (x0, x1), (y0, y1) = bounds
        particles = np.column_stack(
            [rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)]
        )
        return cls(particles, **kwargs)

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_weights - self.log_weights.max())
        return w / w.sum()

    def predict(self, v: float, w: float, dt: float) -> None:
        n = len(self.particles)
        v_noisy = v + self.rng.normal(0.0, self.motion_noise[0], n)
        w_noisy = w + self.rng.normal(0.0, self.motion_noise[1], n)
        self.particles = motion_model(self.particles, v_noisy, w_noisy, dt)

    def update(self, z: ArrayLike, landmark: ArrayLike) -> None:
        expected = range_bearing(self.particles, landmark)
        error = np.asarray(z, dtype=float) - expected
        error[:, 1] = wrap_angle(error[:, 1])
        self.log_weights += -0.5 * np.sum((error / self.measurement_noise) ** 2, axis=1)
        self.log_weights -= self.log_weights.max()  # stabilité numérique

    def effective_sample_size(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w**2))

    def resample(self) -> None:
        """Rééchantillonnage systématique (faible variance)."""
        n = len(self.particles)
        positions = (self.rng.random() + np.arange(n)) / n
        indices = np.searchsorted(np.cumsum(self.weights), positions)
        self.particles = self.particles[np.minimum(indices, n - 1)]
        self.log_weights = np.full(n, -np.log(n))

    def estimate(self) -> np.ndarray:
        """Moyenne pondérée (moyenne circulaire pour le cap)."""
        w = self.weights
        x, y = w @ self.particles[:, 0], w @ self.particles[:, 1]
        theta = np.arctan2(w @ np.sin(self.particles[:, 2]), w @ np.cos(self.particles[:, 2]))
        return np.array([x, y, theta])
