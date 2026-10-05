"""Flocking de Reynolds (boids) : séparation, alignement, cohésion — règles purement locales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BoidsParams:
    perception_radius: float = 3.0
    separation_radius: float = 0.6
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.4
    min_speed: float = 0.5  # vitesse de croisière minimale : un essaim ne fait pas de surplace
    max_speed: float = 1.0
    max_acceleration: float = 2.0


def _clip_norm(v: np.ndarray, max_norm: float, min_norm: float = 0.0) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    target = np.clip(norms, min_norm, max_norm)
    return v * target / np.maximum(norms, 1e-12)


def boids_step(
    positions: ArrayLike, velocities: ArrayLike, params: BoidsParams, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    offsets = p[None, :, :] - p[:, None, :]  # offsets[i, j] = p_j - p_i
    dist = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(dist, np.inf)

    neighbors = dist < params.perception_radius
    count = neighbors.sum(axis=1, keepdims=True)
    has_neighbors = count > 0
    safe_count = np.maximum(count, 1)
    cohesion = np.where(
        has_neighbors, (neighbors[..., None] * offsets).sum(axis=1) / safe_count, 0.0
    )
    alignment = np.where(has_neighbors, (neighbors @ v) / safe_count - v, 0.0)
    close = dist < params.separation_radius
    separation = -(close[..., None] * offsets / np.square(dist)[..., None]).sum(axis=1)

    acceleration = (
        params.separation_weight * separation
        + params.alignment_weight * alignment
        + params.cohesion_weight * cohesion
    )
    v = _clip_norm(
        v + dt * _clip_norm(acceleration, params.max_acceleration),
        params.max_speed,
        params.min_speed,
    )
    return p + dt * v, v


def polarization(velocities: ArrayLike) -> float:
    """Paramètre d'ordre dans [0, 1] : 1 quand tous les agents vont dans la même direction."""
    v = np.asarray(velocities, dtype=float)
    headings = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return float(np.linalg.norm(headings.mean(axis=0)))
