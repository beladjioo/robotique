"""Modèles de mouvement (odométrie unicycle) et de mesure (distance/gisement sur amer)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robocore import wrap_angle


def motion_model(pose: ArrayLike, v: ArrayLike, w: ArrayLike, dt: float) -> np.ndarray:
    """Propagation unicycle par point milieu. Vectorisé : ``pose`` peut être ``(N, 3)``."""
    pose = np.asarray(pose, dtype=float)
    theta_mid = pose[..., 2] + 0.5 * np.asarray(w) * dt
    out = np.empty_like(pose)
    out[..., 0] = pose[..., 0] + v * dt * np.cos(theta_mid)
    out[..., 1] = pose[..., 1] + v * dt * np.sin(theta_mid)
    out[..., 2] = wrap_angle(pose[..., 2] + np.asarray(w) * dt)
    return out


def motion_jacobians(
    pose: ArrayLike, v: float, w: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobiennes ``F = df/dx`` (3x3) et ``V = df/du`` (3x2, ``u = (v, w)``) du modèle de mouvement."""
    theta_mid = float(pose[2]) + 0.5 * w * dt
    c, s = np.cos(theta_mid), np.sin(theta_mid)
    F = np.array([[1.0, 0.0, -v * dt * s], [0.0, 1.0, v * dt * c], [0.0, 0.0, 1.0]])
    V = np.array([[dt * c, -0.5 * v * dt**2 * s], [dt * s, 0.5 * v * dt**2 * c], [0.0, dt]])
    return F, V


def range_bearing(pose: ArrayLike, landmark: ArrayLike) -> np.ndarray:
    """Mesure attendue ``(distance, gisement)`` d'un amer. Vectorisé sur les poses ``(N, 3)``."""
    pose = np.asarray(pose, dtype=float)
    dx = landmark[0] - pose[..., 0]
    dy = landmark[1] - pose[..., 1]
    return np.stack([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - pose[..., 2])], axis=-1)


def range_bearing_jacobian(pose: ArrayLike, landmark: ArrayLike) -> np.ndarray:
    """Jacobienne 2x3 de :func:`range_bearing` par rapport à la pose."""
    dx = landmark[0] - pose[0]
    dy = landmark[1] - pose[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]])
