"""Modèles cinématiques de robots mobiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from robocore import wrap_angle


@dataclass(frozen=True)
class DifferentialDrive:
    """Robot à deux roues motrices coaxiales (type TurtleBot)."""

    wheel_radius: float
    wheel_base: float  # entraxe des roues

    def forward(self, omega_left: float, omega_right: float) -> tuple[float, float]:
        """Vitesses de roues (rad/s) vers torseur ``(v, w)`` du robot."""
        r = self.wheel_radius
        return r * (omega_right + omega_left) / 2.0, r * (
            omega_right - omega_left
        ) / self.wheel_base

    def inverse(self, v: float, w: float) -> tuple[float, float]:
        """Torseur ``(v, w)`` vers vitesses de roues ``(gauche, droite)`` en rad/s."""
        half = 0.5 * w * self.wheel_base
        return (v - half) / self.wheel_radius, (v + half) / self.wheel_radius


def unicycle_step(pose: ArrayLike, v: float, w: float, dt: float) -> np.ndarray:
    """Intègre exactement le modèle unicycle sur ``dt`` à ``(v, w)`` constants (arc de cercle)."""
    x, y, theta = np.asarray(pose, dtype=float)
    if abs(w) < 1e-9:
        x += v * dt * np.cos(theta)
        y += v * dt * np.sin(theta)
    else:
        x += v / w * (np.sin(theta + w * dt) - np.sin(theta))
        y -= v / w * (np.cos(theta + w * dt) - np.cos(theta))
    return np.array([x, y, wrap_angle(theta + w * dt)])
