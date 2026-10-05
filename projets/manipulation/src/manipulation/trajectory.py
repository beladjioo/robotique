"""Génération de trajectoires articulaires : polynômes quintiques et profils trapézoïdaux."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def quintic_coefficients(
    q0: ArrayLike,
    qf: ArrayLike,
    duration: float,
    v0: ArrayLike = 0.0,
    vf: ArrayLike = 0.0,
    a0: ArrayLike = 0.0,
    af: ArrayLike = 0.0,
) -> np.ndarray:
    """Coefficients ``c0..c5`` (lignes) du polynôme ``q(t) = sum c_k t^k`` pour chaque articulation."""
    if duration <= 0:
        raise ValueError("duration doit être strictement positive")
    T = duration
    M = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
            [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
        ],
        dtype=float,
    )
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    b = np.vstack(np.broadcast_arrays(q0, v0, a0, qf, vf, af)).astype(float)
    return np.linalg.solve(M, b)


class QuinticTrajectory:
    """Trajectoire point-à-point C2 (position, vitesse et accélération continues)."""

    def __init__(self, q0, qf, duration, v0=0.0, vf=0.0, a0=0.0, af=0.0) -> None:
        self.duration = float(duration)
        self.coefficients = quintic_coefficients(q0, qf, duration, v0, vf, a0, af)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne ``(q, qd, qdd)`` à l'instant ``t`` (saturé dans [0, duration])."""
        t = float(np.clip(t, 0.0, self.duration))
        c = self.coefficients
        powers = t ** np.arange(6)
        q = powers @ c
        qd = (np.arange(1, 6) * powers[:5]) @ c[1:]
        qdd = (np.array([2, 6, 12, 20]) * powers[:4]) @ c[2:]
        return q, qd, qdd


class TrapezoidalProfile:
    """Profil de vitesse trapézoïdal (ou triangulaire si la distance est trop courte)."""

    def __init__(self, distance: float, v_max: float, a_max: float) -> None:
        if v_max <= 0 or a_max <= 0:
            raise ValueError("v_max et a_max doivent être strictement positifs")
        self.sign = 1.0 if distance >= 0 else -1.0
        self.distance = abs(float(distance))
        self.a_max = float(a_max)
        if self.distance >= v_max**2 / a_max:
            self.t_accel = v_max / a_max
            self.v_peak = float(v_max)
            self.t_cruise = (self.distance - v_max**2 / a_max) / v_max
        else:
            self.t_accel = float(np.sqrt(self.distance / a_max))
            self.v_peak = a_max * self.t_accel
            self.t_cruise = 0.0
        self.duration = 2 * self.t_accel + self.t_cruise

    def sample(self, t: float) -> tuple[float, float, float]:
        """Retourne ``(s, v, a)`` à l'instant ``t``."""
        a, ta, tc, T = self.a_max, self.t_accel, self.t_cruise, self.duration
        if t <= 0:
            s, v, acc = 0.0, 0.0, 0.0
        elif t < ta:
            s, v, acc = 0.5 * a * t**2, a * t, a
        elif t < ta + tc:
            s, v, acc = 0.5 * a * ta**2 + self.v_peak * (t - ta), self.v_peak, 0.0
        elif t < T:
            remaining = T - t
            s, v, acc = self.distance - 0.5 * a * remaining**2, a * remaining, -a
        else:
            s, v, acc = self.distance, 0.0, 0.0
        return self.sign * s, self.sign * v, self.sign * acc
