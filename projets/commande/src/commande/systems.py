"""Systèmes de référence pour la commande."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class DCMotor:
    """Moteur à courant continu. État ``(omega, courant)``, entrée : tension (V).

    ``J w' = K i - b w`` et ``L i' = V - R i - K w``. Valeurs par défaut : petit moteur
    de robot mobile.
    """

    inertia: float = 1e-4  # J (kg.m^2)
    friction: float = 1e-5  # b (N.m.s)
    torque_constant: float = 0.05  # K (N.m/A = V.s/rad)
    resistance: float = 1.5  # R (ohm)
    inductance: float = 1e-3  # L (H)

    def dynamics(self, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        omega, current = x
        voltage = float(np.asarray(u).ravel()[0])
        d_omega = (self.torque_constant * current - self.friction * omega) / self.inertia
        d_current = (
            voltage - self.resistance * current - self.torque_constant * omega
        ) / self.inductance
        return np.array([d_omega, d_current])


@dataclass(frozen=True)
class CartPole:
    """Pendule inversé sur chariot (masse ponctuelle au bout d'une tige sans masse).

    État ``(x, x', theta, theta')`` avec ``theta = 0`` pendule vertical vers le haut ;
    entrée : force horizontale sur le chariot (N).
    """

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    length: float = 0.5
    gravity: float = 9.81

    def dynamics(self, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        _, x_dot, theta, theta_dot = x
        force = float(np.asarray(u).ravel()[0])
        M, m, l, g = self.cart_mass, self.pole_mass, self.length, self.gravity
        s, c = np.sin(theta), np.cos(theta)
        denom = M + m * s * s
        x_ddot = (force + m * l * s * theta_dot**2 - m * g * s * c) / denom
        theta_ddot = ((M + m) * g * s - c * (force + m * l * s * theta_dot**2)) / (l * denom)
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def linearized(self) -> tuple[np.ndarray, np.ndarray]:
        """Modèle linéaire analytique autour de l'équilibre haut."""
        M, m, l, g = self.cart_mass, self.pole_mass, self.length, self.gravity
        A = np.array(
            [[0, 1, 0, 0], [0, 0, -m * g / M, 0], [0, 0, 0, 1], [0, 0, (M + m) * g / (l * M), 0]],
            dtype=float,
        )
        B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (l * M)]])
        return A, B
