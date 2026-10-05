"""Modèle dynamique d'un quadrirotor dans le plan vertical (y, z)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PlanarQuadrotor:
    """État ``(y, z, phi, y', z', phi')`` ; commande ``(poussée totale, moment de roulis)``.

    ``y'' = -u1 sin(phi) / m``, ``z'' = u1 cos(phi) / m - g``, ``phi'' = u2 / I``.
    Valeurs par défaut proches d'un nano-drone de type Crazyflie.
    """

    mass: float = 0.18
    inertia: float = 2.5e-4
    arm_length: float = 0.086
    gravity: float = 9.81
    max_rotor_thrust: float = 1.5  # N par paire de rotors

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity

    def dynamics(self, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        _, _, phi, vy, vz, phi_dot = x
        thrust, moment = u
        return np.array(
            [
                vy,
                vz,
                phi_dot,
                -thrust * np.sin(phi) / self.mass,
                thrust * np.cos(phi) / self.mass - self.gravity,
                moment / self.inertia,
            ]
        )

    def mix(self, thrust: float, moment: float) -> tuple[float, float]:
        """Poussée/moment vers poussées des rotors (gauche, droit), saturées."""
        left = 0.5 * thrust - moment / (2.0 * self.arm_length)
        right = 0.5 * thrust + moment / (2.0 * self.arm_length)
        clip = lambda f: float(np.clip(f, 0.0, self.max_rotor_thrust))  # noqa: E731
        return clip(left), clip(right)

    def unmix(self, left: float, right: float) -> tuple[float, float]:
        return left + right, self.arm_length * (right - left)
