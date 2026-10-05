"""Contrôleur en cascade (position -> attitude) pour quadrirotor plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from drones.quadrotor import PlanarQuadrotor
from drones.trajectory import Reference


@dataclass
class CascadedController:
    """Boucle externe PD en position avec anticipation d'accélération, boucle interne PD en roulis.

    La boucle interne (≈ 30 rad/s) est réglée nettement plus rapide que la boucle externe
    (≈ 3 rad/s), ce qui justifie la séparation des échelles de temps.
    """

    model: PlanarQuadrotor
    kp_y: float = 6.0
    kd_y: float = 4.5
    kp_z: float = 20.0
    kd_z: float = 9.0
    kp_phi: float = 900.0
    kd_phi: float = 54.0
    max_tilt: float = np.deg2rad(30.0)

    def compute(self, state: ArrayLike, ref: Reference) -> tuple[float, float]:
        y, z, phi, vy, vz, phi_dot = state
        g, m = self.model.gravity, self.model.mass
        accel_y = (
            ref.acceleration[0]
            + self.kd_y * (ref.velocity[0] - vy)
            + self.kp_y * (ref.position[0] - y)
        )
        accel_z = (
            ref.acceleration[1]
            + self.kd_z * (ref.velocity[1] - vz)
            + self.kp_z * (ref.position[1] - z)
        )
        phi_cmd = float(np.clip(-accel_y / g, -self.max_tilt, self.max_tilt))
        thrust = m * (g + accel_z) / max(np.cos(phi), 0.5)  # compense l'inclinaison
        moment = self.model.inertia * (self.kp_phi * (phi_cmd - phi) - self.kd_phi * phi_dot)
        # Passage par le mélangeur : la commande réellement applicable tient compte des saturations.
        return self.model.unmix(*self.model.mix(thrust, moment))
