"""Localisation par filtre de Kalman étendu sur amers connus."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from estimation.models import motion_jacobians, motion_model, range_bearing, range_bearing_jacobian
from robocore import wrap_angle


class EKFLocalizer:
    """EKF à état ``(x, y, theta)``.

    ``motion_noise`` : écarts-types ``(sigma_v, sigma_w)`` de l'odométrie ;
    ``measurement_noise`` : écarts-types ``(sigma_distance, sigma_gisement)``.
    """

    def __init__(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        *,
        motion_noise: tuple[float, float],
        measurement_noise: tuple[float, float],
    ) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.M = np.diag(np.square(motion_noise))
        self.R = np.diag(np.square(measurement_noise))

    def predict(self, v: float, w: float, dt: float) -> None:
        F, V = motion_jacobians(self.x, v, w, dt)
        self.x = motion_model(self.x, v, w, dt)
        self.P = F @ self.P @ F.T + V @ self.M @ V.T

    def update(self, z: ArrayLike, landmark: ArrayLike) -> np.ndarray:
        """Corrige l'état avec une mesure ``(distance, gisement)``. Retourne l'innovation."""
        innovation = np.asarray(z, dtype=float) - range_bearing(self.x, landmark)
        innovation[1] = wrap_angle(innovation[1])
        H = range_bearing_jacobian(self.x, landmark)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # forme de Joseph (reste symétrique)
        return innovation

    def mahalanobis(self, z: ArrayLike, landmark: ArrayLike) -> float:
        """Distance de Mahalanobis d'une mesure — utile pour rejeter les aberrations."""
        innovation = np.asarray(z, dtype=float) - range_bearing(self.x, landmark)
        innovation[1] = wrap_angle(innovation[1])
        H = range_bearing_jacobian(self.x, landmark)
        S = H @ self.P @ H.T + self.R
        return float(np.sqrt(innovation @ np.linalg.solve(S, innovation)))
