"""Modèle de caméra sténopé (convention OpenCV : x à droite, y vers le bas, z vers l'avant)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0  # distorsion radiale
    k2: float = 0.0

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov: float) -> PinholeCamera:
        """Caméra idéale à pixels carrés à partir du champ de vision horizontal (rad)."""
        f = 0.5 * width / np.tan(0.5 * horizontal_fov)
        return cls(f, f, 0.5 * width, 0.5 * height, width, height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def distort(self, xn: np.ndarray) -> np.ndarray:
        """Applique la distorsion radiale à des coordonnées normalisées ``(N, 2)``."""
        r2 = np.sum(xn**2, axis=-1, keepdims=True)
        return xn * (1.0 + self.k1 * r2 + self.k2 * r2**2)

    def undistort(self, xd: np.ndarray, iterations: int = 20) -> np.ndarray:
        """Inverse de :meth:`distort` par point fixe (suffisant pour les distorsions usuelles)."""
        xn = xd.copy()
        for _ in range(iterations):
            r2 = np.sum(xn**2, axis=-1, keepdims=True)
            xn = xd / (1.0 + self.k1 * r2 + self.k2 * r2**2)
        return xn

    def project(self, points_cam: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Projette des points 3D du repère caméra ``(N, 3)``.

        Retourne les pixels ``(N, 2)`` et un masque des points devant la caméra et dans l'image.
        """
        P = np.atleast_2d(np.asarray(points_cam, dtype=float))
        z = P[:, 2]
        in_front = z > 1e-9
        xn = P[:, :2] / np.where(in_front, z, 1.0)[:, None]
        xd = self.distort(xn)
        uv = np.column_stack([self.fx * xd[:, 0] + self.cx, self.fy * xd[:, 1] + self.cy])
        visible = (
            in_front
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.height)
        )
        return uv, visible

    def backproject(self, uv: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """Pixels ``(N, 2)`` + profondeurs ``z`` vers points 3D du repère caméra."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = np.column_stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy])
        xn = self.undistort(xd)
        depth = np.broadcast_to(np.asarray(depth, dtype=float), (len(uv),))
        return np.column_stack([xn * depth[:, None], depth])
