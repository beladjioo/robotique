"""Chaînes cinématiques série décrites par paramètres de Denavit-Hartenberg (standard)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Transformation homogène d'un segment selon la convention DH standard.

    ``A = Rot_z(theta) . Trans_z(d) . Trans_x(a) . Rot_x(alpha)``
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class DHLink:
    """Un segment DH. La variable articulaire s'ajoute à ``theta`` (rotoïde) ou à ``d`` (prismatique)."""

    a: float
    alpha: float
    d: float = 0.0
    theta_offset: float = 0.0
    prismatic: bool = False

    def transform(self, q: float) -> np.ndarray:
        if self.prismatic:
            return dh_transform(self.a, self.alpha, self.d + q, self.theta_offset)
        return dh_transform(self.a, self.alpha, self.d, self.theta_offset + q)


class SerialChain:
    """Bras manipulateur série : cinématique directe, jacobienne et butées articulaires."""

    def __init__(
        self,
        links: Sequence[DHLink],
        *,
        base: ArrayLike | None = None,
        tool: ArrayLike | None = None,
        joint_limits: ArrayLike | None = None,
        name: str = "chaine",
    ) -> None:
        self.links = tuple(links)
        self.name = name
        self.base = np.eye(4) if base is None else np.asarray(base, dtype=float)
        self.tool = np.eye(4) if tool is None else np.asarray(tool, dtype=float)
        n = len(self.links)
        if joint_limits is None:
            self.joint_limits = np.column_stack([np.full(n, -np.inf), np.full(n, np.inf)])
        else:
            self.joint_limits = np.asarray(joint_limits, dtype=float).reshape(n, 2)

    @property
    def n_joints(self) -> int:
        return len(self.links)

    def _as_q(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.n_joints:
            raise ValueError(
                f"{self.name}: {self.n_joints} articulations attendues, {q.size} reçues"
            )
        return q

    def frames(self, q: ArrayLike) -> list[np.ndarray]:
        """Poses ``T_0_i`` de tous les repères DH (``i = 0..n``), repère de base inclus."""
        q = self._as_q(q)
        frames = [self.base]
        for link, qi in zip(self.links, q, strict=True):
            frames.append(frames[-1] @ link.transform(qi))
        return frames

    def forward_kinematics(self, q: ArrayLike) -> np.ndarray:
        """Pose 4x4 de l'outil dans le repère de base."""
        return self.frames(q)[-1] @ self.tool

    def jacobian(self, q: ArrayLike) -> np.ndarray:
        """Jacobienne géométrique 6xn ``[v; w]`` exprimée dans le repère de base."""
        frames = self.frames(q)
        p_end = (frames[-1] @ self.tool)[:3, 3]
        J = np.zeros((6, self.n_joints))
        for i, link in enumerate(self.links):
            z = frames[i][:3, 2]
            p = frames[i][:3, 3]
            if link.prismatic:
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_end - p)
                J[3:, i] = z
        return J

    def clip_to_limits(self, q: ArrayLike) -> np.ndarray:
        q = self._as_q(q)
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def within_limits(self, q: ArrayLike) -> bool:
        q = self._as_q(q)
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def manipulability(self, q: ArrayLike, *, position_only: bool = False) -> float:
        """Indice de manipulabilité de Yoshikawa (produit des valeurs singulières de J).

        Vaut ``sqrt(det(J J^T))`` pour un bras redondant et s'annule en singularité.
        ``position_only=True`` ne considère que la partie translation de la jacobienne.
        """
        J = self.jacobian(q)
        if position_only:
            J = J[:3]
        return float(np.prod(np.linalg.svd(J, compute_uv=False)))
