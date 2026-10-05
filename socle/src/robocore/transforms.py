"""Transformations rigides en 2D et 3D.

Conventions :
- repères directs (main droite), angles en radians, unités SI ;
- quaternions au format ``(w, x, y, z)`` ;
- ``T_a_b`` désigne la pose du repère *b* exprimée dans le repère *a*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- 2D
def rot2(theta: float) -> FloatArray:
    """Matrice de rotation plane 2x2."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def se2(x: float, y: float, theta: float) -> FloatArray:
    """Matrice homogène 3x3 d'une pose plane ``(x, y, theta)``."""
    T = np.eye(3)
    T[:2, :2] = rot2(theta)
    T[:2, 2] = (x, y)
    return T


def se2_to_pose(T: ArrayLike) -> tuple[float, float, float]:
    """Extrait ``(x, y, theta)`` d'une matrice homogène 3x3."""
    T = np.asarray(T, dtype=float)
    return float(T[0, 2]), float(T[1, 2]), float(np.arctan2(T[1, 0], T[0, 0]))


# --------------------------------------------------------------------------- 3D
def rotx(angle: float) -> FloatArray:
    """Rotation élémentaire autour de l'axe x."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(angle: float) -> FloatArray:
    """Rotation élémentaire autour de l'axe y."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(angle: float) -> FloatArray:
    """Rotation élémentaire autour de l'axe z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: ArrayLike) -> FloatArray:
    """Matrice antisymétrique telle que ``skew(a) @ b == cross(a, b)``."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(W: ArrayLike) -> FloatArray:
    """Opération inverse de :func:`skew`."""
    W = np.asarray(W, dtype=float)
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def so3_exp(omega: ArrayLike) -> FloatArray:
    """Exponentielle de SO(3) (formule de Rodrigues) d'un vecteur rotation."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3) + skew(omega)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: ArrayLike) -> FloatArray:
    """Logarithme de SO(3) : vecteur rotation ``axe * angle`` avec angle dans [0, pi]."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * vee(R - R.T)
    if np.pi - theta < 1e-6:
        # Proche de pi : l'axe se lit sur la diagonale de (R + I) / 2 = a a^T.
        A = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(A[k, k])
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def homogeneous(R: ArrayLike, p: ArrayLike) -> FloatArray:
    """Assemble une matrice homogène 4x4 à partir d'une rotation et d'une translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def invert_homogeneous(T: ArrayLike) -> FloatArray:
    """Inverse exacte d'une transformation rigide 4x4 (sans inversion numérique)."""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    return homogeneous(R.T, -R.T @ p)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> FloatArray:
    """Angles roulis/tangage/lacet (convention ZYX) vers matrice de rotation."""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def matrix_to_rpy(R: ArrayLike) -> tuple[float, float, float]:
    """Matrice de rotation vers ``(roll, pitch, yaw)`` (convention ZYX)."""
    R = np.asarray(R, dtype=float)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-9:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:  # blocage de cardan : seul roll - yaw (ou roll + yaw) est observable
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def quat_to_matrix(q: ArrayLike) -> FloatArray:
    """Quaternion ``(w, x, y, z)`` (normalisé à la volée) vers matrice de rotation."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: ArrayLike) -> FloatArray:
    """Matrice de rotation vers quaternion ``(w, x, y, z)`` avec ``w >= 0`` (méthode de Shepperd)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)
