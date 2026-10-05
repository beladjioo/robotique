"""Géométrie projective : homographies, projection, triangulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robocore import homogeneous, invert_homogeneous


def to_homogeneous(points: ArrayLike) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.hstack([points, np.ones((len(points), 1))])


def from_homogeneous(points: ArrayLike) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points[:, :-1] / points[:, -1:]


def normalize_points(points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Normalisation de Hartley : centrage et distance moyenne sqrt(2) à l'origine."""
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
    return from_homogeneous(to_homogeneous(points) @ T.T), T


def estimate_homography(src: ArrayLike, dst: ArrayLike) -> np.ndarray:
    """Homographie ``dst ~ H src`` par DLT normalisée (au moins 4 correspondances)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) < 4 or src.shape != dst.shape:
        raise ValueError("au moins 4 correspondances de même taille sont nécessaires")
    src_n, T_src = normalize_points(src)
    dst_n, T_dst = normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    H_n = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return H / H[2, 2]


def apply_homography(H: ArrayLike, points: ArrayLike) -> np.ndarray:
    return from_homogeneous(to_homogeneous(points) @ np.asarray(H, dtype=float).T)


def look_at(eye: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Pose ``T_monde_camera`` d'une caméra placée en ``eye`` et visant ``target``."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("la direction de visée est colinéaire au vecteur « up »")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return homogeneous(np.column_stack([x, y, z]), eye)


def projection_matrix(K: ArrayLike, T_world_cam: ArrayLike) -> np.ndarray:
    """Matrice de projection 3x4 ``P = K [R | t]`` qui envoie un point monde vers l'image."""
    T_cam_world = invert_homogeneous(T_world_cam)
    return np.asarray(K, dtype=float) @ T_cam_world[:3, :]


def triangulate_points(P1: ArrayLike, P2: ArrayLike, x1: ArrayLike, x2: ArrayLike) -> np.ndarray:
    """Triangulation linéaire (DLT) de correspondances ``(N, 2)`` vues par deux caméras."""
    P1, P2 = np.asarray(P1, dtype=float), np.asarray(P2, dtype=float)
    points = []
    for (u1, v1), (u2, v2) in zip(np.asarray(x1, float), np.asarray(x2, float), strict=True):
        A = np.array(
            [u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]]
        )
        _, _, Vt = np.linalg.svd(A)
        points.append(Vt[-1, :3] / Vt[-1, 3])
    return np.array(points)


def reprojection_error(P: ArrayLike, points_3d: ArrayLike, pixels: ArrayLike) -> np.ndarray:
    """Erreur de reprojection (pixels) de chaque point."""
    projected = from_homogeneous(to_homogeneous(points_3d) @ np.asarray(P, dtype=float).T)
    return np.linalg.norm(projected - np.asarray(pixels, dtype=float), axis=1)
