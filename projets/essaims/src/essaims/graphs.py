"""Graphes de communication entre robots (matrices d'adjacence symétriques)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ring_graph(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def complete_graph(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


def disk_graph(positions: ArrayLike, radius: float) -> np.ndarray:
    """Graphe de proximité : deux robots communiquent s'ils sont à moins de ``radius``."""
    p = np.asarray(positions, dtype=float)
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    return ((dist <= radius) & ~np.eye(len(p), dtype=bool)).astype(float)


def laplacian(adjacency: ArrayLike) -> np.ndarray:
    A = np.asarray(adjacency, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def algebraic_connectivity(adjacency: ArrayLike) -> float:
    """Valeur de Fiedler (2e plus petite valeur propre du laplacien) : > 0 si connexe.

    Elle fixe la vitesse de convergence du consensus.
    """
    eigenvalues = np.linalg.eigvalsh(laplacian(adjacency))
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def is_connected(adjacency: ArrayLike, tol: float = 1e-9) -> bool:
    return algebraic_connectivity(adjacency) > tol
