"""Outils pour systèmes linéaires : linéarisation, discrétisation, commandabilité."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from robocore import expm, numerical_jacobian


def linearize(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: ArrayLike, u0: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Linéarisation numérique ``A = df/dx``, ``B = df/du`` autour de ``(x0, u0)``."""
    x0 = np.asarray(x0, dtype=float)
    u0 = np.atleast_1d(np.asarray(u0, dtype=float))
    A = numerical_jacobian(lambda x: f(x, u0), x0)
    B = numerical_jacobian(lambda u: f(x0, u), u0)
    return A, B


def discretize(A: ArrayLike, B: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrétisation exacte avec bloqueur d'ordre zéro."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).reshape(A.shape[0], -1)
    n, m = B.shape
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A
    block[:n, n:] = B
    E = expm(block * dt)
    return E[:n, :n], E[:n, n:]


def controllability_matrix(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).reshape(A.shape[0], -1)
    blocks = [B]
    for _ in range(A.shape[0] - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def is_controllable(A: ArrayLike, B: ArrayLike) -> bool:
    A = np.asarray(A, dtype=float)
    return bool(np.linalg.matrix_rank(controllability_matrix(A, B)) == A.shape[0])
