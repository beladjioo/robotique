"""Outils numériques génériques (sans SciPy)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def numerical_jacobian(
    func: Callable[[np.ndarray], ArrayLike], x: ArrayLike, eps: float = 1e-6
) -> np.ndarray:
    """Jacobienne de ``func`` en ``x`` par différences finies centrées.

    Retourne une matrice ``(m, n)`` où ``m`` est la taille de la sortie et ``n`` celle de ``x``.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(func(x), dtype=float))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx.flat[i] = eps
        f_plus = np.atleast_1d(np.asarray(func(x + dx), dtype=float))
        f_minus = np.atleast_1d(np.asarray(func(x - dx), dtype=float))
        J[:, i] = (f_plus - f_minus).ravel() / (2.0 * eps)
    return J


def expm(A: ArrayLike, order: int = 18) -> np.ndarray:
    """Exponentielle de matrice par « scaling and squaring » + série de Taylor.

    Suffisant pour les petites matrices de la robotique (discrétisation de modèles
    linéaires, exponentielles de Lie). Pour des matrices mal conditionnées, préférer
    ``scipy.linalg.expm``.
    """
    A = np.asarray(A, dtype=float)
    norm = np.linalg.norm(A, ord=np.inf)
    squarings = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    A_scaled = A / (2.0**squarings)
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, order + 1):
        term = term @ A_scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result
