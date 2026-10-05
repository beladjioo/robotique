"""Régulateur linéaire quadratique (LQR), continu et discret, sans SciPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _prepare(A, B, Q, R):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).reshape(A.shape[0], -1)
    Q = np.asarray(Q, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return A, B, Q, R


def lqr(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """LQR continu : retourne ``(K, P)`` avec ``u = -K x``.

    L'équation de Riccati algébrique est résolue via le sous-espace stable de la matrice
    hamiltonienne (méthode des vecteurs propres de Mac Farlane-Potter).
    """
    A, B, Q, R = _prepare(A, B, Q, R)
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    H = np.block([[A, -B @ R_inv @ B.T], [-Q, -A.T]])
    eigvals, eigvecs = np.linalg.eig(H)
    stable = eigvecs[:, eigvals.real < 0]
    if stable.shape[1] != n:
        raise np.linalg.LinAlgError("pas de solution stabilisante (système non stabilisable ?)")
    P = np.real(stable[n:] @ np.linalg.inv(stable[:n]))
    P = 0.5 * (P + P.T)
    return R_inv @ B.T @ P, P


def dlqr(
    A: ArrayLike,
    B: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """LQR discret : retourne ``(K, P)`` avec ``u_k = -K x_k``.

    L'équation de Riccati discrète est résolue par l'algorithme de doublement
    (SDA, convergence quadratique).
    """
    A, B, Q, R = _prepare(A, B, Q, R)
    n = A.shape[0]
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()
    I = np.eye(n)
    for _ in range(max_iterations):
        W = np.linalg.inv(I + Gk @ Hk)
        A_next = Ak @ W @ Ak
        G_next = Gk + Ak @ W @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ W @ Ak
        converged = np.linalg.norm(H_next - Hk) <= tolerance * max(1.0, np.linalg.norm(H_next))
        Ak, Gk, Hk = A_next, G_next, H_next
        if converged:
            break
    else:
        raise np.linalg.LinAlgError("Riccati discrète : pas de convergence")
    P = 0.5 * (Hk + Hk.T)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
