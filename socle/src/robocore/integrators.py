"""Intégrateurs numériques à pas fixe pour systèmes ``x' = f(x, u)``.

La commande ``u`` est supposée constante sur le pas (bloqueur d'ordre zéro), ce qui
correspond au fonctionnement d'un contrôleur numérique réel.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Dynamics = Callable[[np.ndarray, np.ndarray], np.ndarray]


def euler_step(f: Dynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Un pas d'Euler explicite (ordre 1)."""
    x = np.asarray(x)
    return x + dt * f(x, u)


def rk4_step(f: Dynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Un pas de Runge-Kutta classique (ordre 4)."""
    x = np.asarray(x)
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
