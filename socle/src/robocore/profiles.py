"""Profils temporels normalisés pour la génération de trajectoires."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def min_jerk(tau: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profil à jerk minimal ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5`` sur ``tau`` dans [0, 1].

    Retourne ``(s, ds/dtau, d2s/dtau2)``. Vitesse et accélération sont nulles aux deux
    extrémités. Pour une durée ``T`` : ``v = (pf - p0) * ds / T`` et
    ``a = (pf - p0) * d2s / T**2``. ``tau`` est saturé dans [0, 1].
    """
    t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    ds = 30 * t**2 - 60 * t**3 + 30 * t**4
    dds = 60 * t - 180 * t**2 + 120 * t**3
    return s, ds, dds
