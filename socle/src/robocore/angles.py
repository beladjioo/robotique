"""Utilitaires sur les angles (radians)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def wrap_angle(angle: ArrayLike) -> np.ndarray | float:
    """Ramène un angle (ou un tableau d'angles) dans l'intervalle [-pi, pi)."""
    wrapped = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def angle_diff(a: ArrayLike, b: ArrayLike) -> np.ndarray | float:
    """Différence angulaire signée ``a - b`` ramenée dans [-pi, pi)."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
