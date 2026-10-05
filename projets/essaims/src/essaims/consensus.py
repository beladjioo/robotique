"""Protocoles de consensus et de formation distribués.

Chaque robot n'utilise que les états de ses voisins : ``x_i' = -sum_j a_ij (x_i - x_j)``,
soit ``x' = -L x``. Sur un graphe non orienté connexe, tous convergent vers la moyenne initiale.
Pas de temps stable : ``dt < 2 / lambda_max(L)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from essaims.graphs import laplacian


def consensus_step(states: ArrayLike, adjacency: ArrayLike, dt: float) -> np.ndarray:
    """Un pas de consensus ; ``states`` est ``(n,)`` ou ``(n, d)``."""
    x = np.asarray(states, dtype=float)
    return x - dt * laplacian(adjacency) @ x


def formation_step(
    positions: ArrayLike, adjacency: ArrayLike, offsets: ArrayLike, dt: float, gain: float = 1.0
) -> np.ndarray:
    """Commande de formation par déplacements relatifs : ``p_i - p_j -> d_i - d_j``.

    C'est un consensus sur ``p - d`` : la formation se forme autour du barycentre initial.
    """
    p = np.asarray(positions, dtype=float)
    d = np.asarray(offsets, dtype=float)
    return p - dt * gain * laplacian(adjacency) @ (p - d)
