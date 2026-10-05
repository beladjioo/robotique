"""Carte d'occupation 2D.

Convention : la cellule ``(ligne, colonne)`` couvre ``x`` dans
``[ox + col*res, ox + (col+1)*res)`` et ``y`` dans ``[oy + ligne*res, oy + (ligne+1)*res)``.
La ligne 0 est donc en bas (``y`` minimal), comme pour ``nav_msgs/OccupancyGrid`` de ROS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

Cell = tuple[int, int]


class OccupancyGrid:
    def __init__(
        self,
        occupied: ArrayLike,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.occupied = np.asarray(occupied, dtype=bool)
        if self.occupied.ndim != 2:
            raise ValueError("la carte doit être un tableau 2D")
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

    @classmethod
    def from_ascii(
        cls,
        text: str,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
        obstacle: str = "#",
    ) -> OccupancyGrid:
        """Construit une carte depuis un dessin ASCII (la première ligne du texte est en haut)."""
        lines = [line for line in text.strip("\n").splitlines()]
        width = max(len(line) for line in lines)
        rows = [[ch == obstacle for ch in line.ljust(width)] for line in lines]
        return cls(np.flipud(np.array(rows, dtype=bool)), resolution, origin)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.occupied.shape[0] and 0 <= c < self.occupied.shape[1]

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.occupied[cell]

    def world_to_cell(self, x: float, y: float) -> Cell:
        return (
            int(np.floor((y - self.origin[1]) / self.resolution)),
            int(np.floor((x - self.origin[0]) / self.resolution)),
        )

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        """Centre de la cellule en coordonnées monde."""
        r, c = cell
        return (
            self.origin[0] + (c + 0.5) * self.resolution,
            self.origin[1] + (r + 0.5) * self.resolution,
        )

    def inflate(self, radius: float) -> OccupancyGrid:
        """Dilate les obstacles d'un disque de rayon ``radius`` (en mètres).

        Planifier sur la carte gonflée permet de traiter le robot comme un point.
        """
        cells = int(np.ceil(radius / self.resolution))
        occ = self.occupied
        out = occ.copy()
        H, W = occ.shape
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if dr * dr + dc * dc > cells * cells:
                    continue
                i0, i1 = max(0, -dr), H - max(0, dr)
                j0, j1 = max(0, -dc), W - max(0, dc)
                if i0 < i1 and j0 < j1:
                    out[i0:i1, j0:j1] |= occ[i0 + dr : i1 + dr, j0 + dc : j1 + dc]
        return OccupancyGrid(out, self.resolution, self.origin)
