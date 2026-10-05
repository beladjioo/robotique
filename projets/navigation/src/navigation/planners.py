"""Planification de chemin sur carte d'occupation."""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np
from numpy.typing import ArrayLike

from navigation.grid import Cell, OccupancyGrid

_SQRT2 = math.sqrt(2.0)


def _heuristic(a: Cell, b: Cell, diagonal: bool) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if diagonal:  # distance octile : admissible et cohérente en 8-connexité
        return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)
    return float(dr + dc)


def astar(
    grid: OccupancyGrid, start: Cell, goal: Cell, *, allow_diagonal: bool = True
) -> list[Cell] | None:
    """Plus court chemin de ``start`` à ``goal`` (cellules), ou ``None`` s'il n'existe pas.

    En 8-connexité, les diagonales ne « coupent » jamais le coin d'un obstacle.
    """
    if not (grid.is_free(start) and grid.is_free(goal)):
        return None
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
    if allow_diagonal:
        moves += [(1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2)]

    counter = itertools.count()  # départage les égalités sans comparer les cellules
    open_heap = [(_heuristic(start, goal, allow_diagonal), next(counter), start)]
    g_score: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell | None] = {start: None}
    closed: set[Cell] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while (prev := parent[path[-1]]) is not None:
                path.append(prev)
            return path[::-1]
        closed.add(current)
        r, c = current
        for dr, dc, cost in moves:
            neighbor = (r + dr, c + dc)
            if not grid.is_free(neighbor):
                continue
            if dr and dc and not (grid.is_free((r + dr, c)) and grid.is_free((r, c + dc))):
                continue
            candidate = g_score[current] + cost
            if candidate < g_score.get(neighbor, math.inf):
                g_score[neighbor] = candidate
                parent[neighbor] = current
                f = candidate + _heuristic(neighbor, goal, allow_diagonal)
                heapq.heappush(open_heap, (f, next(counter), neighbor))
    return None


def _bresenham(a: Cell, b: Cell) -> list[Cell]:
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    cells = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    """Vrai si le segment discret (Bresenham) entre ``a`` et ``b`` ne traverse que du libre."""
    return all(grid.is_free(cell) for cell in _bresenham(a, b))


def shortcut_path(grid: OccupancyGrid, path: list[Cell]) -> list[Cell]:
    """Supprime les points intermédiaires inutiles par tests de visibilité (glouton)."""
    if len(path) <= 2:
        return list(path)
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1
        result.append(path[j])
        i = j
    return result


def path_length(points: ArrayLike) -> float:
    """Longueur d'une polyligne ``(N, 2)``."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_path(points: ArrayLike, spacing: float) -> np.ndarray:
    """Rééchantillonne une polyligne avec un pas au plus égal à ``spacing``."""
    pts = np.asarray(points, dtype=float)
    out = [pts[0]]
    for a, b in itertools.pairwise(pts):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)
