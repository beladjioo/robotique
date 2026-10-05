"""Contrôleurs de suivi de chemin."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from navigation.planners import resample_path


class PurePursuit:
    """Suivi de chemin « pure pursuit » pour robot de type unicycle.

    Le robot vise un point du chemin situé à la distance ``lookahead`` et suit l'arc
    de cercle qui y mène : courbure ``k = 2 y_L / L^2`` (``y_L`` latéral dans le repère robot).
    """

    def __init__(
        self,
        path: ArrayLike,
        *,
        lookahead: float = 0.5,
        speed: float = 0.5,
        goal_tolerance: float = 0.05,
        max_angular_speed: float = 2.0,
    ) -> None:
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        self.path = resample_path(path, lookahead / 5.0) if len(path) > 1 else path
        self.lookahead = float(lookahead)
        self.speed = float(speed)
        self.goal_tolerance = float(goal_tolerance)
        self.max_angular_speed = float(max_angular_speed)
        self._progress = 0

    def reset(self) -> None:
        self._progress = 0

    def compute(self, pose: ArrayLike) -> tuple[float, float, bool]:
        """Retourne ``(v, w, but_atteint)`` pour la pose ``(x, y, theta)``."""
        x, y, theta = np.asarray(pose, dtype=float)
        position = np.array([x, y])
        dist_goal = float(np.linalg.norm(self.path[-1] - position))
        if dist_goal < self.goal_tolerance:
            return 0.0, 0.0, True

        # Progression monotone le long du chemin : jamais de retour en arrière.
        remaining = self.path[self._progress :]
        self._progress += int(np.argmin(np.linalg.norm(remaining - position, axis=1)))

        target = self.path[-1]
        for point in self.path[self._progress :]:
            if np.linalg.norm(point - position) >= self.lookahead:
                target = point
                break

        dx, dy = target - position
        x_local = np.cos(theta) * dx + np.sin(theta) * dy
        y_local = -np.sin(theta) * dx + np.cos(theta) * dy
        curvature = 2.0 * y_local / max(x_local**2 + y_local**2, 1e-9)

        v = self.speed * float(np.clip(dist_goal / self.lookahead, 0.2, 1.0))
        w = v * curvature
        if abs(w) > self.max_angular_speed:  # on ralentit pour conserver la courbure
            w = float(np.sign(w)) * self.max_angular_speed
            v = abs(w / curvature)
        return float(v), float(w), False
