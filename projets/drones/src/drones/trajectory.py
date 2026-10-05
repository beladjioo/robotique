"""Trajectoires lisses entre points de passage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from robocore import min_jerk


@dataclass(frozen=True)
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class WaypointTrajectory:
    """Enchaîne des segments à jerk minimal (arrêt à chaque point de passage)."""

    def __init__(self, waypoints: ArrayLike, segment_duration: float) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float)
        if len(self.waypoints) < 2:
            raise ValueError("au moins deux points de passage sont nécessaires")
        self.segment_duration = float(segment_duration)

    @property
    def duration(self) -> float:
        return self.segment_duration * (len(self.waypoints) - 1)

    def sample(self, t: float) -> Reference:
        T = self.segment_duration
        index = int(np.clip(t // T, 0, len(self.waypoints) - 2))
        start, end = self.waypoints[index], self.waypoints[index + 1]
        s, ds, dds = min_jerk((t - index * T) / T)
        delta = end - start
        return Reference(start + delta * s, delta * ds / T, delta * dds / T**2)
