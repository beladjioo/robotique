"""Conversions pures (sans dépendance ROS) entre conventions ROS et bibliothèques du dépôt."""

from __future__ import annotations

import math

import numpy as np


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Lacet -> quaternion au format ROS ``(x, y, z, w)``."""
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Quaternion ROS ``(x, y, z, w)`` -> lacet (rad)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def occupancy_to_ros_data(occupied: np.ndarray) -> list[int]:
    """Carte booléenne (ligne 0 = bas) -> champ ``data`` de ``nav_msgs/OccupancyGrid`` (0 ou 100)."""
    return [100 if cell else 0 for cell in np.asarray(occupied, dtype=bool).ravel(order="C")]
