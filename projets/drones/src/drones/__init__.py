"""Branche drones : robotique aérienne.

Première pierre : quadrirotor plan (y, z, roulis), mélangeur moteurs, contrôleur
en cascade position -> attitude et trajectoires à jerk minimal entre points de passage.
"""

from drones.controller import CascadedController
from drones.quadrotor import PlanarQuadrotor
from drones.trajectory import Reference, WaypointTrajectory

__all__ = ["CascadedController", "PlanarQuadrotor", "Reference", "WaypointTrajectory"]
