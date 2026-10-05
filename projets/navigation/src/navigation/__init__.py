"""Branche navigation : robotique mobile autonome.

Première pierre : carte d'occupation, planification A* (avec raccourcissement),
modèle cinématique différentiel et suivi de chemin « pure pursuit ».
"""

from navigation import demo_maps
from navigation.controllers import PurePursuit
from navigation.grid import Cell, OccupancyGrid
from navigation.models import DifferentialDrive, unicycle_step
from navigation.planners import astar, line_of_sight, path_length, resample_path, shortcut_path

__all__ = [
    "Cell",
    "DifferentialDrive",
    "OccupancyGrid",
    "PurePursuit",
    "astar",
    "demo_maps",
    "line_of_sight",
    "path_length",
    "resample_path",
    "shortcut_path",
    "unicycle_step",
]
