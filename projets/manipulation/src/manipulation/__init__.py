"""Branche manipulation : bras manipulateurs série.

Première pierre : cinématique directe (Denavit-Hartenberg), jacobienne géométrique,
cinématique inverse numérique (moindres carrés amortis) et trajectoires articulaires.
"""

from manipulation.ik import IKResult, pose_error, solve_ik
from manipulation.kinematics import DHLink, SerialChain, dh_transform
from manipulation.robots import planar_2r, planar_3r, ur5e
from manipulation.trajectory import QuinticTrajectory, TrapezoidalProfile, quintic_coefficients

__all__ = [
    "DHLink",
    "IKResult",
    "QuinticTrajectory",
    "SerialChain",
    "TrapezoidalProfile",
    "dh_transform",
    "planar_2r",
    "planar_3r",
    "pose_error",
    "quintic_coefficients",
    "solve_ik",
    "ur5e",
]
