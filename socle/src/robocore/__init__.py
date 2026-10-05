"""robocore — socle mathématique partagé par toutes les branches du dépôt robotique."""

from robocore.angles import angle_diff, wrap_angle
from robocore.integrators import euler_step, rk4_step
from robocore.numerics import expm, numerical_jacobian
from robocore.profiles import min_jerk
from robocore.transforms import (
    homogeneous,
    invert_homogeneous,
    matrix_to_quat,
    matrix_to_rpy,
    quat_to_matrix,
    rot2,
    rotx,
    roty,
    rotz,
    rpy_to_matrix,
    se2,
    se2_to_pose,
    skew,
    so3_exp,
    so3_log,
    vee,
)

__version__ = "0.1.0"

__all__ = [
    "angle_diff",
    "euler_step",
    "expm",
    "homogeneous",
    "invert_homogeneous",
    "matrix_to_quat",
    "matrix_to_rpy",
    "min_jerk",
    "numerical_jacobian",
    "quat_to_matrix",
    "rk4_step",
    "rot2",
    "rotx",
    "roty",
    "rotz",
    "rpy_to_matrix",
    "se2",
    "se2_to_pose",
    "skew",
    "so3_exp",
    "so3_log",
    "vee",
    "wrap_angle",
]
