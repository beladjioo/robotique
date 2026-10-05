"""Branche vision : perception visuelle pour la robotique.

Première pierre : caméra sténopé avec distorsion radiale, géométrie projective
(homographie DLT normalisée, triangulation linéaire) et pose de caméra « look-at ».
"""

from vision.camera import PinholeCamera
from vision.geometry import (
    apply_homography,
    estimate_homography,
    from_homogeneous,
    look_at,
    normalize_points,
    projection_matrix,
    reprojection_error,
    to_homogeneous,
    triangulate_points,
)

__all__ = [
    "PinholeCamera",
    "apply_homography",
    "estimate_homography",
    "from_homogeneous",
    "look_at",
    "normalize_points",
    "projection_matrix",
    "reprojection_error",
    "to_homogeneous",
    "triangulate_points",
]
