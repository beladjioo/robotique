"""Démo : reconstruction 3D d'un cube par une paire stéréo bruitée.

python projets/vision/examples/demo_stereo.py
"""

import itertools

import numpy as np

from vision import PinholeCamera, look_at, projection_matrix, reprojection_error, triangulate_points

camera = PinholeCamera.from_fov(1280, 720, np.deg2rad(75))
baseline = 0.12  # 12 cm, comme une caméra stéréo de type ZED / RealSense
target = np.array([0.0, 0.0, 0.3])
P_left = projection_matrix(camera.K, look_at([0.0, -1.5, 0.6], target))
P_right = projection_matrix(camera.K, look_at([baseline, -1.5, 0.6], target + [baseline, 0, 0]))

cube = np.array(list(itertools.product([-0.1, 0.1], [-0.1, 0.1], [0.2, 0.4])))
rng = np.random.default_rng(0)


def observe(P):
    pixels = np.hstack([cube, np.ones((len(cube), 1))]) @ P.T
    return pixels[:, :2] / pixels[:, 2:] + rng.normal(0.0, 0.5, (len(cube), 2))


left, right = observe(P_left), observe(P_right)
reconstructed = triangulate_points(P_left, P_right, left, right)
errors_mm = np.linalg.norm(reconstructed - cube, axis=1) * 1000
print(
    f"Erreur 3D : moyenne {errors_mm.mean():.1f} mm, max {errors_mm.max():.1f} mm "
    f"(bruit image 0,5 px, base {baseline * 100:.0f} cm)"
)
print(
    f"Erreur de reprojection moyenne : {reprojection_error(P_left, reconstructed, left).mean():.2f} px"
)
