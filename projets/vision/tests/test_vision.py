import numpy as np
import pytest

from robocore import invert_homogeneous
from vision import (
    PinholeCamera,
    apply_homography,
    estimate_homography,
    look_at,
    projection_matrix,
    reprojection_error,
    triangulate_points,
)

RNG = np.random.default_rng(0)


def test_project_backproject_roundtrip_with_distortion():
    cam = PinholeCamera(600, 610, 320, 240, 640, 480, k1=-0.2, k2=0.05)
    points = np.column_stack(
        [RNG.uniform(-1, 1, 50), RNG.uniform(-0.7, 0.7, 50), RNG.uniform(2, 6, 50)]
    )
    uv, visible = cam.project(points)
    assert visible.all()
    np.testing.assert_allclose(cam.backproject(uv, points[:, 2]), points, atol=1e-9)


def test_points_behind_camera_are_not_visible():
    cam = PinholeCamera.from_fov(640, 480, np.deg2rad(90))
    _, visible = cam.project([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
    assert visible.tolist() == [False, True, False]


def test_homography_is_recovered_exactly():
    H_true = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, -12.0], [1e-4, 2e-4, 1.0]])
    src = RNG.uniform(0, 640, size=(20, 2))
    dst = apply_homography(H_true, src)
    np.testing.assert_allclose(estimate_homography(src, dst), H_true, rtol=1e-6, atol=1e-9)


def test_look_at_centers_target():
    cam = PinholeCamera.from_fov(640, 480, np.deg2rad(70))
    T_world_cam = look_at(eye=[3.0, -2.0, 1.5], target=[0.0, 0.0, 0.5])
    target_cam = invert_homogeneous(T_world_cam) @ np.array([0.0, 0.0, 0.5, 1.0])
    uv, visible = cam.project(target_cam[:3])
    assert visible[0]
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-9)
    # L'axe y de la caméra pointe vers le bas du monde.
    assert T_world_cam[2, 1] < 0


def test_stereo_triangulation():
    cam = PinholeCamera.from_fov(640, 480, np.deg2rad(70))
    P1 = projection_matrix(cam.K, look_at([0.0, -3.0, 1.0], [0.0, 0.0, 0.5]))
    P2 = projection_matrix(cam.K, look_at([1.0, -3.0, 1.0], [0.0, 0.0, 0.5]))
    points = RNG.uniform(-0.5, 0.5, size=(30, 3)) + [0.0, 0.0, 0.5]
    x1 = np.hstack([points, np.ones((30, 1))]) @ P1.T
    x2 = np.hstack([points, np.ones((30, 1))]) @ P2.T
    x1, x2 = x1[:, :2] / x1[:, 2:], x2[:, :2] / x2[:, 2:]
    noisy1 = x1 + RNG.normal(0, 0.3, x1.shape)
    noisy2 = x2 + RNG.normal(0, 0.3, x2.shape)
    estimated = triangulate_points(P1, P2, noisy1, noisy2)
    assert np.median(np.linalg.norm(estimated - points, axis=1)) < 0.02
    assert reprojection_error(P1, estimated, noisy1).mean() < 1.0
    np.testing.assert_allclose(triangulate_points(P1, P2, x1, x2), points, atol=1e-8)


def test_homography_requires_four_points():
    with pytest.raises(ValueError):
        estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)))
