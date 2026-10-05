import numpy as np
import pytest

from robocore import (
    angle_diff,
    expm,
    homogeneous,
    invert_homogeneous,
    matrix_to_quat,
    matrix_to_rpy,
    min_jerk,
    numerical_jacobian,
    quat_to_matrix,
    rk4_step,
    rpy_to_matrix,
    se2,
    se2_to_pose,
    skew,
    so3_exp,
    so3_log,
    wrap_angle,
)

RNG = np.random.default_rng(0)


def random_rotation():
    q = RNG.normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def test_wrap_angle_scalar_and_array():
    assert wrap_angle(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    out = wrap_angle(np.array([2 * np.pi, -3 * np.pi / 2]))
    np.testing.assert_allclose(out, [0.0, np.pi / 2], atol=1e-12)
    assert angle_diff(np.pi - 0.1, -np.pi + 0.1) == pytest.approx(-0.2)


def test_se2_roundtrip():
    pose = (1.0, -2.0, 0.7)
    assert se2_to_pose(se2(*pose)) == pytest.approx(pose)


@pytest.mark.parametrize("_", range(5))
def test_rotation_conversions_roundtrip(_):
    R = random_rotation()
    np.testing.assert_allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-12)
    np.testing.assert_allclose(rpy_to_matrix(*matrix_to_rpy(R)), R, atol=1e-12)
    np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_so3_log_near_pi():
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    for angle in (np.pi, np.pi - 1e-7):
        R = so3_exp(axis * angle)
        np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-6)


def test_invert_homogeneous():
    T = homogeneous(random_rotation(), [0.3, -1.0, 2.0])
    np.testing.assert_allclose(T @ invert_homogeneous(T), np.eye(4), atol=1e-12)


def test_rk4_exponential_decay():
    x = np.array([1.0])
    for _ in range(100):
        x = rk4_step(lambda s, u: -s, x, None, 0.01)
    assert x[0] == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_expm_matches_rodrigues_and_scalar():
    omega = np.array([0.3, -1.2, 2.5])
    np.testing.assert_allclose(expm(skew(omega)), so3_exp(omega), atol=1e-10)
    assert expm([[3.0]])[0, 0] == pytest.approx(np.exp(3.0), rel=1e-10)


def test_numerical_jacobian():
    def f(x):
        return np.array([x[0] ** 2 * x[1], np.sin(x[1])])

    x = np.array([1.5, 0.3])
    expected = np.array([[2 * x[0] * x[1], x[0] ** 2], [0.0, np.cos(x[1])]])
    np.testing.assert_allclose(numerical_jacobian(f, x), expected, atol=1e-8)


def test_min_jerk_boundaries():
    s, ds, dds = min_jerk(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ds[[0, 2]], 0.0)
    np.testing.assert_allclose(dds[[0, 2]], 0.0)
