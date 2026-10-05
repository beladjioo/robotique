import numpy as np
import pytest

from manipulation import (
    QuinticTrajectory,
    TrapezoidalProfile,
    planar_2r,
    planar_3r,
    solve_ik,
    ur5e,
)
from robocore import numerical_jacobian, so3_log


def test_planar_2r_forward_kinematics_matches_closed_form():
    chain = planar_2r(1.0, 0.5)
    q = np.array([0.3, -0.8])
    p = chain.forward_kinematics(q)[:3, 3]
    expected = [np.cos(0.3) + 0.5 * np.cos(-0.5), np.sin(0.3) + 0.5 * np.sin(-0.5), 0.0]
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_jacobian_matches_finite_differences():
    chain = ur5e()
    q = np.array([0.1, -1.2, 1.4, -0.3, 0.9, 0.2])
    J = chain.jacobian(q)
    J_pos = numerical_jacobian(lambda x: chain.forward_kinematics(x)[:3, 3], q)
    np.testing.assert_allclose(J[:3], J_pos, atol=1e-7)

    R0 = chain.forward_kinematics(q)[:3, :3]

    def rotation_delta(x):
        return so3_log(chain.forward_kinematics(x)[:3, :3] @ R0.T)

    J_rot = numerical_jacobian(rotation_delta, q)
    np.testing.assert_allclose(J[3:], J_rot, atol=1e-6)


def test_ik_full_pose_on_ur5e():
    chain = ur5e()
    q_true = np.array([0.4, -1.0, 1.2, -0.7, 1.1, 0.3])
    target = chain.forward_kinematics(q_true)
    result = solve_ik(chain, target, q_true + 0.3)
    assert result.success
    np.testing.assert_allclose(chain.forward_kinematics(result.q), target, atol=1e-3)


def test_ik_position_only_on_redundant_arm():
    chain = planar_3r()
    result = solve_ik(chain, [1.2, 0.9, 0.0], np.zeros(3))
    assert result.success
    np.testing.assert_allclose(
        chain.forward_kinematics(result.q)[:3, 3], [1.2, 0.9, 0.0], atol=1e-4
    )


def test_ik_reports_failure_for_unreachable_target():
    result = solve_ik(planar_2r(), [3.0, 0.0, 0.0], [0.1, 0.1], max_iterations=100)
    assert not result.success
    assert result.position_error == pytest.approx(1.0, abs=1e-2)


def test_ik_respects_joint_limits():
    chain = planar_2r()
    chain.joint_limits[:] = [[-0.5, 0.5], [-0.5, 0.5]]
    result = solve_ik(chain, [0.0, 2.0, 0.0], [0.0, 0.0], max_iterations=50)
    assert chain.within_limits(result.q)


def test_manipulability_vanishes_at_singularity():
    chain = planar_2r()
    assert chain.manipulability([0.3, 0.0], position_only=True) == pytest.approx(0.0, abs=1e-9)
    assert chain.manipulability([0.3, np.pi / 2], position_only=True) == pytest.approx(1.0)


def test_quintic_boundary_conditions():
    traj = QuinticTrajectory([0.0, 1.0], [1.0, -1.0], duration=2.0, v0=[0.5, 0.0])
    q, qd, qdd = traj.sample(0.0)
    np.testing.assert_allclose(q, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(qd, [0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(qdd, [0.0, 0.0], atol=1e-12)
    q, qd, qdd = traj.sample(2.0)
    np.testing.assert_allclose(q, [1.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(qd, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(qdd, [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("distance", [5.0, -5.0, 0.2])
def test_trapezoidal_profile(distance):
    profile = TrapezoidalProfile(distance, v_max=1.0, a_max=2.0)
    s_end, v_end, _ = profile.sample(profile.duration + 1.0)
    assert s_end == pytest.approx(distance)
    assert v_end == 0.0
    speeds = [abs(profile.sample(t)[1]) for t in np.linspace(0, profile.duration, 200)]
    assert max(speeds) <= 1.0 + 1e-12
