import numpy as np
import pytest

from drones import CascadedController, PlanarQuadrotor, Reference, WaypointTrajectory
from robocore import rk4_step


def simulate(trajectory, duration, x0=None, dt=0.002):
    quad = PlanarQuadrotor()
    ctrl = CascadedController(quad)
    x = np.zeros(6) if x0 is None else np.asarray(x0, dtype=float)
    states, thrusts = [x], []
    for k in range(int(duration / dt)):
        u = ctrl.compute(x, trajectory.sample(k * dt))
        thrusts.append(u[0])
        x = rk4_step(quad.dynamics, x, np.asarray(u), dt)
        states.append(x)
    return np.array(states), np.array(thrusts), ctrl


def test_hover_is_an_equilibrium():
    quad = PlanarQuadrotor()
    ctrl = CascadedController(quad)
    ref = Reference(np.array([0.0, 1.0]), np.zeros(2), np.zeros(2))
    u = ctrl.compute([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], ref)
    assert u == pytest.approx((quad.hover_thrust, 0.0))
    np.testing.assert_allclose(quad.dynamics([0, 1, 0, 0, 0, 0], u), 0.0, atol=1e-12)


def test_mixer_roundtrip():
    quad = PlanarQuadrotor()
    assert quad.unmix(*quad.mix(1.2, 0.01)) == pytest.approx((1.2, 0.01))


def test_waypoint_trajectory_is_smooth_and_stops_at_waypoints():
    traj = WaypointTrajectory([[0, 0], [1, 2], [3, 2]], segment_duration=2.0)
    ref = traj.sample(2.0)
    np.testing.assert_allclose(ref.position, [1, 2])
    np.testing.assert_allclose(ref.velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(traj.sample(10.0).position, [3, 2])


def test_tracks_waypoints_within_limits():
    traj = WaypointTrajectory([[0.0, 0.0], [1.0, 1.5], [2.0, 1.0]], segment_duration=3.0)
    states, thrusts, ctrl = simulate(traj, duration=8.0)
    np.testing.assert_allclose(states[-1, :2], [2.0, 1.0], atol=0.01)
    assert np.abs(states[:, 2]).max() < ctrl.max_tilt
    assert thrusts.min() >= 0.0
    # Erreur de suivi bornée tout au long du vol.
    t = np.arange(len(states)) * 0.002
    reference = np.array([traj.sample(ti).position for ti in t])
    assert np.linalg.norm(states[:, :2] - reference, axis=1).max() < 0.1


def test_recovers_from_initial_tilt():
    hold = WaypointTrajectory([[0.0, 1.0], [0.0, 1.0]], segment_duration=1.0)
    states, _, _ = simulate(hold, duration=4.0, x0=[0.0, 1.0, 0.4, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(states[-1], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=0.01)
