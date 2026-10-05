import numpy as np
import pytest

from estimation import (
    EKFLocalizer,
    ParticleFilter,
    motion_jacobians,
    motion_model,
    range_bearing,
    range_bearing_jacobian,
    simulate_run,
)
from robocore import numerical_jacobian, wrap_angle

LANDMARKS = np.array([[2.0, 1.0], [-1.5, 3.5], [4.0, 5.0], [0.5, 6.5], [-3.0, 0.5]])
MOTION_NOISE = (0.05, 0.03)
MEAS_NOISE = (0.1, 0.03)


def test_analytic_jacobians_match_finite_differences():
    pose = np.array([0.4, -0.3, 0.8])
    F, V = motion_jacobians(pose, 0.7, 0.4, 0.1)
    np.testing.assert_allclose(
        F, numerical_jacobian(lambda p: motion_model(p, 0.7, 0.4, 0.1), pose), atol=1e-8
    )
    np.testing.assert_allclose(
        V, numerical_jacobian(lambda u: motion_model(pose, u[0], u[1], 0.1), [0.7, 0.4]), atol=1e-8
    )
    H = range_bearing_jacobian(pose, LANDMARKS[0])
    np.testing.assert_allclose(
        H, numerical_jacobian(lambda p: range_bearing(p, LANDMARKS[0]), pose), atol=1e-7
    )


def run_ekf(run):
    ekf = EKFLocalizer(
        run.true_poses[0], np.eye(3) * 1e-4, motion_noise=MOTION_NOISE, measurement_noise=MEAS_NOISE
    )
    estimates = [ekf.x.copy()]
    for (v, w), seen in zip(run.odometry, run.measurements, strict=True):
        ekf.predict(v, w, run.dt)
        for i, z in seen:
            ekf.update(z, run.landmarks[i])
        estimates.append(ekf.x.copy())
    return np.array(estimates), ekf


def dead_reckoning(run):
    poses = [run.true_poses[0]]
    for v, w in run.odometry:
        poses.append(motion_model(poses[-1], v, w, run.dt))
    return np.array(poses)


def rmse(estimate, truth):
    return float(np.sqrt(np.mean(np.sum((estimate[:, :2] - truth[:, :2]) ** 2, axis=1))))


def test_ekf_beats_dead_reckoning():
    run = simulate_run(
        LANDMARKS,
        motion_noise=MOTION_NOISE,
        measurement_noise=MEAS_NOISE,
        rng=np.random.default_rng(1),
    )
    estimates, ekf = run_ekf(run)
    ekf_error = rmse(estimates, run.true_poses)
    odom_error = rmse(dead_reckoning(run), run.true_poses)
    assert ekf_error < 0.15
    assert ekf_error < 0.3 * odom_error
    heading_error = np.abs(wrap_angle(estimates[:, 2] - run.true_poses[:, 2]))
    assert heading_error.max() < 0.15
    np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)


def test_ekf_update_reduces_uncertainty():
    ekf = EKFLocalizer(
        [0.0, 0.0, 0.0], np.eye(3) * 0.5, motion_noise=MOTION_NOISE, measurement_noise=MEAS_NOISE
    )
    before = np.trace(ekf.P)
    ekf.update(range_bearing(ekf.x, LANDMARKS[0]), LANDMARKS[0])
    assert np.trace(ekf.P) < before


def test_particle_filter_global_localization():
    rng = np.random.default_rng(3)
    run = simulate_run(
        LANDMARKS, steps=150, motion_noise=MOTION_NOISE, measurement_noise=MEAS_NOISE, rng=rng
    )
    pf = ParticleFilter.uniform(
        3000,
        ((-4.0, 5.0), (-1.0, 7.0)),
        motion_noise=(0.1, 0.06),
        measurement_noise=(0.3, 0.1),
        rng=rng,
    )
    for (v, w), seen in zip(run.odometry, run.measurements, strict=True):
        pf.predict(v, w, run.dt)
        for i, z in seen:
            pf.update(z, run.landmarks[i])
        if pf.effective_sample_size() < 0.5 * len(pf.particles):
            pf.resample()
    estimate = pf.estimate()
    assert np.linalg.norm(estimate[:2] - run.true_poses[-1, :2]) < 0.3
    assert abs(wrap_angle(estimate[2] - run.true_poses[-1, 2])) < 0.2


def test_resample_keeps_particle_count_and_resets_weights():
    pf = ParticleFilter(
        np.zeros((10, 3)),
        motion_noise=(0.1, 0.1),
        measurement_noise=(0.1, 0.1),
        rng=np.random.default_rng(0),
    )
    pf.log_weights = np.log(np.linspace(0.01, 1.0, 10))
    pf.resample()
    assert pf.particles.shape == (10, 3)
    assert pf.effective_sample_size() == pytest.approx(10.0)
