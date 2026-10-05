import numpy as np
import pytest

from locomotion import (
    GAITS,
    HopfCPG,
    LinearInvertedPendulum,
    dcm_walk,
    foot_trajectory,
    leg_forward_kinematics,
    leg_inverse_kinematics,
)


def test_capture_point_brings_com_to_rest():
    lipm = LinearInvertedPendulum(com_height=0.8)
    x0, v0 = 0.0, 0.6
    foot = lipm.capture_point(x0, v0)
    x, v = lipm.propagate(x0, v0, foot, 3.0)
    assert x == pytest.approx(foot, abs=1e-3)
    assert v == pytest.approx(0.0, abs=1e-3)
    assert lipm.orbital_energy(x0, v0, foot) == pytest.approx(0.0, abs=1e-12)


def test_lipm_propagation_matches_numerical_integration():
    lipm = LinearInvertedPendulum(com_height=0.9)
    x, v, dt = 0.05, 0.2, 1e-5
    for _ in range(int(0.4 / dt)):
        v += lipm.omega**2 * (x - 0.0) * dt
        x += v * dt
    assert lipm.propagate(0.05, 0.2, 0.0, 0.4) == pytest.approx((x, v), rel=1e-3)


def test_dcm_walk_is_periodic():
    lipm = LinearInvertedPendulum(com_height=0.8)
    walk = dcm_walk(lipm, step_length=0.3, step_duration=0.5, n_steps=10)
    np.testing.assert_allclose(np.diff(walk.footsteps), 0.3, atol=1e-9)
    mean_speed = (walk.com[-1] - walk.com[0]) / (walk.times[-1] - walk.times[0])
    assert mean_speed == pytest.approx(0.3 / 0.5, rel=0.1)


def test_dcm_walk_recovers_from_push():
    lipm = LinearInvertedPendulum(com_height=0.8)
    walk = dcm_walk(lipm, step_length=0.3, step_duration=0.5, n_steps=10, push=(2.2, 0.4))
    steps = np.diff(walk.footsteps)
    assert steps[4] > 0.35  # le pas qui suit la poussée s'allonge pour la rattraper
    np.testing.assert_allclose(steps[6:], 0.3, atol=1e-9)  # puis la marche nominale reprend
    # Le CdM reste toujours à distance raisonnable du pied d'appui (pas de chute).
    assert np.max(np.abs(np.diff(walk.com))) < 0.1


@pytest.mark.parametrize("gait", ["trot", "bond", "marche"])
def test_cpg_converges_to_gait(gait):
    cpg = HopfCPG(gait, rng=np.random.default_rng(1))
    for _ in range(2000):
        cpg.step(0.005)
    assert cpg.phase_error() < 0.05
    np.testing.assert_allclose(np.abs(cpg.z), 1.0, rtol=0.02)


def test_cpg_gait_transition():
    cpg = HopfCPG("trot", rng=np.random.default_rng(2))
    for _ in range(4000):  # trot parfaitement convergé : cas le plus difficile pour la transition
        cpg.step(0.005)
    assert cpg.phase_error() < 1e-6
    for gait in ("amble", "bond", "marche", "trot"):
        assert gait in GAITS
        cpg.set_gait(gait)
        for _ in range(1000):
            cpg.step(0.005)
        assert cpg.phase_error() < 0.05, gait


def test_leg_inverse_kinematics_roundtrip():
    for x, z in [(0.05, -0.3), (-0.1, -0.25), (0.0, -0.38)]:
        for knee_forward in (False, True):
            hip, knee = leg_inverse_kinematics(x, z, 0.2, 0.2, knee_forward=knee_forward)
            np.testing.assert_allclose(
                leg_forward_kinematics(hip, knee, 0.2, 0.2), [x, z], atol=1e-12
            )
    with pytest.raises(ValueError):
        leg_inverse_kinematics(0.0, -0.5, 0.2, 0.2)


def test_foot_trajectory_is_continuous():
    phases = np.linspace(0, 2 * np.pi, 2001)
    points = np.array(
        [foot_trajectory(p, step_length=0.1, step_height=0.05, stance_height=0.3) for p in phases]
    )
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 1e-3
    assert points[:, 1].max() == pytest.approx(-0.25, abs=1e-4)
