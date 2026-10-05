import numpy as np
import pytest

from commande import PID, CartPole, DCMotor, discretize, dlqr, is_controllable, linearize, lqr
from robocore import rk4_step


def test_scalar_riccati_known_solutions():
    K, P = lqr([[0.0]], [[1.0]], [[1.0]], [[1.0]])
    assert P[0, 0] == pytest.approx(1.0) and K[0, 0] == pytest.approx(1.0)
    K, P = dlqr([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    golden = (1 + np.sqrt(5)) / 2
    assert P[0, 0] == pytest.approx(golden) and K[0, 0] == pytest.approx(golden / (1 + golden))


def test_cartpole_linearization_matches_analytic():
    pole = CartPole()
    A, B = linearize(pole.dynamics, np.zeros(4), [0.0])
    A_ref, B_ref = pole.linearized()
    np.testing.assert_allclose(A, A_ref, atol=1e-6)
    np.testing.assert_allclose(B, B_ref, atol=1e-6)
    assert is_controllable(A, B)


def test_dlqr_satisfies_riccati_and_stabilizes():
    A, B = CartPole().linearized()
    Ad, Bd = discretize(A, B, 0.01)
    Q, R = np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]])
    K, P = dlqr(Ad, Bd, Q, R)
    residual = (
        Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad) + Q
    )
    assert np.abs(residual).max() < 1e-6 * np.abs(P).max()
    assert np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))) < 1.0


def test_continuous_and_discrete_lqr_agree_for_small_dt():
    A, B = CartPole().linearized()
    Q, R = np.eye(4), np.array([[1.0]])
    K_c, _ = lqr(A, B, Q, R)
    dt = 1e-3
    Ad, Bd = discretize(A, B, dt)
    K_d, _ = dlqr(Ad, Bd, Q * dt, R * dt)
    np.testing.assert_allclose(K_d, K_c, rtol=2e-2)


def test_lqr_balances_nonlinear_cartpole():
    pole = CartPole()
    A, B = pole.linearized()
    K, _ = lqr(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    x = np.array([0.0, 0.0, 0.25, 0.0])
    dt = 0.01
    for _ in range(1000):
        u = -K @ x
        x = rk4_step(pole.dynamics, x, u, dt)
    np.testing.assert_allclose(x, 0.0, atol=1e-3)


def test_pid_speed_control_of_dc_motor():
    motor = DCMotor()
    pid = PID(kp=0.05, ki=2.0, output_limits=(-12.0, 12.0))
    x, dt = np.zeros(2), 1e-4
    for _ in range(int(1.0 / dt)):
        voltage = pid.update(100.0, x[0], dt)
        x = rk4_step(motor.dynamics, x, [voltage], dt)
    assert x[0] == pytest.approx(100.0, rel=0.01)


def test_pid_anti_windup_limits_integral():
    pid = PID(kp=1.0, ki=10.0, output_limits=(-1.0, 1.0))
    for _ in range(1000):
        out = pid.update(100.0, 0.0, 0.01)  # consigne inatteignable : sortie saturée
    assert out == 1.0
    assert abs(pid.integral) <= 1.0 + 1e-9
    # Dès que l'erreur change de signe, la sortie quitte la saturation immédiatement.
    assert pid.update(0.0, 5.0, 0.01) < 0.0


def test_pid_has_no_derivative_kick_on_setpoint_change():
    pid = PID(kp=0.0, kd=1.0)
    pid.update(0.0, 0.0, 0.01)
    assert pid.update(10.0, 0.0, 0.01) == 0.0
