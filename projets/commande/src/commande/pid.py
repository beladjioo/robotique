"""Régulateur PID « industriel »."""

from __future__ import annotations

import math


class PID:
    """PID à temps discret prêt pour le matériel.

    - dérivée calculée sur la **mesure** (pas de pic lors d'un changement de consigne) ;
    - filtre passe-bas du premier ordre sur la dérivée (constante ``derivative_tau``) ;
    - saturation de sortie et **anti-emballement** par intégration conditionnelle.

    Le même algorithme existe en C++ embarqué dans ``projets/embarque``.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        *,
        output_limits: tuple[float, float] = (-math.inf, math.inf),
        derivative_tau: float = 0.0,
    ) -> None:
        if output_limits[0] > output_limits[1]:
            raise ValueError("output_limits : borne basse > borne haute")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limits = output_limits
        self.derivative_tau = derivative_tau
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0  # terme intégral déjà multiplié par ki
        self._derivative = 0.0
        self._previous_measurement: float | None = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt doit être strictement positif")
        error = setpoint - measurement

        raw_derivative = 0.0
        if self._previous_measurement is not None:
            raw_derivative = -(measurement - self._previous_measurement) / dt
        self._previous_measurement = measurement
        alpha = dt / (self.derivative_tau + dt)
        self._derivative += alpha * (raw_derivative - self._derivative)

        proportional = self.kp * error
        derivative = self.kd * self._derivative
        candidate = self.integral + self.ki * error * dt
        low, high = self.output_limits
        unsaturated = proportional + candidate + derivative
        output = min(max(unsaturated, low), high)
        # On n'intègre que si la sortie n'est pas saturée, ou si l'erreur aide à en sortir.
        if (
            output == unsaturated
            or (unsaturated > high and error < 0)
            or (unsaturated < low and error > 0)
        ):
            self.integral = candidate
        else:
            output = min(max(proportional + self.integral + derivative, low), high)
        return output
