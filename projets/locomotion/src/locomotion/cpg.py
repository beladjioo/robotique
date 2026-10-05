"""Générateurs centraux de rythme (CPG) à oscillateurs de Hopf couplés.

Chaque patte est un oscillateur complexe ``z = r e^{i phi}`` :
``z' = (alpha (mu - |z|^2) + i omega) z + k sum_j (e^{i d_ij} z_j - z_i)``
où ``d_ij = phi_i* - phi_j*`` est le déphasage désiré. Le cycle limite a pour amplitude
``sqrt(mu)`` et les déphasages convergent vers ceux de l'allure choisie.
"""

from __future__ import annotations

import numpy as np

from robocore import rk4_step, wrap_angle

# Phases des pattes dans l'ordre (avant-gauche, avant-droite, arrière-gauche, arrière-droite).
GAITS: dict[str, tuple[float, float, float, float]] = {
    "marche": (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi),
    "trot": (0.0, np.pi, np.pi, 0.0),
    "amble": (0.0, np.pi, 0.0, np.pi),
    "bond": (0.0, 0.0, np.pi, np.pi),
}


class HopfCPG:
    def __init__(
        self,
        gait: str = "trot",
        *,
        frequency: float = 1.5,
        amplitude: float = 1.0,
        convergence: float = 20.0,
        coupling: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.frequency = frequency
        self.mu = amplitude**2
        self.alpha = convergence
        self.coupling = coupling
        rng = rng if rng is not None else np.random.default_rng()
        self.z = 0.1 * (rng.normal(size=4) + 1j * rng.normal(size=4))
        self._target = np.asarray(GAITS[gait], dtype=float)
        self.set_gait(gait, transition_time=0.0)

    def set_gait(self, gait: str, *, transition_time: float = 1.0) -> None:
        """Change d'allure à la volée.

        Les déphasages désirés glissent continûment de l'allure courante vers la nouvelle
        pendant ``transition_time`` secondes. Un changement instantané pourrait laisser le
        réseau bloqué : les déphasages de 0 ou pi d'une allure sont des équilibres (instables)
        du couplage d'une autre allure.
        """
        self.gait = gait
        target = np.asarray(GAITS[gait], dtype=float)
        self._start = self._target.copy()
        self._target = self._start + wrap_angle(target - self._start)  # chemin le plus court
        self._transition_time = transition_time
        self._elapsed = 0.0
        self._update_coupling(1.0 if transition_time <= 0 else 0.0)

    def _update_coupling(self, progress: float) -> None:
        phases = (1.0 - progress) * self._start + progress * self._target
        delta = phases[:, None] - phases[None, :]
        self._coupling_matrix = np.exp(1j * delta) * (1 - np.eye(len(phases)))

    def derivative(self, z: np.ndarray, _u=None) -> np.ndarray:
        radius2 = np.abs(z) ** 2
        intrinsic = (self.alpha * (self.mu - radius2) + 1j * 2 * np.pi * self.frequency) * z
        coupling = self._coupling_matrix @ z - (len(z) - 1) * z
        return intrinsic + self.coupling * coupling

    def step(self, dt: float) -> np.ndarray:
        if self._elapsed < self._transition_time:
            self._elapsed += dt
            self._update_coupling(min(1.0, self._elapsed / self._transition_time))
        self.z = rk4_step(self.derivative, self.z, None, dt)
        return self.outputs()

    def outputs(self) -> np.ndarray:
        """Signal rythmique de chaque patte (partie réelle), à mapper vers les articulations."""
        return self.z.real.copy()

    def phases(self) -> np.ndarray:
        return np.angle(self.z)

    def relative_phases(self) -> np.ndarray:
        """Phases relatives à la patte avant-gauche, dans [0, 2 pi)."""
        return np.mod(self.phases() - self.phases()[0], 2 * np.pi)

    def phase_error(self) -> float:
        """Écart maximal entre déphasages courants et désirés (rad)."""
        desired = np.asarray(GAITS[self.gait])
        return float(np.max(np.abs(wrap_angle(self.relative_phases() - (desired - desired[0])))))
