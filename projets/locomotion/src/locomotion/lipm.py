"""Pendule inversé linéaire (LIPM) et marche par contrôle du DCM.

Le centre de masse (CdM) reste à hauteur constante ``z_c`` ; dans le plan sagittal :
``x'' = omega^2 (x - p)`` avec ``omega = sqrt(g / z_c)`` et ``p`` le point d'appui (ZMP).
La composante divergente du mouvement (DCM, ou point de capture) ``xi = x + x'/omega``
vérifie ``xi' = omega (xi - p)`` : c'est elle qu'il faut contrôler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LinearInvertedPendulum:
    com_height: float
    gravity: float = 9.81

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.gravity / self.com_height))

    def propagate(self, x0: float, v0: float, zmp: float, t: float) -> tuple[float, float]:
        """Solution analytique ``(x(t), v(t))`` à point d'appui fixe."""
        w = self.omega
        c, s = np.cosh(w * t), np.sinh(w * t)
        return zmp + (x0 - zmp) * c + v0 / w * s, (x0 - zmp) * w * s + v0 * c

    def capture_point(self, x: float, v: float) -> float:
        """Où poser le pied pour arrêter complètement le CdM."""
        return x + v / self.omega

    def orbital_energy(self, x: float, v: float, zmp: float) -> float:
        """Énergie orbitale (conservée à appui fixe) ; négative : le CdM ne franchit pas l'appui."""
        return 0.5 * v**2 - 0.5 * self.omega**2 * (x - zmp) ** 2


@dataclass
class WalkResult:
    times: np.ndarray
    com: np.ndarray
    com_velocity: np.ndarray
    footsteps: list[float] = field(default_factory=list)


def dcm_walk(
    lipm: LinearInvertedPendulum,
    *,
    step_length: float,
    step_duration: float,
    n_steps: int,
    x0: float = 0.0,
    v0: float = 0.0,
    push: tuple[float, float] | None = None,
    samples_per_step: int = 50,
) -> WalkResult:
    """Marche sagittale : chaque pied est posé à ``xi_fin - b`` avec ``b = L / (e^{omega T} - 1)``.

    Ce décalage constant produit une marche périodique de pas ``L`` ; une poussée
    ``push = (instant, delta_v)`` est absorbée automatiquement par le placement des pas suivants.
    """
    w = lipm.omega
    offset = step_length / (np.exp(w * step_duration) - 1.0)
    x, v = x0, v0
    foot = lipm.capture_point(x, v) - offset
    footsteps, times, com, vel = [foot], [], [], []
    dt = step_duration / samples_per_step
    pushed = push is None
    for step in range(n_steps):
        for k in range(samples_per_step):
            t = (step * samples_per_step + k) * dt
            if not pushed and t >= push[0]:
                v += push[1]
                pushed = True
            times.append(t)
            com.append(x)
            vel.append(v)
            x, v = lipm.propagate(x, v, foot, dt)
        foot = lipm.capture_point(x, v) - offset
        footsteps.append(foot)
    return WalkResult(np.array(times), np.array(com), np.array(vel), footsteps)
