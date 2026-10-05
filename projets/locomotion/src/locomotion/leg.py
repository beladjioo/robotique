"""Cinématique d'une jambe plane à deux segments (hanche, genou) et trajectoire de pied."""

from __future__ import annotations

import numpy as np


def leg_forward_kinematics(hip: float, knee: float, thigh: float, shank: float) -> np.ndarray:
    """Position du pied ``(x, z)`` dans le repère hanche (z vers le haut, angles depuis la verticale basse)."""
    x = thigh * np.sin(hip) + shank * np.sin(hip + knee)
    z = -thigh * np.cos(hip) - shank * np.cos(hip + knee)
    return np.array([x, z])


def leg_inverse_kinematics(
    x: float, z: float, thigh: float, shank: float, *, knee_forward: bool = False
) -> tuple[float, float]:
    """Angles ``(hanche, genou)`` qui placent le pied en ``(x, z)``.

    Par défaut le genou plie « vers l'arrière » (``knee < 0``), comme chez un quadrupède
    à genoux inversés ; ``knee_forward=True`` donne l'autre solution.
    """
    d2 = x * x + z * z
    cos_knee = (d2 - thigh**2 - shank**2) / (2 * thigh * shank)
    if not -1.0 <= cos_knee <= 1.0:
        raise ValueError("position de pied hors d'atteinte")
    knee = np.arccos(cos_knee) * (1.0 if knee_forward else -1.0)
    hip = np.arctan2(x, -z) - np.arctan2(shank * np.sin(knee), thigh + shank * np.cos(knee))
    return float(hip), float(knee)


def foot_trajectory(
    phase: float, *, step_length: float, step_height: float, stance_height: float, duty: float = 0.5
) -> np.ndarray:
    """Pied ``(x, z)`` sur un cycle : appui (glissement arrière au sol) puis vol (arc vers l'avant).

    ``phase`` dans [0, 2 pi) ; ``duty`` est la fraction du cycle passée en appui.
    """
    s = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    if s < duty:  # appui : le pied recule à vitesse constante
        u = s / duty
        return np.array([step_length * (0.5 - u), -stance_height])
    u = (s - duty) / (1 - duty)  # vol : retour vers l'avant avec levée sinusoïdale
    x = step_length * (-0.5 + (u - np.sin(2 * np.pi * u) / (2 * np.pi)))
    return np.array([x, -stance_height + step_height * np.sin(np.pi * u)])
