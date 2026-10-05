"""Modèles de robots prêts à l'emploi."""

from __future__ import annotations

import numpy as np

from manipulation.kinematics import DHLink, SerialChain


def planar_2r(l1: float = 1.0, l2: float = 1.0) -> SerialChain:
    """Bras plan à deux articulations rotoïdes (cas d'école, solution analytique connue)."""
    return SerialChain([DHLink(a=l1, alpha=0.0), DHLink(a=l2, alpha=0.0)], name="plan-2R")


def planar_3r(l1: float = 1.0, l2: float = 0.8, l3: float = 0.3) -> SerialChain:
    """Bras plan redondant pour la position (3 articulations pour 2 coordonnées)."""
    return SerialChain(
        [DHLink(a=l1, alpha=0.0), DHLink(a=l2, alpha=0.0), DHLink(a=l3, alpha=0.0)], name="plan-3R"
    )


def ur5e() -> SerialChain:
    """Cobot 6 axes UR5e (paramètres DH standard publiés par Universal Robots)."""
    half_pi = np.pi / 2
    links = [
        DHLink(a=0.0, alpha=half_pi, d=0.1625),
        DHLink(a=-0.425, alpha=0.0),
        DHLink(a=-0.3922, alpha=0.0),
        DHLink(a=0.0, alpha=half_pi, d=0.1333),
        DHLink(a=0.0, alpha=-half_pi, d=0.0997),
        DHLink(a=0.0, alpha=0.0, d=0.0996),
    ]
    limits = np.tile([-2 * np.pi, 2 * np.pi], (6, 1))
    return SerialChain(links, joint_limits=limits, name="UR5e")
