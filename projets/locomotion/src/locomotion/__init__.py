"""Branche locomotion : robots à pattes et humanoïdes.

Première pierre : pendule inversé linéaire (LIPM), point de capture et marche par
contrôle de la composante divergente du mouvement (DCM), générateurs centraux de rythme
(CPG) pour les allures quadrupèdes, cinématique de jambe plane.
"""

from locomotion.cpg import GAITS, HopfCPG
from locomotion.leg import foot_trajectory, leg_forward_kinematics, leg_inverse_kinematics
from locomotion.lipm import LinearInvertedPendulum, WalkResult, dcm_walk

__all__ = [
    "GAITS",
    "HopfCPG",
    "LinearInvertedPendulum",
    "WalkResult",
    "dcm_walk",
    "foot_trajectory",
    "leg_forward_kinematics",
    "leg_inverse_kinematics",
]
