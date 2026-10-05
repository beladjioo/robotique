"""Branche commande : automatique pour la robotique.

Première pierre : PID industriel (anti-emballement, dérivée filtrée sur la mesure),
LQR continu et discret, discrétisation, et deux systèmes de référence :
moteur à courant continu et pendule inversé sur chariot.
"""

from commande.linear import controllability_matrix, discretize, is_controllable, linearize
from commande.lqr import dlqr, lqr
from commande.pid import PID
from commande.systems import CartPole, DCMotor

__all__ = [
    "PID",
    "CartPole",
    "DCMotor",
    "controllability_matrix",
    "discretize",
    "dlqr",
    "is_controllable",
    "linearize",
    "lqr",
]
