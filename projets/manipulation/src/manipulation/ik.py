"""Cinématique inverse numérique par moindres carrés amortis (Levenberg-Marquardt / DLS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from manipulation.kinematics import SerialChain
from robocore import so3_log


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Erreur de pose 6D ``[dp; dw]`` exprimée dans le repère de base.

    ``dw`` est le vecteur rotation qui amène l'orientation courante sur la cible.
    """
    dp = target[:3, 3] - current[:3, 3]
    dw = so3_log(target[:3, :3] @ current[:3, :3].T)
    return np.concatenate([dp, dw])


def solve_ik(
    chain: SerialChain,
    target: ArrayLike,
    q0: ArrayLike,
    *,
    position_only: bool = False,
    damping: float = 0.05,
    max_iterations: int = 300,
    position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-3,
    max_step: float = 0.5,
) -> IKResult:
    """Résout la cinématique inverse à partir de l'estimation initiale ``q0``.

    ``target`` est une pose 4x4, ou une position 3D (implique ``position_only=True``).
    Le pas ``dq = J^T (J J^T + lambda^2 I)^-1 e`` reste bien défini près des singularités ;
    il est borné par ``max_step`` et les butées articulaires sont respectées.
    """
    target = np.asarray(target, dtype=float)
    if target.shape == (3,):
        position = target
        target = np.eye(4)
        target[:3, 3] = position
        position_only = True
    if target.shape != (4, 4):
        raise ValueError("target doit être une pose 4x4 ou une position 3D")

    q = chain.clip_to_limits(q0)
    pos_err = rot_err = np.inf
    for iteration in range(max_iterations + 1):
        error = pose_error(chain.forward_kinematics(q), target)
        pos_err = float(np.linalg.norm(error[:3]))
        rot_err = float(np.linalg.norm(error[3:]))
        if pos_err < position_tolerance and (position_only or rot_err < orientation_tolerance):
            return IKResult(q, True, iteration, pos_err, rot_err)
        if iteration == max_iterations:
            break
        J = chain.jacobian(q)
        if position_only:
            J, error = J[:3], error[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), error)
        step = np.linalg.norm(dq)
        if step > max_step:
            dq *= max_step / step
        q = chain.clip_to_limits(q + dq)
    return IKResult(q, False, max_iterations, pos_err, rot_err)
