"""Environnements minimalistes respectant l'API Gymnasium.

``reset(seed=None) -> (observation, info)`` et
``step(action) -> (observation, récompense, terminé, tronqué, info)`` : ils pourront être
enveloppés tels quels dans ``gymnasium.Env`` quand la branche passera à MuJoCo / Isaac Lab.
"""

from __future__ import annotations

import numpy as np


class GridWorld:
    """Labyrinthe discret. ``S`` départ, ``G`` but, ``#`` mur. Récompense -1 par pas."""

    ACTIONS = ((1, 0), (-1, 0), (0, -1), (0, 1))  # bas, haut, gauche, droite (lignes du texte)

    def __init__(self, layout: str, *, max_steps: int = 100) -> None:
        rows = [line.strip() for line in layout.strip().splitlines()]
        self.walls = np.array([[ch == "#" for ch in row] for row in rows])
        self.n_rows, self.n_cols = self.walls.shape
        self.start = next(
            (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "S"
        )
        self.goal = next(
            (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "G"
        )
        self.max_steps = max_steps
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(self.ACTIONS)
        self.position = self.start
        self._steps = 0

    def encode(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.n_cols + cell[1]

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        self.position = self.start
        self._steps = 0
        return self.encode(self.position), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        dr, dc = self.ACTIONS[action]
        r, c = self.position[0] + dr, self.position[1] + dc
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols and not self.walls[r, c]:
            self.position = (r, c)
        self._steps += 1
        terminated = self.position == self.goal
        truncated = not terminated and self._steps >= self.max_steps
        return self.encode(self.position), -1.0, terminated, truncated, {}


class PointMassReach:
    """Masse ponctuelle 2D poussée par une force bornée vers une cible aléatoire.

    Observation : ``(cible - position, vitesse)`` ; action : force dans [-1, 1]^2.
    Avec ``randomize=True``, masse et frottement sont tirés à chaque épisode
    (randomisation de domaine : la politique apprise doit être robuste au modèle).
    """

    observation_dim = 4
    action_dim = 2

    def __init__(self, *, randomize: bool = False, dt: float = 0.1, max_steps: int = 100) -> None:
        self.randomize = randomize
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.mass, self.damping = 1.0, 0.1

    def _observation(self) -> np.ndarray:
        return np.concatenate([self.goal - self.position, self.velocity])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.goal = self.rng.uniform(-1.0, 1.0, size=2)
        if self.randomize:
            self.mass = self.rng.uniform(0.5, 2.0)
            self.damping = self.rng.uniform(0.0, 0.5)
        self._steps = 0
        return self._observation(), {"mass": self.mass, "damping": self.damping}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        force = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.velocity += self.dt * (force / self.mass - self.damping * self.velocity)
        self.position += self.dt * self.velocity
        self._steps += 1
        distance = float(np.linalg.norm(self.goal - self.position))
        reward = -distance - 0.01 * float(force @ force)
        truncated = self._steps >= self.max_steps
        return self._observation(), reward, False, truncated, {"distance": distance}
