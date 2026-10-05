"""Algorithmes d'apprentissage : Q-learning tabulaire et méthode de l'entropie croisée."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from apprentissage.envs import GridWorld


def q_learning(
    env: GridWorld,
    *,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Q-learning epsilon-glouton. Retourne la table Q et le retour de chaque épisode."""
    rng = rng if rng is not None else np.random.default_rng()
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:  # départage aléatoire des ex aequo
                action = int(rng.choice(np.flatnonzero(Q[state] == Q[state].max())))
            next_state, reward, terminated, truncated, _ = env.step(action)
            target = reward + (0.0 if terminated else gamma * Q[next_state].max())
            Q[state, action] += alpha * (target - Q[state, action])
            state, total, done = next_state, total + reward, terminated or truncated
        returns.append(total)
    return Q, returns


def greedy_rollout(env: GridWorld, Q: np.ndarray) -> list[int]:
    """Suit la politique gloutonne ; retourne la suite d'états visités."""
    state, _ = env.reset()
    states, done = [state], False
    while not done:
        state, _, terminated, truncated, _ = env.step(int(np.argmax(Q[state])))
        states.append(state)
        done = terminated or truncated
    return states


class LinearPolicy:
    """Politique déterministe ``a = clip(W o + b, -1, 1)`` à paramètres aplatis."""

    def __init__(self, observation_dim: int, action_dim: int, params: np.ndarray | None = None):
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.n_params = action_dim * (observation_dim + 1)
        self.params = np.zeros(self.n_params) if params is None else np.asarray(params, dtype=float)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        W = self.params[: self.action_dim * self.observation_dim].reshape(
            self.action_dim, self.observation_dim
        )
        b = self.params[self.action_dim * self.observation_dim :]
        return np.clip(W @ observation + b, -1.0, 1.0)


def evaluate_policy(
    env, policy: Callable[[np.ndarray], np.ndarray], *, episodes: int, seed: int
) -> float:
    """Retour moyen sur ``episodes`` épisodes à graines fixées (évaluation reproductible)."""
    total = 0.0
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
    return total / episodes


def cross_entropy_method(
    objective: Callable[[np.ndarray], float],
    dim: int,
    *,
    iterations: int = 30,
    population: int = 40,
    elite_fraction: float = 0.2,
    initial_std: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Optimisation boîte noire : échantillonne, garde l'élite, réajuste la gaussienne.

    Retourne la moyenne finale et l'historique du meilleur score par itération.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mean, std = np.zeros(dim), np.full(dim, initial_std)
    n_elite = max(1, int(population * elite_fraction))
    history = []
    for _ in range(iterations):
        candidates = mean + std * rng.normal(size=(population, dim))
        scores = np.array([objective(c) for c in candidates])
        elite = candidates[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        history.append(float(scores.max()))
    return mean, history
