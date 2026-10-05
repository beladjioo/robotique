"""Démo : Q-learning dans un labyrinthe, puis CEM sur une tâche d'atteinte randomisée.

python projets/apprentissage/examples/demo_apprentissage.py
"""

import numpy as np

from apprentissage import (
    GridWorld,
    LinearPolicy,
    PointMassReach,
    cross_entropy_method,
    evaluate_policy,
    greedy_rollout,
    q_learning,
)

maze = GridWorld("""
S..#....
.#.#.##.
.#...#..
.####.#.
......#G
""")
Q, returns = q_learning(maze, episodes=400, rng=np.random.default_rng(0))
print(
    f"Q-learning : retour moyen {np.mean(returns[:20]):.0f} (début) -> {np.mean(returns[-20:]):.0f} (fin), "
    f"chemin glouton de {len(greedy_rollout(maze, Q)) - 1} pas"
)

env = PointMassReach(randomize=True)
policy = LinearPolicy(env.observation_dim, env.action_dim)


def objective(params):
    policy.params = params
    return evaluate_policy(env, policy, episodes=4, seed=0)


before = evaluate_policy(env, LinearPolicy(4, 2), episodes=50, seed=1000)
params, history = cross_entropy_method(
    objective, policy.n_params, iterations=25, population=40, rng=np.random.default_rng(0)
)
policy.params = params
after = evaluate_policy(env, policy, episodes=50, seed=1000)
print(
    f"CEM : retour {before:.1f} -> {after:.1f} sur 50 épisodes de test (masses et frottements aléatoires)"
)
W = params[:8].reshape(2, 4)
print("Politique apprise (gains sur [erreur_x, erreur_y, vx, vy]) :\n", np.round(W, 2))
