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

MAZE = """
S..#....
.#.#.##.
.#...#..
.####.#.
......#G
"""


def test_q_learning_finds_shortest_path():
    env = GridWorld(MAZE)
    Q, returns = q_learning(env, episodes=400, rng=np.random.default_rng(0))
    states = greedy_rollout(env, Q)
    assert states[-1] == env.encode(env.goal)
    assert len(states) - 1 == 15  # longueur du plus court chemin (calculée à la main / BFS)
    assert np.mean(returns[-50:]) > np.mean(returns[:50])


def test_gridworld_api_and_walls():
    env = GridWorld(MAZE)
    state, info = env.reset()
    assert state == 0 and info == {}
    state, reward, terminated, truncated, _ = env.step(3)  # droite
    assert (state, reward, terminated, truncated) == (1, -1.0, False, False)
    env.reset()
    env.step(2)  # gauche : bord de la carte, le robot ne bouge pas
    assert env.position == env.start


def test_point_mass_domain_randomization():
    env = PointMassReach(randomize=True)
    masses = {round(env.reset(seed=s)[1]["mass"], 6) for s in range(10)}
    assert len(masses) == 10
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)


def test_cem_learns_reaching_policy():
    env = PointMassReach(randomize=True)
    policy = LinearPolicy(env.observation_dim, env.action_dim)

    def objective(params):
        policy.params = params
        return evaluate_policy(env, policy, episodes=4, seed=0)

    baseline = objective(np.zeros(policy.n_params))
    best, history = cross_entropy_method(
        objective, policy.n_params, iterations=15, population=30, rng=np.random.default_rng(0)
    )
    policy.params = best
    learned = evaluate_policy(env, policy, episodes=20, seed=1000)  # épisodes jamais vus
    assert learned > 0.5 * baseline  # retours négatifs : au moins deux fois moins d'erreur
    assert history[-1] > history[0]
