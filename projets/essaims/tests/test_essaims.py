import itertools

import numpy as np
import pytest

from essaims import (
    BoidsParams,
    algebraic_connectivity,
    auction_assignment,
    boids_step,
    complete_graph,
    consensus_step,
    disk_graph,
    formation_step,
    is_connected,
    laplacian,
    polarization,
    ring_graph,
)


def test_graph_properties():
    assert is_connected(ring_graph(6))
    A = ring_graph(6)
    A[0, 1] = A[1, 0] = A[3, 4] = A[4, 3] = 0.0  # coupe l'anneau en deux
    assert not is_connected(A)
    assert algebraic_connectivity(complete_graph(5)) == pytest.approx(5.0)
    np.testing.assert_allclose(laplacian(ring_graph(4)).sum(axis=1), 0.0)
    np.testing.assert_array_equal(disk_graph([[0, 0], [1, 0], [3, 0]], 1.5)[0], [0, 1, 0])


def test_consensus_reaches_average():
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, size=(8, 2))
    average = x.mean(axis=0)
    for _ in range(2000):
        x = consensus_step(x, ring_graph(8), dt=0.1)
    np.testing.assert_allclose(x, np.tile(average, (8, 1)), atol=1e-6)


def test_formation_converges_to_shape():
    n = 6
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hexagon = np.column_stack([np.cos(angles), np.sin(angles)])
    p = np.random.default_rng(1).uniform(-5, 5, size=(n, 2))
    centroid = p.mean(axis=0)
    for _ in range(3000):
        p = formation_step(p, ring_graph(n), hexagon, dt=0.05)
    np.testing.assert_allclose(p - p.mean(axis=0), hexagon, atol=1e-6)
    np.testing.assert_allclose(p.mean(axis=0), centroid, atol=1e-9)


def test_boids_align_and_keep_distance():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 4, size=(25, 2))
    v = rng.normal(size=(25, 2))
    initial = polarization(v)
    params = BoidsParams()
    for _ in range(1500):
        p, v = boids_step(p, v, params, dt=0.05)
    assert initial < 0.5
    assert polarization(v) > 0.95
    dist = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(25) * 1e9
    assert dist.min() > 0.2


def brute_force_best(benefit):
    n_agents, n_tasks = benefit.shape
    return max(
        sum(benefit[i, t] for i, t in enumerate(perm))
        for perm in itertools.permutations(range(n_tasks), n_agents)
    )


@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (1, 3)])
def test_auction_is_optimal_on_integer_benefits(shape):
    rng = np.random.default_rng(3)
    benefit = rng.integers(0, 50, size=shape).astype(float)
    assignment = auction_assignment(benefit, epsilon=1.0 / (shape[0] + 1))
    assert len(set(assignment.tolist())) == shape[0]
    total = benefit[np.arange(shape[0]), assignment].sum()
    assert total == brute_force_best(benefit)


def test_auction_assigns_robots_to_closest_targets():
    robots = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    targets = np.array([[0.0, 9.0], [1.0, 1.0], [9.0, 1.0]])
    distance = np.linalg.norm(robots[:, None] - targets[None], axis=-1)
    np.testing.assert_array_equal(auction_assignment(-distance), [1, 2, 0])
