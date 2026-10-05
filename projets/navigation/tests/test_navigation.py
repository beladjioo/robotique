import math

import numpy as np
import pytest

from navigation import (
    DifferentialDrive,
    OccupancyGrid,
    PurePursuit,
    astar,
    demo_maps,
    line_of_sight,
    path_length,
    shortcut_path,
    unicycle_step,
)


def empty_grid(rows=10, cols=10):
    return OccupancyGrid(np.zeros((rows, cols), dtype=bool))


def test_astar_straight_and_diagonal():
    grid = empty_grid()
    assert len(astar(grid, (0, 0), (0, 9))) == 10
    diagonal = astar(grid, (0, 0), (9, 9))
    assert len(diagonal) == 10
    assert astar(grid, (0, 0), (9, 9), allow_diagonal=False)[-1] == (9, 9)


def test_astar_returns_none_when_blocked():
    occ = np.zeros((5, 5), dtype=bool)
    occ[:, 2] = True
    assert astar(OccupancyGrid(occ), (0, 0), (0, 4)) is None
    assert astar(OccupancyGrid(occ), (0, 2), (0, 4)) is None  # départ dans un obstacle


def test_astar_never_cuts_corners():
    grid = OccupancyGrid.from_ascii(
        """
        ..
        #.
        """.replace(" ", "")
    )
    # Ligne 0 = bas : (0,0) est l'obstacle, aller de (0,1) à (1,0) impose 2 pas droits.
    path = astar(grid, (0, 1), (1, 0))
    assert path == [(0, 1), (1, 1), (1, 0)]


def test_astar_is_optimal_around_wall():
    occ = np.zeros((7, 7), dtype=bool)
    occ[1:6, 3] = True
    path = astar(OccupancyGrid(occ), (3, 0), (3, 6))
    cost = sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
    # 2 pas droits au sommet du mur + 2 x (1 droit + 2 diagonales) de part et d'autre
    assert cost == pytest.approx(4 + 4 * math.sqrt(2))


def test_grid_coordinates_roundtrip_and_inflation():
    grid = OccupancyGrid(np.zeros((4, 6), dtype=bool), resolution=0.5, origin=(-1.0, 2.0))
    assert grid.world_to_cell(*grid.cell_to_world((3, 5))) == (3, 5)
    assert grid.world_to_cell(-0.9, 2.1) == (0, 0)

    occ = np.zeros((9, 9), dtype=bool)
    occ[4, 4] = True
    inflated = OccupancyGrid(occ).inflate(2.0)
    assert inflated.occupied.sum() == 13  # disque discret de rayon 2
    assert inflated.occupied[4, 6] and not inflated.occupied[6, 6]


def test_shortcut_keeps_path_collision_free():
    grid = OccupancyGrid.from_ascii(demo_maps.WAREHOUSE, demo_maps.WAREHOUSE_RESOLUTION)
    start = grid.world_to_cell(*demo_maps.WAREHOUSE_START)
    goal = grid.world_to_cell(*demo_maps.WAREHOUSE_GOAL)
    path = astar(grid, start, goal)
    short = shortcut_path(grid, path)
    assert short[0] == start and short[-1] == goal
    assert len(short) < len(path)
    assert all(line_of_sight(grid, a, b) for a, b in zip(short, short[1:], strict=False))
    assert path_length(short) <= path_length(path)


def test_differential_drive_roundtrip():
    robot = DifferentialDrive(wheel_radius=0.033, wheel_base=0.16)
    wl, wr = robot.inverse(0.2, 0.5)
    assert robot.forward(wl, wr) == pytest.approx((0.2, 0.5))


def test_unicycle_full_circle_returns_home():
    pose = np.array([1.0, 2.0, 0.3])
    w = 0.5
    for _ in range(100):
        pose = unicycle_step(pose, 1.0, w, 2 * math.pi / w / 100)
    np.testing.assert_allclose(pose, [1.0, 2.0, 0.3], atol=1e-9)


def test_plan_and_follow_end_to_end():
    grid = OccupancyGrid.from_ascii(demo_maps.WAREHOUSE, demo_maps.WAREHOUSE_RESOLUTION)
    planning_grid = grid.inflate(0.2)
    start = planning_grid.world_to_cell(*demo_maps.WAREHOUSE_START)
    goal = planning_grid.world_to_cell(*demo_maps.WAREHOUSE_GOAL)
    cells = shortcut_path(planning_grid, astar(planning_grid, start, goal))
    waypoints = [grid.cell_to_world(c) for c in cells]

    controller = PurePursuit(waypoints, lookahead=0.4, speed=0.6)
    pose = np.array([*waypoints[0], 0.0])
    reached = False
    for _ in range(4000):
        v, w, reached = controller.compute(pose)
        if reached:
            break
        pose = unicycle_step(pose, v, w, 0.02)
        assert grid.is_free(grid.world_to_cell(pose[0], pose[1])), "collision !"
    assert reached
    assert np.linalg.norm(pose[:2] - waypoints[-1]) < 0.05
