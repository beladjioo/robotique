import math

import numpy as np
import pytest

from robotique_ros.conversions import occupancy_to_ros_data, quaternion_to_yaw, yaw_to_quaternion


@pytest.mark.parametrize("yaw", [0.0, 0.5, -2.0, math.pi - 1e-6])
def test_yaw_quaternion_roundtrip(yaw):
    assert quaternion_to_yaw(*yaw_to_quaternion(yaw)) == pytest.approx(yaw)


def test_occupancy_grid_is_row_major_from_origin():
    occupied = np.array([[True, False, False], [False, False, True]])  # ligne 0 = bas
    assert occupancy_to_ros_data(occupied) == [100, 0, 0, 0, 0, 100]
