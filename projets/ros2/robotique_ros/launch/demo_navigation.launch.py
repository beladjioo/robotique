"""Démo complète : simulateur + planificateur A* + suivi pure pursuit (+ RViz en option).

ros2 launch robotique_ros demo_navigation.launch.py rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory("robotique_ros"), "config", "demo.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="false", description="Lancer RViz"),
            Node(package="robotique_ros", executable="unicycle_sim", name="unicycle_sim"),
            Node(package="robotique_ros", executable="path_planner", name="path_planner"),
            Node(
                package="robotique_ros",
                executable="pure_pursuit",
                name="pure_pursuit",
                parameters=[{"lookahead": 0.4, "speed": 0.5}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
