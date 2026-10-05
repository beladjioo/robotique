"""Planifie un chemin A* sur la carte de démonstration et publie ``map`` et ``plan`` (QoS persistante)."""

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from navigation import OccupancyGrid, astar, demo_maps, shortcut_path
from robotique_ros.conversions import occupancy_to_ros_data


class PathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("path_planner")
        self.declare_parameter("robot_radius", 0.2)
        self.declare_parameter("start_x", demo_maps.WAREHOUSE_START[0])
        self.declare_parameter("start_y", demo_maps.WAREHOUSE_START[1])
        self.declare_parameter("goal_x", demo_maps.WAREHOUSE_GOAL[0])
        self.declare_parameter("goal_y", demo_maps.WAREHOUSE_GOAL[1])
        # « transient local » : un abonné tardif (RViz, pure_pursuit) reçoit quand même la carte et le plan.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGridMsg, "map", latched)
        self.path_pub = self.create_publisher(Path, "plan", latched)
        self.grid = OccupancyGrid.from_ascii(demo_maps.WAREHOUSE, demo_maps.WAREHOUSE_RESOLUTION)
        self.publish_map()
        self.publish_plan()

    def publish_map(self) -> None:
        msg = OccupancyGridMsg()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.grid.resolution
        msg.info.height, msg.info.width = self.grid.shape
        msg.info.origin.position.x, msg.info.origin.position.y = self.grid.origin
        msg.info.origin.orientation.w = 1.0
        msg.data = occupancy_to_ros_data(self.grid.occupied)
        self.map_pub.publish(msg)

    def publish_plan(self) -> None:
        planning_grid = self.grid.inflate(self.get_parameter("robot_radius").value)
        start = planning_grid.world_to_cell(
            self.get_parameter("start_x").value, self.get_parameter("start_y").value
        )
        goal = planning_grid.world_to_cell(
            self.get_parameter("goal_x").value, self.get_parameter("goal_y").value
        )
        cells = astar(planning_grid, start, goal)
        if cells is None:
            self.get_logger().error("Aucun chemin entre le départ et le but")
            return
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        for cell in shortcut_path(planning_grid, cells):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = self.grid.cell_to_world(cell)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)
        self.get_logger().info(f"Plan publié : {len(msg.poses)} points de passage")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
