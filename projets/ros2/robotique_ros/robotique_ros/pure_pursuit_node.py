"""Suit le chemin reçu sur ``plan`` à partir de ``odom`` et publie ``cmd_vel``."""

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from navigation import PurePursuit
from robotique_ros.conversions import quaternion_to_yaw


class PurePursuitNode(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        self.declare_parameter("lookahead", 0.4)
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("goal_tolerance", 0.05)
        self.declare_parameter("rate_hz", 20.0)
        self.controller: PurePursuit | None = None
        self.pose: np.ndarray | None = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "plan", self._on_plan, latched)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._on_timer)

    def _on_plan(self, msg: Path) -> None:
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not points:
            self.controller = None
            return
        self.controller = PurePursuit(
            points,
            lookahead=self.get_parameter("lookahead").value,
            speed=self.get_parameter("speed").value,
            goal_tolerance=self.get_parameter("goal_tolerance").value,
        )
        self.get_logger().info(f"Nouveau chemin de {len(points)} points")

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.pose = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                quaternion_to_yaw(q.x, q.y, q.z, q.w),
            ]
        )

    def _on_timer(self) -> None:
        cmd = Twist()
        if self.controller is not None and self.pose is not None:
            v, w, reached = self.controller.compute(self.pose)
            cmd.linear.x, cmd.angular.z = v, w
            if reached:
                self.get_logger().info("But atteint", once=True)
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
