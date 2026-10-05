"""Simulateur cinématique minimal : intègre ``cmd_vel`` et publie ``odom`` + TF odom -> base_link.

Permet de tester toute la chaîne de navigation sans Gazebo.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from navigation import demo_maps, unicycle_step
from robotique_ros.conversions import yaw_to_quaternion


class UnicycleSimNode(Node):
    def __init__(self) -> None:
        super().__init__("unicycle_sim")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("initial_x", demo_maps.WAREHOUSE_START[0])
        self.declare_parameter("initial_y", demo_maps.WAREHOUSE_START[1])
        self.declare_parameter("initial_yaw", 0.0)
        self.pose = np.array(
            [
                self.get_parameter("initial_x").value,
                self.get_parameter("initial_y").value,
                self.get_parameter("initial_yaw").value,
            ],
            dtype=float,
        )
        self.v = self.w = 0.0
        self.dt = 1.0 / self.get_parameter("rate_hz").value
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(self.dt, self._on_timer)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.v, self.w = msg.linear.x, msg.angular.z

    def _on_timer(self) -> None:
        self.pose = unicycle_step(self.pose, self.v, self.w, self.dt)
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quaternion(float(self.pose[2]))

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(self.pose[0])
        odom.pose.pose.position.y = float(self.pose[1])
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = qx, qy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = float(self.pose[0])
        tf.transform.translation.y = float(self.pose[1])
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnicycleSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
