# gripper_loop_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sereact_humanoid_msg.msg import Teleop
import time

class GripperLoopNode(Node):
    def __init__(self):
        super().__init__('gripper_loop_node')
        self.pub = self.create_publisher(Teleop, '/gripper/target', 10)
        self.timer = self.create_timer(2.0, self.timer_callback)  # every 2 seconds
        self.closed = False
        self.get_logger().info("Gripper loop node started.")

    def timer_callback(self):
        msg = Teleop()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.normalized_finger_distance = 0.0 if self.closed else 1.0  # 0 = closed, 1 = open
        msg.finger_distance = msg.normalized_finger_distance
        msg.active = True

        # other fields can be zero-filled or empty
        msg.head_mat = []
        msg.left_wrist = []
        msg.right_wrist = []
        msg.left_qpos = []
        msg.right_qpos = []
        msg.left_hand_tips = []
        msg.right_hand_tips = []
        msg.joint_state_target = []
        msg.left_hand_joint_names = []
        msg.right_hand_joint_names = []

        self.pub.publish(msg)
        self.get_logger().info(
            f"{'Closed' if self.closed else 'Opened'} gripper (normalized={msg.normalized_finger_distance})"
        )
        self.closed = not self.closed

def main(args=None):
    rclpy.init(args=args)
    node = GripperLoopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()