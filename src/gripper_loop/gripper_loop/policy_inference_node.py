import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from sereact_humanoid_msg.msg import Teleop, MultiModalImageStamped, ObservationAction, FingerGripperState
from cv_bridge import CvBridge
import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from PIL import Image as PILImage
from act_evaluator import ACTPolicyEvaluator  # assuming your class is in act_evaluator.py

class PolicyInferenceNode(Node):
    def __init__(self):
        super().__init__('policy_inference_node')
        self.get_logger().info("Initializing Policy Inference Node...")

        # Load policy
        self.evaluator = ACTPolicyEvaluator(
            pretrained_policy_path="outputs/train",
            device="cuda"
        )

        self.bridge = CvBridge()

        # Placeholders for state
        self.current_rgb = None
        self.current_pose = None
        self.current_joint_state = None
        self.current_gripper = None

        # Subscriptions
        self.create_subscription(MultiModalImageStamped, '/multi_modal_image', self.image_callback, 10)
        self.create_subscription(ObservationAction, '/observation_action', self.obs_callback, 10)
        self.create_subscription(FingerGripperState, '/gripper_state', self.gripper_callback, 10)

        # Publisher
        self.teleop_pub = self.create_publisher(Teleop, '/gripper/target', 10)

        # Timer
        self.timer = self.create_timer(0.5, self.timer_callback)  # every 0.5s

    def image_callback(self, msg: MultiModalImageStamped):
        try:
            for topic, img in zip(msg.topics, msg.images):
                if topic == "/camera/wrist_back/color/image_rect_raw":
                    cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="rgb8")
                    self.current_rgb = cv_image
                    break
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def obs_callback(self, msg: ObservationAction):
        self.current_pose = np.array(msg.pos_xyzquat_right)
        self.current_joint_state = np.array(msg.joint_state)

    def gripper_callback(self, msg: FingerGripperState):
        self.current_gripper = np.array([msg.position_normalized])

    def timer_callback(self):
        # Check if we have all data
        if any(x is None for x in [self.current_rgb, self.current_pose, self.current_joint_state, self.current_gripper]):
            self.get_logger().warn("Waiting for all inputs...")
            return

        # Assemble state vector
        state = np.concatenate([self.current_pose, self.current_joint_state, self.current_gripper])

        try:
            action, pos, quat, gripper = self.evaluator.get_next_action(state, self.current_rgb)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # Publish Teleop
        msg = Teleop()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.finger_distance = gripper
        msg.normalized_finger_distance = gripper
        msg.active = True
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

        self.teleop_pub.publish(msg)
        self.get_logger().info(f"Published gripper action: {gripper:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
