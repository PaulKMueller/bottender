# smallvla_inference_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sereact_humanoid_msg.msg import AlignedFrame, Teleop
from cv_bridge import CvBridge

import torch
from lerobot.models import SmallVLA
from lerobot.data.transforms import TransformFactory

import numpy as np
import cv2

class SmallVLAInferenceNode(Node):
    def __init__(self):
        super().__init__('smallvla_inference_node')

        # === Load pretrained SmallVLA ===
        self.model = SmallVLA.from_pretrained("lerobot/smallvla")  # huggingface model name
        self.model = self.model.cuda()
        self.model.eval()

        # === Build image transform ===
        self.transform = TransformFactory.build_default_transform(resize_size=224)

        # === ROS2 subscribers/publishers ===
        self.bridge = CvBridge()

        self.create_subscription(AlignedFrame, '/robot/aligned_frame', self.callback, 10)
        self.teleop_pub = self.create_publisher(Teleop, '/teleop', 10)
        self.gripper_pub = self.create_publisher(Teleop, '/gripper/target', 10)

        self.get_logger().info("SmallVLA inference node initialized.")

    def callback(self, msg: AlignedFrame):
        try:
            # --- Extract wrist_back RGB image ---
            idx = msg.topics.index('/camera/wrist_back/color/image_rect_raw')
            img_msg = msg.images[idx]
            rgb_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')

            # Normalize & resize using lerobot transform
            rgb_tensor = self.transform(rgb_image).unsqueeze(0).cuda()  # shape: (1, C, H, W)

            # --- Extract current robot TCP pose ---
            obs_action = msg.observation_action[0]
            tcp_pose = np.array(obs_action.pos_xyzquat_right, dtype=np.float32)  # shape: (7,)
            tcp_tensor = torch.tensor(tcp_pose).unsqueeze(0).cuda()

            # --- Extract current gripper state ---
            gripper_state = np.array([msg.gripper_state[0].position_normalized], dtype=np.float32)
            gripper_tensor = torch.tensor(gripper_state).unsqueeze(0).cuda()

            # --- Build model input dictionary ---
            obs = {
                "images": [rgb_tensor],
                "robot_state": torch.cat([tcp_tensor, gripper_tensor], dim=1)
            }

            # --- Run inference ---
            with torch.no_grad():
                action = self.model(obs)

            # Output is (batch_size, 10) → unpack
            action_np = action[0].cpu().numpy()

            # --- Publish to /teleop ---
            teleop_msg = Teleop()
            teleop_msg.right_wrist = action_np[:7].tolist()  # xyz + scalar-first quaternion
            teleop_msg.normalized_finger_distance = float(action_np[7])  # gripper
            self.teleop_pub.publish(teleop_msg)
            self.gripper_pub.publish(teleop_msg)

            self.get_logger().info("Published action: " + str(action_np[:7]))

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SmallVLAInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
