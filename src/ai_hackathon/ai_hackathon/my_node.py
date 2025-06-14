# smallvla_inference_node.py

# The following import is crucial to register all custom lerobot classes
# with the transformers library before they are used.
import lerobot
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sereact_humanoid_msg.msg import AlignedFrame, Teleop
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2

# Correct import for SmolVLA in current lerobot version
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class SmallVLAInferenceNode(Node):
    def __init__(self):
        super().__init__('smallvla_inference_node')
        self.get_logger().info("Initializing SmolVLAInferenceNode...")

        # === Load pretrained SmolVLA using the simple blog post approach ===
        try:
            self.get_logger().info("Loading SmolVLA policy from 'lerobot/smolvla_base'...")
            
            # Simple two-line loading from the blog post - now works with fixed versions!
            self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.policy = self.policy.cuda()
                self.get_logger().info("Policy moved to GPU")
            
            self.policy.eval()
            self.get_logger().info("SmolVLA policy loaded successfully.")

        except Exception as e:
            self.get_logger().error(f"Failed to load SmolVLA policy: {e}")
            raise e

        # === ROS2 subscribers/publishers ===
        self.get_logger().info("Creating ROS2 subscribers and publishers...")
        self.bridge = CvBridge()

        self.create_subscription(AlignedFrame, '/robot/aligned_frame', self.callback, 10)
        self.teleop_pub = self.create_publisher(Teleop, '/teleop', 10)
        self.gripper_pub = self.create_publisher(Teleop, '/gripper/target', 10)

        self.get_logger().info("SmolVLA inference node initialized successfully.")

    def callback(self, msg: AlignedFrame):
        self.get_logger().debug("Received a new frame. Starting inference callback.")
        try:
            # --- Extract wrist_back RGB image ---
            idx = msg.topics.index('/camera/wrist_back/color/image_rect_raw')
            img_msg = msg.images[idx]
            rgb_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')

            # Resize image to expected input size (512x512 for SmolVLA)
            rgb_image_resized = cv2.resize(rgb_image, (512, 512))
            
            # Convert to tensor and normalize
            rgb_tensor = torch.from_numpy(rgb_image_resized).permute(2, 0, 1).float() / 255.0
            rgb_tensor = rgb_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                rgb_tensor = rgb_tensor.cuda()

            # --- Extract current robot TCP pose ---
            obs_action = msg.observation_action[0]
            tcp_pose = np.array(obs_action.pos_xyzquat_right, dtype=np.float32)  # shape: (7,)
            
            # --- Extract current gripper state ---
            gripper_state = np.array([msg.gripper_state[0].position_normalized], dtype=np.float32)
            
            # Combine robot state (TCP pose + gripper)
            robot_state = np.concatenate([tcp_pose, gripper_state])  # shape: (8,)
            robot_state_tensor = torch.tensor(robot_state).unsqueeze(0)
            if torch.cuda.is_available():
                robot_state_tensor = robot_state_tensor.cuda()

            # --- Build model input dictionary ---
            task_description = "pick and place the object"
            batch = {
                "observation.images.top": rgb_tensor,  # Use standardized camera name
                "observation.state": robot_state_tensor,
                "task": [task_description] 
            }

            # --- Prepare inputs for the model ---
            normalized_batch = self.policy.normalize_inputs(batch)
            images, img_masks = self.policy.prepare_images(normalized_batch)
            state = self.policy.prepare_state(normalized_batch)
            lang_tokens, lang_masks = self.policy.prepare_language(normalized_batch)

            # --- Run inference ---
            with torch.no_grad():
                action = self.policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

            # Output is (batch_size, action_dim) → unpack
            action_np = action[0].cpu().numpy()
            self.get_logger().info(f"Predicted action: {action_np}")

            # --- Publish to /teleop (TCP pose only) ---
            teleop_msg = Teleop()
            
            # SmolVLA typically outputs 6-DOF actions (xyz + rotation)
            if len(action_np) >= 7:
                # Extract xyz (first 3) and quaternion (next 4)
                xyz = action_np[:3]
                quat = action_np[3:7]  # Assuming SmolVLA outputs (x,y,z,w) quaternion
                
                # Convert to scalar-first quaternion (w,x,y,z) as required by spec
                if len(quat) == 4:
                    # If SmolVLA outputs (x,y,z,w), reorder to (w,x,y,z)
                    quat_scalar_first = [quat[3], quat[0], quat[1], quat[2]]  # w,x,y,z
                    teleop_msg.right_wrist = xyz.tolist() + quat_scalar_first
                else:
                    self.get_logger().warning("Quaternion dimension is not 4, using current pose")
                    teleop_msg.right_wrist = tcp_pose.tolist()
            else:
                # Fallback: use current TCP pose if action is shorter
                self.get_logger().warning(f"Action dimension ({len(action_np)}) is less than 7. Using current pose.")
                teleop_msg.right_wrist = tcp_pose.tolist()

            # --- Publish to /gripper/target (gripper only) ---
            gripper_msg = Teleop()
            if len(action_np) > 7:
                gripper_msg.normalized_finger_distance = float(action_np[7])  # gripper from action
            else:
                gripper_msg.normalized_finger_distance = float(gripper_state[0])  # keep current

            # Publish to respective topics
            self.teleop_pub.publish(teleop_msg)
            self.gripper_pub.publish(gripper_msg)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}", exc_info=True)

def main(args=None):
    rclpy.init(args=args)
    node = SmallVLAInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
