# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ACT Policy Evaluator Class

This class provides a clean interface for loading and running ACT policies
for robotic control tasks.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"`
```
"""

import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from PIL import Image
import io
from scipy.spatial.transform import Rotation as R


class ACTPolicyEvaluator:
    """
    A class for evaluating ACT policies with support for multi-modal observations
    including RGB images, depth images, masks, and state information.
    """
    
    def __init__(self, pretrained_policy_path, device="cuda"):
        """
        Initialize the ACT policy evaluator.
        
        Args:
            pretrained_policy_path (str): Path to the pretrained policy model
            device (str): Device to run the policy on ("cuda" or "cpu")
        """
        self.device = device
        self.policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        self.policy.reset()
        
        print("Policy input features:", self.policy.config.input_features)
        print("Policy output features:", self.policy.config.output_features)
    
    def extract_tcp_components(self, action):
        """
        Extract position, rotation (as quaternion), and gripper state from TCP action array.
        
        Args:
            action: torch.Tensor of shape [1, 10] containing:
                    - TCP position (x,y,z) in meters [3d]
                    - rotation in 6d representation (range -1 to 1) [6d]  
                    - gripper state (0 to 1) [1d]
        
        Returns:
            position: np.array of shape [3] - TCP position in meters
            quaternion: np.array of shape [4] - rotation as quaternion (w, x, y, z)
            gripper_state: float - gripper state value
        """
        # Convert to numpy and squeeze to remove batch dimension
        action_np = action.squeeze().cpu().numpy()
        
        # Extract position (first 3 elements)
        position = action_np[:3]
        
        # Extract 6D rotation representation (elements 3-8)
        rotation_6d = action_np[3:9]
        
        # Extract gripper state (last element)
        gripper_state = action_np[9]
        
        # Convert 6D rotation to quaternion using scipy
        quaternion = self.rotation_6d_to_quaternion(rotation_6d)
        
        return position, quaternion, gripper_state

    def rotation_6d_to_quaternion(self, rotation_6d):
        """
        Convert 6D rotation representation to quaternion (w, x, y, z) using scipy.
        """
        # Reshape to two 3D vectors
        v1 = rotation_6d[:3]
        v2 = rotation_6d[3:6]
        
        # Normalize the first vector
        v1 = v1 / np.linalg.norm(v1)
        
        # Make the second vector orthogonal to the first (Gram-Schmidt)
        v2 = v2 - np.dot(v1, v2) * v1
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute the third vector as cross product
        v3 = np.cross(v1, v2)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack([v1, v2, v3])
        
        # Use scipy to convert rotation matrix to quaternion
        scipy_rotation = R.from_matrix(rotation_matrix)
        quaternion_xyzw = scipy_rotation.as_quat()  # scipy returns (x, y, z, w)
        
        # Reorder to (w, x, y, z) format
        quaternion = np.array([quaternion_xyzw[3], quaternion_xyzw[0], 
                              quaternion_xyzw[1], quaternion_xyzw[2]])
        
        return quaternion
    
    def preprocess_observation(self, state, rgb_image):
        """
        Preprocess raw observation data into the format expected by the policy.
        
        Args:
            state: numpy array or list of state values
            rgb_image: PIL Image or numpy array of RGB image
            
        Returns:
            dict: Preprocessed observation dictionary ready for the policy
        """
        # Convert state to numpy array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Convert image to numpy array if it's a PIL Image
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        
        # Convert to torch tensors
        state_tensor = torch.from_numpy(state).to(torch.float32)
        rgb_tensor = torch.from_numpy(rgb_image).to(torch.float32) / 255.0
        
        # Convert RGB from HWC to CHW format
        if rgb_tensor.dim() == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)
        
        # Move to device
        state_tensor = state_tensor.to(self.device, non_blocking=True)
        rgb_tensor = rgb_tensor.to(self.device, non_blocking=True)
        
        # Add batch dimension
        state_tensor = state_tensor.unsqueeze(0)
        rgb_tensor = rgb_tensor.unsqueeze(0)
        
        # Create observation dictionary
        observation = {
            "observation.state": state_tensor,
            "observation.images.wrist_back": rgb_tensor
        }
        
        return observation
    
    def get_next_action(self, state, rgb_image, return_components=True):
        """
        Get the next action from the policy given current observations.
        
        Args:
            state: numpy array or list of state values
            rgb_image: PIL Image or numpy array of RGB image
            return_components (bool): If True, also return position, quaternion, and gripper state
            
        Returns:
            If return_components is True:
                tuple: (raw_action, position, quaternion, gripper_state)
            If return_components is False:
                numpy.ndarray: Raw action array
        """
        # Preprocess the observation
        observation = self.preprocess_observation(state, rgb_image)
        
        # Get action from policy
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        
        # Convert to numpy
        numpy_action = action.squeeze(0).to("cpu").numpy()
        
        if return_components:
            position, quaternion, gripper_state = self.extract_tcp_components(action)
            return numpy_action, position, quaternion, gripper_state
        else:
            return numpy_action


# Example usage:
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = ACTPolicyEvaluator(
        pretrained_policy_path="outputs/train",
        device="cuda"
    )
    
    # Example of how to use it (you would replace these with actual data)
    state = np.random.randn(10)  # Example state
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Get next action with components
    action, position, quaternion, gripper = evaluator.get_next_action(
        state, rgb_image
    )
    
    print(f"Position: {position}")
    print(f"Quaternion: {quaternion}")
    print(f"Gripper state: {gripper}")
    
    # # Or just get raw action
    # raw_action = evaluator.get_next_action(
    #     state, rgb_image, return_components=False
    # )
