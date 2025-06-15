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

"""This script demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path
import shutil

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


dataset_path = "/home/ubuntu/datasets"
torch.backends.cudnn.benchmark = True

def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for best model checkpoint
    best_model_dir = output_directory / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for periodic checkpoints
    checkpoints_dir = output_directory / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

   # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(dataset_path)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features and "mask" not in key and "depth" not in key}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features)

    cfg.chunk_size = 16
    cfg.n_action_steps = 16

    print("Dataset stats: ", dataset_metadata.stats)

    for key in input_features:
        if key not in dataset_metadata.stats:
            print(f"Warning: {key} not found in dataset stats. ")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std=[0.229, 0.224, 0.225]
    dataset_stats = dataset_metadata.stats.copy()
    dataset_stats.setdefault('observation.images.wrist_back', {})
    dataset_stats.setdefault('observation.masks.wrist_back', {})

    print("Image features, ", cfg.image_features)


    dataset_stats['observation.images.wrist_back']['mean'] = torch.tensor(imagenet_mean)[:, None, None]
    dataset_stats['observation.images.wrist_back']['std'] = torch.tensor(imagenet_std)[:, None, None]
    dataset_stats['observation.masks.wrist_back']['mean'] = torch.tensor(imagenet_mean)[:, None, None]
    dataset_stats['observation.masks.wrist_back']['std'] = torch.tensor(imagenet_std)[:, None, None]


    # We can now instantiate our policy with this config and the dataset stats.
    policy = ACTPolicy(cfg, dataset_stats=dataset_stats)
    policy.train()
    policy.to(device)

    camera_key = dataset_metadata.camera_keys[0]  # Use the first camera key for this example.

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        #"observation.images.wrist_back": [-0.1, 0.0],
        #"observation.masks.wrist_back": [-0.1, 0.0],
        #"observation.state": [0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(dataset_path, delta_timestamps=delta_timestamps, tolerance_s=0.1)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=16,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        persistent_workers=True
    )

    # Initialize best loss tracking
    best_loss = float('inf')
    checkpoint_freq = 100
    periodic_checkpoint_freq = 500  # Save checkpoint every 500 steps
    
    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # drop the is pad
            batch = {k: v for k, v in batch.items() if not k.endswith("wrist_back_is_pad")}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss = loss.item()
            
            # Update best loss if current loss is better
            if current_loss < best_loss:
                best_loss = current_loss
            
            # Check if it's time to checkpoint (every 100 steps) and if we have the best model
            if step % checkpoint_freq == 0 and current_loss == best_loss:
                print(f"New best model at step {step} with loss: {best_loss:.6f}")
                
                # Save the best model checkpoint
                policy.save_pretrained(best_model_dir)
                print(f"Best model saved to {best_model_dir}")
            
            # Save periodic checkpoint every 500 steps
            if step % periodic_checkpoint_freq == 0 and step > 0:
                periodic_checkpoint_dir = checkpoints_dir / f"checkpoint_step_{step}"
                periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(periodic_checkpoint_dir)
                print(f"Periodic checkpoint saved at step {step} to {periodic_checkpoint_dir}")

            if step % log_freq == 0:
                print(f"step: {step} loss: {current_loss:.3f} (best: {best_loss:.6f})")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Copy the best model to the final output directory
    if best_model_dir.exists():
        # Remove existing files in output directory if any (except checkpoints and best_model dirs)
        for item in output_directory.iterdir():
            if item != best_model_dir and item != checkpoints_dir:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # Copy best model files to main output directory
        for item in best_model_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, output_directory / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_directory / item.name)
        
        print(f"Final best model (loss: {best_loss:.6f}) saved to {output_directory}")
        print(f"All periodic checkpoints saved in {checkpoints_dir}")
    else:
        print("No best model found, saving final model...")
        policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
