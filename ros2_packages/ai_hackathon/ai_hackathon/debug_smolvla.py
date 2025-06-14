#!/usr/bin/env python3

import torch
import numpy as np

# Correct import for SmolVLA in current lerobot version
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

def main():
    print("=== SmolVLA Debug Script ===")
    
    try:
        print("Loading SmolVLA policy from 'lerobot/smolvla_base'...")
        
        # Try loading without config override first
        print("Attempting direct load...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print("SUCCESS: SmolVLA policy loaded directly!")
        
        print(f"Policy type: {type(policy)}")
        print(f"Policy config: {policy.config}")
        print(f"VLM model name: {policy.config.vlm_model_name}")
        
        return True
        
    except Exception as e:
        print(f"Direct loading failed: {e}")
        
        # Try with a minimal config
        print("\n=== Trying with minimal config ===")
        try:
            config = SmolVLAConfig()
            # Don't set vlm_model_name, use default
            print(f"Default VLM model: {config.vlm_model_name}")
            
            # Try to load just the config part
            print("Loading policy with default config...")
            policy = SmolVLAPolicy(config)
            print("SUCCESS: SmolVLA policy created with default config!")
            return True
            
        except Exception as e2:
            print(f"Minimal config also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    main() 