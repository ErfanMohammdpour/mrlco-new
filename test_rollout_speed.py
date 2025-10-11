#!/usr/bin/env python3
"""
Test script to measure rollout collection speed.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from drl_experiments.configs import *
from drl_experiments.policy import DRLPolicy
from drl_experiments.rollout import collect_rollout

def test_rollout_speed():
    """Test how fast rollout collection is."""
    print("Testing rollout collection speed...")
    
    # Create environment
    resources = Resources(mec_process_capable=1000, mobile_process_capable=100)
    env = OffloadingEnvironment(resources)
    
    # Create policy
    policy = DRLPolicy(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        scope_name="drl_policy"
    )
    
    # Initialize policy
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Test single rollout
    print("Testing single rollout...")
    start_time = time.time()
    
    try:
        rollout = collect_rollout(env, policy, task_id=1)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Single rollout took: {duration:.2f} seconds")
        print(f"Rollout length: {rollout['length']}")
        print(f"Actions: {len(rollout['actions'])}")
        
        # Estimate time for full epoch
        tasks_per_epoch = 5
        rollouts_per_task = 1
        total_rollouts = tasks_per_epoch * rollouts_per_task
        
        estimated_epoch_time = duration * total_rollouts
        print(f"Estimated time for {total_rollouts} rollouts: {estimated_epoch_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in rollout collection: {e}")
        import traceback
        traceback.print_exc()
    
    sess.close()

if __name__ == "__main__":
    test_rollout_speed()
