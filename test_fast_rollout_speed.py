#!/usr/bin/env python3
"""
Test fast rollout speed.
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
from drl_experiments.fast_rollout import collect_fast_rollouts, batch_fast_rollouts

def test_fast_rollout_speed():
    """Test how fast the new rollout collection is."""
    print("Testing FAST rollout collection speed...")
    
    # Create environment
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    # Use only first 3 maps for quick test
    test_graph_paths = TRAIN_GRAPH_PATHS[:3]
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,
        graph_number=10,  # Small number for quick test
        graph_file_paths=test_graph_paths,
        time_major=time_major
    )
    
    # Create policy
    policy = DRLPolicy(
        obs_dim=17,
        action_dim=2,
        encoder_units=64,  # Smaller for quick test
        decoder_units=64,
        num_layers=1
    )
    
    # Initialize policy
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Test fast rollout collection
    print("Testing fast rollout collection...")
    start_time = time.time()
    
    try:
        # Test with 2 tasks, 1 rollout each
        task_ids = [1, 2]
        rollouts = collect_fast_rollouts(env, policy, task_ids, 1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Fast rollout collection took: {duration:.2f} seconds")
        print(f"Collected {len(rollouts)} rollouts")
        
        if rollouts:
            print(f"First rollout length: {rollouts[0]['length']}")
            print(f"First rollout actions: {len(rollouts[0]['actions'])}")
        
        # Test batching
        batch_data = batch_fast_rollouts(rollouts)
        if batch_data:
            print(f"Batch data shapes:")
            for key, value in batch_data.items():
                print(f"  {key}: {value.shape}")
        
        # Estimate time for full epoch
        tasks_per_epoch = 2
        rollouts_per_task = 1
        total_rollouts = tasks_per_epoch * rollouts_per_task
        
        estimated_epoch_time = duration * (total_rollouts / len(task_ids))
        print(f"Estimated time for {total_rollouts} rollouts: {estimated_epoch_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in fast rollout collection: {e}")
        import traceback
        traceback.print_exc()
    
    sess.close()

if __name__ == "__main__":
    test_fast_rollout_speed()
