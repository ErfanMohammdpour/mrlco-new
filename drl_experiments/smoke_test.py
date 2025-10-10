"""
Smoke test script for DRL experiments.
Runs a minimal training and evaluation to verify the implementation works.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from drl_experiments.configs import *
from drl_experiments.policy import DRLPolicy
from drl_experiments.rollout import collect_rollout


def smoke_test():
    """Run smoke test with minimal parameters."""
    print("Running DRL smoke test...")
    
    # Set seed
    np.random.seed(42)
    tf.set_random_seed(42)
    
    # Create environment with minimal data
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    # Use only first 3 maps for smoke test
    test_graph_paths = TRAIN_GRAPH_PATHS[:3]
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,
        graph_number=10,  # Small number for quick test
        graph_file_paths=test_graph_paths,
        time_major=time_major
    )
    
    print(f"Environment created with {len(test_graph_paths)} maps")
    
    # Create policy
    policy = DRLPolicy(
        obs_dim=17,
        action_dim=2,
        encoder_units=64,  # Smaller for quick test
        decoder_units=64,
        num_layers=1
    )
    
    print("Policy created successfully")
    
    # Test rollout collection
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Test single rollout
        try:
            rollout = collect_rollout(env, policy, task_id=0)
            print(f"Rollout collected successfully: length={rollout['length']}")
            print(f"Actions shape: {rollout['actions'].shape}")
            print(f"Rewards shape: {rollout['rewards'].shape}")
            print(f"Values shape: {rollout['values'].shape}")
        except Exception as e:
            print(f"Error in rollout collection: {e}")
            return False
        
        # Test policy evaluation
        try:
            obs = rollout['obs']
            actions = rollout['actions']
            
            print(f"Debug - obs shape: {obs.shape}")
            print(f"Debug - actions shape: {actions.shape}")
            print(f"Debug - actions dtype: {actions.dtype}")
            print(f"Debug - actions values: {actions}")
            
            # Reshape for evaluation
            obs_batch = obs[np.newaxis, :, :]  # [1, T, F]
            actions_batch = actions[np.newaxis, :]  # [1, T]
            
            print(f"Debug - obs_batch shape: {obs_batch.shape}")
            print(f"Debug - actions_batch shape: {actions_batch.shape}")
            
            log_probs, entropy, values = policy.evaluate_actions(obs_batch, actions_batch)
            print(f"Policy evaluation successful:")
            print(f"  Log probs shape: {log_probs.shape}")
            print(f"  Entropy shape: {entropy.shape}")
            print(f"  Values shape: {values.shape}")
        except Exception as e:
            print(f"Error in policy evaluation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ Smoke test passed!")
    return True


if __name__ == "__main__":
    success = smoke_test()
    if success:
        print("\n🎉 DRL implementation is working correctly!")
        print("You can now run full training with:")
        print("python -m drl_experiments.train_drl --num_epochs 2 --tasks_per_epoch 3 --rollouts_per_task 1")
    else:
        print("\n❌ Smoke test failed. Please check the implementation.")
        sys.exit(1)
