#!/usr/bin/env python3
"""Test script to verify the dimension fix in MRLCO PPO update"""

import tensorflow as tf
import numpy as np
import sys

# Test the _train_step function with correct dimensions
def test_train_step():
    print("Testing MRLCO._train_step with correct dimensions...")
    
    # Import after TF setup
    from meta_algos.MRLCO import MRLCO
    from policies.distributions.categorical_pd import CategoricalPd
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from baselines.linear_baseline import LinearFeatureBaseline
    
    # Create a minimal policy for testing
    dist = CategoricalPd(dim=2)
    policy = MetaSeq2SeqPolicy(
        meta_batch_size=2,
        obs_dim=17,
        encoder_units=128,
        decoder_units=128,
        vocab_size=2
    )
    # Set distribution after creation
    policy.distribution = dist
    
    # Create MRLCO algorithm
    algo = MRLCO(
        policy=policy,
        meta_batch_size=2,
        meta_sampler=None,  # Not needed for this test
        meta_sampler_process=None,  # Not needed for this test
        inner_lr=1e-3,
        outer_lr=1e-3,
        num_inner_grad_steps=1,
        clip_value=0.3,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Create test data with correct shapes
    batch_size = 50
    seq_len = 20
    vocab_size = 2
    
    # Prepare test tensors
    old_logits = tf.random.normal([batch_size, seq_len, vocab_size])
    old_v = tf.random.normal([batch_size, seq_len])
    observations = tf.random.normal([batch_size, seq_len, 17])
    actions = tf.random.uniform([batch_size, seq_len], 0, 2, dtype=tf.int32)
    decoder_inputs = tf.random.uniform([batch_size, seq_len], 0, 2, dtype=tf.int32)
    decoder_full_length = tf.constant([seq_len] * batch_size, dtype=tf.int32)
    advs = tf.random.normal([batch_size, seq_len])  # Correct shape: [batch, seq]
    r = tf.random.normal([batch_size, seq_len])     # Correct shape: [batch, seq]
    
    print(f"Test tensor shapes:")
    print(f"  old_logits: {old_logits.shape}")
    print(f"  old_v: {old_v.shape}")
    print(f"  observations: {observations.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  advs: {advs.shape}")
    print(f"  r: {r.shape}")
    
    # Run the training step
    try:
        vf_loss, surr_obj, lr_mean, clipped_obj = algo._train_step(
            0, old_logits, old_v, observations, actions,
            decoder_inputs, decoder_full_length, advs, r
        )
        
        print(f"\n✓ Training step successful!")
        print(f"  Value loss: {vf_loss.numpy():.4f}")
        print(f"  Surrogate objective: {surr_obj.numpy():.4f}")
        print(f"  Likelihood ratio mean: {lr_mean.numpy():.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training step failed with error:")
        print(f"  {str(e)}")
        return False

if __name__ == "__main__":
    # Set random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Run test
    success = test_train_step()
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS: Dimension fix is working correctly!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("FAILURE: Dimension issues remain")
        print("="*50)
        sys.exit(1)