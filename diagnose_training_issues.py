"""
Diagnostic script to identify training issues with Graph2Seq integration
"""

import tensorflow as tf
import numpy as np
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy


def diagnose_policy_outputs():
    """Diagnose issues with policy outputs and gradients."""
    
    print("Diagnosing Graph2Seq Policy Training Issues")
    print("="*60)
    
    # Create a simple test case
    tf.reset_default_graph()
    
    # Parameters
    meta_batch_size = 2
    obs_dim = 17
    encoder_units = 128
    decoder_units = 128
    vocab_size = 2
    seq_length = 20
    batch_size = 4
    
    # Create policy
    policy = MetaSeq2SeqPolicy(
        meta_batch_size=meta_batch_size,
        obs_dim=obs_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        vocab_size=vocab_size
    )
    
    # Create test data
    observations = np.random.randn(batch_size, seq_length, obs_dim).astype(np.float32)
    actions = np.random.randint(0, vocab_size, size=(batch_size, seq_length)).astype(np.int32)
    decoder_full_length = np.array([seq_length] * batch_size, dtype=np.int32)
    
    # Get outputs from policy
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Test forward pass
        print("\n1. Testing Forward Pass:")
        print("-"*40)
        
        sample_actions, sample_logits, sample_vf = sess.run(
            [policy.core_policy.network.sample_decoder_prediction,
             policy.core_policy.network.sample_decoder_logits,
             policy.core_policy.network.sample_vf],
            feed_dict={
                policy.core_policy.obs: observations,
                policy.core_policy.decoder_full_length: decoder_full_length
            }
        )
        
        print(f"Sample actions shape: {sample_actions.shape}")
        print(f"Sample logits shape: {sample_logits.shape}")
        print(f"Sample logits range: [{np.min(sample_logits):.4f}, {np.max(sample_logits):.4f}]")
        print(f"Sample logits mean: {np.mean(sample_logits):.4f}")
        print(f"Sample logits std: {np.std(sample_logits):.4f}")
        print(f"Sample value function shape: {sample_vf.shape}")
        print(f"Sample value function mean: {np.mean(sample_vf):.4f}")
        
        # Test likelihood ratio calculation
        print("\n2. Testing Likelihood Ratio Calculation:")
        print("-"*40)
        
        # Create old and new logits
        old_logits = np.random.randn(batch_size, seq_length, vocab_size).astype(np.float32)
        new_logits = sample_logits
        
        # Create placeholders for testing
        old_logits_ph = tf.placeholder(tf.float32, shape=[None, None, vocab_size])
        new_logits_ph = tf.placeholder(tf.float32, shape=[None, None, vocab_size])
        actions_ph = tf.placeholder(tf.int32, shape=[None, None])
        
        # Calculate likelihood ratio
        from policies.distributions.categorical_pd import CategoricalPd
        dist = CategoricalPd(vocab_size)
        likelihood_ratio = dist.likelihood_ratio_sym(actions_ph, old_logits_ph, new_logits_ph)
        
        lr_value = sess.run(likelihood_ratio, feed_dict={
            old_logits_ph: old_logits,
            new_logits_ph: new_logits,
            actions_ph: actions
        })
        
        print(f"Likelihood ratio shape: {lr_value.shape}")
        print(f"Likelihood ratio range: [{np.min(lr_value):.4f}, {np.max(lr_value):.4f}]")
        print(f"Likelihood ratio mean: {np.mean(lr_value):.4f}")
        print(f"Likelihood ratio std: {np.std(lr_value):.4f}")
        
        # Check if likelihood ratios are all 1.0 (indicating identical distributions)
        if np.allclose(lr_value, 1.0):
            print("WARNING: All likelihood ratios are ~1.0, indicating old and new policies are identical!")
        
        # Test gradient flow
        print("\n3. Testing Gradient Flow:")
        print("-"*40)
        
        # Create a simple loss
        dummy_loss = tf.reduce_mean(sample_logits)
        
        # Get gradients
        trainable_vars = policy.core_policy.get_trainable_variables()
        print(f"Number of trainable variables: {len(trainable_vars)}")
        
        # Check for Graph2Seq variables
        graph2seq_vars = [v for v in trainable_vars if 'aggregator' in v.name.lower() or 'graph2seq' in v.name.lower()]
        print(f"Number of Graph2Seq-related variables: {len(graph2seq_vars)}")
        
        if graph2seq_vars:
            print("\nGraph2Seq variables found:")
            for v in graph2seq_vars[:5]:  # Show first 5
                print(f"  - {v.name}: {v.shape}")
        
        # Compute gradients
        grads = tf.gradients(dummy_loss, trainable_vars)
        grad_values = sess.run(grads, feed_dict={
            policy.core_policy.obs: observations,
            policy.core_policy.decoder_full_length: decoder_full_length
        })
        
        # Check for None gradients
        none_grad_vars = []
        zero_grad_vars = []
        for i, (grad, var) in enumerate(zip(grad_values, trainable_vars)):
            if grad is None:
                none_grad_vars.append(var.name)
            elif np.allclose(grad, 0.0):
                zero_grad_vars.append(var.name)
        
        print(f"\nVariables with None gradients: {len(none_grad_vars)}")
        if none_grad_vars:
            print("Examples:", none_grad_vars[:3])
            
        print(f"\nVariables with zero gradients: {len(zero_grad_vars)}")
        if zero_grad_vars:
            print("Examples:", zero_grad_vars[:3])
            
        # Test encoder outputs
        print("\n4. Testing Encoder Outputs:")
        print("-"*40)
        
        encoder_outputs, encoder_state = sess.run(
            [policy.core_policy.network.encoder_outputs,
             policy.core_policy.network.encoder_state],
            feed_dict={
                policy.core_policy.obs: observations,
                policy.core_policy.decoder_full_length: decoder_full_length
            }
        )
        
        print(f"Encoder outputs shape: {encoder_outputs.shape}")
        print(f"Encoder outputs range: [{np.min(encoder_outputs):.4f}, {np.max(encoder_outputs):.4f}]")
        print(f"Encoder outputs mean: {np.mean(encoder_outputs):.4f}")
        print(f"Encoder outputs std: {np.std(encoder_outputs):.4f}")
        
        if isinstance(encoder_state, tuple):
            print(f"Encoder state is tuple with {len(encoder_state)} elements")
            print(f"First state c shape: {encoder_state[0].c.shape}")
            print(f"First state h shape: {encoder_state[0].h.shape}")
        
    print("\n" + "="*60)
    print("Diagnosis Complete")
    

def analyze_loss_calculation():
    """Analyze why losses are near zero."""
    
    print("\n\nAnalyzing Loss Calculation")
    print("="*60)
    
    # Simulate PPO loss calculation
    batch_size = 10
    seq_length = 20
    vocab_size = 2
    
    # Create synthetic data
    advantages = np.random.randn(batch_size, seq_length).astype(np.float32)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Normalize
    
    # Likelihood ratios close to 1.0 (as might happen early in training)
    likelihood_ratios = np.ones((batch_size, seq_length)) + np.random.randn(batch_size, seq_length) * 0.01
    
    # Calculate clipped objective
    clip_value = 0.3
    clipped_ratios = np.clip(likelihood_ratios, 1.0 - clip_value, 1.0 + clip_value)
    
    obj1 = likelihood_ratios * advantages
    obj2 = clipped_ratios * advantages
    clipped_obj = np.minimum(obj1, obj2)
    
    # Calculate loss (negative of objective)
    loss = -np.mean(clipped_obj)
    
    print(f"Advantages mean: {np.mean(advantages):.6f}")
    print(f"Advantages std: {np.std(advantages):.6f}")
    print(f"Likelihood ratios mean: {np.mean(likelihood_ratios):.6f}")
    print(f"Likelihood ratios std: {np.std(likelihood_ratios):.6f}")
    print(f"Clipped objective mean: {np.mean(clipped_obj):.6f}")
    print(f"Final loss: {loss:.9f}")
    
    # Show what happens with very small likelihood ratio changes
    print("\nWith very small LR changes (1e-9 std):")
    tiny_lr = np.ones((batch_size, seq_length)) + np.random.randn(batch_size, seq_length) * 1e-9
    tiny_obj = tiny_lr * advantages
    tiny_loss = -np.mean(tiny_obj)
    print(f"Loss with tiny LR changes: {tiny_loss:.12f}")
    

if __name__ == "__main__":
    diagnose_policy_outputs()
    analyze_loss_calculation()