"""
Verify that Graph2Seq aggregator parameters are properly included in the optimization.
"""
import tensorflow as tf
import numpy as np
from policies.meta_seq2seq_policy import Seq2SeqNetwork, Seq2SeqPolicy
from meta_algos.ppo_offloading import PPO


def verify_aggregator_parameter_inclusion():
    """Verify aggregator parameters are included in PPO optimization."""
    print("Verifying Graph2Seq Aggregator Parameter Inclusion")
    print("="*60)
    
    tf.reset_default_graph()
    
    # Create policy
    obs_dim = 5
    encoder_units = 128
    decoder_units = 128
    vocab_size = 3
    
    policy = Seq2SeqPolicy(
        obs_dim=obs_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        vocab_size=vocab_size
    )
    
    # Create PPO algorithm
    ppo = PPO(
        policy=policy,
        meta_sampler=None,  # Not needed for this test
        meta_sampler_process=None,  # Not needed for this test
        lr=1e-4,
        num_inner_grad_steps=4,
        clip_value=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Get all trainable variables
    all_trainable = policy.network.get_trainable_variables()
    
    print(f"\nTotal trainable variables: {len(all_trainable)}")
    
    # Categorize variables
    encoder_vars = []
    aggregator_vars = []
    decoder_vars = []
    other_vars = []
    
    for var in all_trainable:
        var_name = var.name.lower()
        if 'aggregator' in var_name:
            aggregator_vars.append(var)
        elif 'encoder' in var_name:
            encoder_vars.append(var)
        elif 'decoder' in var_name:
            decoder_vars.append(var)
        else:
            other_vars.append(var)
    
    print(f"\nVariable breakdown:")
    print(f"  Encoder variables: {len(encoder_vars)}")
    print(f"  Aggregator variables: {len(aggregator_vars)}")
    print(f"  Decoder variables: {len(decoder_vars)}")
    print(f"  Other variables: {len(other_vars)}")
    
    # Show aggregator variables
    if aggregator_vars:
        print(f"\nAggregator variables found:")
        for var in aggregator_vars:
            print(f"  - {var.name}: shape={var.shape}")
    else:
        print("\n[WARNING] No aggregator variables found!")
        
    # Check if aggregator variables are in the PPO loss computation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Create dummy data
        batch_size = 8
        seq_len = 10
        
        obs_data = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
        action_data = np.random.randint(0, vocab_size, (batch_size, seq_len))
        old_logits_data = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        adv_data = np.random.randn(batch_size, seq_len).astype(np.float32)
        old_v_data = np.random.randn(batch_size, seq_len).astype(np.float32)
        r_data = np.random.randn(batch_size, seq_len).astype(np.float32)
        length_data = np.full(batch_size, seq_len, dtype=np.int32)
        
        feed_dict = {
            ppo.obs: obs_data,
            ppo.decoder_inputs: action_data,
            ppo.actions: action_data,
            ppo.old_logits: old_logits_data,
            ppo.advs: adv_data,
            ppo.old_v: old_v_data,
            ppo.r: r_data,
            ppo.decoder_full_length: length_data
        }
        
        # Compute gradients
        grads = tf.gradients(ppo.total_loss, all_trainable)
        grad_values = sess.run(grads, feed_dict)
        
        # Check which variables have non-zero gradients
        vars_with_grads = []
        vars_without_grads = []
        
        for var, grad in zip(all_trainable, grad_values):
            if grad is not None and np.any(grad != 0):
                vars_with_grads.append(var.name)
            else:
                vars_without_grads.append(var.name)
        
        # Check aggregator gradients specifically
        aggregator_grads_count = 0
        for var in aggregator_vars:
            if var.name in vars_with_grads:
                aggregator_grads_count += 1
                
        print(f"\n[RESULT] Aggregator gradient flow:")
        print(f"  {aggregator_grads_count}/{len(aggregator_vars)} aggregator variables have gradients")
        
        if aggregator_grads_count == len(aggregator_vars) and len(aggregator_vars) > 0:
            print("\n[SUCCESS] All aggregator parameters are included in the loss computation!")
        elif len(aggregator_vars) == 0:
            print("\n[INFO] No explicit aggregator variables found - they may be embedded in encoder scope")
        else:
            print("\n[WARNING] Some aggregator parameters may not be receiving gradients!")
            
        # Show encoder variables with gradients
        encoder_with_grads = [v for v in vars_with_grads if 'encoder' in v.lower()]
        print(f"\n[INFO] {len(encoder_with_grads)} encoder-related variables have gradients")
        
        # List first few encoder variables
        print("\nSample encoder variables with gradients:")
        for var_name in encoder_with_grads[:5]:
            print(f"  - {var_name}")


if __name__ == "__main__":
    verify_aggregator_parameter_inclusion()