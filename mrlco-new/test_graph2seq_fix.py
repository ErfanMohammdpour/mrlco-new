"""
Test the Graph2Seq encoder fix for TensorFlow graph mode compatibility.
"""
import tensorflow as tf
import numpy as np
from policies.meta_seq2seq_policy import Seq2SeqPolicy, MetaSeq2SeqPolicy


def test_graph2seq_fix():
    """Test that the Graph2Seq encoder works in TensorFlow graph mode."""
    print("Testing Graph2Seq Encoder Fix")
    print("="*60)
    
    # Reset graph
    tf.reset_default_graph()
    
    # Test parameters
    batch_size = 8
    seq_len = 10
    obs_dim = 17
    encoder_units = 128
    decoder_units = 128
    vocab_size = 3
    
    try:
        # Test 1: Create single policy
        print("\n1. Testing single policy creation...")
        policy = Seq2SeqPolicy(
            obs_dim=obs_dim,
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            vocab_size=vocab_size,
            name='test_policy'
        )
        print("   [OK] Single policy created successfully")
        
        # Test 2: Run forward pass
        print("\n2. Testing forward pass...")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Create test data
            test_obs = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
            test_dec_inputs = np.random.randint(0, vocab_size, (batch_size, seq_len))
            test_dec_targets = test_dec_inputs.copy()
            test_length = np.full(batch_size, seq_len, dtype=np.int32)
            
            feed_dict = {
                policy.obs: test_obs,
                policy.decoder_inputs: test_dec_inputs,
                policy.decoder_targets: test_dec_targets,
                policy.decoder_full_length: test_length
            }
            
            # Run network
            outputs = sess.run(policy.network.decoder_logits, feed_dict)
            print(f"   [OK] Forward pass successful, output shape: {outputs.shape}")
            
            # Test 3: Get actions
            print("\n3. Testing action generation...")
            actions, logits, values = policy.get_actions(test_obs)
            print(f"   [OK] Actions generated, shape: {actions.shape}")
            print(f"   [OK] Logits shape: {logits.shape}")
            print(f"   [OK] Values shape: {values.shape}")
            
        # Test 4: Create meta policy
        print("\n4. Testing meta policy creation...")
        meta_batch_size = 5
        meta_policy = MetaSeq2SeqPolicy(
            meta_batch_size=meta_batch_size,
            obs_dim=obs_dim,
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            vocab_size=vocab_size
        )
        print(f"   [OK] Meta policy created with {meta_batch_size} tasks")
        
        # Test 5: Meta policy operations
        print("\n5. Testing meta policy operations...")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Sync parameters
            meta_policy.async_parameters()
            print("   [OK] Parameters synchronized")
            
            # Generate actions for multiple tasks
            meta_obs = [np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32) 
                       for _ in range(meta_batch_size)]
            
            meta_actions, meta_logits, meta_values = meta_policy.get_actions(meta_obs)
            print(f"   [OK] Meta actions generated for {len(meta_actions)} tasks")
            
        print("\n" + "="*60)
        print("[SUCCESS] All tests passed! Graph2Seq encoder is working correctly.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_graph2seq_fix()
    if not success:
        print("\nPlease check the error above and fix any remaining issues.")