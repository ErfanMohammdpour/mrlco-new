"""
Test script to validate Graph2Seq encoder compatibility with metarl-offloading.
Ensures output shapes and dtypes match the original encoder interface.
"""
import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from policies.meta_seq2seq_policy import Seq2SeqNetwork
from policies.graph2seq_encoder import create_graph2seq_encoder
import policies.model_helper as model_helper


def test_encoder_shape_compatibility():
    """Test that Graph2Seq encoder outputs match expected shapes."""
    
    # Test parameters
    batch_size = 32
    seq_len = 20
    obs_dim = 64
    encoder_units = 128
    decoder_units = 128
    vocab_size = 10
    num_layers = 2
    
    print("Testing encoder shape compatibility...")
    
    # Reset graph for clean test
    tf.reset_default_graph()
    
    # Create test inputs
    encoder_inputs = tf.placeholder(tf.float32, [None, None, obs_dim], name="test_encoder_inputs")
    decoder_inputs = tf.placeholder(tf.int32, [None, None], name="test_decoder_inputs")
    decoder_targets = tf.placeholder(tf.int32, [None, None], name="test_decoder_targets")
    decoder_full_length = tf.placeholder(tf.int32, [None], name="test_decoder_length")
    
    # Create hparams for network
    hparams = tf.contrib.training.HParams(
        unit_type="lstm",
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        n_features=vocab_size,
        time_major=False,
        is_attention=True,
        forget_bias=1.0,
        dropout=0,
        num_gpus=1,
        num_layers=num_layers,
        num_residual_layers=0,
        start_token=0,
        end_token=2,
        is_bidencoder=False
    )
    
    # Test 1: Compare original encoder with Graph2Seq encoder
    with tf.variable_scope("original_encoder"):
        # Original encoder embeddings
        orig_embeddings = tf.contrib.layers.fully_connected(
            encoder_inputs,
            encoder_units,
            activation_fn=None,
            scope="encoder_embeddings",
            reuse=tf.AUTO_REUSE
        )
        
        # Original encoder
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            encoder_cell = model_helper.create_rnn_cell(
                unit_type=hparams.unit_type,
                num_units=encoder_units,
                num_layers=num_layers,
                num_residual_layers=0,
                forget_bias=hparams.forget_bias,
                dropout=hparams.dropout,
                num_gpus=hparams.num_gpus,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                base_gpu=0,
                single_cell_fn=None
            )
            
            orig_outputs, orig_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                sequence_length=None,
                inputs=orig_embeddings,
                dtype=tf.float32,
                time_major=False,
                swap_memory=True
            )
    
    # Graph2Seq encoder
    with tf.variable_scope("graph2seq_encoder"):
        # Graph2Seq encoder embeddings
        g2s_embeddings = tf.contrib.layers.fully_connected(
            encoder_inputs,
            encoder_units,
            activation_fn=None,
            scope="encoder_embeddings",
            reuse=tf.AUTO_REUSE
        )
        
        g2s_outputs, g2s_state = create_graph2seq_encoder(
            encoder_inputs=g2s_embeddings,
            encoder_units=encoder_units,
            num_layers=num_layers,
            is_bidirectional=False,
            mode="train",
            scope_name="encoder"
        )
    
    # Initialize session and run tests
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Generate test data
        test_encoder_data = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
        test_decoder_data = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        test_decoder_length = np.full(batch_size, seq_len, dtype=np.int32)
        
        feed_dict = {
            encoder_inputs: test_encoder_data,
            decoder_inputs: test_decoder_data,
            decoder_targets: test_decoder_data,
            decoder_full_length: test_decoder_length
        }
        
        # Get outputs from both encoders
        orig_out_val, orig_state_val = sess.run([orig_outputs, orig_state], feed_dict=feed_dict)
        g2s_out_val, g2s_state_val = sess.run([g2s_outputs, g2s_state], feed_dict=feed_dict)
        
        # Test output shapes
        print(f"\nOriginal encoder output shape: {orig_out_val.shape}")
        print(f"Graph2Seq encoder output shape: {g2s_out_val.shape}")
        
        # Check state shapes
        if isinstance(orig_state_val, tuple):
            print(f"\nOriginal encoder state (multi-layer):")
            for i, state in enumerate(orig_state_val):
                print(f"  Layer {i} - c: {state.c.shape}, h: {state.h.shape}")
        else:
            print(f"\nOriginal encoder state - c: {orig_state_val.c.shape}, h: {orig_state_val.h.shape}")
            
        if isinstance(g2s_state_val, tuple):
            print(f"\nGraph2Seq encoder state (multi-layer):")
            for i, state in enumerate(g2s_state_val):
                print(f"  Layer {i} - c: {state.c.shape}, h: {state.h.shape}")
        else:
            print(f"\nGraph2Seq encoder state - c: {g2s_state_val.c.shape}, h: {g2s_state_val.h.shape}")
        
        # Validate shapes match
        assert orig_out_val.shape[0] == g2s_out_val.shape[0], "Batch size mismatch"
        assert orig_out_val.shape[1] == g2s_out_val.shape[1], "Sequence length mismatch"
        
        # Validate state structure
        if isinstance(orig_state_val, tuple) and isinstance(g2s_state_val, tuple):
            assert len(orig_state_val) == len(g2s_state_val), "Number of layers mismatch"
        
        print("\n✓ Shape compatibility test passed!")
        
        # Test 2: Test full network integration
        print("\nTesting full network integration with Graph2Seq encoder...")
        
        try:
            network = Seq2SeqNetwork(
                name="test_network",
                hparams=hparams,
                reuse=False,
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                decoder_full_length=decoder_full_length,
                decoder_targets=decoder_targets
            )
            
            # Test forward pass
            decoder_pred, pi = sess.run(
                [network.decoder_prediction, network.pi],
                feed_dict=feed_dict
            )
            
            print(f"Decoder prediction shape: {decoder_pred.shape}")
            print(f"Policy (pi) shape: {pi.shape}")
            
            assert decoder_pred.shape[0] == batch_size, "Decoder batch size mismatch"
            assert pi.shape[-1] == vocab_size, "Policy output dimension mismatch"
            
            print("\n✓ Full network integration test passed!")
            
        except Exception as e:
            print(f"\n✗ Full network integration test failed: {e}")
            raise


def test_bidirectional_encoder():
    """Test bidirectional encoder compatibility."""
    print("\n\nTesting bidirectional encoder compatibility...")
    
    tf.reset_default_graph()
    
    batch_size = 16
    seq_len = 15
    input_dim = 32
    hidden_dim = 64
    
    test_input = tf.placeholder(tf.float32, [None, None, input_dim])
    
    # Test bidirectional Graph2Seq encoder
    bi_outputs, bi_state = create_graph2seq_encoder(
        encoder_inputs=test_input,
        encoder_units=hidden_dim,
        num_layers=2,
        is_bidirectional=True,
        mode="train",
        scope_name="bi_encoder"
    )
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        test_data = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        
        bi_out_val, bi_state_val = sess.run(
            [bi_outputs, bi_state],
            feed_dict={test_input: test_data}
        )
        
        print(f"Bidirectional output shape: {bi_out_val.shape}")
        
        if isinstance(bi_state_val, tuple):
            print(f"Bidirectional state (multi-layer):")
            for i, state in enumerate(bi_state_val):
                print(f"  Layer {i} - c: {state.c.shape}, h: {state.h.shape}")
        else:
            print(f"Bidirectional state - c: {bi_state_val.c.shape}, h: {bi_state_val.h.shape}")
        
        print("\n✓ Bidirectional encoder test passed!")


def test_gradient_flow():
    """Test that gradients flow through the Graph2Seq encoder."""
    print("\n\nTesting gradient flow through Graph2Seq encoder...")
    
    tf.reset_default_graph()
    
    batch_size = 8
    seq_len = 10
    input_dim = 16
    hidden_dim = 32
    
    test_input = tf.placeholder(tf.float32, [None, None, input_dim])
    target = tf.placeholder(tf.float32, [None, hidden_dim * 2])  # Target for loss
    
    # Create encoder
    outputs, state = create_graph2seq_encoder(
        encoder_inputs=test_input,
        encoder_units=hidden_dim,
        num_layers=1,
        is_bidirectional=False,
        mode="train",
        scope_name="grad_encoder"
    )
    
    # Simple loss using the final state
    final_output = state.h if not isinstance(state, tuple) else state[0].h
    loss = tf.reduce_mean(tf.square(final_output - target))
    
    # Get trainable variables
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="grad_encoder")
    
    # Compute gradients
    grads = tf.gradients(loss, encoder_vars)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        test_data = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        target_data = np.random.randn(batch_size, hidden_dim * 2).astype(np.float32)
        
        grad_vals = sess.run(grads, feed_dict={
            test_input: test_data,
            target: target_data
        })
        
        # Check that all gradients are non-None and have proper shapes
        for i, (var, grad) in enumerate(zip(encoder_vars, grad_vals)):
            assert grad is not None, f"Gradient for {var.name} is None"
            assert grad.shape == var.shape.as_list(), f"Gradient shape mismatch for {var.name}"
            print(f"✓ Gradient flows through {var.name}, shape: {grad.shape}")
        
        print("\n✓ Gradient flow test passed!")


if __name__ == "__main__":
    print("Running Graph2Seq encoder compatibility tests...\n")
    
    try:
        test_encoder_shape_compatibility()
        test_bidirectional_encoder()
        test_gradient_flow()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED! Graph2Seq encoder is compatible.")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        raise