"""
Test tensor operations in Graph2Seq encoder to ensure TensorFlow compatibility.
"""
import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from policies.graph2seq_encoder import Graph2SeqEncoderAdapter


def test_tensor_operations():
    """Test individual tensor operations that might cause issues."""
    print("Testing Tensor Operations in Graph Mode")
    print("="*60)
    
    tf.reset_default_graph()
    
    # Test parameters
    batch_size = 4
    seq_len = 6
    feature_dim = 8
    hidden_dim = 16
    
    # Create test input
    test_input = tf.placeholder(tf.float32, [None, None, feature_dim], name="test_input")
    
    print("\n1. Testing sequence_to_graph conversion...")
    
    with tf.Session() as sess:
        # Create adapter
        adapter = Graph2SeqEncoderAdapter(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            bidirectional=False,
            mode='train'
        )
        
        # Test sequence_to_graph
        fw_adj, bw_adj, features, batch_nodes = adapter.sequence_to_graph(test_input)
        
        # Create test data
        test_data = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
        
        # Run the graph conversion
        result = sess.run({
            'fw_adj': fw_adj,
            'bw_adj': bw_adj,
            'features': features,
            'batch_nodes': batch_nodes,
            'input_shape': tf.shape(test_input)
        }, feed_dict={test_input: test_data})
        
        print(f"   Input shape: {test_data.shape}")
        print(f"   Forward adjacency shape: {result['fw_adj'].shape}")
        print(f"   Features shape: {result['features'].shape}")
        print(f"   Batch nodes shape: {result['batch_nodes'].shape}")
        
        # Verify shapes
        expected_total_nodes = batch_size * seq_len
        assert result['fw_adj'].shape == (expected_total_nodes, seq_len), \
            f"Adjacency shape mismatch: {result['fw_adj'].shape} != {(expected_total_nodes, seq_len)}"
        assert result['features'].shape == (expected_total_nodes, feature_dim), \
            f"Features shape mismatch: {result['features'].shape} != {(expected_total_nodes, feature_dim)}"
        assert result['batch_nodes'].shape == (batch_size, seq_len), \
            f"Batch nodes shape mismatch: {result['batch_nodes'].shape} != {(batch_size, seq_len)}"
        
        print("   [OK] All shapes correct!")
        
        # Test adjacency structure
        print("\n2. Checking adjacency structure...")
        # First sequence should have nodes 0-5 connecting to each other
        first_seq_adj = result['fw_adj'][:seq_len, :]
        print(f"   First sequence adjacency (first 3 nodes):")
        print(first_seq_adj[:3])
        
        # Verify first sequence connects to nodes 0-5
        expected_first_seq = np.tile(np.arange(seq_len), (seq_len, 1))
        assert np.array_equal(first_seq_adj, expected_first_seq), \
            "First sequence adjacency incorrect"
        
        # Second sequence should connect to nodes 6-11
        second_seq_adj = result['fw_adj'][seq_len:2*seq_len, :]
        expected_second_seq = np.tile(np.arange(seq_len, 2*seq_len), (seq_len, 1))
        assert np.array_equal(second_seq_adj, expected_second_seq), \
            "Second sequence adjacency incorrect"
        
        print("   [OK] Adjacency structure is correct!")
        
        print("\n3. Testing full encode operation...")
        # Test full encode
        outputs, state = adapter.encode(test_input)
        
        # Initialize variables before running
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        enc_result = sess.run({
            'outputs': outputs,
            'state': state if not isinstance(state, tuple) else state[0]
        }, feed_dict={test_input: test_data})
        
        print(f"   Encoder outputs shape: {enc_result['outputs'].shape}")
        print(f"   Encoder state c shape: {enc_result['state'].c.shape}")
        print(f"   Encoder state h shape: {enc_result['state'].h.shape}")
        
        # Verify output shapes
        assert enc_result['outputs'].shape == (batch_size, seq_len, 2 * hidden_dim), \
            f"Encoder output shape mismatch: {enc_result['outputs'].shape}"
        assert enc_result['state'].c.shape == (batch_size, 2 * hidden_dim), \
            f"Encoder state shape mismatch: {enc_result['state'].c.shape}"
        
        print("   [OK] Full encode operation successful!")
        
    print("\n" + "="*60)
    print("[SUCCESS] All tensor operations work correctly in graph mode!")
    print("="*60)


if __name__ == "__main__":
    test_tensor_operations()