import tensorflow as tf
import numpy as np
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies.meta_seq2seq_policy import tf2_dynamic_decode, TF2BasicDecoder


class TestHelperTraining:
    """Mock training helper for testing"""
    def __init__(self, inputs, sequence_length):
        self.inputs = inputs
        self.sequence_length = sequence_length


class TestCellWithTupleState(tf.keras.layers.Layer):
    """Test RNN cell that returns states as lists (to simulate the issue)"""
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.dense = tf.keras.layers.Dense(units)
    
    def __call__(self, inputs, states):
        # Simulate multi-layer LSTM state structure
        # States come in as nested tuples, but we return as lists to simulate the bug
        (layer1_state, layer2_state), context = states
        
        # Process input
        output = self.dense(inputs)
        
        # Simulate state updates that return lists instead of tuples
        new_layer1_state = [layer1_state[0] + 0.1, layer1_state[1] + 0.1]  # Returns list
        new_layer2_state = [layer2_state[0] + 0.1, layer2_state[1] + 0.1]  # Returns list
        new_context = context + 0.1
        
        # Return output and new states (as lists to trigger the original bug)
        return output, ([new_layer1_state, new_layer2_state], new_context)


class TestTF2DynamicDecode(unittest.TestCase):
    
    def test_state_structure_consistency(self):
        """Test that state structure remains consistent across iterations"""
        tf.random.set_seed(42)
        
        # Setup test parameters
        batch_size = 2
        sequence_length = 5
        input_dim = 10
        hidden_dim = 16
        
        # Create test inputs
        inputs = tf.random.normal([batch_size, sequence_length, input_dim])
        seq_lengths = tf.constant([sequence_length] * batch_size, dtype=tf.int32)
        
        # Create mock helper
        helper = TestHelperTraining(inputs, seq_lengths)
        
        # Create test cell
        cell = TestCellWithTupleState(hidden_dim)
        
        # Create initial state as nested tuples (matching the error structure)
        initial_state = (
            (
                tf.zeros([batch_size, hidden_dim]),  # layer 1, state c
                tf.zeros([batch_size, hidden_dim])   # layer 1, state h
            ),
            (
                tf.zeros([batch_size, hidden_dim]),  # layer 2, state c
                tf.zeros([batch_size, hidden_dim])   # layer 2, state h
            )
        ), tf.zeros([batch_size, hidden_dim * 2])  # attention context
        
        # Create decoder
        decoder = TF2BasicDecoder(cell, helper, initial_state)
        
        # Run dynamic decode
        outputs, final_state, _ = tf2_dynamic_decode(decoder)
        
        # Verify state structure is preserved
        try:
            tf.nest.assert_same_structure(initial_state, final_state)
            structure_matches = True
        except (ValueError, TypeError):
            structure_matches = False
        
        self.assertTrue(structure_matches,
                       "Final state structure doesn't match initial state structure")
        
        # Verify state is still tuples at all levels
        self.assertIsInstance(final_state, tuple, "Final state should be a tuple")
        self.assertIsInstance(final_state[0], tuple, "First element of state should be a tuple")
        self.assertIsInstance(final_state[0][0], tuple, "Nested state should be a tuple")
        self.assertIsInstance(final_state[0][1], tuple, "Nested state should be a tuple")
        
        # Verify shapes are preserved
        def check_shapes(initial, final, path=""):
            if isinstance(initial, tf.Tensor) and isinstance(final, tf.Tensor):
                self.assertEqual(initial.shape, final.shape, 
                               f"Shape mismatch at {path}: {initial.shape} vs {final.shape}")
            elif isinstance(initial, tuple) and isinstance(final, tuple):
                for i, (init_elem, final_elem) in enumerate(zip(initial, final)):
                    check_shapes(init_elem, final_elem, f"{path}[{i}]")
        
        check_shapes(initial_state, final_state)
        
        print("✓ State structure consistency test passed")
        print(f"  Initial state type structure: {self._get_type_structure(initial_state)}")
        print(f"  Final state type structure: {self._get_type_structure(final_state)}")
    
    def _get_type_structure(self, obj):
        """Helper to visualize type structure"""
        if isinstance(obj, tf.Tensor):
            return f"Tensor{obj.shape}"
        elif isinstance(obj, tuple):
            return f"tuple({', '.join(self._get_type_structure(x) for x in obj)})"
        elif isinstance(obj, list):
            return f"list[{', '.join(self._get_type_structure(x) for x in obj)}]"
        else:
            return str(type(obj))
    
    def test_single_iteration_state_preservation(self):
        """Test that state structure is preserved after just one iteration"""
        tf.random.set_seed(42)
        
        # Minimal test with just 1 timestep
        batch_size = 1
        sequence_length = 1
        input_dim = 5
        hidden_dim = 8
        
        # Create test inputs for single step
        inputs = tf.random.normal([batch_size, sequence_length, input_dim])
        seq_lengths = tf.constant([sequence_length] * batch_size, dtype=tf.int32)
        
        # Create mock helper
        helper = TestHelperTraining(inputs, seq_lengths)
        
        # Create test cell
        cell = TestCellWithTupleState(hidden_dim)
        
        # Create initial state
        initial_state = (
            (
                tf.zeros([batch_size, hidden_dim]),
                tf.zeros([batch_size, hidden_dim])
            ),
            (
                tf.zeros([batch_size, hidden_dim]),
                tf.zeros([batch_size, hidden_dim])
            )
        ), tf.zeros([batch_size, hidden_dim * 2])
        
        # Create decoder
        decoder = TF2BasicDecoder(cell, helper, initial_state)
        
        # Run dynamic decode
        outputs, final_state, _ = tf2_dynamic_decode(decoder)
        
        # Check structure preservation
        try:
            tf.nest.assert_same_structure(initial_state, final_state)
            structure_matches = True
        except (ValueError, TypeError):
            structure_matches = False
            
        self.assertTrue(structure_matches,
                       "State structure changed after single iteration")
        
        # Verify tuple types are preserved
        self.assertEqual(type(initial_state), type(final_state))
        self.assertEqual(type(initial_state[0]), type(final_state[0]))
        self.assertEqual(type(initial_state[0][0]), type(final_state[0][0]))
        
        print("✓ Single iteration state preservation test passed")


if __name__ == '__main__':
    unittest.main()