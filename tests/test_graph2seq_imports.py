"""
Test script to verify all Graph2Seq modules are properly imported and self-contained.
"""
import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os

# Add the policies directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Graph2Seq module imports...")
print("="*60)

try:
    # Test individual module imports
    print("\n1. Testing individual module imports:")
    
    from policies.graph2seq_modules import inits
    print("   [OK] inits module imported successfully")
    
    from policies.graph2seq_modules import layers
    print("   [OK] layers module imported successfully")
    
    from policies.graph2seq_modules import pooling
    print("   [OK] pooling module imported successfully")
    
    from policies.graph2seq_modules import neigh_samplers
    print("   [OK] neigh_samplers module imported successfully")
    
    from policies.graph2seq_modules import aggregators
    print("   [OK] aggregators module imported successfully")
    
    # Test specific class imports
    print("\n2. Testing specific class imports:")
    
    from policies.graph2seq_modules.neigh_samplers import UniformNeighborSampler
    print("   [OK] UniformNeighborSampler imported successfully")
    
    from policies.graph2seq_modules.aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator
    print("   [OK] Aggregator classes imported successfully")
    
    from policies.graph2seq_modules.inits import glorot, zeros
    print("   [OK] Initialization functions imported successfully")
    
    from policies.graph2seq_modules.layers import Layer, Dense
    print("   [OK] Layer classes imported successfully")
    
    # Test graph2seq_encoder import
    print("\n3. Testing graph2seq_encoder import:")
    
    from policies.graph2seq_encoder import Graph2SeqEncoderAdapter, create_graph2seq_encoder
    print("   [OK] Graph2SeqEncoderAdapter imported successfully")
    print("   [OK] create_graph2seq_encoder imported successfully")
    
    # Test basic functionality
    print("\n4. Testing basic functionality:")
    
    import tensorflow as tf
    import numpy as np
    
    # Test initializers
    test_shape = [10, 20]
    glorot_var = glorot(test_shape)
    zeros_var = zeros(test_shape)
    print("   [OK] Initializers work correctly")
    
    # Test Layer creation
    test_layer = Layer()
    print("   [OK] Layer instantiation works")
    
    # Test encoder adapter creation
    adapter = Graph2SeqEncoderAdapter(
        input_dim=64,
        hidden_dim=128,
        num_layers=2,
        bidirectional=False,
        mode='train'
    )
    print("   [OK] Graph2SeqEncoderAdapter instantiation works")
    
    print("\n" + "="*60)
    print("[OK] ALL IMPORTS SUCCESSFUL! Graph2Seq modules are self-contained.")
    print("="*60)
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    print("="*60)
    raise
except Exception as e:
    print(f"\n[ERROR] Unexpected Error: {e}")
    print("="*60)
    raise