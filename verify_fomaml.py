#!/usr/bin/env python3
"""
Verify FOMAML implementation
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Clear any existing graph
tf.reset_default_graph()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_fomaml_implementation():
    """Verify that FOMAML is properly implemented"""
    print("=" * 60)
    print("FOMAML Implementation Verification")
    print("=" * 60)
    
    # 1. Check if FOMAML class exists and can be imported
    try:
        from meta_algos.FOMAML import FOMAML
        print("✅ FOMAML class imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FOMAML: {e}")
        return False
    
    # 2. Check if FOMAML has required methods
    required_methods = ['adapt_task', 'meta_update', 'evaluate_adapted_policy']
    for method in required_methods:
        if hasattr(FOMAML, method):
            print(f"✅ FOMAML has {method} method")
        else:
            print(f"❌ FOMAML missing {method} method")
            return False
    
    # 3. Check if meta_trainer.py uses FOMAML
    try:
        with open('meta_trainer.py', 'r') as f:
            content = f.read()
            if 'from meta_algos.FOMAML import FOMAML' in content:
                print("✅ meta_trainer.py imports FOMAML")
            else:
                print("❌ meta_trainer.py does not import FOMAML")
                return False
                
            if 'algo = FOMAML(' in content:
                print("✅ meta_trainer.py uses FOMAML algorithm")
            else:
                print("❌ meta_trainer.py does not use FOMAML algorithm")
                return False
                
            if 'split_support_query' in content:
                print("✅ meta_trainer.py uses support/query splitting")
            else:
                print("❌ meta_trainer.py does not use support/query splitting")
                return False
                
    except Exception as e:
        print(f"❌ Error reading meta_trainer.py: {e}")
        return False
    
    # 4. Check if sampler has split_support_query method
    try:
        from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
        if hasattr(Seq2SeqMetaSampler, 'split_support_query'):
            print("✅ Seq2SeqMetaSampler has split_support_query method")
        else:
            print("❌ Seq2SeqMetaSampler missing split_support_query method")
            return False
    except ImportError as e:
        print(f"❌ Failed to import Seq2SeqMetaSampler: {e}")
        return False
    
    # 5. Check if training loop has inner and outer loops
    try:
        with open('meta_trainer.py', 'r') as f:
            content = f.read()
            if 'Inner Loop: Task Adaptation' in content:
                print("✅ Training loop has inner loop for task adaptation")
            else:
                print("❌ Training loop missing inner loop")
                return False
                
            if 'Outer Loop: Meta-Update' in content:
                print("✅ Training loop has outer loop for meta-update")
            else:
                print("❌ Training loop missing outer loop")
                return False
    except Exception as e:
        print(f"❌ Error checking training loop: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 FOMAML Implementation Verification PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify_fomaml_implementation()
    
    if not success:
        print("\n💥 FOMAML Implementation Verification FAILED!")
        print("Please check the errors above and fix them.")
        sys.exit(1)
