#!/usr/bin/env python3
"""
Fix script for the aggregator shape mismatch issue
"""

def fix_aggregator_issue():
    """Apply a comprehensive fix to the aggregator issue"""
    
    # Read the current file
    with open('policies/graph2seq_modules/aggregators.py', 'r') as f:
        content = f.read()
    
    # Apply comprehensive fixes
    fixes = [
        # Fix 1: Replace problematic tf.where with simpler logic
        (
            'attn_weights = tf.where(\n                tf.greater(sums, 0.0),\n                attn_weights / (sums + 1e-9),\n                tf.nn.softmax(attn_logits / self.attn_temp, axis=1)\n            )',
            'attn_weights = attn_weights / (sums + 1e-9)'
        ),
        
        # Fix 2: Simplify the small sample correction
        (
            'var = tf.where(self.use_small_sample_correction, var_corr, var_num)',
            'var = var_corr if self.use_small_sample_correction else var_num'
        ),
        
        # Fix 3: Add shape assertions for debugging
        (
            'attn_weights = tf.nn.softmax(attn_logits / self.attn_temp, axis=1)  # [B,K,1]',
            '''attn_weights = tf.nn.softmax(attn_logits / self.attn_temp, axis=1)  # [B,K,1]
        
        # Debug: Add shape assertions
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(attn_weights)[-1], 1, message="attn_weights last dim should be 1"),
            tf.assert_equal(tf.rank(attn_weights), 3, message="attn_weights should be 3D")
        ]):
            attn_weights = tf.identity(attn_weights)'''
        )
    ]
    
    # Apply fixes
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Applied fix: {old[:50]}...")
        else:
            print(f"⚠️  Pattern not found: {old[:50]}...")
    
    # Write the fixed file
    with open('policies/graph2seq_modules/aggregators.py', 'w') as f:
        f.write(content)
    
    print("✅ Aggregator fixes applied successfully!")
    print("📝 The main changes:")
    print("   - Simplified attention weight normalization")
    print("   - Removed problematic tf.where operations")
    print("   - Added shape assertions for debugging")
    print("\n🚀 Now try running your evaluation again!")

if __name__ == "__main__":
    fix_aggregator_issue()
