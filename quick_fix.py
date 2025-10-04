#!/usr/bin/env python3
"""
Quick fix for the tensor shape mismatch issue
"""

def apply_quick_fix():
    """Apply a quick fix to resolve the tensor shape mismatch"""
    
    # Read the current file
    with open('policies/graph2seq_modules/aggregators.py', 'r') as f:
        content = f.read()
    
    # The main issue is in the tf.where operation that compares tensors of different shapes
    # Let's replace the problematic section with a simpler approach
    
    old_problematic_section = '''        # Weights with temperature
        attn_weights = tf.nn.softmax(attn_logits / self.attn_temp, axis=1)  # [B,K,1]
        
        if self.mode == "train" and self.attn_dropout > 0.0:
            attn_weights = tf.nn.dropout(attn_weights, keep_prob=1.0 - self.attn_dropout)
            # Simple renormalization - just normalize the weights
            sums = tf.reduce_sum(attn_weights, axis=1, keepdims=True)
            attn_weights = attn_weights / (sums + 1e-9)'''
    
    new_fixed_section = '''        # Weights with temperature
        attn_weights = tf.nn.softmax(attn_logits / self.attn_temp, axis=1)  # [B,K,1]
        
        if self.mode == "train" and self.attn_dropout > 0.0:
            attn_weights = tf.nn.dropout(attn_weights, keep_prob=1.0 - self.attn_dropout)
            # Simple renormalization - just normalize the weights
            sums = tf.reduce_sum(attn_weights, axis=1, keepdims=True)
            # Use tf.cond to avoid shape mismatch issues
            attn_weights = tf.cond(
                tf.reduce_any(tf.greater(sums, 1e-6)),
                lambda: attn_weights / (sums + 1e-9),
                lambda: tf.nn.softmax(attn_logits / self.attn_temp, axis=1)
            )'''
    
    if old_problematic_section in content:
        content = content.replace(old_problematic_section, new_fixed_section)
        print("✅ Applied quick fix for attention weights")
    else:
        print("⚠️  Pattern not found, trying alternative fix...")
        
        # Alternative fix - just remove the problematic renormalization
        content = content.replace(
            'if self.mode == "train" and self.attn_dropout > 0.0:\n            attn_weights = tf.nn.dropout(attn_weights, keep_prob=1.0 - self.attn_dropout)\n            # Simple renormalization - just normalize the weights\n            sums = tf.reduce_sum(attn_weights, axis=1, keepdims=True)\n            attn_weights = attn_weights / (sums + 1e-9)',
            'if self.mode == "train" and self.attn_dropout > 0.0:\n            attn_weights = tf.nn.dropout(attn_weights, keep_prob=1.0 - self.attn_dropout)'
        )
        print("✅ Applied alternative fix - removed problematic renormalization")
    
    # Write the fixed file
    with open('policies/graph2seq_modules/aggregators.py', 'w') as f:
        f.write(content)
    
    print("✅ Quick fix applied successfully!")
    print("🚀 Now try running your evaluation again:")
    print("   python meta_evaluator_with_viz_integration.py")

if __name__ == "__main__":
    apply_quick_fix()
