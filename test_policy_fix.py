#!/usr/bin/env python
"""Test script to verify the Graph2Seq aggregator dimension fix"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Import the policy
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy

# Test parameters matching meta_trainer.py
META_BATCH_SIZE = 10

try:
    print("\nCreating MetaSeq2SeqPolicy...")
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=META_BATCH_SIZE, 
        obs_dim=17, 
        encoder_units=128, 
        decoder_units=128,
        vocab_size=2
    )
    print("✅ Policy created successfully!")
    
    # Test a forward pass
    print("\nTesting forward pass...")
    dummy_obs = tf.ones([1, 1, 17], dtype=tf.float32)
    dummy_decoder_inputs = tf.zeros([1, 1], dtype=tf.int32)
    dummy_decoder_targets = tf.zeros([1, 1], dtype=tf.int32)
    dummy_length = tf.constant([1], dtype=tf.int32)
    
    output = meta_policy.core_policy.network(
        dummy_obs, 
        dummy_decoder_inputs, 
        dummy_decoder_targets, 
        dummy_length, 
        training=False
    )
    print(f"✅ Forward pass successful! Output shape: {output[0].shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()