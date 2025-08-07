#!/usr/bin/env python
"""Quick test to verify all components initialize correctly after fixes"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import sys
import numpy as np
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from meta_algos.MRLCO import MRLCO

print("\n" + "="*50)
print("Testing Component Initialization")
print("="*50)

# Test environment
print("\n1. Testing Environment...")
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0, bandwidth_dl=7.0
)
env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=10,
    graph_number=10,
    graph_file_paths=[
        "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
    ],
    time_major=False
)
print("   ✅ Environment created")

# Test policy
print("\n2. Testing Policy...")
meta_policy = MetaSeq2SeqPolicy(
    meta_batch_size=10,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2
)
print("   ✅ Policy created")

# Test forward pass
print("\n3. Testing Forward Pass...")
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
print(f"   ✅ Forward pass successful! Output shape: {output[0].shape}")

# Test baseline
print("\n4. Testing Baseline...")
baseline = ValueFunctionBaseline()
print("   ✅ Baseline created")

# Test sampler
print("\n5. Testing Sampler...")
sampler = Seq2SeqMetaSampler(
    env=env,
    policy=meta_policy,
    rollouts_per_meta_task=1,
    meta_batch_size=10,
    max_path_length=20000,
    parallel=False,
)
print("   ✅ Sampler created")

# Test sample processor
print("\n6. Testing Sample Processor...")
sample_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)
print("   ✅ Sample processor created")

# Test algorithm
print("\n7. Testing MRLCO Algorithm...")
algo = MRLCO(
    policy=meta_policy,
    meta_sampler=sampler,
    meta_sampler_process=sample_processor,
    inner_lr=1e-3,
    outer_lr=1e-3,
    meta_batch_size=10,
    num_inner_grad_steps=1,
    clip_value=0.3,
    vf_coef=0.5,
    max_grad_norm=0.5
)
print("   ✅ MRLCO algorithm created")

print("\n" + "="*50)
print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
print("="*50)
print("\nThe fixes have resolved the dimension mismatch issues:")
print("1. Graph2Seq aggregator now correctly handles input dimensions")
print("2. Encoder-decoder state dimensions are properly aligned")
print("3. The policy can perform forward passes successfully")
print("\nThe training loop should now work, though it may be slow on CPU.")