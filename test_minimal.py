#!/usr/bin/env python
"""Minimal test to verify TF2 training loop works"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting minimal test...")

import tensorflow as tf
import numpy as np
print(f"TensorFlow version: {tf.__version__}")

# Import everything needed
from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from meta_algos.MRLCO import MRLCO

print("Creating environment with minimal settings...")
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0, bandwidth_dl=7.0
)

env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=2,  # Minimal
    graph_number=2,  # Minimal
    graph_file_paths=[
        "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
    ],
    time_major=False
)

print("Getting greedy solution...")
action, greedy_finish_time = env.greedy_solution()

print("Creating components...")
baseline = ValueFunctionBaseline()
META_BATCH_SIZE = 1

policy = MetaSeq2SeqPolicy(
    meta_batch_size=META_BATCH_SIZE,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2
)

sampler = Seq2SeqMetaSampler(
    env=env,
    policy=policy,
    rollouts_per_meta_task=1,
    meta_batch_size=META_BATCH_SIZE,
    max_path_length=20000,
    parallel=False,
)

sample_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)

algo = MRLCO(
    policy=policy,
    meta_batch_size=META_BATCH_SIZE,
    meta_sampler=sampler,
    meta_sampler_process=sample_processor,
    inner_lr=1e-3,
    outer_lr=1e-3
)

print("\n" + "="*60)
print("Starting training iterations...")
print("="*60)

# Run 2 iterations
for itr in range(2):
    print(f"\n---------------- Iteration {itr} ----------------")
    print("Sampling...")
    
    task_ids = sampler.update_tasks()
    paths = sampler.obtain_samples(log=False, log_prefix='')
    
    print("Processing samples...")
    samples_data = sample_processor.process_samples(paths, log=False, log_prefix='')
    
    print("Updating policy...")
    policy_losses, value_losses = algo.UpdatePPOTarget(samples_data, batch_size=500)
    print(f"Policy loss: {np.mean(policy_losses):.4f}")
    
    print("Resampling...")
    new_paths = sampler.obtain_samples(log=False, log_prefix='')
    new_samples_data = sample_processor.process_samples(new_paths, log=False, log_prefix='')
    
    print("Meta update...")
    algo.UpdateMetaPolicy()
    
    print(f"✅ Iteration {itr} completed!")

print("\n" + "="*60)
print("✅ TEST SUCCESSFUL - TF2 TRAINING WORKS!")
print("="*60)