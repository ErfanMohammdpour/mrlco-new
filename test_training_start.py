#!/usr/bin/env python
"""Test script to verify if training loop starts successfully"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import numpy as np
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import utils.logger as logger
from env.mec_offloading_envs.offloading_env import Resources
from env.mec_offloading_envs.offloading_env import OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from baselines.vf_baseline import ValueFunctionBaseline
from meta_algos.MRLCO import MRLCO
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor

# Hyperparameters - reduced for quick testing
META_BATCH_SIZE = 2  # Reduced from 10
num_parallel = 2  # Reduced from 10

# Initialize environment
res_data = Resources(
    mec_process_capable=10.0 * 1024 * 1024,
    mobile_process_capable=1.0 * 1024 * 1024,
    bandwidth_up=1.0, bandwidth_dl=7.0
)

print("\nCreating environment...")
env = OffloadingEnvironment(
    user_num=3,  # Reduced from 10
    data_num=3,  # Reduced from 100
    dim_num=17,
    resource_data=res_data
)

print("Creating policy...")
meta_policy = MetaSeq2SeqPolicy(
    meta_batch_size=META_BATCH_SIZE,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2
)

print("Creating baseline...")
baseline = ValueFunctionBaseline()

print("Creating sampler...")
sampler = Seq2SeqMetaSampler(
    env=env,
    policy=meta_policy,
    batch_size=3000,  # Reduced from 6000
    meta_batch_size=META_BATCH_SIZE,
    num_parallel=num_parallel
)

print("Creating sample processor...")
sample_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)

print("\nCreating MRLCO algorithm...")
algo = MRLCO(
    policy=meta_policy,
    meta_learner=None,
    inner_lr=1e-3,
    meta_batch_size=META_BATCH_SIZE,
    num_inner_grad_steps=1,
    inner_batch_size=500,
    learning_rate=1e-3,
    num_ppo_steps=1,  # Reduced from 3
    clip_value=0.3,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_inner_step=1
)

print("\n" + "="*50)
print("Starting training - testing iteration 0 only...")
print("="*50)

try:
    # Test just the first iteration
    logger.log("********* Iteration 0 ************")
    
    # Sample
    logger.log("Obtaining samples...")
    paths = sampler.obtain_samples(log=True, log_prefix='')
    
    logger.log("Processing samples...")
    samples = sample_processor.process_samples(paths, log='all', log_prefix='')
    
    # Inner gradient update
    logger.log("Computing inner gradients...")
    algo.UpdatePPO(samples)
    
    print("\n✅ Training iteration 0 completed successfully!")
    print("The training loop is working properly.")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()