#!/usr/bin/env python
"""Quick test to check if iteration 0 completes after the fix."""

import sys
import numpy as np
import tensorflow as tf
from meta_trainer import Trainer
from env.mec_offloading_envs.offloading_env import Resources
from utils import logger
from meta_algos.MRLCO import MRLCO
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline

# Tiny params for quick testing
META_BATCH_SIZE = 1
BATCH_SIZE = 2
GRAPH_NUMBER = 2
num_iterations = 1  # Only test iteration 0

logger.configure(dir=None, format_strs=['stdout'])

# Create environment
env = Resources(
    batch_size=BATCH_SIZE,
    graph_number=GRAPH_NUMBER,
    distance_range=100
)

# Create policy
meta_policy = MetaSeq2SeqPolicy(
    meta_batch_size=META_BATCH_SIZE,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2,
    lstm_layers=2
)

# Create sampler
sampler = Seq2SeqMetaSampler(
    env=env,
    policy=meta_policy,
    rollouts_per_meta_task=1,
    meta_batch_size=META_BATCH_SIZE,
    max_path_length=20,
    parallel=False
)

# Create sampler processor
baseline = ValueFunctionBaseline()
sampler_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)

# Create MRLCO algorithm
algo = MRLCO(
    policy=meta_policy,
    meta_batch_size=META_BATCH_SIZE
)

# Create trainer
trainer = Trainer(
    algo=algo,
    env=env,
    sampler=sampler,
    sampler_processor=sampler_processor,
    policy=meta_policy,
    n_itr=num_iterations
)

print("\n" + "="*50)
print("Starting training with fixed evaluation loop...")
print("="*50)

try:
    avg_ret, avg_loss, avg_latencies = trainer.train()
    print("\n" + "="*50)
    print("✅ SUCCESS! Iteration 0 completed successfully!")
    print(f"Average return: {avg_ret[-1] if avg_ret else 'N/A'}")
    print(f"Average loss: {avg_loss[-1] if avg_loss else 'N/A'}")
    print(f"Average latency: {avg_latencies[-1] if avg_latencies else 'N/A'}")
    print("="*50)
except Exception as e:
    print("\n" + "="*50)
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("="*50)
    sys.exit(1)