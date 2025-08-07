#!/usr/bin/env python
"""Minimal test to find where the training loop hangs."""

import sys
import os
import numpy as np
import tensorflow as tf

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Setting environment...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Import required modules
print("\n1. Importing modules...")
from env.mec_offloading_envs.offloading_env import Resources
from utils import logger
from meta_algos.MRLCO import MRLCO
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline

print("✓ All imports successful")

# Ultra minimal params
META_BATCH_SIZE = 1
BATCH_SIZE = 1
GRAPH_NUMBER = 1

logger.configure(dir=None, format_strs=['stdout'])

print("\n2. Creating resource cluster...")
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0,
    bandwidth_dl=7.0
)
print("✓ Resource cluster created")

print("\n3. Creating offloading environment...")
from env.mec_offloading_envs.offloading_env import OffloadingEnvironment

# Get graph file paths
graph_file_paths = [
    "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
]

env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=BATCH_SIZE,
    graph_number=GRAPH_NUMBER,
    graph_file_paths=graph_file_paths,
    time_major=False
)
print(f"✓ Environment created: batch_size={BATCH_SIZE}, graph_number={GRAPH_NUMBER}")

print("\n4. Creating policy...")
meta_policy = MetaSeq2SeqPolicy(
    meta_batch_size=META_BATCH_SIZE,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2,
    lstm_layers=2
)
print("✓ Policy created")

print("\n5. Creating sampler...")
sampler = Seq2SeqMetaSampler(
    env=env,
    policy=meta_policy,
    rollouts_per_meta_task=1,
    meta_batch_size=META_BATCH_SIZE,
    max_path_length=20,
    parallel=False
)
print("✓ Sampler created")

print("\n6. Creating sampler processor...")
baseline = ValueFunctionBaseline()
sampler_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)
print("✓ Sampler processor created")

print("\n7. Creating MRLCO algorithm...")
algo = MRLCO(policy=meta_policy, meta_batch_size=META_BATCH_SIZE)
print("✓ Algorithm created")

print("\n8. Testing sampling (this is where it might hang)...")
print("   Calling sampler.obtain_samples()...")
sys.stdout.flush()

try:
    # Test just the sampling part
    paths = sampler.obtain_samples(log=False, log_prefix='')
    print(f"✓ Sampling successful! Got {len(paths)} meta-tasks")
    
    print("\n9. Testing sample processing...")
    samples_data = sampler_processor.process_samples(paths, log=False, log_prefix='')
    print(f"✓ Processing successful! Got {len(samples_data)} processed samples")
    
    print("\n10. Testing inner gradient step...")
    policy_losses, value_losses = algo.UpdatePPOTarget(samples_data)
    print(f"✓ Inner update successful! Policy losses: {len(policy_losses)}")
    
    print("\n11. Testing evaluation sampling...")
    new_paths = sampler.obtain_samples(log=False, log_prefix='')
    print(f"✓ Evaluation sampling successful!")
    
    print("\n12. Testing evaluation processing...")
    new_samples_data = sampler_processor.process_samples(new_paths, log=False, log_prefix='')
    print(f"✓ Evaluation processing successful! Got {len(new_samples_data)} samples")
    
    print("\n13. Testing meta-policy update...")
    algo.UpdateMetaPolicy()
    print("✓ Meta-policy update successful!")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED! The training loop should work.")
    print("="*50)
    
except KeyboardInterrupt:
    print("\n❌ Test interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR at current step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)