import tensorflow as tf
import numpy as np
import sys

print(f"TensorFlow version: {tf.__version__}")

# Disable GPU for faster testing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from baselines.vf_baseline import ValueFunctionBaseline

print("Creating minimal environment...")
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0, 
    bandwidth_dl=7.0
)

env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=2,  # Small batch
    graph_number=2,  # Small number
    graph_file_paths=[
        "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
    ],
    time_major=False
)

print("Environment created!")

# Hardcoded values from meta_trainer.py
OBS_DIM = 17
VOCAB_SIZE = 2
print(f"Using obs_dim={OBS_DIM}, vocab_size={VOCAB_SIZE}")

print("\nCreating baseline...")
baseline = ValueFunctionBaseline()

print("Creating policy with META_BATCH_SIZE=1...")
META_BATCH_SIZE = 1

try:
    with tf.name_scope("seq2seq_policy"):
        policy = MetaSeq2SeqPolicy(
            meta_batch_size=META_BATCH_SIZE,
            obs_dim=OBS_DIM,
            encoder_units=128,
            decoder_units=128,
            vocab_size=VOCAB_SIZE
        )
    print("✓ Policy created successfully!")
except Exception as e:
    print(f"✗ Failed to create policy: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting policy forward pass...")
try:
    # Create dummy input
    batch_size = 2
    seq_len = 20
    obs = np.random.randn(batch_size, seq_len, OBS_DIM).astype(np.float32)
    
    # Test get_actions
    print(f"Input shape: {obs.shape}")
    actions, logits, values = policy.get_actions([obs])  # Wrap in list for meta-batch
    print(f"Actions shape: {actions[0].shape}")
    print(f"Logits shape: {logits[0].shape}")
    print(f"Values shape: {values[0].shape}")
    print("✓ Policy forward pass successful!")
    
except Exception as e:
    print(f"✗ Policy forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!")