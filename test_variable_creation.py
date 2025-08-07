"""Test if variable creation in tf.function is fixed."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np

print("Testing variable creation fix...")

# Import the necessary modules
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from meta_algos.MRLCO import MRLCO

# Set minimal parameters
hparams = {
    'n_actions': 2,
    'n_features': 2,
    'encoder_hidden_unit': 128,
    'encoder_units': 128,
    'decoder_hidden_unit': 128,
    'decoder_units': 128,
    'num_layers': 2,
    'learning_rate': 1e-3,
    'inner_lr': 1e-3,
    'encoder_type': 'graph2seq',
    'decoder_type': 'luong',
    'attention': True,
    'hidden_size': 128,
    'vocab_size': 2,
    'embedding_size': 128,
    'inner_unit': 128,
    'clip_value': 0.3,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'is_bidir_encoder': False,
    'is_attention': True,
    'start_symbol': 0,
    'end_symbol': 1
}

print("Creating policy...")
# Create meta policy
policy = MetaSeq2SeqPolicy(
    meta_batch_size=2,
    obs_dim=17,
    encoder_units=128,
    decoder_units=128,
    vocab_size=2
)

print("Creating algorithm...")
algo = MRLCO(
    policy=policy,
    meta_batch_size=2,
    meta_sampler=None,
    meta_sampler_process=None,
    outer_lr=1e-3,
    inner_lr=1e-3,
    num_inner_grad_steps=1,
    clip_value=0.3,
    vf_coef=0.5,
    max_grad_norm=0.5
)

print("Building policies...")
dummy_input = np.zeros((100, 20, 17), dtype=np.float32)
decoder_inputs = np.zeros((100, 20), dtype=np.int32)
actions = np.zeros((100, 20), dtype=np.int32)
decoder_full_length = np.ones((100,), dtype=np.int32) * 20

# Build policy by calling it once
policy.meta_policies[0].network(
    dummy_input, 
    decoder_inputs, 
    actions, 
    decoder_full_length,
    training=False
)

print("\nCreating test data...")
# Create dummy task samples
task_samples = []
for i in range(2):
    samples = {
        'observations': np.random.randn(100, 20, 17).astype(np.float32),
        'actions': np.random.randint(0, 2, (100, 20)).astype(np.int32),
        'decoder_inputs': np.zeros((100, 20), dtype=np.int32),
        'decoder_full_length': np.ones((100,), dtype=np.int32) * 20,
        'rewards': np.random.randn(100).astype(np.float32),
        'returns': np.random.randn(100).astype(np.float32),
        'advantages': np.random.randn(100).astype(np.float32),
        'logprobs': np.random.randn(100).astype(np.float32),
        'values': np.random.randn(100).astype(np.float32),
        'logits': np.random.randn(100, 20, 2).astype(np.float32)  # Added logits field
    }
    task_samples.append(samples)

print("\nTrying to update PPO target (this will trigger tf.function)...")
try:
    policy_losses, value_losses = algo.UpdatePPOTarget(task_samples, batch_size=100)
    print("✓ SUCCESS: No variable creation error!")
    print(f"  Policy losses: {policy_losses}")
    print(f"  Value losses: {value_losses}")
except ValueError as e:
    if "tf.function only supports singleton tf.Variables" in str(e):
        print("✗ FAILED: Variable creation error still present!")
        print(f"  Error: {e}")
        exit(1)
    else:
        raise

print("\nAll tests passed! The variable creation issue is fixed.")