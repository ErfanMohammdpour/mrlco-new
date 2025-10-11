"""
Configuration file for DRL PPO experiments.
Contains hyperparameters and data splits for training and evaluation.
"""

import os

# Data splits - train on all maps except 2, 21, 25 (for evaluation)
TRAIN_TASK_IDS = [i for i in range(1, 26) if i not in [2, 21, 25]]  # Maps 1-25, skip 2,21,25
EVAL_TASK_IDS = [2, 21, 25]      # Evaluation maps

# Training hyperparameters (simplified for faster training)
tasks_per_epoch = 5              # Reduced from 10 to 5
rollouts_per_task = 1            # Reduced from 3 to 1
ppo_epochs = 2                   # Reduced from 4 to 2
minibatch_size = 512             # Reduced from 2048 to 512

# PPO hyperparameters (match original project)
gamma = 0.99                     # Discount factor (same as original)
gae_lambda = 0.95                # GAE lambda parameter
clip_ratio = 0.2                 # PPO clipping ratio
ent_coef = 0.02                  # Entropy coefficient
vf_coef = 0.5                    # Value function coefficient
max_grad_norm = 0.5              # Gradient clipping norm
lr = 3e-4                        # Learning rate (same as original)

# Training configuration (for 2500 episodes)
num_epochs = 2500                # Total episodes (like original)
seed = 42                        # Random seed
time_major = False               # Expect [B,T,F] format

# Model architecture
encoder_units = 128              # Encoder hidden units
decoder_units = 128              # Decoder hidden units
num_layers = 1                   # Number of encoder layers (avoid tuple state issues)

# Optional aggregator
use_attentive_agg = False        # Use AttentiveStatisticsAggregator instead of MeanAggregator

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "env", "mec_offloaing_envs", "data", "meta_offloading_20")

# Graph file paths for training (all maps except eval maps)
TRAIN_GRAPH_PATHS = [
    os.path.join(data_dir, f"offload_random20_{i+1}", "random.20.") 
    for i in TRAIN_TASK_IDS
]

# Graph file paths for evaluation (3 maps)
EVAL_GRAPH_PATHS = [
    os.path.join(data_dir, f"offload_random20_{i+1}", "random.20.") 
    for i in EVAL_TASK_IDS
]

# Output directories
output_dir = os.path.join(base_dir, "drl_runs")
eval_dir = os.path.join(base_dir, "drl_eval")

# Logging
log_interval = 10                # Log every N epochs
save_interval = 5                # Save checkpoint every 5 epochs (more frequent)
