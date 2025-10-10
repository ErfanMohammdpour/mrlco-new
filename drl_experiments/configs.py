"""
Configuration file for DRL PPO experiments.
Contains hyperparameters and data splits for training and evaluation.
"""

import os

# Data splits - train on all maps except 21, 2 (for evaluation)
TRAIN_TASK_IDS = [i for i in range(25) if i not in [21, 2]]  # All maps except eval maps (0-24, skip 21,2)
EVAL_TASK_IDS = [21, 2]      # Evaluation maps

# Training hyperparameters (adjusted for DRL PPO)
tasks_per_epoch = 10             # Number of tasks to sample per epoch
rollouts_per_task = 3            # Number of rollouts per task
ppo_epochs = 4                   # PPO epochs per batch
minibatch_size = 2048            # Minibatch size for PPO updates (larger for stability)

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
save_interval = 20               # Save checkpoint every N epochs
