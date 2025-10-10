"""
DRL Experiments Package

This package contains implementations for vanilla Deep Reinforcement Learning (PPO)
experiments on the task offloading problem, designed to work alongside the existing
Meta-RL (MRLCO) codebase.

Key Components:
- configs.py: Configuration and hyperparameters
- policy.py: DRL policy with autoregressive decoder
- rollout.py: Rollout collection utilities
- gae.py: Generalized Advantage Estimation
- ppo.py: Proximal Policy Optimization implementation
- train_drl.py: Main training script
- eval_drl.py: Evaluation script with zero-shot and fine-tuning modes
- attentive_stats_agg.py: Optional attentive aggregator

Usage:
    Training:
        python -m drl_experiments.train_drl --config drl_experiments.configs
    
    Evaluation:
        python -m drl_experiments.eval_drl --ckpt drl_runs/ckpt_epoch_XXX.ckpt --mode zero_shot
        python -m drl_experiments.eval_drl --ckpt drl_runs/ckpt_epoch_XXX.ckpt --mode finetune --finetune_steps 20

The training uses the same 22 maps as MRLCO for fair comparison, and evaluates
on 3 unseen maps (default: {21, 2, 25}).
"""

from .configs import *
from .policy import DRLPolicy
from .rollout import collect_rollout, collect_rollouts, batch_rollouts
from .gae import compute_gae, compute_gae_tf, normalize_advantages
from .ppo import compute_ppo_loss, train_one_epoch, compute_makespan
from .attentive_stats_agg import AttentiveStatisticsAggregator, create_attentive_aggregator

__version__ = "1.0.0"
__author__ = "DRL Experiments Team"
