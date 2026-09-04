"""Frozen LEARNING_PROTOCOL.md log fields and eval helpers. Numpy only."""

from __future__ import annotations

import numpy as np

from spec.learning_ops import composite_query_objective
from spec.split_loader import split_version

REQUIRED_LOG_FIELDS = (
    "seed",
    "split_version",
    "support_graphs_per_meta_task",
    "support_trajectories_per_meta_task",
    "query_graph_count",
    "k_steps",
    "outer_update_count",
    "entropy_coefficient",
    "value_clip_epsilon",
    "ppo_batch_size_trajectories",
    "meta_batch_size_distributions",
    "inner_learning_rate",
    "outer_learning_rate",
    "outer_update_method",
    "hyperparameter_provenance.policy",
)


def require_sliced_task(task):
    if not isinstance(task, dict) or "dist_index" not in task or "graph_indices" not in task:
        raise ValueError(
            "task must be dict with dist_index and graph_indices; "
            "integer set_task clears the slice and leaks the full 100-graph pool"
        )
    return task


def protocol_log_kvs(
    seed,
    k_steps,
    outer_update_count,
    support_graphs_per_meta_task=20,
    support_trajectories_per_meta_task=20,
    query_graph_count=80,
    entropy_coefficient=0.0,
    value_clip_epsilon=0.2,
    ppo_batch_size_trajectories=20,
    meta_batch_size_distributions=10,
    inner_learning_rate=5e-4,
    outer_learning_rate=5e-4,
    outer_update_method="mrlco_first_order_mean_pseudogradient",
    hyperparameter_provenance_policy="fixed_literature_derived_defaults",
):
    return {
        "seed": int(seed),
        "split_version": split_version(),
        "support_graphs_per_meta_task": int(support_graphs_per_meta_task),
        "support_trajectories_per_meta_task": int(support_trajectories_per_meta_task),
        "query_graph_count": int(query_graph_count),
        "k_steps": int(k_steps),
        "outer_update_count": int(outer_update_count),
        "entropy_coefficient": float(entropy_coefficient),
        "value_clip_epsilon": float(value_clip_epsilon),
        "ppo_batch_size_trajectories": int(ppo_batch_size_trajectories),
        "meta_batch_size_distributions": int(meta_batch_size_distributions),
        "inner_learning_rate": float(inner_learning_rate),
        "outer_learning_rate": float(outer_learning_rate),
        "outer_update_method": str(outer_update_method),
        "hyperparameter_provenance.policy": str(hyperparameter_provenance_policy),
    }


def query_metrics_from_samples(samples_data):
    rewards = samples_data["rewards"]
    latency = samples_data["finish_time"]
    out = {
        "query_mean_return": composite_query_objective(rewards),
        "query_mean_latency": float(np.mean(latency)),
        "validation_query_composite_objective": composite_query_objective(rewards),
    }
    if "energy" in samples_data:
        energy = np.asarray(samples_data["energy"])
        if energy.ndim == 2:
            out["query_mean_energy"] = float(np.mean(energy.sum(axis=-1)))
        else:
            out["query_mean_energy"] = float(np.mean(energy))
    return out
