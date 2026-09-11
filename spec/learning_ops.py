"""Pure-numpy learning contracts from LEARNING_PROTOCOL.md."""

from __future__ import annotations

import numpy as np


def clipped_value_prediction(v_old, v_new, epsilon: float):
    """PPO value clip: v_old + clip(v_new - v_old, -eps, eps)."""
    v_old = np.asarray(v_old, dtype=np.float64)
    v_new = np.asarray(v_new, dtype=np.float64)
    delta = np.clip(v_new - v_old, -float(epsilon), float(epsilon))
    return v_old + delta


def mean_pseudogradient(theta0, adapted, alpha: float, k_steps: int):
    """mean((theta0 - theta_i) / (alpha * k_steps)) for one outer apply_gradients."""
    if k_steps < 1:
        raise ValueError("k_steps must be a positive optimizer-step count")
    scale = float(alpha) * float(k_steps)
    n = len(adapted)
    if n < 1:
        raise ValueError("adapted parameter list is empty")
    grads = []
    for j, th0 in enumerate(theta0):
        acc = np.zeros_like(th0, dtype=np.float64)
        for theta_i in adapted:
            acc += (np.asarray(th0, dtype=np.float64) - np.asarray(theta_i[j], dtype=np.float64)) / scale
        grads.append(acc / float(n))
    return grads


def select_support_rows(n, k, rng):
    n = int(n)
    k = int(k)
    if n < k:
        raise ValueError("need %d trajectories, got %d" % (k, n))
    if n == k:
        return np.arange(n)
    return rng.choice(n, k, replace=False)


def instance_ids_from_order(n, n_instances):
    """Paths cycle support graphs: traj i belongs to instance i % n_instances."""
    n = int(n)
    n_instances = int(n_instances)
    if n < 1:
        raise ValueError("need at least one trajectory")
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    if n < n_instances:
        raise ValueError("need %d instances, got %d trajectories" % (n_instances, n))
    return np.arange(n, dtype=np.int64) % n_instances


def pomo_episode_advantages(episode_returns, instance_ids):
    """POMO/RLOO: A_i = R_i - mean(R of same instance). Mean-zero per instance."""
    returns = np.asarray(episode_returns, dtype=np.float64).reshape(-1)
    ids = np.asarray(instance_ids).reshape(-1)
    if returns.shape[0] != ids.shape[0]:
        raise ValueError(
            "returns/instance_ids length mismatch: %d vs %d" % (returns.shape[0], ids.shape[0])
        )
    adv = np.zeros_like(returns)
    for uid in np.unique(ids):
        mask = ids == uid
        group = returns[mask]
        adv[mask] = group - np.mean(group)
    return adv


def broadcast_token_advantages(episode_adv, n_tokens):
    adv = np.asarray(episode_adv, dtype=np.float64).reshape(-1)
    n_tokens = int(n_tokens)
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")
    return np.repeat(adv[:, None], n_tokens, axis=1)


def apply_pomo_token_advantages(rewards, n_instances):
    """Replace GAE with POMO advantages from undiscounted episode return."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError("rewards must be (n_traj, n_tokens), got %s" % (rewards.shape,))
    ids = instance_ids_from_order(rewards.shape[0], n_instances)
    ep_a = pomo_episode_advantages(rewards.sum(axis=-1), ids)
    return broadcast_token_advantages(ep_a, rewards.shape[1])


def select_elite_per_instance(costs, instance_ids, k):
    """One lowest-cost row per instance. Requires exactly k unique instances."""
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    ids = np.asarray(instance_ids).reshape(-1)
    k = int(k)
    if costs.shape[0] != ids.shape[0]:
        raise ValueError("costs/instance_ids length mismatch")
    order = []
    seen = set()
    for uid in ids.tolist():
        key = int(uid)
        if key not in seen:
            seen.add(key)
            order.append(key)
    if len(order) != k:
        raise ValueError("elite select needs %d instances, got %d" % (k, len(order)))
    picks = []
    for uid in order:
        mask = np.where(ids == uid)[0]
        picks.append(int(mask[int(np.argmin(costs[mask]))]))
    return np.asarray(picks, dtype=np.int64)


def shuffled_minibatch_slices(n, batch_size, rng):
    """Yield index arrays for one shuffled epoch. Leftover last slice is allowed."""
    n = int(n)
    batch_size = int(batch_size)
    if n < 1:
        raise ValueError("need at least one trajectory")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = rng.permutation(n)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield order[start:end]
        start = end


def expected_adam_apply_count(n, batch_size, k_steps):
    n = int(n)
    batch_size = int(batch_size)
    k_steps = int(k_steps)
    if k_steps < 0:
        raise ValueError("k_steps cannot be negative")
    if k_steps == 0:
        return 0
    slices = int(np.ceil(float(n) / float(batch_size)))
    return slices * k_steps


def composite_query_objective(rewards):
    """Mean per-trajectory return. Token rewards already encode frozen 0.5/0.5 J."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim == 0:
        return float(rewards)
    if rewards.ndim == 1:
        return float(np.mean(rewards))
    returns = rewards.sum(axis=-1)
    return float(np.mean(returns))
