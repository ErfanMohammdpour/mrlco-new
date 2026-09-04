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
