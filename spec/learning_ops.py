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
