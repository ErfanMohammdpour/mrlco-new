#!/usr/bin/env python3
"""The single objective/metric/selection contract (v1).

Problem this file fixes
-----------------------
`spec/learning_ops.composite_query_objective` sums token rewards *without* the
shaping discount:

    sum_t r_t = J_0 + (1 - gamma) * sum_{t<N} J_t - gamma * J_N

while PPO optimises the discounted return

    sum_t gamma^(t-1) r_t = J_0 - gamma^N * J_N

Those are different functionals of the same rollout, so the logged and
checkpoint-selecting scalar was not the quantity the optimiser maximises. This
module defines ONE criterion, gives it a versioned name, and exposes the pure
helpers used by the trainer, the held-out evaluator and the tests.

The contract
------------
* reward mode ``latency_only``: ``J_t = L_t / L_scale``, ``r_t = J_{t-1} - gamma*J_t``.
* the optimised/persisted criterion is the **discounted return**
  ``R = mean_trajectories sum_t gamma^(t-1) r_t`` (higher is better);
* because ``N`` (tokens per plan) is fixed and ``J_0`` is a per-graph constant
  that no policy can change, ``R`` and the terminal plan objective ``J_N`` are
  related by an exact affine map, so ranking checkpoints by ``R`` is the same as
  ranking them by ``-J_N`` (see ``affine_relation`` / ``ranking_agrees``);
* the seconds-valued ``query_mean_latency`` and the legacy undiscounted sum stay
  available for continuity, but are explicitly labelled as companions, never as
  the criterion.
"""

from __future__ import annotations

import math

import numpy as np

SCHEMA = "objective_contract_v1"
REWARD_MODE = "latency_only"
SHAPING_DISCOUNT = 0.99
TASKS_PER_GRAPH = 20
TOKENS_PER_PLAN = TASKS_PER_GRAPH

#: CSV column that carries the criterion (higher is better).
METRIC_PRIMARY = "discounted_return"
#: terminal plan objective J_N (derived; the "what the plan costs" view).
METRIC_TERMINAL = "plan_objective_terminal"
#: the pre-contract scalar, kept only so old runs stay readable.
METRIC_LEGACY = "legacy_undiscounted_sum"

SELECTION_METRIC_DEFAULT = METRIC_PRIMARY
SELECTION_METRICS = (METRIC_PRIMARY, METRIC_LEGACY)

#: held-out metric keys, as produced by ``spec.eval_protocol.query_metrics_from_samples``.
SAMPLES_DATA_KEYS = {
    METRIC_PRIMARY: "query_discounted_return",
    METRIC_LEGACY: "query_legacy_undiscounted_sum",
    METRIC_TERMINAL: "query_mean_latency",
}


def selection_value(metric: str, metrics: dict) -> float:
    """Extract the selection scalar for `metric` from a metrics dict."""
    metric = str(metric)
    if metric not in SELECTION_METRICS:
        raise ValueError(
            "selection metric must be one of %s, got %r" % (list(SELECTION_METRICS), metric)
        )
    key = SAMPLES_DATA_KEYS[metric]
    if key not in metrics:
        raise KeyError("metrics dict has no %r (%s)" % (key, metric))
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError("%s is not finite: %r" % (metric, metrics[key]))
    return value


def higher_is_better(metric: str = METRIC_PRIMARY) -> bool:
    """Both supported criteria are rewards, so both are maximised."""
    if str(metric) not in SELECTION_METRICS:
        raise ValueError("unknown selection metric %r" % (metric,))
    return True


def discounted_return(rewards, discount: float = SHAPING_DISCOUNT) -> float:
    """Mean over trajectories of ``sum_t gamma^(t-1) r_t`` (PPO's objective)."""
    arr = np.asarray(rewards, dtype=np.float64)
    return float(np.mean(_per_trajectory_discounted(arr, discount)))


def legacy_undiscounted_sum(rewards) -> float:
    """The pre-contract scalar: mean over trajectories of ``sum_t r_t``."""
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.ndim <= 1:
        return float(np.mean(arr))
    return float(np.mean(arr.sum(axis=-1)))


def _per_trajectory_discounted(arr: np.ndarray, discount: float) -> np.ndarray:
    discount = float(discount)
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1], got %s" % discount)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("rewards must be [trajectories, tokens], got %s" % (arr.shape,))
    n = arr.shape[-1]
    weights = discount ** np.arange(n, dtype=np.float64)
    return arr @ weights


def telescoped_from_potentials(potentials, discount: float = SHAPING_DISCOUNT) -> float:
    """``J_0 - gamma^N J_N`` for one plan's potential series ``J_0..J_N``."""
    j = np.asarray(potentials, dtype=np.float64).reshape(-1)
    if j.size < 2:
        raise ValueError("need at least J_0 and J_1, got %d values" % j.size)
    n = j.size - 1
    return float(j[0] - (float(discount) ** n) * j[-1])


def legacy_sum_from_potentials(potentials, discount: float = SHAPING_DISCOUNT) -> float:
    """``sum_t r_t`` for a potential series, i.e. the pre-contract scalar.

    ``sum_{t=1..N} (J_{t-1} - gamma J_t) = J_0 + (1-gamma) sum_{t<N} J_t - gamma J_N``
    """
    j = np.asarray(potentials, dtype=np.float64).reshape(-1)
    if j.size < 2:
        raise ValueError("need at least J_0 and J_1, got %d values" % j.size)
    gamma = float(discount)
    middle = float(np.sum(j[1:-1])) if j.size > 2 else 0.0
    return float(j[0] + (1.0 - gamma) * middle - gamma * j[-1])


def terminal_objective_from_return(
    return_value: float, j0: float, discount: float = SHAPING_DISCOUNT, n_tokens: int = TOKENS_PER_PLAN
) -> float:
    """Invert the telescoping identity: ``J_N = (J_0 - R) / gamma^N``."""
    n_tokens = int(n_tokens)
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")
    return float((float(j0) - float(return_value)) / (float(discount) ** n_tokens))


def affine_relation(returns, terminals, discount: float = SHAPING_DISCOUNT,
                    n_tokens: int = TOKENS_PER_PLAN) -> dict:
    """Exact ``R = mean(J_0) - gamma^N * mean(J_N)`` check over a graph set.

    `returns`/`terminals` are per-graph means (the same graphs for every
    checkpoint). Returns the implied constant and the residual so callers can
    assert the relation instead of assuming it.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    t = np.asarray(terminals, dtype=np.float64).reshape(-1)
    if r.size != t.size:
        raise ValueError("returns and terminals must have the same length")
    gamma_n = float(discount) ** int(n_tokens)
    implied_j0 = float(np.mean(r + gamma_n * t))
    residual = float(np.max(np.abs(r - (implied_j0 - gamma_n * t)))) if r.size else 0.0
    return {
        "implied_mean_j0": implied_j0,
        "gamma_pow_n": gamma_n,
        "max_abs_residual": residual,
        "exact": bool(residual <= 1e-9),
    }


def ranking_agrees(values_a, values_b, *, higher_is_better_a: bool = True,
                   higher_is_better_b: bool = True) -> bool:
    """True when two scalar series induce the same (strict) ordering."""
    a = [float(v) for v in values_a]
    b = [float(v) for v in values_b]
    if len(a) != len(b):
        raise ValueError("series must have equal length")

    def sign(x):
        return (1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    def rank_key(values, higher):
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=bool(higher))
        return order

    return rank_key(a, higher_is_better_a) == rank_key(b, higher_is_better_b)


def contract_log_kvs() -> dict:
    """Self-describing KVs: written into every CSV row that reports the criterion."""
    return {
        "objective_contract/schema": SCHEMA,
        "objective_contract/reward_mode": REWARD_MODE,
        "objective_contract/discount": float(SHAPING_DISCOUNT),
        "objective_contract/selection_metric": SELECTION_METRIC_DEFAULT,
        "objective_contract/tokens_per_plan": int(TOKENS_PER_PLAN),
    }
