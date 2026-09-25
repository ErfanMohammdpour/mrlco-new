#!/usr/bin/env python3
"""Phase P1 pilot metrics: additive, scheduler-replay-free instrumentation.

Pure helpers used by the trainer to log the pilot dashboard:
  * action mix (local/MEC/V2V) from the rollout actions that were already sampled;
  * policy/value loss, KL, clip fraction and gradient norm from the PPO update;
  * validation baselines (all-MEC and greedy) and the policy gaps;
  * parent-child co-location, cross-location edges, task counts and a resource
    utilisation summary from the SAME ScheduleResults the rollout produced;
  * the action-collapse flag.

Nothing here schedules; all inputs come from existing results/data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES
from env.mec_offloaing_envs.scheduler.model import Location

ACTION_KEYS = {
    "local": Location.UE,
    "mec": Location.MEC,
    "v2v": Location.HELPER,
}
MEC_COLLAPSE_SHARE = 0.95
COLLAPSE_WINDOW = 5


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("%s is not finite: %r" % (name, value))
    return number


def flatten_actions(samples_data_batch: Iterable) -> list[int]:
    """Every action token from the processed rollout samples (no new sampling)."""
    out: list[int] = []
    for samples in samples_data_batch or ():
        actions = samples.get("actions") if hasattr(samples, "get") else None
        if actions is None:
            continue
        for row in actions:
            for value in row:
                out.append(int(value))
    return out


def action_fractions(actions: Sequence[int]) -> dict[str, float]:
    total = len(actions)
    if total == 0:
        return {name: 0.0 for name in ACTION_KEYS}
    return {
        name: sum(1 for a in actions if int(a) == loc.to_action()) / float(total)
        for name, loc in ACTION_KEYS.items()
    }


def plan_summary(result, edges: Sequence[Sequence[int]] | None, n_tasks: int) -> dict:
    """Co-location / cross-location / task counts / utilisation from one result.

    `edges` are `[src, dst, bytes]` records of the scheduled graph. Zero-hop
    edges (same location) have no TRANSFER record, so the graph edges are needed
    to count them; co-location rate is measured over ALL parent-child edges.
    """
    locations = {int(tid): rec.location for tid, rec in result.tasks.items()}
    counts = {name: 0 for name in ACTION_KEYS}
    for loc in locations.values():
        for name, expected in ACTION_KEYS.items():
            if loc == expected:
                counts[name] += 1
    n = max(1, int(n_tasks))
    cross = 0
    total_edges = 0
    for edge in edges or ():
        src, dst = int(edge[0]), int(edge[1])
        if src not in locations or dst not in locations:
            continue
        total_edges += 1
        if locations[src] != locations[dst]:
            cross += 1
    makespan = float(result.makespan_seconds)
    utils = []
    for name in RESOURCE_NAMES:
        busy = sum(
            float(iv.end) - float(iv.start)
            for iv in result.resource_intervals
            if iv.resource == name
        )
        utils.append(busy / makespan if makespan > 0 else 0.0)
    return {
        "task_fraction/local": counts["local"] / n,
        "task_fraction/mec": counts["mec"] / n,
        "task_fraction/v2v": counts["v2v"] / n,
        "co_location_rate": (1.0 - cross / total_edges) if total_edges else 1.0,
        "cross_location_edges": float(cross),
        "total_edges": float(total_edges),
        "utilization_mean": sum(utils) / len(utils),
        "utilization_max": max(utils),
    }


def validation_gaps(policy_latency: float, all_mec_latency: float, greedy_latency: float) -> dict:
    p = _finite("policy_latency", policy_latency)
    m = _finite("all_mec_latency", all_mec_latency)
    g = _finite("greedy_latency", greedy_latency)
    return {
        "validation_gap_to_all_mec": p - m,
        "validation_gap_to_greedy": p - g,
        "validation_all_mec_latency": m,
        "validation_greedy_latency": g,
    }


def collapse_flag(
    mec_share_series: Sequence[float],
    policy_vs_all_mec_gap: float | None,
    policy_vs_greedy_gap: float | None,
    *,
    window: int = COLLAPSE_WINDOW,
    share_threshold: float = MEC_COLLAPSE_SHARE,
) -> bool:
    """Action-collapse stop: last `window` MEC shares > threshold and the policy
    has not beaten all-MEC (gap >= 0) or is worse than Greedy (gap > 0)."""
    if len(mec_share_series) < window:
        return False
    recent = list(mec_share_series)[-window:]
    if not all(s > share_threshold for s in recent):
        return False
    if policy_vs_all_mec_gap is None or policy_vs_greedy_gap is None:
        return False
    return policy_vs_all_mec_gap >= 0.0 or policy_vs_greedy_gap > 0.0


def update_metric_kvs(
    *,
    actions: Sequence[int],
    policy_loss_mean: float | None = None,
    value_loss_mean: float | None = None,
    approx_kl: float | None = None,
    clip_fraction: float | None = None,
    grad_norm: float | None = None,
) -> dict[str, float]:
    """Flat CSV KVs for one iteration; absent PPO metrics are simply omitted."""
    kvs = {"action_fraction/%s" % name: value for name, value in action_fractions(actions).items()}
    for key, value in (
        ("policy/policy_loss_mean", policy_loss_mean),
        ("policy/value_loss_mean", value_loss_mean),
        ("policy/approx_kl", approx_kl),
        ("policy/clip_fraction", clip_fraction),
        ("policy/grad_norm", grad_norm),
    ):
        if value is not None:
            kvs[key] = _finite(key, value)
    return kvs
