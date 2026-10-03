#!/usr/bin/env python3
"""v2 CRN evaluation loop with ACTUAL R (environmental realization) and S (policy sampling) axes.

Executed without TensorFlow by substituting a documented **numpy reference policy** for the TF
PPO policy. The PROTOCOL is the frozen one (`automotive_crn_gumbel_v1`); only the network that
would produce the logits is replaced. The TF policy path remains NOT RUN and is reported as
such — this module never claims to have evaluated a trained policy.

Definitions (frozen):
    R = the environmental-realization axis. Replicate r rebuilds the world with an
        r-identity-keyed stochastic environment (realized link process, helper draws) while the
        exogenous randomness is keyed by STABLE identities, never by RNG call order.
    S = the policy-sampling axis. Sample s draws the action vector with Gumbel noise keyed by
        (protocol_id, r, s, graph_id), so every candidate and checkpoint sees the SAME noise
        for a given (r, s) — that is the CRN pairing.

Aggregation is NESTED: average over s inside a replicate, then over the R replicate means; the
standard error is computed on the replicate means (R independent units). Best-of-S selection is
never used. Deterministic baselines are marked `deterministic: true` and are still evaluated on
every (r, s) cell so the pairing is exact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "spec" / "automotive_training" / "reports" / "v2_system_model"

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2.adapters import plan_map_from_actions, pure_plan  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    spec_from_fractions, v2_metrics,
)
from spec.automotive_training.v2.crn import (  # noqa: E402
    DEFAULT_R_SELECT, DEFAULT_S_SELECT, PROTOCOL_ID, gumbel_argmax, gumbel_noise,
    nested_aggregate, paired_delta, protocol_sha, seed_pair,
)
from spec.automotive_training.v2.energy import schedule_energy  # noqa: E402
from spec.automotive_training.v2.link_model import make_process  # noqa: E402
from spec.automotive_training.v2.world import (  # noqa: E402
    V2World, V2WorldConfig, build_world,
)

#: candidate baselines. `all_*` are pure-location references; `greedy_cd` is a declared
#: latency-aware coordinate-descent baseline with a 1-sweep computational budget.
DETERMINISTIC_CANDIDATES = ("all_ue", "all_mec", "all_helper", "greedy_cd")
STOCHASTIC_CANDIDATE = "gumbel_reference_policy"

#: extra plan-level quantities that are aggregated alongside the metrics
EXTRA_KEYS = ("helper_task_fraction", "v2v_airtime_s")
#: every quantity that is aggregated over the R and S axes
AGG_KEYS = ("episode_latency_s", "world_makespan_s", "system_joules",
            "requester_joules", "mobile_joules", "deadline_miss_rate",
            "cpu_wait_mean_s", "wasted_work_bytes") + EXTRA_KEYS

METRIC_KEYS = ("episode_latency_s", "world_makespan_s", "system_joules",
               "requester_joules", "mobile_joules", "deadline_miss_rate",
               "cpu_wait_mean_s", "wasted_work_bytes")


class EvalLoopError(RuntimeError):
    """Raised on an invalid evaluation protocol or candidate."""


@dataclass(frozen=True)
class EvalProtocol:
    protocol_id: str = PROTOCOL_ID
    r_select: int = DEFAULT_R_SELECT
    s_select: int = DEFAULT_S_SELECT
    background_dags: int = 0
    mec_workers: int = 1
    link_regime: str = "stable"
    tokens: int = 20
    greedy_sweeps: int = 1
    energy_primary_scope: str = "system"

    def __post_init__(self) -> None:
        if int(self.r_select) < 1 or int(self.s_select) < 1:
            raise EvalLoopError("R and S must both be >= 1")
        if str(self.protocol_id) != PROTOCOL_ID:
            raise EvalLoopError("the CRN protocol is frozen: expected %r" % PROTOCOL_ID)

    def world_config(self) -> V2WorldConfig:
        return V2WorldConfig(background_dags=int(self.background_dags),
                             mec_workers=int(self.mec_workers))

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def replicate_world(graph, protocol: EvalProtocol, replicate: int,
                    dataset_graph_id: int = 0) -> tuple:
    """Build the world for environmental realization `replicate` (identity-keyed seeds)."""
    graph_id = str(getattr(graph, "graph_id", "g"))
    env_seed, helper_seed = seed_pair(protocol.protocol_id, replicate, 0, graph_id)
    world = build_world(graph, slot_id=replicate, world_id="crn_r%d_%s" % (replicate, graph_id),
                        dataset_graph_id=dataset_graph_id,
                        config=protocol.world_config(), helper_seed=helper_seed)
    link_process = None
    if str(protocol.link_regime) != "stable":
        link_process = make_process(protocol.link_regime, env_seed)
    return world, link_process


def _score(world: V2World, plan_map: Mapping, *, link_process=None) -> dict:
    result = world.with_foreground_plan_map(plan_map).schedule(
        link_process=link_process, validate=True)
    world.annotate(result)
    ledger = schedule_energy(result)
    metrics = v2_metrics(result, ledger)
    timings = [tm for tm in result.timings.values() if tm.dag_id == world.foreground_id]
    return {
        "episode_latency_s": float(result.episode_latency_s),
        "world_makespan_s": float(result.world_makespan_s),
        "system_joules": float(ledger.system_joules),
        "requester_joules": float(ledger.requester_joules),
        "mobile_joules": float(ledger.mobile_joules),
        "helper_task_fraction": float(metrics.v2v_task_fraction),
        "deadline_miss_rate": 0.0,
        "cpu_wait_mean_s": float(result.queue_stats["cpu_wait_mean_s"]),
        "wasted_work_bytes": float(result.queue_stats["wasted_work_bytes_total"]),
        "v2v_airtime_s": float(metrics.v2v_airtime_s),
        "location_mix": {loc: sum(1 for tm in timings if tm.location == loc)
                         for loc in ("UE", "MEC", "HELPER")},
    }


def baseline_plans(world: V2World, graph, protocol: EvalProtocol) -> dict:
    """The declared baseline panel, all on the SAME v2 world/scheduler/objective."""
    tokens = [int(t) for t in pure_plan(graph, 0).keys()]
    plans = {
        "all_ue": pure_plan(graph, 0),
        "all_mec": pure_plan(graph, 1),
        "all_helper": pure_plan(graph, 2),
    }
    # latency-aware greedy coordinate descent with a declared budget of `greedy_sweeps`
    chosen = {t: 0 for t in tokens}
    for _ in range(max(1, int(protocol.greedy_sweeps))):
        for tok in tokens:
            best = None
            for action in (0, 1, 2):
                trial = dict(chosen)
                trial[tok] = action
                value = _score(world, trial)["episode_latency_s"]
                if best is None or value < best[0]:
                    best = (value, action)
            chosen[tok] = best[1]
    plans["greedy_cd"] = chosen
    return plans


def reference_policy_logits(world: V2World, graph, protocol: EvalProtocol) -> np.ndarray:
    """[tokens, 3] decision-time scores of a numpy reference policy (NOT a trained network).

    For each token the score of an action is the negative foreground episode latency of the
    plan that places that action on the token and leaves everything else local. Only nominal
    (plan-time) rates are used, so the logits contain no realized future.
    """
    tokens = [int(t) for t in pure_plan(graph, 0).keys()]
    base = {t: 0 for t in tokens}
    out = np.zeros((len(tokens), 3), dtype=np.float64)
    for i, tok in enumerate(tokens):
        for action in (0, 1, 2):
            trial = dict(base)
            trial[tok] = action
            out[i, action] = -_score(world, trial)["episode_latency_s"]
    out -= out.max(axis=1, keepdims=True)      # numerically stable logits
    return out


def sample_policy_actions(logits: np.ndarray, protocol: EvalProtocol, replicate: int,
                          sample: int, graph_id: str) -> np.ndarray:
    noise = gumbel_noise(protocol.protocol_id, replicate, sample, graph_id,
                         tokens=int(logits.shape[0]), vocab=int(logits.shape[1]))
    return gumbel_argmax(logits, noise)


def evaluate_candidate(graph, name: str, protocol: EvalProtocol,
                       dataset_graph_id: int = 0) -> dict:
    """Evaluate ONE candidate on the full R x S grid and nest-aggregate the metrics."""
    graph_id = str(getattr(graph, "graph_id", "g"))
    per_replicate = {key: [] for key in AGG_KEYS}
    cells_by_replicate = {key: [] for key in AGG_KEYS}
    cells = []
    for r in range(int(protocol.r_select)):
        world, link_process = replicate_world(graph, protocol, r, dataset_graph_id)
        plans = baseline_plans(world, graph, protocol)
        if name in plans:
            plan_map = plans[name]
            deterministic = True
            logits = None
        elif name == STOCHASTIC_CANDIDATE:
            logits = reference_policy_logits(world, graph, protocol)
            deterministic = False
            plan_map = None
        else:
            raise EvalLoopError("unknown candidate %r (have %s + %s)"
                                % (name, list(plans), STOCHASTIC_CANDIDATE))
        replicate_metrics = {key: [] for key in AGG_KEYS}
        for s in range(int(protocol.s_select)):
            if deterministic:
                actions_plan = plan_map
            else:
                actions = sample_policy_actions(logits, protocol, r, s, graph_id)
                actions_plan = plan_map_from_actions(graph, list(int(a) for a in actions))
            metrics = _score(world, actions_plan, link_process=link_process)
            metrics["candidate"] = name
            metrics["r"] = r
            metrics["s"] = s
            metrics["deterministic"] = bool(deterministic)
            cells.append(metrics)
            for key in METRIC_KEYS:
                replicate_metrics[key].append(float(metrics[key]))
            replicate_metrics["helper_task_fraction"].append(
                float(metrics["helper_task_fraction"]))
            replicate_metrics["v2v_airtime_s"].append(float(metrics["v2v_airtime_s"]))
        for key, values in replicate_metrics.items():
            per_replicate[key].append(statistics.fmean(values) if values else 0.0)
            cells_by_replicate[key].append(list(values))
    aggregate = {}
    for key, rows in cells_by_replicate.items():
        # NESTED aggregation: R rows of S samples. Averaging over s first and then over the R
        # replicate means is what makes the standard error use R independent units; best-of-S
        # would report min_s instead and is never used.
        agg = nested_aggregate(rows)
        aggregate[key] = agg.as_dict() if hasattr(agg, "as_dict") else agg
    return {
        "candidate": name,
        "deterministic": name not in (STOCHASTIC_CANDIDATE,),
        "r_select": int(protocol.r_select),
        "s_select": int(protocol.s_select),
        "cells": cells,
        "replicate_means": per_replicate,
        "nested": aggregate,
    }


def run_evaluation(graphs: Sequence, *, protocol: EvalProtocol | None = None,
                   candidates: Sequence[str] = (),
                   dataset_graph_ids: Sequence[int] | None = None,
                   reference_candidate: str = "all_mec") -> dict:
    protocol = protocol or EvalProtocol()
    names = list(candidates) or list(DETERMINISTIC_CANDIDATES) + [STOCHASTIC_CANDIDATE]
    ids = list(dataset_graph_ids) if dataset_graph_ids is not None else list(range(len(graphs)))
    out = {
        "schema": "v2_crn_evaluation_v1",
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol_sha(),
        "engine": "v2 shared scheduler + energy ledger + constraints (CPU, numpy)",
        "policy_substitution": (
            "the TF PPO policy is REPLACED by a numpy reference policy over the same action "
            "space; the CRN protocol, the world, the scheduler and the objective are the real "
            "v2 ones. The trained-policy path is NOT RUN (TensorFlow absent)"),
        "graphs": [],
        "paired_deltas_vs_%s" % reference_candidate: {},
    }
    totals: dict = {}
    for gi, graph in enumerate(graphs):
        graph_id = str(getattr(graph, "graph_id", "g%d" % gi))
        per_graph = {"graph_id": graph_id, "dataset_graph_id": int(ids[gi]), "candidates": {}}
        for name in names:
            per_graph["candidates"][name] = evaluate_candidate(
                graph, name, protocol, dataset_graph_id=int(ids[gi]))
        if reference_candidate not in per_graph["candidates"]:
            raise EvalLoopError("reference candidate %r is not in the panel"
                                % reference_candidate)
        ref = per_graph["candidates"][reference_candidate]["replicate_means"]
        per_graph["paired_deltas"] = {}
        for name in names:
            if name == reference_candidate:
                continue
            cand = per_graph["candidates"][name]["replicate_means"]
            per_graph["paired_deltas"][name] = {}
            for key in AGG_KEYS:
                per_graph["paired_deltas"][name][key] = paired_delta(cand[key], ref[key])
        out["graphs"].append(per_graph)
        for name in names:
            totals.setdefault(name, {key: [] for key in AGG_KEYS})
            for key in AGG_KEYS:
                totals[name][key].append(
                    per_graph["candidates"][name]["nested"][key]["mean"])
    out["panel_mean_over_graphs"] = {
        name: {key: float(statistics.fmean(vals)) for key, vals in metrics.items()}
        for name, metrics in totals.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument("--r", type=int, default=DEFAULT_R_SELECT)
    ap.add_argument("--s", type=int, default=DEFAULT_S_SELECT)
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--link-regime", default="stable")
    ap.add_argument("--json", default=str(REPORTS / "V2_CRN_EVALUATION.json"))
    args = ap.parse_args()
    ds = load_dataset()
    graphs = ds.validation_query()[: int(args.graphs)]
    protocol = EvalProtocol(r_select=int(args.r), s_select=int(args.s),
                            background_dags=int(args.background),
                            link_regime=str(args.link_regime))
    out = run_evaluation(graphs, protocol=protocol)
    for name, metrics in sorted(out["panel_mean_over_graphs"].items()):
        print("%-24s latency=%.6f s  system=%.3f J  helper_frac=%.3f" % (
            name, metrics["episode_latency_s"], metrics["system_joules"],
            metrics["helper_task_fraction"]))
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("written", path, "(R=%d, S=%d, cells=%d)"
          % (protocol.r_select, protocol.s_select,
             sum(len(g["candidates"][c]["cells"]) for g in out["graphs"]
                 for c in g["candidates"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
