"""Per-stage learning audit dumps. Not a paper result. Does not change frozen 3500.

Writes compact trajectory records, current-batch graphs, inner/outer losses,
and a health verdict after every trainer stage.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np


ACTION_NAMES = {0: "Local", 1: "MEC", 2: "V2V"}
INNER_K = 3
COLLAPSE_FRAC = 0.95
LOSS_ABS_WARN = 50.0


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def _as_float_list(arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return [jsonable(float(x)) for x in a]


def _as_int_list(arr):
    a = np.asarray(arr).reshape(-1)
    return [int(x) for x in a]


def mean_token_entropy(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / np.sum(p, axis=-1, keepdims=True)
    ent = -np.sum(p * np.log(p + 1e-12), axis=-1)
    return float(np.mean(ent))


def action_histogram(actions):
    flat = np.asarray(actions).reshape(-1).astype(np.int32)
    counts = {name: 0 for name in ACTION_NAMES.values()}
    counts["other"] = 0
    total = int(flat.size)
    for v in flat.tolist():
        name = ACTION_NAMES.get(int(v))
        if name is None:
            counts["other"] += 1
        else:
            counts[name] += 1
    fracs = {}
    for key, n in counts.items():
        fracs[key] = (float(n) / float(total)) if total else 0.0
    return {"counts": counts, "fracs": fracs, "n": total}


def serialize_graph(tg, dist_id, env_slot, graph_index):
    tasks = []
    for i, task in enumerate(tg.task_list):
        tasks.append(
            {
                "index": i,
                "id_name": str(task.id_name),
                "processing_data_size": float(task.processing_data_size),
                "transmission_data_size": float(task.transmission_data_size),
                "depth": int(task.depth),
                "heft_score": float(task.heft_score),
            }
        )
    pred = [[int(x) for x in sorted(s)] for s in tg.pre_task_sets]
    succ = [[int(x) for x in sorted(s)] for s in tg.succ_task_sets]
    return {
        "distribution_id": int(dist_id),
        "env_slot": int(env_slot),
        "graph_index": int(graph_index),
        "task_number": int(tg.task_number),
        "prioritize_sequence": [int(x) for x in tg.prioritize_sequence],
        "dependency": np.asarray(tg.dependency, dtype=np.float64).tolist(),
        "tasks": tasks,
        "predecessors": pred,
        "successors": succ,
    }


def task_spec_records(task_specs, env=None):
    rows = []
    for slot, spec in enumerate(task_specs):
        if isinstance(spec, dict):
            env_slot = int(spec["dist_index"])
            gidx = [int(x) for x in np.asarray(spec["graph_indices"]).tolist()]
        else:
            env_slot = int(spec)
            gidx = []
        dist_id = env_slot
        if env is not None and getattr(env, "distribution_ids", None) is not None:
            dist_id = int(env.distribution_ids[env_slot])
        rows.append(
            {
                "meta_task": slot,
                "env_slot": env_slot,
                "distribution_id": dist_id,
                "graph_indices": gidx,
                "n_graphs": len(gidx),
            }
        )
    return rows


def compact_paths(paths, task_specs, env=None):
    specs = task_spec_records(task_specs, env)
    rows = []
    for meta_i, spec in enumerate(specs):
        task_paths = paths[meta_i] if isinstance(paths, dict) else paths[meta_i]
        gidx = spec["graph_indices"]
        n_g = len(gidx) if gidx else 1
        for traj_i, path in enumerate(task_paths):
            actions = np.asarray(path["actions"])
            rewards = np.asarray(path["rewards"], dtype=np.float64)
            finish = path.get("finish_time", None)
            energy = path.get("energy", None)
            logits = path.get("logits", None)
            values = path.get("values", None)
            graph_index = gidx[traj_i % n_g] if gidx else traj_i
            rec = {
                "meta_task": meta_i,
                "traj_i": traj_i,
                "repeat": int(traj_i // n_g) if gidx else 0,
                "distribution_id": spec["distribution_id"],
                "env_slot": spec["env_slot"],
                "graph_index": int(graph_index),
                "actions": _as_int_list(actions),
                "rewards": _as_float_list(rewards),
                "return_sum": float(np.sum(rewards)),
                "finish_time": jsonable(np.asarray(finish).reshape(-1)[-1] if finish is not None else None),
                "energy_sum": float(np.sum(np.asarray(energy, dtype=np.float64))) if energy is not None else None,
                "entropy": mean_token_entropy(logits) if logits is not None else None,
                "value_mean": float(np.mean(values)) if values is not None else None,
            }
            rows.append(rec)
    return rows


def summarize_paths(rows):
    if not rows:
        return {"n_traj": 0}
    returns = np.array([r["return_sum"] for r in rows], dtype=np.float64)
    lat = np.array([r["finish_time"] for r in rows if r["finish_time"] is not None], dtype=np.float64)
    all_actions = []
    for r in rows:
        all_actions.extend(r["actions"])
    hist = action_histogram(all_actions)
    energies = [r["energy_sum"] for r in rows if r["energy_sum"] is not None]
    ents = [r["entropy"] for r in rows if r["entropy"] is not None]
    by_dist = Counter(r["distribution_id"] for r in rows)
    return {
        "n_traj": len(rows),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "latency_mean": float(np.mean(lat)) if lat.size else None,
        "energy_mean": float(np.mean(energies)) if energies else None,
        "entropy_mean": float(np.mean(ents)) if ents else None,
        "action_hist": hist,
        "distribution_traj_counts": {str(k): int(v) for k, v in sorted(by_dist.items())},
        "has_nan_reward": bool(np.any(~np.isfinite(returns))),
    }


def flatten_losses(nested):
    out = []
    for item in nested:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_losses(item))
        else:
            out.append(float(item))
    return out


def health_verdict(summary, *, inner_policy_losses=None, inner_value_losses=None, k_steps=INNER_K):
    flags = []
    hist = summary.get("action_hist") or {}
    fracs = hist.get("fracs") or {}
    max_name = None
    max_frac = 0.0
    for name in ACTION_NAMES.values():
        frac = float(fracs.get(name, 0.0))
        if frac > max_frac:
            max_frac = frac
            max_name = name
    if max_frac >= COLLAPSE_FRAC:
        flags.append("action_collapse_%s_%.3f" % (max_name, max_frac))
    if summary.get("has_nan_reward"):
        flags.append("nan_or_inf_return")
    pol = flatten_losses(inner_policy_losses or [])
    val = flatten_losses(inner_value_losses or [])
    if any(not math.isfinite(x) for x in pol + val):
        flags.append("nan_or_inf_loss")
    if any(abs(x) >= LOSS_ABS_WARN for x in pol):
        flags.append("policy_loss_abs_ge_%s" % LOSS_ABS_WARN)
    n_tasks = len(inner_policy_losses or [])
    apply_ok = True
    if inner_policy_losses:
        for item in inner_policy_losses:
            n_apply = len(item) if isinstance(item, (list, tuple)) else 1
            if n_apply != int(k_steps):
                apply_ok = False
                flags.append("inner_apply_count_%s_!=_%s" % (n_apply, k_steps))
                break
    if summary.get("n_traj", 0) == 0:
        flags.append("empty_trajs")
    ok = len(flags) == 0
    return {
        "ok": ok,
        "flags": flags,
        "n_meta_tasks_logged": n_tasks,
        "inner_apply_ok": apply_ok,
        "max_action": max_name,
        "max_action_frac": max_frac,
        "policy_loss_mean": float(np.mean(pol)) if pol else None,
        "value_loss_mean": float(np.mean(val)) if val else None,
    }


class TrainAuditWriter:
    """One JSONL line per trainer stage, plus per-iter traj/graph dumps."""

    def __init__(self, run_dir):
        self.run_dir = os.fspath(run_dir)
        self.audit_root = os.path.join(self.run_dir, "audit")
        os.makedirs(self.audit_root, exist_ok=True)
        self._health_path = os.path.join(self.audit_root, "health.jsonl")
        self._iter_dir = None
        self._current_itr = None

    def _iter_path(self, itr):
        return os.path.join(self.audit_root, "iter_%04d" % int(itr))

    def begin_iter(self, itr):
        self._current_itr = int(itr)
        self._iter_dir = self._iter_path(itr)
        os.makedirs(self._iter_dir, exist_ok=True)
        self.stage(itr, "begin_iter", {"itr": int(itr)})

    def stage(self, itr, name, payload):
        rec = {"itr": int(itr), "stage": str(name), "payload": jsonable(payload)}
        path = os.path.join(self._iter_path(itr), "stages.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def write_json(self, itr, filename, obj):
        path = os.path.join(self._iter_path(itr), filename)
        with open(path, "w") as f:
            f.write(json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n")
        return path

    def write_jsonl(self, itr, filename, rows):
        path = os.path.join(self._iter_path(itr), filename)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        return path

    def dump_graphs(self, itr, env, task_specs):
        recs = []
        for spec in task_spec_records(task_specs, env):
            env_slot = spec["env_slot"]
            for gi in spec["graph_indices"]:
                tg = env.task_graphs_batchs[env_slot][int(gi)]
                recs.append(serialize_graph(tg, spec["distribution_id"], env_slot, gi))
        self.write_json(itr, "graphs.json", recs)
        return recs

    def dump_paths(self, itr, filename, paths, task_specs, env=None):
        rows = compact_paths(paths, task_specs, env)
        self.write_jsonl(itr, filename, rows)
        summary = summarize_paths(rows)
        return rows, summary

    def finish_iter(self, itr, summary):
        self.write_json(itr, "summary.json", summary)
        with open(self._health_path, "a") as f:
            f.write(json.dumps(jsonable({"itr": int(itr), **summary.get("health", {})}), sort_keys=True) + "\n")
        self.stage(itr, "end_iter", {"health": summary.get("health")})
