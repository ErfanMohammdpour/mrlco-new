"""Per-profile 2-opt teachers for Phase 4. CPU. paper_result=false."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.hamming2_probe import _gv_path, _n_non, _score
from spec.phase4_campaign import META_TRAIN_IDS, ROOT, RUNS_ROOT, VALIDATION_IDS
from spec.resource_profiles import (
    resource_config_for_profile,
    resources_cluster_for_profile,
    role_profile_ids,
)
from spec.split_loader import meta_train_distribution_ids, validation_distribution_ids
from spec.twopt_expert import TWOPT_N_TOK, iterate_2opt

EXPERT_PROFILES_METHOD_ID = "margo_v0.3_expert_profiles"


def profile_cache_path(split: str, profile_id: str, runs_root=RUNS_ROOT) -> Path:
    safe = str(profile_id).replace("/", "_")
    return Path(runs_root) / ("expert_2opt_mec_%s_%s.npz" % (split, safe))


def _graph_picks(split: str):
    if split == "meta_train":
        dists = list(meta_train_distribution_ids())
    elif split == "validation":
        dists = list(validation_distribution_ids())
    else:
        raise ValueError(split)
    picks = []
    for d in dists:
        for gi in range(100):
            picks.append((int(d), int(gi)))
    return picks


def run_expert_for_profile(profile_id: str, split: str = "meta_train", max_graphs=None):
    """Greedy-from-mec + 2-opt under profile resources. Returns rows + summary."""
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    cluster = resources_cluster_for_profile(profile_id)
    resources = resource_config_for_profile(profile_id)
    picks = _graph_picks(split)
    if max_graphs is not None:
        picks = picks[: int(max_graphs)]
    rows = []
    for dist_id, graph_idx in picks:
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != TWOPT_N_TOK:
            raise ValueError("task_number %s != %d" % (n, TWOPT_N_TOK))

        def score_fn(trial, _tg=tg, _res=resources):
            return _score(_tg, _res, trial)

        plan, result = greedy_from_mec_plan(tg, resources, max_passes=BC_MAX_PASSES)
        actions = [int(a) for _, a in plan]
        t_g = float(result.makespan_seconds)
        t_mec = float(score_fn([1] * n))
        out = iterate_2opt(actions, t_g, score_fn)
        rows.append(
            {
                "dist_id": int(dist_id),
                "graph_idx": int(graph_idx),
                "profile_id": str(profile_id),
                "greedy_T": float(t_g),
                "twopt_T": float(out["t"]),
                "mec_T": float(t_mec),
                "twopt_acts": [int(a) for a in out["actions"]],
                "greedy_n_non": _n_non(actions),
                "twopt_n_non": _n_non(out["actions"]),
                "mix_local": float(np.mean(np.asarray(out["actions"]) == 0)),
                "mix_mec": float(np.mean(np.asarray(out["actions"]) == 1)),
                "mix_v2v": float(np.mean(np.asarray(out["actions"]) == 2)),
            }
        )
    acts = np.stack([np.asarray(r["twopt_acts"], dtype=np.int32) for r in rows], axis=0)
    summary = {
        "profile_id": str(profile_id),
        "split": str(split),
        "n_graphs": len(rows),
        "greedy_T_mean": float(np.mean([r["greedy_T"] for r in rows])),
        "twopt_T_mean": float(np.mean([r["twopt_T"] for r in rows])),
        "mec_T_mean": float(np.mean([r["mec_T"] for r in rows])),
        "mix_local": float(np.mean([r["mix_local"] for r in rows])),
        "mix_mec": float(np.mean([r["mix_mec"] for r in rows])),
        "mix_v2v": float(np.mean([r["mix_v2v"] for r in rows])),
        "twopt_n_non_mean": float(np.mean([r["twopt_n_non"] for r in rows])),
    }
    return rows, acts, summary


def write_profile_cache(split, profile_id, rows, acts, run_dir=None):
    dest = profile_cache_path(split, profile_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        acts=acts,
        t_expert=np.asarray([r["twopt_T"] for r in rows], dtype=np.float64),
        t_greedy=np.asarray([r["greedy_T"] for r in rows], dtype=np.float64),
        t_mec=np.asarray([r["mec_T"] for r in rows], dtype=np.float64),
        n_non=np.asarray([r["twopt_n_non"] for r in rows], dtype=np.int32),
        dist_id=np.asarray([r["dist_id"] for r in rows], dtype=np.int32),
        graph_idx=np.asarray([r["graph_idx"] for r in rows], dtype=np.int32),
        profile_id=np.asarray([profile_id] * len(rows)),
    )
    np.savez_compressed(str(dest), **payload)
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(Path(run_dir) / dest.name), **payload)
    return dest


def run_expert_profiles(
    seed: int,
    smoke: bool = False,
    profiles: list[str] | None = None,
    split: str = "meta_train",
    role: str | None = None,
):
    """Generate 2-opt caches.

    split: meta_train | validation (graph set)
    role: if set, profile list from yaml role (meta_train / validation_heldout / frozen)
    Smoke train: 3 profiles × 32 graphs.
    """
    run_dir = (
        Path(RUNS_ROOT)
        / EXPERT_PROFILES_METHOD_ID
        / ("seed_%d" % int(seed))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if profiles is None:
        if role is not None:
            profiles = list(role_profile_ids(role))
        elif split == "validation":
            profiles = list(role_profile_ids("validation_heldout")) + list(
                role_profile_ids("frozen")
            )
        else:
            profiles = list(role_profile_ids("meta_train"))
    max_graphs = 32 if smoke else None
    if smoke and split == "meta_train" and role is None:
        profiles = ["p_3_3_5", "p_11_7_20", "frozen_7_5_10"]
    summaries = []
    for pid in profiles:
        print("expert_profiles start %s split=%s smoke=%s" % (pid, split, int(smoke)))
        rows, acts, summary = run_expert_for_profile(
            pid, split=split, max_graphs=max_graphs
        )
        dest = write_profile_cache(split, pid, rows, acts, run_dir=run_dir)
        summary["cache"] = str(dest)
        summaries.append(summary)
        print(
            "expert_profiles done %s greedy=%.1f twopt=%.1f mec=%.1f mix_mec=%.3f"
            % (
                pid,
                summary["greedy_T_mean"],
                summary["twopt_T_mean"],
                summary["mec_T_mean"],
                summary["mix_mec"],
            )
        )
    out = {
        "method_id": EXPERT_PROFILES_METHOD_ID,
        "seed": int(seed),
        "smoke": bool(smoke),
        "split": split,
        "role": role,
        "paper_result": False,
        "summaries": summaries,
    }
    tag = "expert_profiles_summary.json"
    if split != "meta_train":
        tag = "expert_profiles_%s_summary.json" % split
    (run_dir / tag).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return run_dir, out
