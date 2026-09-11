"""Phase 5 per-λ 2-opt teachers. CPU. ADR-001 mobile energy. paper_result=false."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.mec_offloaing_envs.scheduler.energy_api import (
    PARETO_LAMBDAS,
    compute_reference_ranges,
    j_lambda,
    lambda_tag,
)
from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.expert_profiles import _graph_picks, profile_cache_path
from spec.hamming2_probe import _gv_path, _n_non
from spec.phase4_campaign import (
    DIAG_EXPERT_ENERGY_METHOD_ID,
    RUNS_ROOT,
    diag_expert_energy_run_dir,
)
from spec.resource_profiles import (
    resource_config_for_profile,
    resources_cluster_for_profile,
    role_profile_ids,
)
from spec.split_loader import meta_test_distribution_ids
from spec.twopt_expert import TWOPT_N_TOK, iterate_2opt

ENERGY_SCOPE = (
    "E = total_mobile_joules = UE + HELPER (compute + radio). "
    "MEC server compute excluded (ADR-001)."
)
SMOKE_GRAPHS = 8
SMOKE_PROFILES = ("frozen_7_5_10",)


def energy_cache_path(split: str, profile_id: str, lam: float, runs_root=RUNS_ROOT) -> Path:
    safe = str(profile_id).replace("/", "_")
    return Path(runs_root) / ("expert_2opt_J%s_%s_%s.npz" % (lambda_tag(lam), split, safe))


def _metric_for_lambda(lam, refs):
    lam = float(lam)

    def metric_fn(result):
        if abs(lam - 1.0) <= 1e-12:
            return float(result.makespan_seconds)
        return float(
            j_lambda(
                result.makespan_seconds,
                result.total_mobile_joules,
                refs,
                lam,
                clip=False,
            )
        )

    return metric_fn


def _score_j(tg, resources, acts, refs, lam):
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    order = [int(tid) for tid in tg.prioritize_sequence]
    plan = list(zip(order, [int(a) for a in acts]))
    result, _, _ = schedule_via_adapter(tg, plan, resources)
    t = float(result.makespan_seconds)
    e = float(result.total_mobile_joules)
    if abs(float(lam) - 1.0) <= 1e-12:
        j = t
    else:
        j = float(j_lambda(t, e, refs, lam, clip=False))
    return t, e, j, result


def _assert_no_metatest(split):
    if split == "meta_test":
        raise ValueError("Phase 5 must not touch meta-test")
    got = set(int(d) for d, _ in _graph_picks(split))
    leak = got & set(int(x) for x in meta_test_distribution_ids())
    if leak:
        raise ValueError("meta-test dist leaked into Phase 5: %s" % leak)


def run_expert_energy_for_profile(profile_id, lam, split="meta_train", max_graphs=None):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    _assert_no_metatest(split)
    cluster = resources_cluster_for_profile(profile_id)
    resources = resource_config_for_profile(profile_id)
    picks = _graph_picks(split)
    if max_graphs is not None:
        picks = picks[: int(max_graphs)]
    rows = []
    n_l_tie = 0
    n_e_tie = 0
    n_mec_lowest_e = 0
    n_ue_highest_e = 0
    for dist_id, graph_idx in picks:
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != TWOPT_N_TOK:
            raise ValueError("task_number %s != %d" % (n, TWOPT_N_TOK))
        refs = compute_reference_ranges(tg, resources)
        if refs.L_scale <= 1e-12:
            n_l_tie += 1
        if refs.E_scale <= 1e-12:
            n_e_tie += 1
        if refs.E_mec <= refs.E_ue + 1e-12 and refs.E_mec <= refs.E_helper + 1e-12:
            n_mec_lowest_e += 1
        if refs.E_ue + 1e-12 >= refs.E_mec and refs.E_ue + 1e-12 >= refs.E_helper:
            n_ue_highest_e += 1
        metric_fn = _metric_for_lambda(lam, refs)

        def score_fn(trial, _tg=tg, _res=resources, _refs=refs, _lam=lam):
            t, _e, j, _r = _score_j(_tg, _res, trial, _refs, _lam)
            return j if abs(float(_lam) - 1.0) > 1e-12 else t

        plan, result = greedy_from_mec_plan(
            tg, resources, max_passes=BC_MAX_PASSES, metric_fn=metric_fn
        )
        actions = [int(a) for _, a in plan]
        t_g = float(result.makespan_seconds)
        e_g = float(result.total_mobile_joules)
        j0 = float(score_fn(actions))
        out = iterate_2opt(actions, j0, score_fn)
        t_ex, e_ex, j_ex, _ = _score_j(tg, resources, out["actions"], refs, lam)
        rows.append(
            {
                "dist_id": int(dist_id),
                "graph_idx": int(graph_idx),
                "profile_id": str(profile_id),
                "lam": float(lam),
                "greedy_T": t_g,
                "greedy_E": e_g,
                "twopt_T": t_ex,
                "twopt_E": e_ex,
                "twopt_J": float(j_ex),
                "twopt_acts": [int(a) for a in out["actions"]],
                "t_allUE": float(refs.L_ue),
                "t_allMEC": float(refs.L_mec),
                "t_allHELPER": float(refs.L_helper),
                "e_allUE": float(refs.E_ue),
                "e_allMEC": float(refs.E_mec),
                "e_allHELPER": float(refs.E_helper),
                "l_scale": float(refs.L_scale),
                "e_scale": float(refs.E_scale),
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
        "lam": float(lam),
        "n_graphs": len(rows),
        "greedy_T_mean": float(np.mean([r["greedy_T"] for r in rows])),
        "twopt_T_mean": float(np.mean([r["twopt_T"] for r in rows])),
        "twopt_E_mean": float(np.mean([r["twopt_E"] for r in rows])),
        "twopt_J_mean": float(np.mean([r["twopt_J"] for r in rows])),
        "mix_local": float(np.mean([r["mix_local"] for r in rows])),
        "mix_mec": float(np.mean([r["mix_mec"] for r in rows])),
        "mix_v2v": float(np.mean([r["mix_v2v"] for r in rows])),
        "frac_mec_lowest_E": float(n_mec_lowest_e) / float(len(rows)),
        "frac_ue_highest_E": float(n_ue_highest_e) / float(len(rows)),
        "n_L_scale_eps": int(n_l_tie),
        "n_E_scale_eps": int(n_e_tie),
        "energy_scope": ENERGY_SCOPE,
    }
    return rows, acts, summary


def write_energy_cache(split, profile_id, lam, rows, acts, run_dir=None):
    dest = energy_cache_path(split, profile_id, lam)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        acts=acts,
        lam=np.float64(lam),
        t_expert=np.asarray([r["twopt_T"] for r in rows], dtype=np.float64),
        e_expert=np.asarray([r["twopt_E"] for r in rows], dtype=np.float64),
        j_expert=np.asarray([r["twopt_J"] for r in rows], dtype=np.float64),
        t_greedy=np.asarray([r["greedy_T"] for r in rows], dtype=np.float64),
        e_greedy=np.asarray([r["greedy_E"] for r in rows], dtype=np.float64),
        t_allUE=np.asarray([r["t_allUE"] for r in rows], dtype=np.float64),
        t_allMEC=np.asarray([r["t_allMEC"] for r in rows], dtype=np.float64),
        t_allHELPER=np.asarray([r["t_allHELPER"] for r in rows], dtype=np.float64),
        e_allUE=np.asarray([r["e_allUE"] for r in rows], dtype=np.float64),
        e_allMEC=np.asarray([r["e_allMEC"] for r in rows], dtype=np.float64),
        e_allHELPER=np.asarray([r["e_allHELPER"] for r in rows], dtype=np.float64),
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


def _lambda1_acts_match(split, profile_id, acts):
    lat = profile_cache_path(split, profile_id)
    if not lat.is_file():
        return {"compared": False, "match": None, "path": str(lat)}
    blob = np.load(str(lat), allow_pickle=False)
    want = np.asarray(blob["acts"], dtype=np.int32)
    got = np.asarray(acts, dtype=np.int32)
    n = min(want.shape[0], got.shape[0])
    match = bool(np.array_equal(want[:n], got[:n]))
    return {
        "compared": True,
        "match": match,
        "n": int(n),
        "n_mismatch": int(np.sum(np.any(want[:n] != got[:n], axis=1))),
        "path": str(lat),
    }


def run_expert_energy(seed, smoke=False, split="meta_train", lambdas=None, profiles=None):
    _assert_no_metatest(split)
    run_dir = diag_expert_energy_run_dir(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    lambdas = list(PARETO_LAMBDAS if lambdas is None else lambdas)
    if profiles is None:
        if split == "validation":
            profiles = list(role_profile_ids("validation_heldout")) + list(
                role_profile_ids("frozen")
            )
        else:
            profiles = list(role_profile_ids("meta_train"))
    max_graphs = SMOKE_GRAPHS if smoke else None
    if smoke:
        profiles = list(SMOKE_PROFILES)
    summaries = []
    lambda1_checks = []
    for pid in profiles:
        for lam in lambdas:
            print(
                "expert_energy start %s lam=%s split=%s smoke=%s"
                % (pid, lambda_tag(lam), split, int(smoke))
            )
            rows, acts, summary = run_expert_energy_for_profile(
                pid, lam, split=split, max_graphs=max_graphs
            )
            dest = write_energy_cache(split, pid, lam, rows, acts, run_dir=run_dir)
            summary["cache"] = str(dest)
            if abs(float(lam) - 1.0) <= 1e-12:
                chk = _lambda1_acts_match(split, pid, acts)
                summary["lambda1_vs_phase4"] = chk
                lambda1_checks.append({"profile_id": pid, **chk})
                print(
                    "expert_energy lambda1_match compared=%s match=%s n=%s"
                    % (chk["compared"], chk["match"], chk.get("n"))
                )
            summaries.append(summary)
            print(
                "expert_energy done %s J%s T=%.1f E=%.1f mix_mec=%.3f mix_local=%.3f"
                % (
                    pid,
                    lambda_tag(lam),
                    summary["twopt_T_mean"],
                    summary["twopt_E_mean"],
                    summary["mix_mec"],
                    summary["mix_local"],
                )
            )
    trend = {}
    for pid in profiles:
        rows = [s for s in summaries if s["profile_id"] == pid]
        rows = sorted(rows, key=lambda s: -float(s["lam"]))
        t_seq = [s["twopt_T_mean"] for s in rows]
        e_seq = [s["twopt_E_mean"] for s in rows]
        t_mono = all(t_seq[i] <= t_seq[i + 1] + 1e-6 for i in range(len(t_seq) - 1))
        e_mono = all(e_seq[i] + 1e-6 >= e_seq[i + 1] for i in range(len(e_seq) - 1))
        trend[pid] = {
            "lam": [s["lam"] for s in rows],
            "T": t_seq,
            "E": e_seq,
            "T_nondecreasing_as_lam_drops": bool(t_mono),
            "E_nonincreasing_as_lam_drops": bool(e_mono),
        }
    out = {
        "method_id": DIAG_EXPERT_ENERGY_METHOD_ID,
        "seed": int(seed),
        "smoke": bool(smoke),
        "split": split,
        "paper_result": False,
        "energy_scope": ENERGY_SCOPE,
        "lambdas": lambdas,
        "summaries": summaries,
        "trend": trend,
        "lambda1_checks": lambda1_checks,
    }
    tag = "expert_energy_summary.json"
    if smoke:
        tag = "expert_energy_smoke_summary.json"
    if split != "meta_train":
        tag = "expert_energy_%s_summary.json" % split
    (run_dir / tag).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("expert_energy_verdict smoke=%s trend=%s" % (int(smoke), json.dumps(trend)))
    return run_dir, out


assert DIAG_EXPERT_ENERGY_METHOD_ID == "margo_v0.3_expert_energy"
assert PARETO_LAMBDAS[0] == 1.0
assert PARETO_LAMBDAS[-1] == 0.0
