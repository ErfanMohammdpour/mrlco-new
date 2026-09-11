"""Motif pair frac_pos from all-MEC start. CPU. No train. Not 3500.

pairsup started from greedy_from_mec (clone, frac_pos=0.070). This probe
starts from all-MEC so 9-way joints have room. No network. No PPO.
No pair search at inference. paper_result=false. Frozen primary remains 3500.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spec.hamming2_probe import H2_N_POOL, _frozen_cluster, _gv_path, stratified_graph_picks
from spec.pair_head import _score, motif_pairs
from spec.pair_sup import pick_improving_joint
from spec.phase4_campaign import META_TRAIN_IDS, SEEDS

PAIRFRAC_N_GRAPHS = 200
PAIRFRAC_N_TOK = 20
PAIRFRAC_CLONE_FRAC = 0.070
PAIRFRAC_CLONE_HI = 0.12
PAIRFRAC_OK = 0.20
PAIRFRAC_HOLD = 0.40


def classify_pairfrac(frac_pos):
    f = float(frac_pos)
    if f < PAIRFRAC_CLONE_HI:
        return "pairfrac_mec_clone_trap"
    if f < PAIRFRAC_OK:
        return "pairfrac_mec_weak"
    if f < PAIRFRAC_HOLD:
        return "pairfrac_mec_ok"
    return "pairfrac_mec_holdout_like"


def proceed_rewrite(frac_pos):
    return float(frac_pos) >= PAIRFRAC_OK


def run_pairfrac_mec(seed, n_graphs=PAIRFRAC_N_GRAPHS, run_dir=None):
    """CPU diagnostic. No GPU. No PPO. paper_result=false. Frozen primary remains 3500."""
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler import resource_config_from_cluster
    from spec.phase4_campaign import diag_pairfrac_run_dir

    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    if int(n_graphs) != PAIRFRAC_N_GRAPHS:
        raise ValueError("pairfrac n_graphs must be %d, got %s" % (PAIRFRAC_N_GRAPHS, n_graphs))

    rng = np.random.RandomState(seed)
    picks = stratified_graph_picks(META_TRAIN_IDS, PAIRFRAC_N_GRAPHS, H2_N_POOL, rng)
    cluster = _frozen_cluster()
    resources = resource_config_from_cluster(cluster)
    if run_dir is None:
        run_dir = diag_pairfrac_run_dir(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_pos = 0
    n_eval = 0
    deltas = []
    rows = []
    for gi, (dist_id, graph_idx) in enumerate(picks):
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != PAIRFRAC_N_TOK:
            raise ValueError("graph %s task_number %s != %d" % (gv, n, PAIRFRAC_N_TOK))
        plan = [1] * n
        t0 = _score(tg, resources, plan)
        order = [int(tid) for tid in tg.prioritize_sequence]
        pairs = motif_pairs(tg.succ_task_sets, tg.pre_task_sets, order)
        n_pairs += len(pairs)

        def score_fn(trial, _tg=tg, _res=resources):
            return _score(_tg, _res, trial)

        n_pos_g = 0
        best_delta = 0.0
        for rec in pairs:
            picked, n_e = pick_improving_joint(plan, rec["i"], rec["j"], t0, score_fn)
            n_eval += n_e
            if picked is None:
                continue
            n_pos += 1
            n_pos_g += 1
            deltas.append(float(picked["delta"]))
            if float(picked["delta"]) > best_delta:
                best_delta = float(picked["delta"])
        rows.append(
            {
                "dist_id": int(dist_id),
                "graph_idx": int(graph_idx),
                "mec_T": float(t0),
                "n_pairs": int(len(pairs)),
                "n_pos": int(n_pos_g),
                "best_delta": float(best_delta),
            }
        )
        if (gi + 1) % 10 == 0 or gi == 0 or gi == len(picks) - 1:
            frac_so_far = float(n_pos) / float(max(n_pairs, 1))
            print(
                "pairfrac_mec %d/%d dist=%d idx=%d mec=%.1f pos=%d pairs=%d frac=%.3f"
                % (
                    gi + 1,
                    len(picks),
                    dist_id,
                    graph_idx,
                    t0,
                    n_pos,
                    n_pairs,
                    frac_so_far,
                )
            )

    frac_pos = float(n_pos) / float(max(n_pairs, 1))
    verdict = classify_pairfrac(frac_pos)
    mean_delta_pos = float(np.mean(deltas)) if deltas else 0.0
    frac_graphs = float(np.mean([1.0 if r["n_pos"] > 0 else 0.0 for r in rows]))
    per = {}
    by = defaultdict(list)
    for r in rows:
        by[str(r["dist_id"])].append(r)
    for did, group in by.items():
        gp = int(sum(x["n_pos"] for x in group))
        gq = int(sum(x["n_pairs"] for x in group))
        per[did] = {
            "n": int(len(group)),
            "mec_T_mean": float(np.mean([x["mec_T"] for x in group])),
            "n_pos": gp,
            "n_pairs": gq,
            "frac_pos": float(gp) / float(max(gq, 1)),
        }
    stats = {
        "method_id": "margo_v0.2_diag_pairfrac_mec",
        "paper_result": False,
        "ppo": False,
        "gpu": False,
        "n_graphs": int(len(rows)),
        "seed": seed,
        "split": "meta_train",
        "start": "all_mec",
        "n_pairs": int(n_pairs),
        "n_pos": int(n_pos),
        "n_eval": int(n_eval),
        "frac_pos": frac_pos,
        "mean_delta_pos": mean_delta_pos,
        "frac_graphs_with_pos": frac_graphs,
        "clone_ref_frac_pos": PAIRFRAC_CLONE_FRAC,
        "proceed_rewrite": bool(proceed_rewrite(frac_pos)),
        "verdict": verdict,
        "per_dist": per,
        "mec_T_mean": float(np.mean([r["mec_T"] for r in rows])),
    }
    (run_dir / "pairfrac_mec.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(
        "pairfrac_verdict=%s frac_pos=%.3f n_pos=%d n_pairs=%d clone_ref=%.3f proceed_rewrite=%s"
        % (verdict, frac_pos, n_pos, n_pairs, PAIRFRAC_CLONE_FRAC, stats["proceed_rewrite"])
    )
    return run_dir, stats
