"""Hamming-2 local-opt probe of greedy_from_mec. CPU. No PPO. Not the frozen 3500 primary.

One-shot: from the current greedy expert plan, try every 2-coordinate relabel.
Does not search a radius-k ball around all-MEC. Does not iterate 2-opt.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.phase4_campaign import META_TRAIN_IDS, ROOT, SEEDS

H2_N_GRAPHS = 200
H2_N_POOL = 100
H2_N_TOK = 20
H2_CASE1_MAX_DELTA = 5.0
H2_CASE2_MIN_DELTA = 30.0
ALTS = (0, 1, 2)


def count_h2_neighbors(actions):
    actions = np.asarray(actions, dtype=np.int32)
    n = int(actions.shape[0])
    n_pair = n * (n - 1) // 2
    return int(n_pair * 2 * 2)


def classify_mean_delta(mean_delta):
    d = float(mean_delta)
    if d < H2_CASE1_MAX_DELTA:
        return "ood_not_teacher"
    if d >= H2_CASE2_MIN_DELTA:
        return "weak_expert"
    return "mixed"


def stratified_graph_picks(dist_ids, n_total, n_pool, rng):
    dist_ids = [int(x) for x in dist_ids]
    n_d = len(dist_ids)
    if n_total < n_d:
        raise ValueError("n_total %s < n_dists %s" % (n_total, n_d))
    if n_pool < 1:
        raise ValueError("n_pool must be positive")
    base = n_total // n_d
    rem = n_total % n_d
    picks = []
    for i, dist_id in enumerate(dist_ids):
        k = base + (1 if i < rem else 0)
        if k > n_pool:
            raise ValueError("need %d graphs from pool %d" % (k, n_pool))
        idx = np.sort(rng.choice(n_pool, size=k, replace=False))
        for g in idx:
            picks.append((int(dist_id), int(g)))
    if len(picks) != n_total:
        raise ValueError("stratified size %d != %d" % (len(picks), n_total))
    return picks


def _gv_path(dist_id, graph_idx):
    return ROOT / (
        "env/mec_offloaing_envs/data/meta_offloading_20/"
        "offload_random20_%d/random.20.%d.gv" % (int(dist_id), int(graph_idx))
    )


def _score(tg, resources, actions):
    from env.mec_offloaing_envs.scheduler import schedule_via_adapter

    order = [int(tid) for tid in tg.prioritize_sequence]
    plan = list(zip(order, [int(a) for a in actions]))
    result, _, _ = schedule_via_adapter(tg, plan, resources)
    return float(result.makespan_seconds)


def _n_non(actions):
    return int(sum(1 for a in actions if int(a) != 1))


def best_h1(tg, resources, actions, t0):
    best_t = float(t0)
    n_better = 0
    n = len(actions)
    for k in range(n):
        for a in ALTS:
            if a == actions[k]:
                continue
            trial = list(actions)
            trial[k] = a
            t = _score(tg, resources, trial)
            if t + 1e-12 < t0:
                n_better += 1
            if t + 1e-12 < best_t:
                best_t = t
    return best_t, n_better


def best_h2(tg, resources, actions, t0):
    best_t = float(t0)
    best_actions = list(actions)
    n_better = 0
    n_eval = 0
    n = len(actions)
    for i in range(n):
        for j in range(i + 1, n):
            for ai in ALTS:
                if ai == actions[i]:
                    continue
                for aj in ALTS:
                    if aj == actions[j]:
                        continue
                    trial = list(actions)
                    trial[i] = ai
                    trial[j] = aj
                    t = _score(tg, resources, trial)
                    n_eval += 1
                    if t + 1e-12 < t0:
                        n_better += 1
                    if t + 1e-12 < best_t:
                        best_t = t
                        best_actions = trial
    if n_eval != count_h2_neighbors(actions):
        raise ValueError("h2 eval count %d != %d" % (n_eval, count_h2_neighbors(actions)))
    return best_t, best_actions, n_better, n_eval


def _frozen_cluster():
    from env.mec_offloaing_envs.offloading_env import Resources

    energy_config = {
        "use_energy": True,
        "reward_mode": "latency_over_all_mec",
        "energy_weight": 0.5,
        "latency_weight": 0.5,
        "rho": 1.0,
        "f_l": 1.0,
        "zeta": 2.0,
        "ptx": 0.1,
        "prx": 0.05,
        "ptx_v2v": 0.06,
        "prx_v2v": 0.03,
        "rho_v2v": 0.7,
        "f_v2v": 1.0,
        "normalize_energy": True,
    }
    return Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        v2v_process_capable=(1.0 * 1024 * 1024),
        v2v_bandwidth=5.0,
        use_energy=True,
        energy_config=energy_config,
    )


def run_hamming2_probe(seed, n_graphs=H2_N_GRAPHS, run_dir=None):
    """CPU diagnostic. No GPU. No PPO. paper_result=false. Frozen primary remains 3500."""
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan, resource_config_from_cluster
    from spec.phase4_campaign import diag_h2_run_dir

    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    if int(n_graphs) != H2_N_GRAPHS:
        raise ValueError("hamming2 n_graphs must be %d, got %s" % (H2_N_GRAPHS, n_graphs))
    if count_h2_neighbors([1] * H2_N_TOK) != 760:
        raise ValueError("frozen h2 neighborhood must be 760")

    rng = np.random.RandomState(seed)
    picks = stratified_graph_picks(META_TRAIN_IDS, H2_N_GRAPHS, H2_N_POOL, rng)
    cluster = _frozen_cluster()
    resources = resource_config_from_cluster(cluster)
    if run_dir is None:
        run_dir = diag_h2_run_dir(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for gi, (dist_id, graph_idx) in enumerate(picks):
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != H2_N_TOK:
            raise ValueError("graph %s task_number %s != %d" % (gv, n, H2_N_TOK))
        expert_plan, expert_res = greedy_from_mec_plan(tg, resources, max_passes=BC_MAX_PASSES)
        actions = [int(a) for _, a in expert_plan]
        t_g = float(expert_res.makespan_seconds)
        t_mec = _score(tg, resources, [1] * n)
        t_h1, n_h1 = best_h1(tg, resources, actions, t_g)
        t_h2, act_h2, n_h2, n_eval = best_h2(tg, resources, actions, t_g)
        row = {
            "dist_id": int(dist_id),
            "graph_idx": int(graph_idx),
            "mec_T": t_mec,
            "greedy_T": t_g,
            "h1_T": float(t_h1),
            "h2_T": float(t_h2),
            "h1_n_better": int(n_h1),
            "h2_n_better": int(n_h2),
            "h2_n_eval": int(n_eval),
            "greedy_n_non": _n_non(actions),
            "h2_n_non": _n_non(act_h2),
            "h1_improved": bool(t_h1 + 1e-12 < t_g),
            "h2_improved": bool(t_h2 + 1e-12 < t_g),
            "delta_h2": float(t_g - t_h2),
        }
        rows.append(row)
        if (gi + 1) % 10 == 0 or gi == 0 or gi == len(picks) - 1:
            print(
                "h2_graph %d/%d dist=%d idx=%d greedy=%.1f h2=%.1f dT=%.2f h1_better=%d"
                % (gi + 1, len(picks), dist_id, graph_idx, t_g, t_h2, t_g - t_h2, n_h1)
            )

    greedy_t = np.asarray([r["greedy_T"] for r in rows], dtype=np.float64)
    h2_t = np.asarray([r["h2_T"] for r in rows], dtype=np.float64)
    mec_t = np.asarray([r["mec_T"] for r in rows], dtype=np.float64)
    h1_t = np.asarray([r["h1_T"] for r in rows], dtype=np.float64)
    mean_delta = float(np.mean(greedy_t - h2_t))
    verdict = classify_mean_delta(mean_delta)
    per = {}
    by = defaultdict(list)
    for r in rows:
        by[str(r["dist_id"])].append(r)
    for did, group in by.items():
        gt = np.asarray([x["greedy_T"] for x in group], dtype=np.float64)
        ht = np.asarray([x["h2_T"] for x in group], dtype=np.float64)
        per[did] = {
            "n": int(len(group)),
            "greedy_T_mean": float(np.mean(gt)),
            "h2_T_mean": float(np.mean(ht)),
            "mean_delta": float(np.mean(gt - ht)),
            "frac_h2_improved": float(np.mean([x["h2_improved"] for x in group])),
        }
    payload = {
        "method_id": "margo_v0.1_diag_hamming2_expert",
        "paper_result": False,
        "ppo": False,
        "gpu": False,
        "n_graphs": int(len(rows)),
        "seed": seed,
        "split": "meta_train",
        "neighborhood": "hamming2_from_greedy_from_mec",
        "h2_neighbors": 760,
        "bc_max_passes": int(BC_MAX_PASSES),
        "mec_T_mean": float(np.mean(mec_t)),
        "greedy_T_mean": float(np.mean(greedy_t)),
        "h1_T_mean": float(np.mean(h1_t)),
        "h2_T_mean": float(np.mean(h2_t)),
        "mean_delta_h2": mean_delta,
        "p50_delta_h2": float(np.median(greedy_t - h2_t)),
        "max_delta_h2": float(np.max(greedy_t - h2_t)),
        "frac_h1_improved": float(np.mean([r["h1_improved"] for r in rows])),
        "frac_h2_improved": float(np.mean([r["h2_improved"] for r in rows])),
        "greedy_n_non_mean": float(np.mean([r["greedy_n_non"] for r in rows])),
        "h2_n_non_mean": float(np.mean([r["h2_n_non"] for r in rows])),
        "verdict": verdict,
        "thresholds": {"ood_not_teacher_lt": H2_CASE1_MAX_DELTA, "weak_expert_gte": H2_CASE2_MIN_DELTA},
        "per_distribution": per,
        "note": "one-shot Hamming-2 from greedy_from_mec; not ball from all-MEC; no PPO; CPU; paper_result=false; frozen primary remains 3500",
    }
    (run_dir / "hamming2_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "hamming2_rows.json").write_text(json.dumps(rows) + "\n")
    print(
        "h2_verdict %s greedy_T=%.1f h2_T=%.1f mec_T=%.1f dT=%.2f frac_h2=%.3f frac_h1=%.3f"
        % (
            verdict,
            payload["greedy_T_mean"],
            payload["h2_T_mean"],
            payload["mec_T_mean"],
            mean_delta,
            payload["frac_h2_improved"],
            payload["frac_h1_improved"],
        )
    )
    return run_dir, payload
