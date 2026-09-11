"""Iterative 2-opt teacher from greedy_from_mec. CPU. No PPO. Not the frozen 3500 primary.

H1 to a Hamming-1 local min, then repeat best Hamming-2 + H1 until H2 empty.
Writes new expert caches for later BC. Does not search a ball from all-MEC.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.hamming2_probe import (
    ALTS,
    H2_N_GRAPHS,
    H2_N_POOL,
    H2_N_TOK,
    _frozen_cluster,
    _gv_path,
    _n_non,
    _score,
    count_h2_neighbors,
    stratified_graph_picks,
)
from spec.phase4_campaign import META_TEST_IDS, META_TRAIN_IDS, ROOT, SEEDS, VALIDATION_IDS

MAX_H1_ROUNDS = 20
MAX_H2_ROUNDS = 8
TWOPT_N_TOK = H2_N_TOK
TWOPT_N_POOL = H2_N_POOL
CACHE_GREEDY = {
    "meta_train": ROOT / "runs" / "phase4" / "expert_greedy_mec_train.npz",
    "validation": ROOT / "runs" / "phase4" / "expert_greedy_mec_validation.npz",
    "meta_test": ROOT / "runs" / "phase4" / "expert_greedy_mec_metatest.npz",
}
CACHE_TWOPT = {
    "meta_train": ROOT / "runs" / "phase4" / "expert_2opt_mec_train.npz",
    "validation": ROOT / "runs" / "phase4" / "expert_2opt_mec_validation.npz",
    "meta_test": ROOT / "runs" / "phase4" / "expert_2opt_mec_metatest.npz",
}


def h1_best_step(actions, t0, score_fn):
    actions = [int(a) for a in actions]
    n = len(actions)
    best_t = float(t0)
    best = list(actions)
    n_eval = 0
    for k in range(n):
        for a in ALTS:
            if a == actions[k]:
                continue
            trial = list(actions)
            trial[k] = a
            t = float(score_fn(trial))
            n_eval += 1
            if t + 1e-12 < best_t:
                best_t = t
                best = trial
    improved = best_t + 1e-12 < float(t0)
    return best, best_t, int(n_eval), bool(improved)


def h1_until_local(actions, t0, score_fn, max_rounds=MAX_H1_ROUNDS):
    actions = [int(a) for a in actions]
    t = float(t0)
    n_eval = 0
    n_moves = 0
    cap = False
    while True:
        if n_moves >= int(max_rounds):
            cap = True
            break
        best, best_t, ne, improved = h1_best_step(actions, t, score_fn)
        n_eval += ne
        if not improved:
            break
        actions = best
        t = best_t
        n_moves += 1
    return actions, t, int(n_eval), int(n_moves), bool(cap)


def h2_best_step(actions, t0, score_fn):
    actions = [int(a) for a in actions]
    n = len(actions)
    best_t = float(t0)
    best = list(actions)
    n_eval = 0
    n_better = 0
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
                    t = float(score_fn(trial))
                    n_eval += 1
                    if t + 1e-12 < float(t0):
                        n_better += 1
                    if t + 1e-12 < best_t:
                        best_t = t
                        best = trial
    expect = count_h2_neighbors(actions)
    if n_eval != expect:
        raise ValueError("h2 eval count %d != %d" % (n_eval, expect))
    improved = best_t + 1e-12 < float(t0)
    return best, best_t, int(n_eval), int(n_better), bool(improved)


def iterate_2opt(actions, t0, score_fn, max_h1_rounds=MAX_H1_ROUNDS, max_h2_rounds=MAX_H2_ROUNDS):
    """H1 local min, then best H2 + H1 until H2 empty or cap."""
    actions = [int(a) for a in actions]
    t = float(t0)
    n_eval = 0
    h1_moves = 0
    h2_moves = 0
    h1_cap = False
    h2_cap = False
    act, t, ne, mv, cap = h1_until_local(actions, t, score_fn, max_rounds=max_h1_rounds)
    n_eval += ne
    h1_moves += mv
    h1_cap = bool(cap)
    t_after_h1 = float(t)
    t_after_first_h2 = float(t)
    first_h2_done = False
    while True:
        if h2_moves >= int(max_h2_rounds):
            h2_cap = True
            break
        best, best_t, ne, _n_better, improved = h2_best_step(act, t, score_fn)
        n_eval += ne
        if not first_h2_done:
            t_after_first_h2 = float(best_t if improved else t)
            first_h2_done = True
        if not improved:
            break
        act = best
        t = best_t
        h2_moves += 1
        act, t, ne, mv, cap = h1_until_local(act, t, score_fn, max_rounds=max_h1_rounds)
        n_eval += ne
        h1_moves += mv
        h1_cap = h1_cap or bool(cap)
    return {
        "actions": [int(a) for a in act],
        "t": float(t),
        "t_after_h1": float(t_after_h1),
        "t_after_first_h2": float(t_after_first_h2),
        "n_eval": int(n_eval),
        "h1_moves": int(h1_moves),
        "h2_moves": int(h2_moves),
        "h1_cap": bool(h1_cap),
        "h2_cap": bool(h2_cap),
    }


def _split_ids(split):
    if split == "meta_train":
        return META_TRAIN_IDS
    if split == "validation":
        return VALIDATION_IDS
    if split == "meta_test":
        return META_TEST_IDS
    raise ValueError("unknown split %s" % split)


def _all_picks(split):
    return [(int(d), int(g)) for d in _split_ids(split) for g in range(TWOPT_N_POOL)]


def _load_greedy_cache(split):
    path = CACHE_GREEDY[split]
    if not path.is_file():
        return None
    blob = np.load(str(path), allow_pickle=False)
    out = {
        "obs": np.asarray(blob["obs"], dtype=np.float32),
        "acts": np.asarray(blob["acts"], dtype=np.int32),
        "t_expert": np.asarray(blob["t_expert"], dtype=np.float64),
        "t_mec": np.asarray(blob["t_mec"], dtype=np.float64),
        "n_non": np.asarray(blob["n_non"], dtype=np.int32),
    }
    if "dist_id" in blob.files:
        out["dist_id"] = np.asarray(blob["dist_id"], dtype=np.int32)
    return out


def _cache_row(cache, dist_ids, dist_id, graph_idx):
    if cache is None or "dist_id" not in cache:
        di = list(dist_ids).index(int(dist_id))
        return di * TWOPT_N_POOL + int(graph_idx)
    di = list(dist_ids).index(int(dist_id))
    row = di * TWOPT_N_POOL + int(graph_idx)
    if int(cache["dist_id"][row]) != int(dist_id):
        raise ValueError("cache dist_id mismatch row=%d" % row)
    return row


def _summarize(rows):
    if not rows:
        raise ValueError("empty 2opt rows")
    greedy_t = np.asarray([r["greedy_T"] for r in rows], dtype=np.float64)
    h1_t = np.asarray([r["h1_T"] for r in rows], dtype=np.float64)
    first_h2_t = np.asarray([r["first_h2_T"] for r in rows], dtype=np.float64)
    tw_t = np.asarray([r["twopt_T"] for r in rows], dtype=np.float64)
    mec_t = np.asarray([r["mec_T"] for r in rows], dtype=np.float64)
    per = {}
    by = defaultdict(list)
    for r in rows:
        by[str(r["dist_id"])].append(r)
    for did, group in by.items():
        gt = np.asarray([x["greedy_T"] for x in group], dtype=np.float64)
        tt = np.asarray([x["twopt_T"] for x in group], dtype=np.float64)
        per[did] = {
            "n": int(len(group)),
            "greedy_T_mean": float(np.mean(gt)),
            "twopt_T_mean": float(np.mean(tt)),
            "mean_delta": float(np.mean(gt - tt)),
            "frac_twopt_improved": float(np.mean([x["twopt_improved"] for x in group])),
        }
    return {
        "n_graphs": int(len(rows)),
        "mec_T_mean": float(np.mean(mec_t)),
        "greedy_T_mean": float(np.mean(greedy_t)),
        "h1_T_mean": float(np.mean(h1_t)),
        "first_h2_T_mean": float(np.mean(first_h2_t)),
        "twopt_T_mean": float(np.mean(tw_t)),
        "mean_delta_h1": float(np.mean(greedy_t - h1_t)),
        "mean_delta_first_h2": float(np.mean(greedy_t - first_h2_t)),
        "mean_delta_twopt": float(np.mean(greedy_t - tw_t)),
        "mean_extra_vs_first_h2": float(np.mean(first_h2_t - tw_t)),
        "p50_delta_twopt": float(np.median(greedy_t - tw_t)),
        "max_delta_twopt": float(np.max(greedy_t - tw_t)),
        "frac_h1_improved": float(np.mean([r["h1_improved"] for r in rows])),
        "frac_first_h2_improved": float(np.mean([r["first_h2_improved"] for r in rows])),
        "frac_twopt_improved": float(np.mean([r["twopt_improved"] for r in rows])),
        "frac_h2_moves_ge2": float(np.mean([r["h2_moves"] >= 2 for r in rows])),
        "h2_moves_mean": float(np.mean([r["h2_moves"] for r in rows])),
        "h1_moves_mean": float(np.mean([r["h1_moves"] for r in rows])),
        "n_eval_mean": float(np.mean([r["n_eval"] for r in rows])),
        "frac_h2_cap": float(np.mean([r["h2_cap"] for r in rows])),
        "greedy_n_non_mean": float(np.mean([r["greedy_n_non"] for r in rows])),
        "twopt_n_non_mean": float(np.mean([r["twopt_n_non"] for r in rows])),
        "per_distribution": per,
    }


def _write_split_cache(split, rows, greedy_cache, run_dir):
    n = len(rows)
    if greedy_cache is None or greedy_cache["obs"].shape[0] != n:
        print("twopt cache skip %s obs mismatch greedy=%s n=%d" % (split, greedy_cache is not None, n))
        return None
    dist_ids = _split_ids(split)
    obs_rows = []
    act_rows = []
    t_ex = []
    t_mec = []
    n_non = []
    dist_id_out = []
    for r in rows:
        row = _cache_row(greedy_cache, dist_ids, r["dist_id"], r["graph_idx"])
        obs_rows.append(greedy_cache["obs"][row])
        act_rows.append(np.asarray(r["twopt_acts"], dtype=np.int32))
        t_ex.append(r["twopt_T"])
        t_mec.append(r["mec_T"])
        n_non.append(r["twopt_n_non"])
        dist_id_out.append(r["dist_id"])
    obs = np.stack(obs_rows, axis=0)
    acts = np.stack(act_rows, axis=0)
    dest = CACHE_TWOPT[split]
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        obs=obs,
        acts=acts,
        t_expert=np.asarray(t_ex, dtype=np.float64),
        t_mec=np.asarray(t_mec, dtype=np.float64),
        n_non=np.asarray(n_non, dtype=np.int32),
        dist_id=np.asarray(dist_id_out, dtype=np.int32),
    )
    np.savez_compressed(str(dest), **payload)
    copy = Path(run_dir) / dest.name
    np.savez_compressed(str(copy), **payload)
    print("twopt cache write %s n=%d" % (dest, n))
    return str(dest)


def _audit_split(split, cluster, resources):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    picks = _all_picks(split)
    dist_ids = _split_ids(split)
    cache = _load_greedy_cache(split)
    use_cache_acts = cache is not None and cache["acts"].shape[0] == len(picks)
    rows = []
    for gi, (dist_id, graph_idx) in enumerate(picks):
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != TWOPT_N_TOK:
            raise ValueError("graph %s task_number %s != %d" % (gv, n, TWOPT_N_TOK))

        def score_fn(trial, _tg=tg, _res=resources):
            return _score(_tg, _res, trial)

        if use_cache_acts:
            row = _cache_row(cache, dist_ids, dist_id, graph_idx)
            actions = [int(a) for a in cache["acts"][row]]
            t_g = float(cache["t_expert"][row])
            t_mec = float(cache["t_mec"][row])
            src = "cache"
        else:
            plan, result = greedy_from_mec_plan(tg, resources, max_passes=BC_MAX_PASSES)
            actions = [int(a) for _, a in plan]
            t_g = float(result.makespan_seconds)
            t_mec = float(score_fn([1] * n))
            src = "greedy_from_mec"
        if len(actions) != n:
            raise ValueError("act length %d != n %d" % (len(actions), n))
        out = iterate_2opt(actions, t_g, score_fn)
        row = {
            "dist_id": int(dist_id),
            "graph_idx": int(graph_idx),
            "split": split,
            "expert_source": src,
            "mec_T": t_mec,
            "greedy_T": t_g,
            "h1_T": float(out["t_after_h1"]),
            "first_h2_T": float(out["t_after_first_h2"]),
            "twopt_T": float(out["t"]),
            "twopt_acts": out["actions"],
            "greedy_n_non": _n_non(actions),
            "twopt_n_non": _n_non(out["actions"]),
            "h1_moves": int(out["h1_moves"]),
            "h2_moves": int(out["h2_moves"]),
            "n_eval": int(out["n_eval"]),
            "h1_cap": bool(out["h1_cap"]),
            "h2_cap": bool(out["h2_cap"]),
            "h1_improved": bool(out["t_after_h1"] + 1e-12 < t_g),
            "first_h2_improved": bool(out["t_after_first_h2"] + 1e-12 < t_g),
            "twopt_improved": bool(out["t"] + 1e-12 < t_g),
            "delta_twopt": float(t_g - out["t"]),
        }
        rows.append(row)
        if (gi + 1) % 10 == 0 or gi == 0 or gi == len(picks) - 1:
            print(
                "twopt %s %d/%d dist=%d idx=%d greedy=%.1f h1=%.1f h2=%.1f tw=%.1f h2m=%d"
                % (
                    split,
                    gi + 1,
                    len(picks),
                    dist_id,
                    graph_idx,
                    t_g,
                    out["t_after_h1"],
                    out["t_after_first_h2"],
                    out["t"],
                    out["h2_moves"],
                )
            )
    summary = _summarize(rows)
    summary["split"] = split
    summary["expert_source"] = "cache" if use_cache_acts else "greedy_from_mec"
    return rows, summary, cache


def run_twopt_expert(seed, run_dir=None):
    """CPU diagnostic. No GPU. No PPO. paper_result=false. Frozen primary remains 3500."""
    from env.mec_offloaing_envs.scheduler import resource_config_from_cluster
    from spec.phase4_campaign import diag_twopt_run_dir

    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    if count_h2_neighbors([1] * TWOPT_N_TOK) != 760:
        raise ValueError("frozen h2 neighborhood must be 760")
    cluster = _frozen_cluster()
    resources = resource_config_from_cluster(cluster)
    if run_dir is None:
        run_dir = diag_twopt_run_dir(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    splits = {}
    caches_written = {}
    for split in ("meta_train", "validation", "meta_test"):
        rows, summary, greedy_cache = _audit_split(split, cluster, resources)
        caches_written[split] = _write_split_cache(split, rows, greedy_cache, run_dir)
        slim = []
        for r in rows:
            slim.append({k: v for k, v in r.items() if k != "twopt_acts"})
        all_rows.extend(slim)
        splits[split] = summary
        print(
            "twopt_split %s greedy=%.1f h1=%.1f first_h2=%.1f tw=%.1f dT=%.2f extra_vs_h2=%.2f h2m=%.2f"
            % (
                split,
                summary["greedy_T_mean"],
                summary["h1_T_mean"],
                summary["first_h2_T_mean"],
                summary["twopt_T_mean"],
                summary["mean_delta_twopt"],
                summary["mean_extra_vs_first_h2"],
                summary["h2_moves_mean"],
            )
        )

    rng = np.random.RandomState(seed)
    h2_picks = set(stratified_graph_picks(META_TRAIN_IDS, H2_N_GRAPHS, H2_N_POOL, rng))
    h2_rows = [
        r
        for r in all_rows
        if r["split"] == "meta_train" and (r["dist_id"], r["graph_idx"]) in h2_picks
    ]
    h2_join = _summarize(h2_rows) if h2_rows else None
    if h2_join is not None:
        h2_join["n_graphs_expected"] = int(H2_N_GRAPHS)
        if int(h2_join["n_graphs"]) != int(H2_N_GRAPHS):
            raise ValueError("h2 join size %s != %s" % (h2_join["n_graphs"], H2_N_GRAPHS))

    payload = {
        "method_id": "margo_v0.1_diag_2opt_expert",
        "paper_result": False,
        "ppo": False,
        "gpu": False,
        "seed": seed,
        "max_h1_rounds": int(MAX_H1_ROUNDS),
        "max_h2_rounds": int(MAX_H2_ROUNDS),
        "bc_max_passes": int(BC_MAX_PASSES),
        "h2_neighbors": 760,
        "splits": splits,
        "h2_join_train200": h2_join,
        "caches": caches_written,
        "note": "iterative 2-opt from greedy_from_mec; H1 local min then repeat best H2; no PPO; CPU; paper_result=false; frozen primary remains 3500",
    }
    (run_dir / "twopt_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "twopt_rows.json").write_text(json.dumps(all_rows) + "\n")
    tr = splits["meta_train"]
    print(
        "twopt_verdict train greedy=%.1f tw=%.1f mec=%.1f dT=%.2f extra_vs_h2=%.2f h2m=%.2f"
        % (
            tr["greedy_T_mean"],
            tr["twopt_T_mean"],
            tr["mec_T_mean"],
            tr["mean_delta_twopt"],
            tr["mean_extra_vs_first_h2"],
            tr["h2_moves_mean"],
        )
    )
    return run_dir, payload
