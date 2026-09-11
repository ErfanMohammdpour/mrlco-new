"""Motif audit of greedy_from_mec expert. CPU. No PPO. Not the frozen 3500 primary.

Question: non-MEC support is one DAG-connected region, or scattered islands?
Hamming-2 pairwise gain is not enough: those pairs may not be path-connected.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.hamming2_probe import (
    H2_N_GRAPHS,
    H2_N_POOL,
    H2_N_TOK,
    _frozen_cluster,
    _gv_path,
    stratified_graph_picks,
)
from spec.phase4_campaign import META_TEST_IDS, META_TRAIN_IDS, ROOT, SEEDS, VALIDATION_IDS

MOTIF_N_TRAIN = H2_N_GRAPHS
MOTIF_N_POOL = H2_N_POOL
MOTIF_N_TOK = H2_N_TOK
CLUSTER_MAX_CC_RATIO = 0.4
CLUSTER_MIN_LOCAL_NBR = 0.5
SCATTER_MIN_CC_RATIO = 0.75
SCATTER_MAX_LOCAL_NBR = 0.35
CACHE_TRAIN = ROOT / "runs" / "phase4" / "expert_greedy_mec_train.npz"
CACHE_VAL = ROOT / "runs" / "phase4" / "expert_greedy_mec_validation.npz"
CACHE_TEST = ROOT / "runs" / "phase4" / "expert_greedy_mec_metatest.npz"


def undirected_adj(succ_sets):
    n = len(succ_sets)
    adj = [set() for _ in range(n)]
    for u, succs in enumerate(succ_sets):
        for v in succs:
            v = int(v)
            adj[u].add(v)
            adj[v].add(u)
    return adj


def connected_components(nodes, adj):
    nodes = set(int(x) for x in nodes)
    seen = set()
    sizes = []
    for s in sorted(nodes):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        sz = 0
        while stack:
            u = stack.pop()
            sz += 1
            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        sizes.append(int(sz))
    return int(len(sizes)), sizes


def directed_edges(succ_sets):
    edges = []
    for u, succs in enumerate(succ_sets):
        for v in succs:
            edges.append((int(u), int(v)))
    return edges


def edge_homophily(act, edges):
    if not edges:
        return float("nan"), 0, 0, 0, 0
    same = 0
    ll = mm = vv = 0
    for u, v in edges:
        au = int(act[u])
        av = int(act[v])
        if au != av:
            continue
        same += 1
        if au == 0:
            ll += 1
        elif au == 1:
            mm += 1
        else:
            vv += 1
    n_e = float(len(edges))
    return float(same) / n_e, int(same), int(ll), int(mm), int(vv)


def local_neighbor_frac(act, adj):
    locals_ = [i for i, a in enumerate(act) if int(a) == 0]
    if not locals_:
        return float("nan")
    hit = 0
    for u in locals_:
        if any(int(act[v]) == 0 for v in adj[u]):
            hit += 1
    return float(hit) / float(len(locals_))


def topo_order(n, pre_sets, succ_sets):
    indeg = [len(pre_sets[i]) for i in range(n)]
    q = [i for i in range(n) if indeg[i] == 0]
    out = []
    while q:
        u = q.pop()
        out.append(u)
        for v in succ_sets[u]:
            v = int(v)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != n:
        raise ValueError("DAG topo incomplete %d/%d" % (len(out), n))
    return out


def longest_compute_path(n, succ_sets, weights, pre_sets):
    order = topo_order(n, pre_sets, succ_sets)
    dist = [float(w) for w in weights]
    parent = [-1] * n
    for u in order:
        for v in succ_sets[u]:
            v = int(v)
            cand = dist[u] + float(weights[v])
            if cand > dist[v] + 1e-12:
                dist[v] = cand
                parent[v] = u
    end = int(np.argmax(np.asarray(dist, dtype=np.float64)))
    path = []
    cur = end
    seen = set()
    while cur != -1:
        if cur in seen:
            raise ValueError("parent cycle in longest path")
        seen.add(cur)
        path.append(int(cur))
        cur = parent[cur]
    path.reverse()
    return path


def path_location_changes(act, path):
    if len(path) < 2:
        return 0, 0.0
    n_ch = 0
    for i in range(1, len(path)):
        if int(act[path[i]]) != int(act[path[i - 1]]):
            n_ch += 1
    return int(n_ch), float(n_ch) / float(len(path) - 1)


def classify_motif(n_non_mean, n_cc_mean, local_nbr_frac):
    n_non_mean = float(n_non_mean)
    n_cc_mean = float(n_cc_mean)
    local_nbr_frac = float(local_nbr_frac)
    if n_non_mean < 0.5:
        return "all_mec"
    ratio = n_cc_mean / max(n_non_mean, 1e-12)
    clustered = (ratio <= CLUSTER_MAX_CC_RATIO) and (local_nbr_frac >= CLUSTER_MIN_LOCAL_NBR)
    scattered = (ratio >= SCATTER_MIN_CC_RATIO) and (local_nbr_frac < SCATTER_MAX_LOCAL_NBR)
    if clustered and not scattered:
        return "region_clustered"
    if scattered and not clustered:
        return "scattered_support"
    return "mixed_structure"


def decoder_acts_to_task_order(order, decoder_acts, n):
    order = [int(tid) for tid in order]
    decoder_acts = [int(a) for a in decoder_acts]
    if len(order) != len(decoder_acts):
        raise ValueError("order/act length mismatch %d %d" % (len(order), len(decoder_acts)))
    if len(order) != n:
        raise ValueError("order length %d != n %d" % (len(order), n))
    act = [1] * n
    seen = set()
    for tid, a in zip(order, decoder_acts):
        if tid in seen or tid < 0 or tid >= n:
            raise ValueError("bad task id %s" % tid)
        seen.add(tid)
        act[tid] = a
    if len(seen) != n:
        raise ValueError("decoder order not a permutation")
    return act


def _cache_row(dist_ids, dist_id, graph_idx, n_pool):
    dist_ids = [int(x) for x in dist_ids]
    di = dist_ids.index(int(dist_id))
    return di * int(n_pool) + int(graph_idx)


def _load_expert_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    blob = np.load(str(path))
    if "acts" not in blob.files or "dist_id" not in blob.files:
        return None
    return {
        "acts": np.asarray(blob["acts"], dtype=np.int32),
        "dist_id": np.asarray(blob["dist_id"], dtype=np.int32),
        "t_expert": np.asarray(blob["t_expert"], dtype=np.float64) if "t_expert" in blob.files else None,
    }


def _graph_metrics(tg, act):
    n = int(tg.task_number)
    if len(act) != n:
        raise ValueError("act length %d != n %d" % (len(act), n))
    succ = tg.succ_task_sets
    pre = tg.pre_task_sets
    adj = undirected_adj(succ)
    edges = directed_edges(succ)
    non = [i for i, a in enumerate(act) if int(a) != 1]
    loc = [i for i, a in enumerate(act) if int(a) == 0]
    n_cc, sizes = connected_components(non, adj)
    n_cc_local, local_sizes = connected_components(loc, adj)
    homo, n_same, n_ll, n_mm, n_vv = edge_homophily(act, edges)
    lnbr = local_neighbor_frac(act, adj)
    weights = [float(tg.task_list[i].processing_data_size) for i in range(n)]
    path = longest_compute_path(n, succ, weights, pre)
    n_ch, ch_frac = path_location_changes(act, path)
    n_non = int(len(non))
    max_cc = int(max(sizes) if sizes else 0)
    return {
        "n_non": n_non,
        "n_cc": int(n_cc),
        "max_cc": max_cc,
        "n_local": int(len(loc)),
        "n_cc_local": int(n_cc_local),
        "max_cc_local": int(max(local_sizes) if local_sizes else 0),
        "edge_homophily": float(homo),
        "n_edges": int(len(edges)),
        "n_same_label_edges": int(n_same),
        "n_ll_edges": int(n_ll),
        "n_mm_edges": int(n_mm),
        "n_vv_edges": int(n_vv),
        "local_nbr_frac": None if np.isnan(lnbr) else float(lnbr),
        "cp_len": int(len(path)),
        "cp_changes": int(n_ch),
        "cp_change_frac": float(ch_frac),
        "cc_ratio": float(n_cc) / float(max(n_non, 1)),
    }


def _summarize(rows):
    if not rows:
        raise ValueError("empty motif rows")
    n_non = np.asarray([r["n_non"] for r in rows], dtype=np.float64)
    n_cc = np.asarray([r["n_cc"] for r in rows], dtype=np.float64)
    lnbr = np.asarray(
        [r["local_nbr_frac"] for r in rows if r["local_nbr_frac"] is not None],
        dtype=np.float64,
    )
    homo = np.asarray([r["edge_homophily"] for r in rows], dtype=np.float64)
    n_non_mean = float(np.mean(n_non))
    n_cc_mean = float(np.mean(n_cc))
    lnbr_mean = float(np.mean(lnbr)) if lnbr.size else float("nan")
    verdict = classify_motif(n_non_mean, n_cc_mean, 0.0 if np.isnan(lnbr_mean) else lnbr_mean)
    ge2 = [r for r in rows if r["n_non"] >= 2]
    frac_single_cc = float(np.mean([r["n_cc"] == 1 for r in ge2])) if ge2 else float("nan")
    per = {}
    by = defaultdict(list)
    for r in rows:
        by[str(r["dist_id"])].append(r)
    for did, group in by.items():
        nn = np.asarray([x["n_non"] for x in group], dtype=np.float64)
        cc = np.asarray([x["n_cc"] for x in group], dtype=np.float64)
        per[did] = {
            "n": int(len(group)),
            "n_non_mean": float(np.mean(nn)),
            "n_cc_mean": float(np.mean(cc)),
            "cc_ratio_mean": float(np.mean(cc / np.maximum(nn, 1.0))),
        }
    return {
        "n_graphs": int(len(rows)),
        "n_non_mean": n_non_mean,
        "n_non_p50": float(np.median(n_non)),
        "n_cc_mean": n_cc_mean,
        "n_cc_p50": float(np.median(n_cc)),
        "max_cc_mean": float(np.mean([r["max_cc"] for r in rows])),
        "cc_ratio_mean": float(np.mean([r["cc_ratio"] for r in rows])),
        "frac_n_non_ge2_single_cc": frac_single_cc,
        "edge_homophily_mean": float(np.mean(homo)),
        "local_nbr_frac_mean": lnbr_mean,
        "n_cc_local_mean": float(np.mean([r["n_cc_local"] for r in rows])),
        "cp_len_mean": float(np.mean([r["cp_len"] for r in rows])),
        "cp_changes_mean": float(np.mean([r["cp_changes"] for r in rows])),
        "cp_change_frac_mean": float(np.mean([r["cp_change_frac"] for r in rows])),
        "verdict": verdict,
        "per_distribution": per,
    }


def _split_picks(split, seed):
    if split == "meta_train":
        rng = np.random.RandomState(int(seed))
        return stratified_graph_picks(META_TRAIN_IDS, MOTIF_N_TRAIN, MOTIF_N_POOL, rng)
    if split == "validation":
        return [(int(d), int(g)) for d in VALIDATION_IDS for g in range(MOTIF_N_POOL)]
    if split == "meta_test":
        return [(int(d), int(g)) for d in META_TEST_IDS for g in range(MOTIF_N_POOL)]
    raise ValueError("unknown split %s" % split)


def _dist_ids_for_split(split):
    if split == "meta_train":
        return META_TRAIN_IDS
    if split == "validation":
        return VALIDATION_IDS
    if split == "meta_test":
        return META_TEST_IDS
    raise ValueError("unknown split %s" % split)


def _cache_for_split(split):
    if split == "meta_train":
        return CACHE_TRAIN
    if split == "validation":
        return CACHE_VAL
    if split == "meta_test":
        return CACHE_TEST
    raise ValueError("unknown split %s" % split)


def _expert_task_act(tg, resources, cache, dist_ids, dist_id, graph_idx):
    n = int(tg.task_number)
    if cache is not None:
        row = _cache_row(dist_ids, dist_id, graph_idx, MOTIF_N_POOL)
        if int(cache["dist_id"][row]) != int(dist_id):
            raise ValueError("cache dist_id mismatch row=%d" % row)
        decoder_acts = cache["acts"][row]
        act = decoder_acts_to_task_order(tg.prioritize_sequence, decoder_acts, n)
        t_ex = None if cache["t_expert"] is None else float(cache["t_expert"][row])
        return act, t_ex, "cache"
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    plan, result = greedy_from_mec_plan(tg, resources, max_passes=BC_MAX_PASSES)
    act = [1] * n
    for tid, a in plan:
        act[int(tid)] = int(a)
    return act, float(result.makespan_seconds), "greedy_from_mec"


def _audit_split(split, seed, cluster, resources):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

    picks = _split_picks(split, seed)
    dist_ids = _dist_ids_for_split(split)
    cache = _load_expert_cache(_cache_for_split(split))
    source = "cache" if cache is not None else "greedy_from_mec"
    rows = []
    for gi, (dist_id, graph_idx) in enumerate(picks):
        gv = _gv_path(dist_id, graph_idx)
        if not gv.is_file():
            raise FileNotFoundError(str(gv))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        n = int(tg.task_number)
        if n != MOTIF_N_TOK:
            raise ValueError("graph %s task_number %s != %d" % (gv, n, MOTIF_N_TOK))
        act, t_ex, used = _expert_task_act(tg, resources, cache, dist_ids, dist_id, graph_idx)
        metrics = _graph_metrics(tg, act)
        metrics.update(
            {
                "dist_id": int(dist_id),
                "graph_idx": int(graph_idx),
                "split": split,
                "expert_source": used,
                "expert_T": t_ex,
            }
        )
        rows.append(metrics)
        if (gi + 1) % 50 == 0 or gi == 0 or gi == len(picks) - 1:
            print(
                "motif %s %d/%d dist=%d idx=%d n_non=%d n_cc=%d homo=%.3f"
                % (
                    split,
                    gi + 1,
                    len(picks),
                    dist_id,
                    graph_idx,
                    metrics["n_non"],
                    metrics["n_cc"],
                    metrics["edge_homophily"],
                )
            )
    summary = _summarize(rows)
    summary["split"] = split
    summary["expert_source"] = source
    return rows, summary


def run_motif_audit(seed, run_dir=None):
    """CPU diagnostic. No GPU. No PPO. paper_result=false. Frozen primary remains 3500."""
    from env.mec_offloaing_envs.scheduler import resource_config_from_cluster
    from spec.phase4_campaign import diag_motif_run_dir

    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    cluster = _frozen_cluster()
    resources = resource_config_from_cluster(cluster)
    if run_dir is None:
        run_dir = diag_motif_run_dir(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    splits = {}
    for split in ("meta_train", "validation", "meta_test"):
        rows, summary = _audit_split(split, seed, cluster, resources)
        all_rows.extend(rows)
        splits[split] = summary
        print(
            "motif_split %s verdict=%s n_non=%.2f n_cc=%.2f cc_ratio=%.3f local_nbr=%.3f homo=%.3f cp_ch=%.2f"
            % (
                split,
                summary["verdict"],
                summary["n_non_mean"],
                summary["n_cc_mean"],
                summary["cc_ratio_mean"],
                summary["local_nbr_frac_mean"],
                summary["edge_homophily_mean"],
                summary["cp_changes_mean"],
            )
        )

    train_v = splits["meta_train"]["verdict"]
    val_v = splits["validation"]["verdict"]
    test_v = splits["meta_test"]["verdict"]
    if train_v == "region_clustered" and val_v == "region_clustered":
        architecture_hint = "region_head"
    elif train_v == "scattered_support" and val_v == "scattered_support":
        architecture_hint = "skip_region_sparse_support"
    else:
        architecture_hint = "mixed_or_ood_structure"
    payload = {
        "method_id": "margo_v0.1_diag_motif_expert",
        "paper_result": False,
        "ppo": False,
        "gpu": False,
        "seed": seed,
        "n_train_graphs": MOTIF_N_TRAIN,
        "n_val_graphs": int(splits["validation"]["n_graphs"]),
        "n_metatest_graphs": int(splits["meta_test"]["n_graphs"]),
        "bc_max_passes": int(BC_MAX_PASSES),
        "splits": splits,
        "train_verdict": train_v,
        "val_verdict": val_v,
        "metatest_verdict": test_v,
        "architecture_hint": architecture_hint,
        "thresholds": {
            "cluster_max_cc_ratio": CLUSTER_MAX_CC_RATIO,
            "cluster_min_local_nbr": CLUSTER_MIN_LOCAL_NBR,
            "scatter_min_cc_ratio": SCATTER_MIN_CC_RATIO,
            "scatter_max_local_nbr": SCATTER_MAX_LOCAL_NBR,
        },
        "note": "motif audit of greedy_from_mec; CC of non-MEC on undirected DAG skeleton; no PPO; CPU; paper_result=false; frozen primary remains 3500",
    }
    (run_dir / "motif_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "motif_rows.json").write_text(json.dumps(all_rows) + "\n")
    print(
        "motif_verdict train=%s val=%s test=%s hint=%s"
        % (train_v, val_v, test_v, architecture_hint)
    )
    return run_dir, payload
