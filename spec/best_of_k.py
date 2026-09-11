"""Best-of-k neural inference. Sampling from π, one schedule() per sample. No training."""

from __future__ import annotations

import hashlib
import json
import time
import types
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import GREEDY_PAD_ACTION, align_greedy_pred, collect_reachability, policy_feed

EVAL_BATCH = 32

ROOT = Path(__file__).resolve().parent.parent
K_SWEEP = (1, 4, 8, 16, 32, 64)
TEMPERATURES_AT_K32 = (0.7, 1.0, 1.3)
DEFAULT_TEMPERATURE = 1.0
N_TOK = 20
ACTION_FROM_NAME = {"UE": 0, "MEC": 1, "HELPER": 2, 0: 0, 1: 1, 2: 2}
TOY_ORACLE_FILES = (
    "01_all-local-chain.yaml",
    "04_ue-to-mec.yaml",
    "05_ue-to-helper.yaml",
    "06_mec-to-ue.yaml",
    "07_mec-to-helper.yaml",
)
PAIR_SEQ_HEURISTIC = {
    "note": "pair_seq bestimp k20 is a heuristic reference, not the method",
    "T_best": 447.9,
    "evals_per_graph": 718,
}
CKPT_PHASE1_MEANAGG = (
    ROOT / "runs" / "phase4" / "margo_v0.3_diag_encoder" / "meanagg_triple" / "seed_0" / "ckpt" / "bc_core.ckpt"
)
CKPT_BC2OPT = (
    ROOT / "runs" / "phase4" / "margo_v0.1_diag_bc_2opt" / "seed_0" / "ckpt" / "bc_core.ckpt"
)


def resolve_bok_ckpt(explicit=None):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError("best-of-k ckpt missing: %s" % path)
        return path
    if CKPT_PHASE1_MEANAGG.is_file():
        return CKPT_PHASE1_MEANAGG
    if CKPT_BC2OPT.is_file():
        return CKPT_BC2OPT
    raise FileNotFoundError(
        "no Phase 1 winner ckpt and no bc_2opt fallback at %s or %s"
        % (CKPT_PHASE1_MEANAGG, CKPT_BC2OPT)
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def toy_graph_from_yaml(doc):
    nodes = list(doc["nodes"])
    id_map = {int(n["task_id"]): i for i, n in enumerate(nodes)}
    n = len(nodes)
    tg = types.SimpleNamespace()
    tg.task_number = n
    tg.task_list = []
    tg.pre_task_sets = [set() for _ in range(n)]
    tg.edge_set = []
    for node in nodes:
        tg.task_list.append(
            types.SimpleNamespace(
                processing_data_size=int(node["compute_workload_bytes"]),
                transmission_data_size=int(node["task_output_bytes"]),
            )
        )
    for edge in doc.get("edges") or []:
        src = id_map[int(edge["src_task_id"])]
        dst = id_map[int(edge["dst_task_id"])]
        nbytes = int(edge["edge_output_bytes"])
        tg.pre_task_sets[dst].add(src)
        tg.edge_set.append([src, 0, 0, nbytes, dst, 0, 0])
    tg.prioritize_sequence = list(range(n))
    acts = [int(ACTION_FROM_NAME[a]) for a in doc["actions"]]
    return tg, np.asarray(acts, dtype=np.int32)


def load_toy_oracle(name):
    import yaml

    path = ROOT / "spec" / "toy_oracles" / name
    doc = yaml.safe_load(path.read_text())
    tg, acts = toy_graph_from_yaml(doc)
    return tg, acts, float(doc["expected"]["makespan_seconds"]), doc


def _sample_feed(policy, obs, fl, reach=None, temperature=DEFAULT_TEMPERATURE, top_p=None):
    fd = policy_feed(policy, obs, fl, reach=reach)
    temp_ph = getattr(policy.network, "sample_softmax_temperature", None)
    if temp_ph is not None:
        fd[temp_ph] = np.float32(temperature)
    top_ph = getattr(policy.network, "sample_top_p", None)
    if top_ph is not None and top_p is not None:
        fd[top_ph] = np.float32(top_p)
    return fd


def greedy_plan(sess, policy, obs, reach=None):
    n = int(obs.shape[0])
    n_tok = int(obs.shape[1])
    rows = []
    for start in range(0, n, EVAL_BATCH):
        sl = slice(start, min(start + EVAL_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        reach_sl = None if reach is None else reach[sl]
        pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict=policy_feed(policy, obs[sl], fl, reach=reach_sl),
        )
        aligned, _trunc = align_greedy_pred(pred, n_tok, pad_value=GREEDY_PAD_ACTION)
        rows.append(aligned)
    out = np.concatenate(rows, axis=0)
    if int(out.max()) > 2:
        raise ValueError("greedy emitted token > 2: max=%s" % int(out.max()))
    if out.shape[1] != n_tok:
        raise ValueError("greedy length %s != %d" % (out.shape, n_tok))
    return out


def _sample_once(sess, policy, obs, reach=None, temperature=DEFAULT_TEMPERATURE, top_p=None):
    n = int(obs.shape[0])
    n_tok = int(obs.shape[1])
    rows = []
    for start in range(0, n, EVAL_BATCH):
        sl = slice(start, min(start + EVAL_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        reach_sl = None if reach is None else reach[sl]
        pred = sess.run(
            policy.network.sample_decoder_prediction,
            feed_dict=_sample_feed(
                policy, obs[sl], fl, reach=reach_sl, temperature=temperature, top_p=top_p
            ),
        )
        aligned, _trunc = align_greedy_pred(pred, n_tok, pad_value=GREEDY_PAD_ACTION)
        rows.append(aligned)
    out = np.concatenate(rows, axis=0)
    if int(out.max()) > 2:
        raise ValueError("sample emitted token > 2: max=%s" % int(out.max()))
    if out.shape[1] != n_tok:
        raise ValueError("sample length %s != %d" % (out.shape, n_tok))
    return out


def sample_plans(sess, policy, obs, k, temperature=DEFAULT_TEMPERATURE, top_p=None, reach=None):
    """Return [B, k, 20]. Slot 0 is greedy so best-of-k ≥ greedy by construction."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    greedy = greedy_plan(sess, policy, obs, reach=reach)
    if k == 1:
        return greedy[:, None, :]
    slots = [greedy]
    for _ in range(k - 1):
        slots.append(
            _sample_once(
                sess, policy, obs, reach=reach, temperature=temperature, top_p=top_p
            )
        )
    return np.stack(slots, axis=1)


def score_plans(graphs, plans, resources, refs_cache=None, cache_prefix=""):
    """Makespan seconds via one schedule_via_adapter per plan. plans [B, k, L] or [B, L]."""
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    plans = np.asarray(plans, dtype=np.int32)
    if plans.ndim == 2:
        plans = plans[:, None, :]
    if plans.ndim != 3:
        raise ValueError("plans rank %d, want 2 or 3" % plans.ndim)
    n_graphs, k, _n_tok = plans.shape
    if n_graphs != len(graphs):
        raise ValueError("graphs %d != plans B %d" % (len(graphs), n_graphs))
    if refs_cache is None:
        refs_cache = {}
    scores = np.zeros((n_graphs, k), dtype=np.float64)
    for i, tg in enumerate(graphs):
        order = [int(tid) for tid in tg.prioritize_sequence]
        n = len(order)
        key_m = (cache_prefix, i, "mec")
        key_u = (cache_prefix, i, "ue")
        if key_m not in refs_cache:
            mec, _, _ = schedule_via_adapter(
                tg, list(zip(order, [1] * n)), resources
            )
            refs_cache[key_m] = float(mec.makespan_seconds)
        if key_u not in refs_cache:
            ue, _, _ = schedule_via_adapter(
                tg, list(zip(order, [0] * n)), resources
            )
            refs_cache[key_u] = float(ue.makespan_seconds)
        for j in range(k):
            acts = [int(a) for a in plans[i, j, :n].tolist()]
            if any(a not in (0, 1, 2) for a in acts):
                raise ValueError("invalid plan actions %s" % acts)
            result, _, _ = schedule_via_adapter(
                tg, list(zip(order, acts)), resources
            )
            scores[i, j] = float(result.makespan_seconds)
    return scores


def prefix_scores(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    k = int(k)
    if k < 1 or k > scores.shape[1]:
        raise ValueError("k=%s not in 1..%d" % (k, scores.shape[1]))
    return scores[:, :k]


def best_index(scores_k):
    return np.argmin(np.asarray(scores_k, dtype=np.float64), axis=1)


def duplicates_ratio(plans):
    plans = np.asarray(plans)
    n, k = plans.shape[0], plans.shape[1]
    ratios = []
    for i in range(n):
        rows = [tuple(plans[i, j].tolist()) for j in range(k)]
        ratios.append(1.0 - (float(len(set(rows))) / float(k)))
    return float(np.mean(ratios))


def _mix_row(plan):
    acts = np.asarray(plan, dtype=np.int32)
    counts = np.bincount(acts.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non = np.sum(acts != 1)
    return {
        "local_frac": float(counts[0]),
        "mec_frac": float(counts[1]),
        "v2v_frac": float(counts[2]),
        "n_non": int(n_non),
    }


def summarize_k(plans, scores, k, expert_T=None, dist_ids=None, wall_s=None):
    sub_p = plans[:, :k]
    sub_s = prefix_scores(scores, k)
    best_i = best_index(sub_s)
    t_best = sub_s[np.arange(sub_s.shape[0]), best_i]
    t_greedy = sub_s[:, 0]
    t_mean = np.mean(sub_s, axis=1)
    t_p10 = np.percentile(sub_s, 10, axis=1)
    best_plans = sub_p[np.arange(sub_p.shape[0]), best_i]
    n_non = np.sum(best_plans != 1, axis=1).astype(np.float64)
    counts = np.bincount(best_plans.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    row = {
        "k": int(k),
        "n_graphs": int(sub_s.shape[0]),
        "evals_per_graph": int(k),
        "T_greedy": float(np.mean(t_greedy)),
        "T_best_k": float(np.mean(t_best)),
        "T_mean_k": float(np.mean(t_mean)),
        "T_p10_k": float(np.mean(t_p10)),
        "T_best_p50": float(np.median(t_best)),
        "T_best_p10": float(np.percentile(t_best, 10)),
        "frac_best_equals_greedy": float(np.mean(best_i == 0)),
        "greedy_local_frac": float(counts[0]),
        "greedy_mec_frac": float(counts[1]),
        "greedy_v2v_frac": float(counts[2]),
        "n_nonmec_p50": float(np.median(n_non)),
        "n_nonmec_mean": float(np.mean(n_non)),
        "duplicates_ratio": duplicates_ratio(sub_p),
        "monotonic_ok": True,
    }
    if wall_s is not None:
        row["wall_clock_s"] = float(wall_s)
        row["seconds_per_graph"] = float(wall_s) / max(float(sub_s.shape[0]), 1.0)
    if expert_T is not None:
        expert_T = np.asarray(expert_T, dtype=np.float64)[: sub_s.shape[0]]
        row["expert_T_mean"] = float(np.mean(expert_T))
        row["oracle_within_samples"] = float(np.mean(np.min(sub_s, axis=1) + 1e-12 < expert_T))
        row["frac_best_beats_expert"] = float(np.mean(t_best + 1e-12 < expert_T))
    if dist_ids is not None:
        dist_ids = np.asarray(dist_ids)[: sub_s.shape[0]]
        per = {}
        for did in sorted(set(int(x) for x in dist_ids.tolist())):
            m = dist_ids == did
            per[str(did)] = {
                "n_graphs": int(np.sum(m)),
                "T_greedy": float(np.mean(t_greedy[m])),
                "T_best_k": float(np.mean(t_best[m])),
                "T_p10_k": float(np.mean(t_p10[m])),
            }
            if expert_T is not None:
                per[str(did)]["expert_T_mean"] = float(np.mean(expert_T[m]))
                per[str(did)]["oracle_within_samples"] = float(
                    np.mean(np.min(sub_s[m], axis=1) + 1e-12 < expert_T[m])
                )
        row["per_distribution"] = per
    return row


def assert_monotone_k(scores, ks=K_SWEEP):
    prev = None
    means = []
    for k in ks:
        if k > scores.shape[1]:
            continue
        m = float(np.mean(np.min(prefix_scores(scores, k), axis=1)))
        means.append((int(k), m))
        if prev is not None and m > prev + 1e-12:
            raise ValueError(
                "T_best_k not monotone: k means %s (RNG not nested?)" % means
            )
        prev = m
    return means


def collect_split(env, max_graphs=None):
    tgs = []
    obs_rows = []
    dist_ids = []
    dist_id_list = list(getattr(env, "distribution_ids", []) or [])
    for dist_i, graphs in enumerate(env.task_graphs_batchs):
        enc = np.asarray(env.encoder_batchs[dist_i])
        dist_id = int(dist_id_list[dist_i]) if dist_i < len(dist_id_list) else dist_i
        for g_i, tg in enumerate(graphs):
            tgs.append(tg)
            obs_rows.append(np.asarray(enc[g_i], dtype=np.float32))
            dist_ids.append(dist_id)
            if max_graphs is not None and len(tgs) >= int(max_graphs):
                return tgs, np.stack(obs_rows, axis=0), np.asarray(dist_ids, dtype=np.int32)
    return tgs, np.stack(obs_rows, axis=0), np.asarray(dist_ids, dtype=np.int32)


def load_expert_T(cache_path, n):
    blob = np.load(str(cache_path), allow_pickle=False)
    t = np.asarray(blob["t_expert"], dtype=np.float64)
    if t.shape[0] < n:
        raise ValueError("expert cache n=%d < %d at %s" % (t.shape[0], n, cache_path))
    return t[:n]


def crosscheck_twopt_score(graphs, plans, scores, resources, rng, n_check=3):
    from spec.hamming2_probe import _score

    n = len(graphs)
    n_check = min(int(n_check), n)
    idxs = rng.choice(n, size=n_check, replace=False)
    for i in idxs:
        best_j = int(np.argmin(scores[i]))
        plan = plans[i, best_j]
        order_n = int(graphs[i].task_number)
        got = float(_score(graphs[i], resources, plan[:order_n].tolist()))
        want = float(scores[i, best_j])
        if abs(got - want) > 1e-6:
            raise ValueError(
                "twopt _score mismatch graph=%d got=%.9f want=%.9f" % (int(i), got, want)
            )
    return [int(x) for x in idxs]


def eval_best_of_k(
    sess,
    policy,
    env,
    resources,
    k_max,
    temperature=DEFAULT_TEMPERATURE,
    top_p=None,
    max_graphs=None,
    expert_cache=None,
    ks=K_SWEEP,
    seed=0,
    split_name="",
):
    tgs, obs, dist_ids = collect_split(env, max_graphs=max_graphs)
    reach = None
    if getattr(policy, "reachability_mask", None) is not None:
        reach = collect_reachability(env)
        if max_graphs is not None:
            reach = reach[: len(tgs)]
    t0 = time.time()
    plans = sample_plans(
        sess, policy, obs, k=int(k_max), temperature=temperature, top_p=top_p, reach=reach
    )
    sample_s = time.time() - t0
    if int(plans.max()) > 2:
        raise ValueError("end_token leaked: max token %s" % int(plans.max()))
    if plans.shape[-1] != int(obs.shape[1]):
        raise ValueError("plan length %s != obs %s" % (plans.shape, obs.shape))
    t1 = time.time()
    refs = {}
    scores = score_plans(tgs, plans, resources, refs_cache=refs, cache_prefix=split_name)
    score_s = time.time() - t1
    wall_s = sample_s + score_s
    expert_T = None
    if expert_cache is not None:
        expert_T = load_expert_T(expert_cache, len(tgs))
    assert_monotone_k(scores, ks=tuple(k for k in ks if k <= int(k_max)))
    greedy_T = float(np.mean(scores[:, 0]))
    best1 = float(np.mean(np.min(prefix_scores(scores, 1), axis=1)))
    if abs(best1 - greedy_T) > 0.1:
        raise ValueError("T_best_1=%.4f != T_greedy=%.4f (sample0 != greedy)" % (best1, greedy_T))
    rng = np.random.RandomState(int(seed) + 17)
    checked = crosscheck_twopt_score(tgs, plans, scores, resources, rng)
    rows = {}
    for k in ks:
        if k > int(k_max):
            continue
        rows[str(k)] = summarize_k(
            plans, scores, k, expert_T=expert_T, dist_ids=dist_ids, wall_s=wall_s * (float(k) / float(k_max))
        )
    return {
        "split": split_name,
        "n_graphs": int(len(tgs)),
        "k_max": int(k_max),
        "temperature": float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "T_greedy": greedy_T,
        "T_best_1": best1,
        "sample_s": float(sample_s),
        "score_s": float(score_s),
        "wall_clock_s": float(wall_s),
        "seconds_per_graph": float(wall_s) / max(float(len(tgs)), 1.0),
        "crosscheck_graphs": checked,
        "by_k": rows,
        "evals_per_graph": int(k_max),
    }


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
