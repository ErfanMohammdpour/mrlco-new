"""STATUS: baseline/ablation only (ADR-007)

ΔT pair-ranker diagnostic on frozen Graph2Seq encoder. No PPO. Not 3500.

Job of the head: rank which motif pairs to search, not predict the joint.
Labels: one-shot gain T_BC - min_9 schedule(pair joint) from the BC plan.
Compare at k=10/20: CE (1-P_BC) vs learned ΔT vs true ΔT (label ceiling).
LSTM decoder unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.pair_head import (
    N_JOINT,
    PAIR_TOP_K,
    _apply_joint,
    _pack_and_fit,
    _score,
    joint_id,
    predict_joint,
    refine_split_scores,
)
from spec.phase4_campaign import SEEDS

PAIR_RANKER_K = (10, 20)
RANKER_LAM = 1.0
RANKER_POS_WEIGHT = 10.0
GAIN_EPS = 1e-9
RANKER_HELP_SEC = 8.0
RANKER_FIT_SEC = 8.0
ONE_PASS_FAIL_SEC = 12.0


def _concat_row_field(rows, key):
    chunks = [np.asarray(r[key], dtype=np.float64) for r in rows if len(r["pairs"])]
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return float("nan")
    return float(a.dot(b) / den)


def recall_at_k(gains, scores, k=PAIR_TOP_K, eps=GAIN_EPS):
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if gains.size != scores.size:
        raise ValueError("recall size mismatch")
    need = np.where(gains > float(eps))[0]
    if need.size == 0:
        return float("nan")
    top = np.argsort(-scores)[: int(k)]
    return float(np.sum(np.isin(need, top))) / float(need.size)


def recall_at_k_graph_mean(rows, scores_concat, k=PAIR_TOP_K, eps=GAIN_EPS):
    scores_concat = np.asarray(scores_concat, dtype=np.float64).reshape(-1)
    recs = []
    off = 0
    for r in rows:
        n = len(r["pairs"])
        if n == 0:
            continue
        g = np.asarray(r["pair_gain"], dtype=np.float64)
        sc = scores_concat[off : off + n]
        off += n
        need = np.where(g > float(eps))[0]
        if need.size == 0:
            continue
        top = np.argsort(-sc)[: int(k)]
        recs.append(float(np.sum(np.isin(need, top))) / float(need.size))
    if off != int(scores_concat.size):
        raise ValueError("recall consume %d != %d" % (off, scores_concat.size))
    if not recs:
        return float("nan")
    return float(np.mean(recs))


def train_weighted_ridge(x, y, lam=RANKER_LAM, pos_weight=RANKER_POS_WEIGHT):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("ridge x/y shape")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    xs = (x - mean) / std
    n, d = xs.shape
    xb = np.concatenate([xs, np.ones((n, 1))], axis=1)
    wrow = np.where(y > GAIN_EPS, float(pos_weight), 1.0)
    xtw = xb.T * wrow
    a = xtw.dot(xb)
    a[np.arange(d), np.arange(d)] += float(lam)
    beta = np.linalg.solve(a, xtw.dot(y))
    return {
        "mean": mean,
        "std": std,
        "beta": beta,
        "lam": float(lam),
        "pos_weight": float(pos_weight),
        "n": int(n),
        "n_pos": int(np.sum(y > GAIN_EPS)),
    }


def predict_ridge(model, x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
    return xb.dot(model["beta"])


def classify_ranker_verdict(ce_k20, learned_k20, gain_k20):
    ce_k20 = float(ce_k20)
    learned_k20 = float(learned_k20)
    gain_k20 = float(gain_k20)
    if (ce_k20 - gain_k20) < ONE_PASS_FAIL_SEC:
        return "one_pass_gain_rank_insufficient"
    if (learned_k20 - gain_k20) <= RANKER_FIT_SEC and (ce_k20 - learned_k20) >= RANKER_HELP_SEC:
        return "delta_ranker_fits"
    if (ce_k20 - learned_k20) >= RANKER_HELP_SEC:
        return "delta_ranker_helps"
    if (learned_k20 - gain_k20) <= RANKER_FIT_SEC:
        return "ranker_fits_labels_ce_already_ok"
    return "delta_labels_not_enough"


def _one_pair_gain(tg, resources, plan, rec, t0):
    i, j = int(rec["i"]), int(rec["j"])
    cur = joint_id(plan[i], plan[j])
    best = float(t0)
    n_eval = 0
    for jid in range(N_JOINT):
        if jid == cur:
            continue
        tt = _score(tg, resources, _apply_joint(plan, i, j, jid))
        n_eval += 1
        if tt + 1e-12 < best:
            best = tt
    return max(0.0, float(t0) - best), n_eval


def attach_pair_gains(rows, resources, tag=""):
    n_eval = 0
    n_pos = 0
    n_pairs = 0
    for i, r in enumerate(rows):
        t0 = _score(r["tg"], resources, r["bc"])
        r["t_bc"] = t0
        gains = []
        for rec in r["pairs"]:
            g, ne = _one_pair_gain(r["tg"], resources, r["bc"], rec, t0)
            gains.append(g)
            n_eval += ne
            n_pairs += 1
            if g > GAIN_EPS:
                n_pos += 1
        r["pair_gain"] = np.asarray(gains, dtype=np.float64)
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(
                "pair_gain_progress split=%s %d/%d eval=%d n_pos=%d"
                % (tag, i + 1, len(rows), n_eval, n_pos)
            )
    return {
        "n_graphs": int(len(rows)),
        "n_pairs": int(n_pairs),
        "n_pos": int(n_pos),
        "frac_pos": float(n_pos) / float(max(n_pairs, 1)),
        "n_eval": int(n_eval),
        "mean_gain_pos": float(
            np.mean(
                [
                    float(g)
                    for r in rows
                    for g in np.asarray(r["pair_gain"], dtype=np.float64)
                    if g > GAIN_EPS
                ]
            )
        )
        if n_pos
        else 0.0,
    }


def _rank_metrics(rows, scores, k=PAIR_TOP_K):
    gains = _concat_row_field(rows, "pair_gain")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "pearson_vs_gain": pearson_corr(gains, scores),
        "recall_at_%d" % int(k): recall_at_k(gains, scores, k=k),
        "recall_at_%d_graph_mean" % int(k): recall_at_k_graph_mean(rows, scores, k=k),
        "n_pos": int(np.sum(gains > GAIN_EPS)),
        "n_pairs": int(gains.size),
    }


def _ce_scores(packed_split, model):
    _pred, prob = predict_joint(model, packed_split["x"])
    rows = packed_split["rows"]
    chunks = []
    off = 0
    for r in rows:
        n = len(r["pairs"])
        if n == 0:
            continue
        sl = slice(off, off + n)
        chunks.append(1.0 - prob[sl][np.arange(n), r["pair_bc"]])
        off += n
    if off != int(prob.shape[0]):
        raise ValueError("ce score consume %d != %d" % (off, prob.shape[0]))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def run_pair_ranker(
    sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed, run_dir
):
    """CE vs learned ΔT vs true-gain ranking at k=10/20. No PPO. Decoder unchanged."""
    if tuple(int(x) for x in PAIR_RANKER_K) != (10, 20):
        raise ValueError("PAIR_RANKER_K must be (10, 20), got %s" % (PAIR_RANKER_K,))
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    packed, ce_model, _rng = _pack_and_fit(
        sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed
    )
    resources = train_env.scheduler_resources
    gain_stats = {}
    for name in ("meta_train", "validation", "meta_test"):
        gain_stats[name] = attach_pair_gains(packed[name]["rows"], resources, tag=name)
    y_train = _concat_row_field(packed["meta_train"]["rows"], "pair_gain")
    ridge = train_weighted_ridge(packed["meta_train"]["x"], y_train)
    print(
        "pair_ridge n=%d n_pos=%d last_corr_train=%.3f"
        % (ridge["n"], ridge["n_pos"], pearson_corr(y_train, predict_ridge(ridge, packed["meta_train"]["x"])))
    )
    methods = {}
    for name in ("validation", "meta_test"):
        rows = packed[name]["rows"]
        ce_sc = _ce_scores(packed[name], ce_model)
        learned_sc = predict_ridge(ridge, packed[name]["x"])
        gain_sc = _concat_row_field(rows, "pair_gain")
        methods[name] = {
            "ce": ce_sc,
            "learned": learned_sc,
            "gain_oracle": gain_sc,
        }
    rank_metrics = {}
    refine = {}
    for method in ("ce", "learned", "gain_oracle"):
        rank_metrics[method] = {}
        refine[method] = {}
        for name in ("validation", "meta_test"):
            sc = methods[name][method]
            rank_metrics[method][name] = _rank_metrics(
                packed[name]["rows"], sc, k=PAIR_TOP_K
            )
            print(
                "pair_rank_metrics method=%s split=%s pearson=%.3f recall@%d=%.3f n_pos=%d"
                % (
                    method,
                    name,
                    rank_metrics[method][name]["pearson_vs_gain"],
                    PAIR_TOP_K,
                    rank_metrics[method][name]["recall_at_%d" % PAIR_TOP_K],
                    rank_metrics[method][name]["n_pos"],
                )
            )
            refine[method][name] = {}
            for k in PAIR_RANKER_K:
                refine[method][name][str(int(k))] = refine_split_scores(
                    packed[name]["rows"],
                    sc,
                    resources,
                    top_k=int(k),
                    tag="%s_%s_k%d" % (method, name, int(k)),
                )
    val_ce = refine["ce"]["validation"]["20"]["pair_T_mean"]
    val_learned = refine["learned"]["validation"]["20"]["pair_T_mean"]
    val_gain = refine["gain_oracle"]["validation"]["20"]["pair_T_mean"]
    verdict = classify_ranker_verdict(val_ce, val_learned, val_gain)
    payload = {
        "paper_result": False,
        "ppo": False,
        "encoder_frozen": True,
        "decoder_replaced": False,
        "k_sweep": list(PAIR_RANKER_K),
        "n_joint": N_JOINT,
        "top_k": PAIR_TOP_K,
        "ridge": {
            "lam": ridge["lam"],
            "pos_weight": ridge["pos_weight"],
            "n": ridge["n"],
            "n_pos": ridge["n_pos"],
        },
        "gain_stats": gain_stats,
        "rank_metrics": rank_metrics,
        "refine": refine,
        "verdict": verdict,
        "softmax_last_loss": ce_model.get("last_loss"),
        "note": "ΔT ridge ranker vs CE 1-P_BC vs true one-shot gain ranking; LSTM decoder unchanged; no PPO; not 3500",
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ranker_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "pair_ranker verdict=%s val_bc=%.1f ce_k20=%.1f learned_k20=%.1f gain_k20=%.1f test_learned_k20=%.1f"
        % (
            verdict,
            refine["learned"]["validation"]["20"]["bc_T_mean"],
            val_ce,
            val_learned,
            val_gain,
            refine["learned"]["meta_test"]["20"]["pair_T_mean"],
        )
    )
    return payload
