"""STATUS: baseline/ablation only (ADR-007)

Pair/motif diagnostic on frozen Graph2Seq encoder. No PPO. Not the frozen 3500 primary.

Auxiliary 9-way joint head. Does not replace the LSTM decoder.
Pairs: direct DAG edge, sibling/fork, join. Labels from 2-opt expert.
Then top-K pairs x 9 joints with real schedule() from the BC plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_BATCH, align_greedy_pred
from spec.phase4_campaign import SEEDS

N_TOK = 20
N_ACT = 3
N_JOINT = 9
PAIR_TOP_K = 20
PAIR_K_SWEEP = (10, 20, 50)
PAIR_EPOCHS = 40
PAIR_BATCH = 256
PAIR_LR = 0.05
ORACLE_N_VAL = 100
ORACLE_MAX_ROUNDS = 8
DECODER_OK_T = 500.0
DECODER_OK_DELTA = 80.0
ENCODER_FAIL_DELTA = 20.0


def joint_id(ai, aj):
    return int(ai) * N_ACT + int(aj)


def split_joint(jid):
    jid = int(jid)
    return jid // N_ACT, jid % N_ACT


def decoder_index_of(order):
    order = [int(tid) for tid in order]
    n = len(order)
    dec = [-1] * n
    for k, tid in enumerate(order):
        if tid < 0 or tid >= n or dec[tid] != -1:
            raise ValueError("bad prioritize_sequence")
        dec[tid] = k
    if any(x < 0 for x in dec):
        raise ValueError("prioritize_sequence not a permutation")
    return dec


def motif_pairs(succ_sets, pre_sets, order):
    """Decoder-index pairs. A pair may be sibling and join at once (diamond)."""
    dec_of = decoder_index_of(order)
    pairs = {}

    def add(tid_a, tid_b, kind):
        da = dec_of[int(tid_a)]
        db = dec_of[int(tid_b)]
        if da == db:
            return
        lo, hi = (da, db) if da < db else (db, da)
        rec = pairs.setdefault(
            (lo, hi),
            {"i": lo, "j": hi, "direct": 0, "sibling": 0, "join": 0},
        )
        rec[kind] = 1

    for u, succs in enumerate(succ_sets):
        children = sorted(int(x) for x in succs)
        for v in children:
            add(u, v, "direct")
        for a in range(len(children)):
            for b in range(a + 1, len(children)):
                add(children[a], children[b], "sibling")
    for _v, pres in enumerate(pre_sets):
        parents = sorted(int(x) for x in pres)
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                add(parents[a], parents[b], "join")
    return list(pairs.values())


def pair_feat(h, z, rec, n_tok=N_TOK):
    i = int(rec["i"])
    j = int(rec["j"])
    hi = h[i]
    hj = h[j]
    extra = np.asarray(
        [
            float(rec["direct"]),
            float(rec["sibling"]),
            float(rec["join"]),
            float(j - i) / float(max(int(n_tok), 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hi, hj, hi * hj, hi - hj, z, extra], axis=0)


def softmax_rows(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def train_joint_softmax(x, y, rng, n_class=N_JOINT, epochs=PAIR_EPOCHS, lr=PAIR_LR, batch=PAIR_BATCH):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    n, d = x.shape
    counts = np.bincount(y, minlength=n_class).astype(np.float64)
    cw = 1.0 / np.maximum(counts, 1.0)
    cw = cw / np.mean(cw)
    w = rng.randn(d, n_class) * 0.01
    b = np.zeros(n_class, dtype=np.float64)
    order = np.arange(n)
    last = None
    for _epoch in range(int(epochs)):
        rng.shuffle(order)
        losses = []
        for start in range(0, n, int(batch)):
            sl = order[start : start + int(batch)]
            xb = x[sl]
            yb = y[sl]
            logits = xb.dot(w) + b
            p = softmax_rows(logits)
            wt = cw[yb]
            losses.append(float(np.mean(-wt * np.log(np.maximum(p[np.arange(len(yb)), yb], 1e-12)))))
            p[np.arange(len(yb)), yb] -= 1.0
            p *= wt[:, None]
            p /= max(float(len(yb)), 1.0)
            w -= float(lr) * xb.T.dot(p)
            b -= float(lr) * p.sum(axis=0)
        last = float(np.mean(losses)) if losses else None
    return {"w": w, "b": b, "class_weight": cw, "n": int(n), "last_loss": last}


def predict_joint(model, x):
    x = np.asarray(x, dtype=np.float64)
    p = softmax_rows(x.dot(model["w"]) + model["b"])
    pred = np.argmax(p, axis=1).astype(np.int32)
    return pred, p


def classify_pair_verdict(bc_t, refine_t, oracle_t=None):
    bc_t = float(bc_t)
    refine_t = float(refine_t)
    delta = bc_t - refine_t
    oracle_delta = None if oracle_t is None else (bc_t - float(oracle_t))
    if oracle_delta is not None and oracle_delta >= DECODER_OK_DELTA and delta < ENCODER_FAIL_DELTA:
        return "pair_search_works_ranker_fails"
    if refine_t <= DECODER_OK_T or delta >= DECODER_OK_DELTA:
        return "decoder_interaction"
    if delta < ENCODER_FAIL_DELTA:
        return "encoder_insufficient"
    return "mixed_gain"


def _score(tg, resources, decoder_acts):
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    order = [int(tid) for tid in tg.prioritize_sequence]
    acts = [int(a) for a in decoder_acts]
    if len(acts) != len(order):
        raise ValueError("plan length mismatch")
    result, _, _ = schedule_via_adapter(tg, list(zip(order, acts)), resources)
    return float(result.makespan_seconds)


def _apply_joint(plan, i, j, jid):
    out = list(plan)
    ai, aj = split_joint(jid)
    out[int(i)] = int(ai)
    out[int(j)] = int(aj)
    return out


def refine_topk(tg, resources, plan, pairs, scores, t0, top_k=PAIR_TOP_K):
    """One pass over top-K pairs. Each pair tries 9 joints. Accept if T drops."""
    plan = [int(a) for a in plan]
    t = float(t0)
    n_eval = 0
    n_accept = 0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    for rank, pi in enumerate(order):
        if rank >= int(top_k):
            break
        rec = pairs[int(pi)]
        i, j = int(rec["i"]), int(rec["j"])
        best_t = t
        best_plan = plan
        for jid in range(N_JOINT):
            trial = _apply_joint(plan, i, j, jid)
            tt = _score(tg, resources, trial)
            n_eval += 1
            if tt + 1e-12 < best_t:
                best_t = tt
                best_plan = trial
        if best_t + 1e-12 < t:
            plan = best_plan
            t = best_t
            n_accept += 1
    return t, plan, n_eval, n_accept


def oracle_motif_search(tg, resources, plan, pairs, t0, max_rounds=ORACLE_MAX_ROUNDS):
    """Best improving motif-pair joint each round. Ceiling if ranking were perfect."""
    plan = [int(a) for a in plan]
    t = float(t0)
    n_eval = 0
    n_accept = 0
    for _ in range(int(max_rounds)):
        best_t = t
        best_plan = plan
        for rec in pairs:
            i, j = int(rec["i"]), int(rec["j"])
            for jid in range(N_JOINT):
                trial = _apply_joint(plan, i, j, jid)
                tt = _score(tg, resources, trial)
                n_eval += 1
                if tt + 1e-12 < best_t:
                    best_t = tt
                    best_plan = trial
        if best_t + 1e-12 < t:
            plan = best_plan
            t = best_t
            n_accept += 1
        else:
            break
    return t, plan, n_eval, n_accept


def _iter_graphs(env):
    for graphs in env.task_graphs_batchs:
        for tg in graphs:
            yield tg


def extract_encoder_and_greedy(sess, env, core_policy, obs, expert_acts):
    n, n_tok = expert_acts.shape
    if n_tok != N_TOK:
        raise ValueError("n_tok %s != %d" % (n_tok, N_TOK))
    hs = []
    preds = []
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        h, greedy = sess.run(
            [
                core_policy.network.encoder_outputs,
                core_policy.network.greedy_decoder_prediction,
            ],
            feed_dict={
                core_policy.obs: obs[sl],
                core_policy.decoder_full_length: fl,
            },
        )
        aligned, _ = align_greedy_pred(greedy, n_tok)
        hs.append(np.asarray(h, dtype=np.float32))
        preds.append(aligned)
    h_all = np.concatenate(hs, axis=0)
    pred = np.concatenate(preds, axis=0)
    if h_all.shape[0] != n or pred.shape[0] != n:
        raise ValueError("extract size mismatch")
    return h_all, pred


def _build_split_rows(env, h_all, bc_acts, expert_acts, extras):
    rows = []
    t_ex = np.asarray(extras["t_expert"], dtype=np.float64)
    t_mec = np.asarray(extras["t_mec"], dtype=np.float64)
    dist = extras.get("dist_id")
    graphs = list(_iter_graphs(env))
    if len(graphs) != h_all.shape[0]:
        raise ValueError("graph count %d != embed %d" % (len(graphs), h_all.shape[0]))
    x_rows = []
    y_rows = []
    meta = []
    for gi, tg in enumerate(graphs):
        n = int(tg.task_number)
        if n != N_TOK:
            raise ValueError("task_number %s != %d" % (n, N_TOK))
        order = [int(tid) for tid in tg.prioritize_sequence]
        pairs = motif_pairs(tg.succ_task_sets, tg.pre_task_sets, order)
        h = h_all[gi]
        z = h.mean(axis=0)
        bc = [int(a) for a in bc_acts[gi]]
        ex = [int(a) for a in expert_acts[gi]]
        pair_x = []
        pair_y = []
        pair_bc = []
        for rec in pairs:
            i, j = int(rec["i"]), int(rec["j"])
            feat = pair_feat(h, z, rec, n_tok=n)
            pair_x.append(feat)
            pair_y.append(joint_id(ex[i], ex[j]))
            pair_bc.append(joint_id(bc[i], bc[j]))
            x_rows.append(feat)
            y_rows.append(joint_id(ex[i], ex[j]))
            meta.append((gi, rec, joint_id(bc[i], bc[j])))
        rows.append(
            {
                "tg": tg,
                "pairs": pairs,
                "bc": bc,
                "expert": ex,
                "t_expert": float(t_ex[gi]),
                "t_mec": float(t_mec[gi]),
                "dist_id": None if dist is None else int(np.asarray(dist)[gi]),
                "pair_x": np.asarray(pair_x, dtype=np.float32) if pair_x else np.zeros((0, h.shape[1] * 5 + 4), dtype=np.float32),
                "pair_y": np.asarray(pair_y, dtype=np.int32),
                "pair_bc": np.asarray(pair_bc, dtype=np.int32),
            }
        )
    x = np.asarray(x_rows, dtype=np.float32) if x_rows else np.zeros((0, 1), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    return rows, x, y, meta


def _pair_metrics(rows, pred, prob, split_name):
    y = np.concatenate([r["pair_y"] for r in rows if r["pair_y"].size], axis=0)
    bc = np.concatenate([r["pair_bc"] for r in rows if r["pair_bc"].size], axis=0)
    if y.shape[0] != pred.shape[0]:
        raise ValueError("metric size mismatch")
    acc = float(np.mean(pred == y)) if y.size else float("nan")
    disagree = y != bc
    n_dis = int(np.sum(disagree))
    acc_dis = float(np.mean(pred[disagree] == y[disagree])) if n_dis else float("nan")
    recalls = []
    n_hit = 0
    n_need = 0
    for r, sl_pred, sl_prob in _row_pred_slices(rows, pred, prob):
        need = np.where(r["pair_y"] != r["pair_bc"])[0]
        n_need += int(need.size)
        if r["pair_y"].size == 0:
            continue
        score = 1.0 - sl_prob[np.arange(sl_prob.shape[0]), r["pair_bc"]]
        top = np.argsort(-score)[:PAIR_TOP_K]
        hit = int(np.sum(np.isin(need, top)))
        n_hit += hit
        if need.size:
            recalls.append(float(hit) / float(need.size))
    return {
        "split": split_name,
        "n_pairs": int(y.size),
        "n_graphs": int(len(rows)),
        "joint_acc": acc,
        "n_expert_ne_bc": n_dis,
        "joint_acc_expert_ne_bc": acc_dis,
        "frac_expert_ne_bc": float(n_dis) / float(max(y.size, 1)),
        "recall_at_%d" % PAIR_TOP_K: float(n_hit) / float(max(n_need, 1)),
        "recall_at_%d_graph_mean" % PAIR_TOP_K: float(np.mean(recalls)) if recalls else float("nan"),
        "n_pairs_per_graph_mean": float(np.mean([len(r["pairs"]) for r in rows])),
    }


def _row_pred_slices(rows, pred, prob):
    off = 0
    for r in rows:
        n = int(r["pair_y"].size)
        sl = slice(off, off + n)
        yield r, pred[sl], prob[sl]
        off += n
    if off != pred.shape[0]:
        raise ValueError("slice consume %d != %d" % (off, pred.shape[0]))


def refine_split_scores(rows, scores_concat, resources, top_k=PAIR_TOP_K, tag=""):
    """One-pass top-k x 9 refine from explicit pair scores. LSTM decoder unchanged."""
    scores_concat = np.asarray(scores_concat, dtype=np.float64).reshape(-1)
    n_expected = int(sum(len(r["pairs"]) for r in rows))
    if int(scores_concat.size) != n_expected:
        raise ValueError("scores size %d != pairs %d" % (scores_concat.size, n_expected))
    t_bc = []
    t_ref = []
    n_evals = []
    n_accs = []
    off = 0
    for r in rows:
        n = len(r["pairs"])
        t0 = r["t_bc"] if "t_bc" in r else _score(r["tg"], resources, r["bc"])
        if n == 0:
            t_bc.append(t0)
            t_ref.append(t0)
            n_evals.append(0)
            n_accs.append(0)
            continue
        score = scores_concat[off : off + n]
        off += n
        t1, _, n_eval, n_accept = refine_topk(
            r["tg"], resources, r["bc"], r["pairs"], score, t0, top_k=top_k
        )
        t_bc.append(t0)
        t_ref.append(t1)
        n_evals.append(n_eval)
        n_accs.append(n_accept)
    if off != int(scores_concat.size):
        raise ValueError("score consume %d != %d" % (off, scores_concat.size))
    t_bc = np.asarray(t_bc, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    out = {
        "n_graphs": int(len(rows)),
        "bc_T_mean": float(np.mean(t_bc)),
        "pair_T_mean": float(np.mean(t_ref)),
        "expert_T_mean": float(np.mean([r["t_expert"] for r in rows])),
        "mec_T_mean": float(np.mean([r["t_mec"] for r in rows])),
        "mean_delta": float(np.mean(t_bc - t_ref)),
        "frac_improved": float(np.mean(t_ref + 1e-12 < t_bc)),
        "n_eval_mean": float(np.mean(n_evals)) if n_evals else 0.0,
        "n_accept_mean": float(np.mean(n_accs)) if n_accs else 0.0,
        "top_k": int(top_k),
    }
    print(
        "pair_refine split=%s n=%d bc_T=%.1f pair_T=%.1f ex=%.1f delta=%.1f improved=%.3f eval=%.1f"
        % (
            tag or "?",
            out["n_graphs"],
            out["bc_T_mean"],
            out["pair_T_mean"],
            out["expert_T_mean"],
            out["mean_delta"],
            out["frac_improved"],
            out["n_eval_mean"],
        )
    )
    return out


def _ce_disagree_scores(rows, pred, prob):
    chunks = []
    for r, _sl_pred, sl_prob in _row_pred_slices(rows, pred, prob):
        n = int(r["pair_y"].size)
        if n == 0:
            continue
        chunks.append(1.0 - sl_prob[np.arange(n), r["pair_bc"]])
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _refine_split(rows, pred, prob, resources, top_k=PAIR_TOP_K, tag=""):
    return refine_split_scores(
        rows, _ce_disagree_scores(rows, pred, prob), resources, top_k=top_k, tag=tag
    )


def _oracle_subset(rows, resources, n_keep, rng):
    n = len(rows)
    if n_keep >= n:
        idx = np.arange(n)
    else:
        idx = np.sort(rng.choice(n, size=int(n_keep), replace=False))
    t_bc = []
    t_or = []
    n_evals = []
    n_accs = []
    for gi in idx:
        r = rows[int(gi)]
        t0 = _score(r["tg"], resources, r["bc"])
        t1, _, n_eval, n_accept = oracle_motif_search(
            r["tg"], resources, r["bc"], r["pairs"], t0
        )
        t_bc.append(t0)
        t_or.append(t1)
        n_evals.append(n_eval)
        n_accs.append(n_accept)
        if len(t_bc) == 1 or len(t_bc) % 25 == 0 or len(t_bc) == len(idx):
            print("pair_oracle_progress %d/%d" % (len(t_bc), len(idx)))
    t_bc = np.asarray(t_bc, dtype=np.float64)
    t_or = np.asarray(t_or, dtype=np.float64)
    out = {
        "n_graphs": int(len(idx)),
        "bc_T_mean": float(np.mean(t_bc)),
        "oracle_T_mean": float(np.mean(t_or)),
        "mean_delta": float(np.mean(t_bc - t_or)),
        "frac_improved": float(np.mean(t_or + 1e-12 < t_bc)),
        "n_eval_mean": float(np.mean(n_evals)) if n_evals else 0.0,
        "n_accept_mean": float(np.mean(n_accs)) if n_accs else 0.0,
        "max_rounds": int(ORACLE_MAX_ROUNDS),
    }
    print(
        "pair_oracle n=%d bc_T=%.1f oracle_T=%.1f delta=%.1f improved=%.3f"
        % (out["n_graphs"], out["bc_T_mean"], out["oracle_T_mean"], out["mean_delta"], out["frac_improved"])
    )
    return out


def _pack_and_fit(sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed):
    from spec.bc_greedy_mec import collect_expert_dataset

    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    rng = np.random.RandomState(seed + 80)
    packed = {}
    for name, env, cache in (
        ("meta_train", train_env, train_cache),
        ("validation", val_env, val_cache),
        ("meta_test", test_env, test_cache),
    ):
        obs, acts, stats, extras = collect_expert_dataset(env, cache_path=cache)
        print("pair_extract split=%s n=%d expert_T=%.1f" % (name, acts.shape[0], stats["expert_T_mean"]))
        h_all, bc_pred = extract_encoder_and_greedy(sess, env, core_policy, obs, acts)
        rows, x, y, _meta = _build_split_rows(env, h_all, bc_pred, acts, extras)
        packed[name] = {"rows": rows, "x": x, "y": y, "bc_pred": bc_pred, "expert": acts, "stats": stats}
        print(
            "pair_built split=%s graphs=%d pairs=%d n_pairs_mean=%.1f"
            % (name, len(rows), int(y.size), float(np.mean([len(r["pairs"]) for r in rows])))
        )
    model = train_joint_softmax(packed["meta_train"]["x"], packed["meta_train"]["y"], rng)
    print("pair_softmax last_loss=%s n=%d" % (model.get("last_loss"), model["n"]))
    return packed, model, rng


def run_pair_head(sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed, run_dir):
    packed, model, rng = _pack_and_fit(
        sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed
    )
    resources = train_env.scheduler_resources
    metrics = {}
    refine = {}
    for name in ("meta_train", "validation", "meta_test"):
        pred, prob = predict_joint(model, packed[name]["x"])
        metrics[name] = _pair_metrics(packed[name]["rows"], pred, prob, name)
        print(
            "pair_metrics split=%s acc=%.3f acc_dis=%.3f recall@%d=%.3f n_dis=%d"
            % (
                name,
                metrics[name]["joint_acc"],
                metrics[name]["joint_acc_expert_ne_bc"],
                PAIR_TOP_K,
                metrics[name]["recall_at_%d" % PAIR_TOP_K],
                metrics[name]["n_expert_ne_bc"],
            )
        )
        if name != "meta_train":
            refine[name] = _refine_split(
                packed[name]["rows"], pred, prob, resources, top_k=PAIR_TOP_K, tag=name
            )
    oracle = _oracle_subset(packed["validation"]["rows"], resources, ORACLE_N_VAL, rng)
    val_bc = refine["validation"]["bc_T_mean"]
    val_ref = refine["validation"]["pair_T_mean"]
    verdict = classify_pair_verdict(val_bc, val_ref, oracle_t=oracle["oracle_T_mean"])
    payload = {
        "paper_result": False,
        "ppo": False,
        "encoder_frozen": True,
        "decoder_replaced": False,
        "n_joint": N_JOINT,
        "top_k": PAIR_TOP_K,
        "pair_kinds": ["direct", "sibling", "join"],
        "metrics": metrics,
        "refine": refine,
        "oracle_val": oracle,
        "verdict": verdict,
        "softmax_last_loss": model.get("last_loss"),
        "note": "auxiliary 9-way pair head on frozen encoder; LSTM decoder unchanged; no PPO; not 3500",
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pair_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "pair_verdict %s val_bc=%.1f val_pair=%.1f val_oracle=%.1f test_bc=%.1f test_pair=%.1f"
        % (
            verdict,
            val_bc,
            val_ref,
            oracle["oracle_T_mean"],
            refine["meta_test"]["bc_T_mean"],
            refine["meta_test"]["pair_T_mean"],
        )
    )
    return payload


def run_pair_ksweep(sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed, run_dir):
    """k=10/20/50 refine + oracle on the same val graphs. No PPO. Decoder unchanged."""
    if tuple(int(x) for x in PAIR_K_SWEEP) != (10, 20, 50):
        raise ValueError("PAIR_K_SWEEP must be (10, 20, 50), got %s" % (PAIR_K_SWEEP,))
    packed, model, rng = _pack_and_fit(
        sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed
    )
    resources = train_env.scheduler_resources
    preds = {}
    for name in ("validation", "meta_test"):
        pred, prob = predict_joint(model, packed[name]["x"])
        preds[name] = (pred, prob)
    curve = {}
    for k in PAIR_K_SWEEP:
        curve[str(int(k))] = {}
        for name in ("validation", "meta_test"):
            pred, prob = preds[name]
            row = _refine_split(
                packed[name]["rows"], pred, prob, resources, top_k=int(k), tag="%s_k%d" % (name, int(k))
            )
            curve[str(int(k))][name] = row
    print("pair_oracle_fair n=%d (same val graphs as k-sweep)" % len(packed["validation"]["rows"]))
    oracle = _oracle_subset(
        packed["validation"]["rows"], resources, len(packed["validation"]["rows"]), rng
    )
    k20 = curve["20"]["validation"]
    payload = {
        "paper_result": False,
        "ppo": False,
        "encoder_frozen": True,
        "decoder_replaced": False,
        "k_sweep": list(PAIR_K_SWEEP),
        "n_joint": N_JOINT,
        "curve": curve,
        "oracle_val": oracle,
        "softmax_last_loss": model.get("last_loss"),
        "note": "k=10/20/50 vs motif oracle on the same validation graphs; LSTM decoder unchanged; no PPO; not 3500",
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ksweep_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "pair_ksweep val_bc=%.1f k10=%.1f k20=%.1f k50=%.1f oracle=%.1f eval_k10=%.0f eval_k20=%.0f eval_k50=%.0f eval_oracle=%.0f"
        % (
            k20["bc_T_mean"],
            curve["10"]["validation"]["pair_T_mean"],
            k20["pair_T_mean"],
            curve["50"]["validation"]["pair_T_mean"],
            oracle["oracle_T_mean"],
            curve["10"]["validation"]["n_eval_mean"],
            k20["n_eval_mean"],
            curve["50"]["validation"]["n_eval_mean"],
            oracle["n_eval_mean"],
        )
    )
    return payload
