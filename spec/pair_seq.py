"""STATUS: baseline/ablation only (ADR-007)

Sequential pair refine diagnostic. Frozen encoder. LSTM unchanged. No PPO. Not 3500.

Question: the 21s CE-k20 vs motif-oracle gap — is it scan-vs-best-improvement
on the same CE top-20, or pairs outside that set after accepts?

Methods on CE scores:
  scan        one-pass ranked accept (paper k=20 control)
  multipass   repeat scan until no accept, max 4
  bestimp     each round take the single best joint among CE top-k, max 8
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.pair_head import (
    N_JOINT,
    ORACLE_MAX_ROUNDS,
    PAIR_TOP_K,
    _apply_joint,
    _ce_disagree_scores,
    _pack_and_fit,
    _score,
    predict_joint,
    refine_topk,
)
from spec.phase4_campaign import SEEDS

SEQ_SCAN_PASSES = 4
SEQ_BESTIMP_K = (20, 50)
SEQ_ORACLE_REF_VAL = 436.6
SEQ_CLOSE_SEC = 8.0
SEQ_HELP_SEC = 8.0


def refine_bestimp(tg, resources, plan, pairs, scores, t0, top_k, max_rounds=ORACLE_MAX_ROUNDS):
    """Best improving joint among CE top-k each round. Candidate set frozen."""
    plan = [int(a) for a in plan]
    t = float(t0)
    n_eval = 0
    n_accept = 0
    n_round = 0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    cand = []
    for rank, pi in enumerate(order):
        if rank >= int(top_k):
            break
        cand.append(pairs[int(pi)])
    for _ in range(int(max_rounds)):
        n_round += 1
        best_t = t
        best_plan = plan
        for rec in cand:
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
    return t, plan, n_eval, n_accept, n_round


def refine_multipass(tg, resources, plan, pairs, scores, t0, top_k, n_pass=SEQ_SCAN_PASSES):
    """Repeat one-pass scan on the same ranked list from the updated plan."""
    plan = [int(a) for a in plan]
    t = float(t0)
    n_eval = 0
    n_accept = 0
    n_used = 0
    for _ in range(int(n_pass)):
        t2, plan, ne, na = refine_topk(
            tg, resources, plan, pairs, scores, t, top_k=top_k
        )
        n_eval += ne
        n_accept += na
        n_used += 1
        t = t2
        if na == 0:
            break
    return t, plan, n_eval, n_accept, n_used


def classify_seq_verdict(scan_k20, multi_k20, best_k20, oracle_ref=SEQ_ORACLE_REF_VAL):
    scan_k20 = float(scan_k20)
    multi_k20 = float(multi_k20)
    best_k20 = float(best_k20)
    oracle_ref = float(oracle_ref)
    if best_k20 <= oracle_ref + SEQ_CLOSE_SEC:
        return "seq_top20_closes_oracle_gap"
    if (scan_k20 - best_k20) >= SEQ_HELP_SEC:
        return "seq_helps_need_more_pairs"
    if (scan_k20 - multi_k20) >= SEQ_HELP_SEC:
        return "multipass_helps"
    return "seq_on_top20_not_enough"


def _row_score_slices(rows, scores_concat):
    scores_concat = np.asarray(scores_concat, dtype=np.float64).reshape(-1)
    n_expected = int(sum(len(r["pairs"]) for r in rows))
    if int(scores_concat.size) != n_expected:
        raise ValueError("scores size %d != pairs %d" % (scores_concat.size, n_expected))
    off = 0
    for r in rows:
        n = len(r["pairs"])
        yield r, scores_concat[off : off + n]
        off += n
    if off != int(scores_concat.size):
        raise ValueError("score consume %d != %d" % (off, scores_concat.size))


def _summarize(rows, t_bc, t_ref, n_evals, n_accs, extra, top_k, tag):
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
    out.update(extra)
    print(
        "pair_seq split=%s n=%d bc_T=%.1f pair_T=%.1f ex=%.1f delta=%.1f improved=%.3f eval=%.1f acc=%.2f"
        % (
            tag,
            out["n_graphs"],
            out["bc_T_mean"],
            out["pair_T_mean"],
            out["expert_T_mean"],
            out["mean_delta"],
            out["frac_improved"],
            out["n_eval_mean"],
            out["n_accept_mean"],
        )
    )
    return out


def _refine_mode(rows, scores_concat, resources, mode, top_k, tag):
    t_bc = []
    t_ref = []
    n_evals = []
    n_accs = []
    n_extra = []
    for gi, (r, sc) in enumerate(_row_score_slices(rows, scores_concat)):
        t0 = r["t_bc"] if "t_bc" in r else _score(r["tg"], resources, r["bc"])
        r["t_bc"] = t0
        extra = 0
        if len(r["pairs"]) == 0:
            t1, n_eval, n_accept = t0, 0, 0
        elif mode == "scan":
            t1, _, n_eval, n_accept = refine_topk(
                r["tg"], resources, r["bc"], r["pairs"], sc, t0, top_k=top_k
            )
        elif mode == "multipass":
            t1, _, n_eval, n_accept, extra = refine_multipass(
                r["tg"], resources, r["bc"], r["pairs"], sc, t0, top_k=top_k
            )
        elif mode == "bestimp":
            t1, _, n_eval, n_accept, extra = refine_bestimp(
                r["tg"], resources, r["bc"], r["pairs"], sc, t0, top_k=top_k
            )
        else:
            raise ValueError("unknown seq mode %s" % mode)
        t_bc.append(t0)
        t_ref.append(t1)
        n_evals.append(n_eval)
        n_accs.append(n_accept)
        n_extra.append(extra)
        if gi == 0 or (gi + 1) % 50 == 0 or gi + 1 == len(rows):
            print("pair_seq_progress split=%s %d/%d" % (tag, gi + 1, len(rows)))
    extra_out = {}
    if mode == "multipass":
        extra_out["n_pass_mean"] = float(np.mean(n_extra)) if n_extra else 0.0
        extra_out["n_pass_max"] = SEQ_SCAN_PASSES
    if mode == "bestimp":
        extra_out["n_round_mean"] = float(np.mean(n_extra)) if n_extra else 0.0
        extra_out["max_rounds"] = int(ORACLE_MAX_ROUNDS)
    return _summarize(rows, t_bc, t_ref, n_evals, n_accs, extra_out, top_k, tag)


def run_pair_seq(
    sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed, run_dir
):
    """Scan vs multipass vs bestimp on frozen CE ranking. No PPO. Decoder unchanged."""
    if int(PAIR_TOP_K) != 20:
        raise ValueError("PAIR_TOP_K must be 20, got %s" % PAIR_TOP_K)
    if tuple(int(x) for x in SEQ_BESTIMP_K) != (20, 50):
        raise ValueError("SEQ_BESTIMP_K must be (20, 50), got %s" % (SEQ_BESTIMP_K,))
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    packed, ce_model, _rng = _pack_and_fit(
        sess, train_env, val_env, test_env, core_policy, train_cache, val_cache, test_cache, seed
    )
    resources = train_env.scheduler_resources
    methods = {}
    for name in ("validation", "meta_test"):
        pred, prob = predict_joint(ce_model, packed[name]["x"])
        scores = _ce_disagree_scores(packed[name]["rows"], pred, prob)
        rows = packed[name]["rows"]
        block = {}
        block["scan_k20"] = _refine_mode(rows, scores, resources, "scan", 20, "%s_scan_k20" % name)
        block["multipass_k20"] = _refine_mode(
            rows, scores, resources, "multipass", 20, "%s_multipass_k20" % name
        )
        for k in SEQ_BESTIMP_K:
            key = "bestimp_k%d" % int(k)
            block[key] = _refine_mode(
                rows, scores, resources, "bestimp", int(k), "%s_%s" % (name, key)
            )
        methods[name] = block
    scan = methods["validation"]["scan_k20"]["pair_T_mean"]
    multi = methods["validation"]["multipass_k20"]["pair_T_mean"]
    best = methods["validation"]["bestimp_k20"]["pair_T_mean"]
    verdict = classify_seq_verdict(scan, multi, best)
    payload = {
        "paper_result": False,
        "ppo": False,
        "encoder_frozen": True,
        "decoder_replaced": False,
        "top_k": PAIR_TOP_K,
        "bestimp_k": list(SEQ_BESTIMP_K),
        "scan_passes": SEQ_SCAN_PASSES,
        "oracle_ref_val": SEQ_ORACLE_REF_VAL,
        "methods": methods,
        "verdict": verdict,
        "softmax_last_loss": ce_model.get("last_loss"),
        "note": "CE scan vs multipass vs bestimp; LSTM decoder unchanged; no PPO; not 3500",
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seq_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "pair_seq verdict=%s val_scan=%.1f val_multi=%.1f val_best20=%.1f val_best50=%.1f oracle_ref=%.1f test_best20=%.1f"
        % (
            verdict,
            scan,
            multi,
            best,
            methods["validation"]["bestimp_k50"]["pair_T_mean"],
            SEQ_ORACLE_REF_VAL,
            methods["meta_test"]["bestimp_k20"]["pair_T_mean"],
        )
    )
    return payload
