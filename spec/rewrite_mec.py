"""STATUS: baseline/ablation only (ADR-007)

All-MEC motif rewrite. Diagnostic. Not 3500.

L_BC clones greedy_from_mec (occupancy). L_joint labels come from all-MEC
9-way schedule() counterfactuals, not the clone-trap greedy_from_mec labels.
Eval: greedy decode, then K=1..3 neural argmax joint applies.
Zero schedule() search at inference. schedule() only reports T.
Control A = bc_unseen val 575 / test 557. paper_result=false.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import (
    BC_BATCH,
    align_greedy_pred,
    collect_expert_dataset,
    greedy_rollout_eval,
)
from spec.pair_head import N_JOINT, _apply_joint, _iter_graphs, _score, motif_pairs, split_joint
from spec.pair_sup import (
    PAIRSUP_A_TEST_T,
    PAIRSUP_A_VAL_T,
    PAIRSUP_EPOCHS,
    PAIRSUP_LAMBDA,
    PAIRSUP_LR,
    _batch_pairs,
    build_pairsup_train_ops,
    collect_joint_labels,
)

REWRITE_LAMBDA = PAIRSUP_LAMBDA
REWRITE_EPOCHS = PAIRSUP_EPOCHS
REWRITE_LR = PAIRSUP_LR
REWRITE_A_VAL_T = PAIRSUP_A_VAL_T
REWRITE_A_TEST_T = PAIRSUP_A_TEST_T
REWRITE_HELP_SEC = 15.0
REWRITE_WEAK_SEC = 5.0
REWRITE_LOCAL_OK = 0.28
REWRITE_LOCAL_HURT = 0.35
REWRITE_K = (1, 2, 3)
REWRITE_MIN_FRAC_POS = 0.20
CACHE_JOINT_ALLMEC = Path("runs") / "phase4" / "rewrite_joint_from_allmec_train.npz"


def classify_rewrite_verdict(val_t, local_frac, a_val_t=REWRITE_A_VAL_T):
    val_t = float(val_t)
    local_frac = float(local_frac)
    a_val_t = float(a_val_t)
    if local_frac >= REWRITE_LOCAL_HURT or val_t >= a_val_t + REWRITE_HELP_SEC:
        return "rewrite_hurts"
    delta = a_val_t - val_t
    if delta >= REWRITE_HELP_SEC and local_frac < REWRITE_LOCAL_OK:
        return "rewrite_helps"
    if delta >= REWRITE_WEAK_SEC:
        return "rewrite_weak"
    return "rewrite_no_gain"


def log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def pick_neural_joint(plan, pairs, logp, min_gain=1e-8):
    """Best motif joint by logπ_i + logπ_j. No schedule()."""
    plan = [int(a) for a in plan]
    logp = np.asarray(logp, dtype=np.float64)
    best = None
    best_gain = float(min_gain)
    for rec in pairs:
        i = int(rec["i"])
        j = int(rec["j"])
        ci = plan[i]
        cj = plan[j]
        cur_s = float(logp[i, ci]) + float(logp[j, cj])
        for jid in range(N_JOINT):
            ai, aj = split_joint(jid)
            if ai == ci and aj == cj:
                continue
            gain = (float(logp[i, ai]) + float(logp[j, aj])) - cur_s
            if gain > best_gain:
                best_gain = gain
                best = {
                    "i": i,
                    "j": j,
                    "ai": int(ai),
                    "aj": int(aj),
                    "jid": int(jid),
                    "gain": float(gain),
                }
    return best


def apply_k_neural(plan, pairs, logp_fn, k):
    """Sequential neural joint applies. logp_fn(plan)->[n_tok, n_act]. No schedule()."""
    plan = [int(a) for a in plan]
    applied = []
    for _ in range(int(k)):
        logp = logp_fn(plan)
        picked = pick_neural_joint(plan, pairs, logp)
        if picked is None:
            break
        plan = _apply_joint(plan, picked["i"], picked["j"], picked["jid"])
        applied.append(picked)
    return plan, applied


def _mix_and_t(acts_rows, t_rows):
    pred = np.asarray(acts_rows, dtype=np.int32)
    t_p = np.asarray(t_rows, dtype=np.float64)
    counts = np.bincount(pred.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non = np.asarray([sum(1 for a in row if int(a) != 1) for row in pred], dtype=np.float64)
    return {
        "greedy_T_mean": float(np.mean(t_p)),
        "greedy_T_p50": float(np.median(t_p)),
        "greedy_local_frac": float(counts[0]),
        "greedy_mec_frac": float(counts[1]),
        "greedy_v2v_frac": float(counts[2]),
        "n_nonmec_p50": float(np.median(n_non)),
        "n_nonmec_mean": float(np.mean(n_non)),
        "n_apply_mean": None,
    }


def teacher_force_logp(sess, policy, obs_row, plan):
    plan_b = np.asarray(plan, dtype=np.int32).reshape(1, -1)
    obs_b = np.asarray(obs_row, dtype=np.float32)
    if obs_b.ndim == 2:
        obs_b = obs_b[None, ...]
    n_tok = int(plan_b.shape[1])
    shift = np.concatenate([np.zeros((1, 1), dtype=np.int32), plan_b[:, :-1]], axis=1)
    logits = sess.run(
        policy.network.decoder_logits,
        feed_dict={
            policy.obs: obs_b,
            policy.decoder_inputs: shift,
            policy.decoder_targets: plan_b,
            policy.decoder_full_length: np.array([n_tok], dtype=np.int32),
        },
    )
    return log_softmax(logits[0])[:, :3]


def greedy_decode_all(sess, policy, obs, n_tok):
    n = int(obs.shape[0])
    preds = []
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        greedy_pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict={policy.obs: obs[sl], policy.decoder_full_length: fl},
        )
        aligned, _ = align_greedy_pred(greedy_pred, n_tok)
        preds.append(aligned)
    return np.concatenate(preds, axis=0)


def eval_rewrite_on_env(sess, env, policy, cache_path=None, split_name="", k_max=3):
    """Greedy then K neural applies. schedule() reports T only. No search."""
    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=cache_path)
    n, n_tok = acts.shape
    pred0 = greedy_decode_all(sess, policy, obs, n_tok)
    graphs = list(_iter_graphs(env))
    if len(graphs) != n:
        raise ValueError("graph count %d != %d" % (len(graphs), n))
    resources = env.scheduler_resources
    k_max = int(k_max)
    t_by_k = {k: [] for k in range(0, k_max + 1)}
    acts_by_k = {k: [] for k in range(0, k_max + 1)}
    n_apply = {k: [] for k in range(1, k_max + 1)}
    for gi, tg in enumerate(graphs):
        plan = [int(a) for a in pred0[gi].tolist()]
        order = [int(tid) for tid in tg.prioritize_sequence]
        pairs = motif_pairs(tg.succ_task_sets, tg.pre_task_sets, order)
        t0 = _score(tg, resources, plan)
        t_by_k[0].append(t0)
        acts_by_k[0].append(list(plan))

        def logp_fn(cur, _obs=obs[gi], _pol=policy, _sess=sess):
            return teacher_force_logp(_sess, _pol, _obs, cur)

        cur = list(plan)
        applied_all = []
        for k in range(1, k_max + 1):
            cur, step = apply_k_neural(cur, pairs, logp_fn, 1)
            applied_all.extend(step)
            t_by_k[k].append(_score(tg, resources, cur))
            acts_by_k[k].append(list(cur))
            n_apply[k].append(len(applied_all))
        if (gi + 1) % 50 == 0 or gi == 0 or gi == n - 1:
            print(
                "rewrite_eval split=%s gi=%d/%d T0=%.1f Tk=%.1f n_apply=%d"
                % (
                    split_name or "?",
                    gi + 1,
                    n,
                    t_by_k[0][-1],
                    t_by_k[k_max][-1],
                    n_apply[k_max][-1],
                )
            )
    t_ex = np.asarray(extras["t_expert"], dtype=np.float64)
    t_m = np.asarray(extras["t_mec"], dtype=np.float64)
    by_k = {}
    for k in range(0, k_max + 1):
        row = _mix_and_t(acts_by_k[k], t_by_k[k])
        row["all_mec_T_mean"] = float(np.mean(t_m))
        row["expert_T_mean"] = float(np.mean(t_ex))
        row["greedy_over_expert"] = float(row["greedy_T_mean"] / max(float(np.mean(t_ex)), 1e-12))
        row["k"] = int(k)
        if k >= 1:
            row["n_apply_mean"] = float(np.mean(n_apply[k]))
        by_k[str(k)] = row
    k0 = by_k["0"]
    best_k = 0
    best_t = k0["greedy_T_mean"]
    for k in range(1, k_max + 1):
        tt = by_k[str(k)]["greedy_T_mean"]
        if tt + 1e-12 < best_t:
            best_t = tt
            best_k = k
    print(
        "rewrite_unseen split=%s n=%d T_k0=%.1f T_bestk=%.1f best_k=%d mix0 L/M/V=%.3f/%.3f/%.3f"
        % (
            split_name or "?",
            n,
            k0["greedy_T_mean"],
            best_t,
            best_k,
            k0["greedy_local_frac"],
            k0["greedy_mec_frac"],
            k0["greedy_v2v_frac"],
        )
    )
    return {
        "split": split_name,
        "expert": {
            "n_graphs": expert_stats["n_graphs"],
            "T_mean": expert_stats["expert_T_mean"],
            "all_mec_T_mean": expert_stats["all_mec_T_mean"],
        },
        "by_k": by_k,
        "best_k": int(best_k),
        "best_T": float(best_t),
        "greedy": k0,
        "pair_search_at_eval": False,
    }


def run_rewrite_train(
    sess,
    env,
    policy,
    rng,
    run_dir,
    load_ckpt,
    expert_cache,
    joint_cache,
    epochs=REWRITE_EPOCHS,
    lam=REWRITE_LAMBDA,
    save_ckpt=None,
):
    """Teacher-force greedy_from_mec + λ joint CE from all-MEC labels."""
    import tensorflow as tf

    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=expert_cache)
    mec_acts = np.ones_like(acts, dtype=np.int32)
    labels_by_gi, label_stats = collect_joint_labels(
        env, mec_acts, extras["t_mec"], cache_path=joint_cache
    )
    frac = float(label_stats["frac_pos"])
    if frac < REWRITE_MIN_FRAC_POS:
        raise ValueError(
            "rewrite all-MEC frac_pos=%.3f < %.2f — clone trap, do not train"
            % (frac, REWRITE_MIN_FRAC_POS)
        )
    n, n_tok = acts.shape
    shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    if load_ckpt:
        policy.load_variables(str(load_ckpt), sess=sess)
        print("rewrite_load_ckpt %s" % load_ckpt)
    ops = build_pairsup_train_ops(policy, lr=REWRITE_LR, lam=lam)
    sess.run(tf.compat.v1.variables_initializer(ops["opt"].variables()))
    n_epochs = int(epochs)
    order = np.arange(n)
    epoch_losses = []
    for epoch in range(n_epochs):
        rng.shuffle(order)
        batch_losses = []
        batch_bc = []
        batch_j = []
        for start in range(0, n, BC_BATCH):
            sl = order[start : start + BC_BATCH]
            packed = _batch_pairs(sl, labels_by_gi)
            fl = np.full((len(sl),), n_tok, dtype=np.int32)
            _, lv, lbc, ljv = sess.run(
                [ops["train"], ops["loss"], ops["loss_bc"], ops["loss_joint"]],
                feed_dict={
                    policy.obs: obs[sl],
                    policy.decoder_inputs: shift[sl],
                    policy.decoder_targets: acts[sl],
                    policy.decoder_full_length: fl,
                    ops["pair_b"]: packed["b"],
                    ops["pair_i"]: packed["i"],
                    ops["pair_j"]: packed["j"],
                    ops["pair_ai"]: packed["ai"],
                    ops["pair_aj"]: packed["aj"],
                    ops["pair_w"]: packed["w"],
                },
            )
            batch_losses.append(float(lv))
            batch_bc.append(float(lbc))
            batch_j.append(float(ljv))
        epoch_losses.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(batch_losses)),
                "loss_bc": float(np.mean(batch_bc)),
                "loss_joint": float(np.mean(batch_j)),
            }
        )
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
            row = epoch_losses[-1]
            print(
                "rewrite_epoch %d/%d loss=%.4f bc=%.4f joint=%.4f"
                % (epoch + 1, n_epochs, row["loss"], row["loss_bc"], row["loss_joint"])
            )
    if save_ckpt:
        ckpt_path = Path(save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_variables(str(ckpt_path), sess=sess)
        print("rewrite_ckpt %s" % ckpt_path)
    rollout = greedy_rollout_eval(sess, env, policy, obs, acts, extras)
    stats = {
        "expert": expert_stats,
        "labels": label_stats,
        "label_start": "all_mec",
        "lam": float(lam),
        "epochs": n_epochs,
        "lr": float(REWRITE_LR),
        "load_ckpt": str(load_ckpt) if load_ckpt else None,
        "epoch_losses": epoch_losses,
        "greedy_rollout": rollout,
        "ppo": False,
        "pair_search_at_eval": False,
        "paper_result": False,
    }
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "rewrite_train.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )
    print(
        "rewrite_train T=%.1f expert=%.1f frac_pos=%.3f mix=%.3f/%.3f/%.3f"
        % (
            rollout["greedy_T_mean"],
            rollout["expert_T_mean"],
            label_stats["frac_pos"],
            rollout["greedy_local_frac"],
            rollout["greedy_mec_frac"],
            rollout["greedy_v2v_frac"],
        )
    )
    return stats
