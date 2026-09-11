"""STATUS: baseline/ablation only (ADR-007)

Pair-supervision on Encoder+Decoder from greedy_from_mec. Diagnostic. Not 3500.

B vs A: A is bc_continue unseen greedy (val T 575). B adds L_joint on motif
pairs whose 9-way schedule() counterfactual beats the greedy_from_mec plan.
Eval is greedy decode only. No pair search at inference. Not the 2-opt clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import BC_BATCH, BC_LR, align_greedy_pred, collect_expert_dataset, greedy_rollout_eval
from spec.pair_head import N_JOINT, _apply_joint, _iter_graphs, _score, joint_id, motif_pairs, split_joint

PAIRSUP_LAMBDA = 0.5
PAIRSUP_EPOCHS = 40
PAIRSUP_LR = BC_LR
PAIRSUP_A_VAL_T = 575.0
PAIRSUP_A_TEST_T = 557.0
PAIRSUP_HELP_SEC = 15.0
PAIRSUP_WEAK_SEC = 5.0
PAIRSUP_LOCAL_OK = 0.28
PAIRSUP_LOCAL_HURT = 0.35
CACHE_JOINT_TRAIN = Path("runs") / "phase4" / "pairsup_joint_from_mec_train.npz"


def classify_pairsup_verdict(val_t, local_frac, a_val_t=PAIRSUP_A_VAL_T):
    val_t = float(val_t)
    local_frac = float(local_frac)
    a_val_t = float(a_val_t)
    if local_frac >= PAIRSUP_LOCAL_HURT or val_t >= a_val_t + PAIRSUP_HELP_SEC:
        return "pairsup_hurts"
    delta = a_val_t - val_t
    if delta >= PAIRSUP_HELP_SEC and local_frac < PAIRSUP_LOCAL_OK:
        return "pairsup_helps"
    if delta >= PAIRSUP_WEAK_SEC:
        return "pairsup_weak"
    return "pairsup_no_gain"


def pick_improving_joint(plan, i, j, t0, score_fn):
    """Return improving (i,j,ai,aj,jid,delta) or None. score_fn(acts)->T."""
    plan = [int(a) for a in plan]
    i = int(i)
    j = int(j)
    t0 = float(t0)
    cur = joint_id(plan[i], plan[j])
    best_t = t0
    best_jid = cur
    n_eval = 0
    for jid in range(N_JOINT):
        if jid == cur:
            continue
        trial = _apply_joint(plan, i, j, jid)
        tt = float(score_fn(trial))
        n_eval += 1
        if tt + 1e-12 < best_t:
            best_t = tt
            best_jid = jid
    if best_jid == cur:
        return None, n_eval
    ai, aj = split_joint(best_jid)
    return {
        "i": i,
        "j": j,
        "ai": int(ai),
        "aj": int(aj),
        "jid": int(best_jid),
        "delta": float(t0 - best_t),
    }, n_eval


def _batch_pairs(batch_gi, labels_by_gi):
    b_idx = []
    i_l = []
    j_l = []
    ai_l = []
    aj_l = []
    for local, gi in enumerate(batch_gi):
        for rec in labels_by_gi.get(int(gi), ()):
            b_idx.append(int(local))
            i_l.append(int(rec["i"]))
            j_l.append(int(rec["j"]))
            ai_l.append(int(rec["ai"]))
            aj_l.append(int(rec["aj"]))
    if not b_idx:
        return {
            "b": np.zeros((1,), dtype=np.int32),
            "i": np.zeros((1,), dtype=np.int32),
            "j": np.zeros((1,), dtype=np.int32),
            "ai": np.zeros((1,), dtype=np.int32),
            "aj": np.zeros((1,), dtype=np.int32),
            "w": np.zeros((1,), dtype=np.float32),
        }
    return {
        "b": np.asarray(b_idx, dtype=np.int32),
        "i": np.asarray(i_l, dtype=np.int32),
        "j": np.asarray(j_l, dtype=np.int32),
        "ai": np.asarray(ai_l, dtype=np.int32),
        "aj": np.asarray(aj_l, dtype=np.int32),
        "w": np.ones((len(b_idx),), dtype=np.float32),
    }


def collect_joint_labels(env, acts, t_expert, cache_path=None):
    """Improving motif joints from greedy_from_mec plans. Not 2-opt clone labels."""
    acts = np.asarray(acts, dtype=np.int32)
    t_expert = np.asarray(t_expert, dtype=np.float64)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.is_file():
        blob = np.load(str(cache_path), allow_pickle=False)
        stats = {
            "n_graphs": int(blob["n_graphs"][0]),
            "n_pairs": int(blob["n_pairs"][0]),
            "n_pos": int(blob["n_pos"][0]),
            "n_eval": int(blob["n_eval"][0]),
            "frac_pos": float(blob["frac_pos"][0]),
            "cache": str(cache_path),
        }
        labels_by_gi = {}
        for k in range(int(blob["gi"].shape[0])):
            gi = int(blob["gi"][k])
            labels_by_gi.setdefault(gi, []).append(
                {
                    "i": int(blob["i"][k]),
                    "j": int(blob["j"][k]),
                    "ai": int(blob["ai"][k]),
                    "aj": int(blob["aj"][k]),
                    "jid": int(blob["jid"][k]),
                    "delta": float(blob["delta"][k]),
                }
            )
        print("pairsup_label_cache hit %s n_pos=%d frac=%.3f" % (cache_path, stats["n_pos"], stats["frac_pos"]))
        return labels_by_gi, stats

    graphs = list(_iter_graphs(env))
    if len(graphs) != acts.shape[0]:
        raise ValueError("graph count %d != acts %d" % (len(graphs), acts.shape[0]))
    resources = env.scheduler_resources
    gi_l = []
    i_l = []
    j_l = []
    ai_l = []
    aj_l = []
    jid_l = []
    delta_l = []
    n_pairs = 0
    n_eval = 0
    labels_by_gi = {}
    for gi, tg in enumerate(graphs):
        plan = [int(a) for a in acts[gi].tolist()]
        t0 = float(t_expert[gi])
        order = [int(tid) for tid in tg.prioritize_sequence]
        pairs = motif_pairs(tg.succ_task_sets, tg.pre_task_sets, order)
        n_pairs += len(pairs)

        def score_fn(trial, _tg=tg, _res=resources):
            return _score(_tg, _res, trial)

        kept = []
        for rec in pairs:
            picked, n_e = pick_improving_joint(plan, rec["i"], rec["j"], t0, score_fn)
            n_eval += n_e
            if picked is None:
                continue
            kept.append(picked)
            gi_l.append(gi)
            i_l.append(picked["i"])
            j_l.append(picked["j"])
            ai_l.append(picked["ai"])
            aj_l.append(picked["aj"])
            jid_l.append(picked["jid"])
            delta_l.append(picked["delta"])
        if kept:
            labels_by_gi[gi] = kept
        if (gi + 1) % 50 == 0 or gi == 0:
            print(
                "pairsup_labels gi=%d/%d pos=%d pairs=%d eval=%d"
                % (gi + 1, len(graphs), len(gi_l), n_pairs, n_eval)
            )
    n_pos = len(gi_l)
    frac_pos = float(n_pos) / float(max(n_pairs, 1))
    stats = {
        "n_graphs": int(len(graphs)),
        "n_pairs": int(n_pairs),
        "n_pos": int(n_pos),
        "n_eval": int(n_eval),
        "frac_pos": frac_pos,
        "mean_delta_pos": float(np.mean(delta_l)) if delta_l else 0.0,
    }
    if frac_pos < 0.20:
        print("pairsup_label_frac_pos_low %.3f — expected >> 2-opt-clone 0.069" % frac_pos)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            gi=np.asarray(gi_l, dtype=np.int32),
            i=np.asarray(i_l, dtype=np.int32),
            j=np.asarray(j_l, dtype=np.int32),
            ai=np.asarray(ai_l, dtype=np.int32),
            aj=np.asarray(aj_l, dtype=np.int32),
            jid=np.asarray(jid_l, dtype=np.int32),
            delta=np.asarray(delta_l, dtype=np.float64),
            n_graphs=np.asarray([stats["n_graphs"]], dtype=np.int32),
            n_pairs=np.asarray([stats["n_pairs"]], dtype=np.int32),
            n_pos=np.asarray([stats["n_pos"]], dtype=np.int32),
            n_eval=np.asarray([stats["n_eval"]], dtype=np.int32),
            frac_pos=np.asarray([frac_pos], dtype=np.float64),
        )
        stats["cache"] = str(cache_path)
        print("pairsup_label_cache write %s frac_pos=%.3f" % (cache_path, frac_pos))
    return labels_by_gi, stats


def build_pairsup_train_ops(policy, lr=PAIRSUP_LR, lam=PAIRSUP_LAMBDA):
    import tensorflow as tf

    logits = policy.network.decoder_logits
    logp = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    pair_b = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pairsup_b")
    pair_i = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pairsup_i")
    pair_j = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pairsup_j")
    pair_ai = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pairsup_ai")
    pair_aj = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pairsup_aj")
    pair_w = tf.compat.v1.placeholder(tf.float32, shape=[None], name="pairsup_w")
    li = tf.gather_nd(logp, tf.stack([pair_b, pair_i], axis=1))
    lj = tf.gather_nd(logp, tf.stack([pair_b, pair_j], axis=1))
    p_range = tf.range(tf.shape(pair_b)[0])
    logp_i = tf.gather_nd(li, tf.stack([p_range, pair_ai], axis=1))
    logp_j = tf.gather_nd(lj, tf.stack([p_range, pair_aj], axis=1))
    wsum = tf.maximum(tf.reduce_sum(pair_w), 1.0)
    loss_joint = -tf.reduce_sum(pair_w * (logp_i + logp_j)) / wsum
    loss_bc = tf.reduce_mean(policy.network.neglogp())
    loss = loss_bc + float(lam) * loss_joint
    opt = tf.compat.v1.train.AdamOptimizer(float(lr), name="pairsup_adam")
    train_op = opt.minimize(loss, var_list=policy.get_trainable_variables())
    return {
        "train": train_op,
        "loss": loss,
        "loss_bc": loss_bc,
        "loss_joint": loss_joint,
        "opt": opt,
        "pair_b": pair_b,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "pair_ai": pair_ai,
        "pair_aj": pair_aj,
        "pair_w": pair_w,
        "lam": float(lam),
        "lr": float(lr),
    }


def run_pairsup_train(
    sess,
    env,
    policy,
    rng,
    run_dir,
    load_ckpt,
    expert_cache,
    joint_cache,
    epochs=PAIRSUP_EPOCHS,
    lam=PAIRSUP_LAMBDA,
    save_ckpt=None,
):
    """Teacher-force greedy_from_mec + λ joint CE. Encoder and decoder trainable."""
    import tensorflow as tf

    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=expert_cache)
    labels_by_gi, label_stats = collect_joint_labels(
        env, acts, extras["t_expert"], cache_path=joint_cache
    )
    n, n_tok = acts.shape
    shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    if load_ckpt:
        policy.load_variables(str(load_ckpt), sess=sess)
        print("pairsup_load_ckpt %s" % load_ckpt)
    ops = build_pairsup_train_ops(policy, lr=PAIRSUP_LR, lam=lam)
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
                "pairsup_epoch %d/%d loss=%.4f bc=%.4f joint=%.4f"
                % (epoch + 1, n_epochs, row["loss"], row["loss_bc"], row["loss_joint"])
            )
    if save_ckpt:
        ckpt_path = Path(save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_variables(str(ckpt_path), sess=sess)
        print("pairsup_ckpt %s" % ckpt_path)
    rollout = greedy_rollout_eval(sess, env, policy, obs, acts, extras)
    stats = {
        "expert": expert_stats,
        "labels": label_stats,
        "lam": float(lam),
        "epochs": n_epochs,
        "lr": float(PAIRSUP_LR),
        "load_ckpt": str(load_ckpt) if load_ckpt else None,
        "epoch_losses": epoch_losses,
        "greedy_rollout": rollout,
        "ppo": False,
        "pair_search_at_eval": False,
        "paper_result": False,
    }
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "pairsup_train.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )
    print(
        "pairsup_train T=%.1f expert=%.1f frac_pos=%.3f mix=%.3f/%.3f/%.3f"
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
