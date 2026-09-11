"""STATUS: baseline/ablation only (ADR-007)

Oracle dist_id embed at every decoder step. Diagnostic. Not 3500.

True dist_id lookup + zero-init residual into decoder inputs. Graph2Seq frozen.
Init bc_continue. Identity (delta=0) must match control A before train.
Val/test IDs are unseen latin slots — untrained table rows. paper_result=false.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_fewshot import encoder_frozen_adapt_vars
from spec.bc_greedy_mec import (
    BC_BATCH,
    BC_LR,
    align_greedy_pred,
    collect_expert_dataset,
)

ORACLE_N_DIST = 26
ORACLE_Z_DIM = 32
ORACLE_EPOCHS = 40
ORACLE_LR = BC_LR
ORACLE_A_VAL_T = 575.0
ORACLE_A_TEST_T = 557.0
ORACLE_TRAIN_REF_T = 464.0
ORACLE_IDENTITY_T_VAL_LO = 568.0
ORACLE_IDENTITY_T_VAL_HI = 582.0
ORACLE_HELP_SEC = 15.0
ORACLE_WEAK_SEC = 5.0
ORACLE_LOCAL_OK = 0.28
ORACLE_LOCAL_HURT = 0.35


def is_oracle_dist_var_name(name):
    return "oracle_dist" in str(name)


def classify_oracle_verdict(val_t, local_frac, train_t=None, a_val_t=ORACLE_A_VAL_T):
    val_t = float(val_t)
    local_frac = float(local_frac)
    a_val_t = float(a_val_t)
    if local_frac >= ORACLE_LOCAL_HURT or val_t >= a_val_t + ORACLE_HELP_SEC:
        return "oracle_hurts"
    delta = a_val_t - val_t
    if delta >= ORACLE_HELP_SEC and local_frac < ORACLE_LOCAL_OK:
        return "oracle_helps"
    if train_t is not None:
        train_delta = float(ORACLE_TRAIN_REF_T) - float(train_t)
        if train_delta >= ORACLE_HELP_SEC and delta < ORACLE_HELP_SEC:
            return "oracle_in_dist_only"
    if delta >= ORACLE_WEAK_SEC:
        return "oracle_weak"
    return "oracle_no_gain"


def dist_ids_for_env(env):
    dist_id_list = list(getattr(env, "distribution_ids", []) or [])
    rows = []
    for dist_i, graphs in enumerate(env.task_graphs_batchs):
        did = int(dist_id_list[dist_i]) if dist_i < len(dist_id_list) else dist_i
        rows.extend([did] * len(graphs))
    return np.asarray(rows, dtype=np.int32)


def resolve_dist_ids(env, extras, n_graphs):
    extras = extras or {}
    if extras.get("dist_id") is not None:
        did = np.asarray(extras["dist_id"], dtype=np.int32).reshape(-1)
        if int(did.shape[0]) == int(n_graphs):
            if np.any(did < 0) or np.any(did >= ORACLE_N_DIST):
                raise ValueError("dist_id out of table %s %s" % (int(did.min()), int(did.max())))
            return did
        print("oracle dist_id cache length mismatch; using env")
    did = dist_ids_for_env(env)
    if int(did.shape[0]) != int(n_graphs):
        raise ValueError("env dist_id count %d != n_graphs %d" % (did.shape[0], n_graphs))
    if np.any(did < 0) or np.any(did >= ORACLE_N_DIST):
        raise ValueError("env dist_id out of table")
    return did


def load_oracle_from_bc_ckpt(policy, ckpt_path, sess, src_scope="core_policy"):
    import joblib

    from spec.kl_bc_anchor import pair_ckpt_to_vars

    loaded = joblib.load(str(ckpt_path))
    dst_vars = [v for v in policy.get_variables() if not is_oracle_dist_var_name(v.name)]
    paired = pair_ckpt_to_vars(loaded, dst_vars, src_scope, policy.name)
    restores = []
    for var, arr in paired:
        arr = np.asarray(arr)
        shape = tuple(int(x) for x in var.shape.as_list())
        if arr.shape != shape:
            raise ValueError("shape mismatch %s ckpt%s var%s" % (var.name, arr.shape, shape))
        restores.append(var.assign(arr))
    sess.run(restores)
    return len(restores)


def oracle_feed(policy, obs, dist_ids, n_tok, shift=None, acts=None):
    fl = np.full((int(obs.shape[0]),), int(n_tok), dtype=np.int32)
    fd = {
        policy.obs: obs,
        policy.decoder_full_length: fl,
        policy.dist_ids: np.asarray(dist_ids, dtype=np.int32).reshape(-1),
    }
    if shift is not None:
        fd[policy.decoder_inputs] = shift
    if acts is not None:
        fd[policy.decoder_targets] = acts
    return fd


def greedy_oracle_eval(sess, env, policy, cache_path=None, split_name=""):
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=cache_path)
    n, n_tok = acts.shape
    dist_ids = resolve_dist_ids(env, extras, n)
    preds = []
    n_trunc = 0
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        greedy_pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict=oracle_feed(policy, obs[sl], dist_ids[sl], n_tok),
        )
        aligned, trunc = align_greedy_pred(greedy_pred, n_tok)
        n_trunc += int(trunc)
        preds.append(aligned)
    pred = np.concatenate(preds, axis=0)
    if n_trunc:
        print("oracle_greedy_align_batches %d" % n_trunc)
    t_policy = []
    n_non = []
    idx = 0
    for graphs in env.task_graphs_batchs:
        for tg in graphs:
            actions = [int(a) for a in pred[idx].tolist()]
            order = [int(tid) for tid in tg.prioritize_sequence]
            if len(actions) != len(order):
                raise ValueError("oracle greedy plan length mismatch")
            result, _, _ = schedule_via_adapter(
                tg, list(zip(order, actions)), env.scheduler_resources
            )
            t_policy.append(float(result.makespan_seconds))
            n_non.append(sum(1 for a in actions if a != 1))
            idx += 1
    if idx != n:
        raise ValueError("oracle eval graph count %d != dataset %d" % (idx, n))
    t_p = np.asarray(t_policy, dtype=np.float64)
    t_ex = np.asarray(extras["t_expert"], dtype=np.float64)
    t_m = np.asarray(extras["t_mec"], dtype=np.float64)
    counts = np.bincount(pred.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non_a = np.asarray(n_non, dtype=np.float64)
    out = {
        "split": split_name,
        "n_graphs": int(n),
        "greedy_token_acc_vs_expert": float(np.mean(pred == acts)),
        "greedy_T_mean": float(np.mean(t_p)),
        "greedy_T_p50": float(np.median(t_p)),
        "greedy_T_p10": float(np.percentile(t_p, 10)),
        "expert_T_mean": float(np.mean(t_ex)),
        "all_mec_T_mean": float(np.mean(t_m)),
        "greedy_over_expert": float(np.mean(t_p) / max(float(np.mean(t_ex)), 1e-12)),
        "greedy_over_mec": float(np.mean(t_p) / max(float(np.mean(t_m)), 1e-12)),
        "frac_greedy_beats_mec": float(np.mean(t_p + 1e-12 < t_m)),
        "frac_greedy_beats_expert": float(np.mean(t_p + 1e-12 < t_ex)),
        "n_nonmec_mean": float(np.mean(n_non_a)),
        "n_nonmec_p50": float(np.median(n_non_a)),
        "frac_n_nonmec_le2": float(np.mean(n_non_a <= 2)),
        "frac_n_nonmec_le3": float(np.mean(n_non_a <= 3)),
        "greedy_local_frac": float(counts[0]),
        "greedy_mec_frac": float(counts[1]),
        "greedy_v2v_frac": float(counts[2]),
        "encoder_frozen": True,
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
    }
    per = {}
    for d in np.unique(dist_ids):
        m = dist_ids == d
        per[str(int(d))] = {
            "n_graphs": int(np.sum(m)),
            "greedy_T_mean": float(np.mean(t_p[m])),
            "expert_T_mean": float(np.mean(t_ex[m])),
        }
    out["per_distribution"] = per
    print(
        "oracle_eval split=%s n=%d T=%.1f mix=%.3f/%.3f/%.3f"
        % (
            split_name or "?",
            n,
            out["greedy_T_mean"],
            out["greedy_local_frac"],
            out["greedy_mec_frac"],
            out["greedy_v2v_frac"],
        )
    )
    return out


def run_oracle_train(
    sess,
    env,
    policy,
    rng,
    run_dir=None,
    load_ckpt=None,
    expert_cache=None,
    epochs=ORACLE_EPOCHS,
    save_ckpt=None,
    identity_row=None,
):
    import tensorflow as tf

    n_epochs = int(epochs)
    if n_epochs < 1:
        raise ValueError("epochs must be positive")
    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=expert_cache)
    n, n_tok = acts.shape
    dist_ids = resolve_dist_ids(env, extras, n)
    shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    n_loaded = 0
    if load_ckpt:
        n_loaded = load_oracle_from_bc_ckpt(policy, load_ckpt, sess)
        print("oracle_load_ckpt n_vars=%d %s" % (n_loaded, load_ckpt))
    frozen, adapt = encoder_frozen_adapt_vars(policy.get_trainable_variables())
    oracle_vars = [v for v in adapt if is_oracle_dist_var_name(v.name)]
    if not oracle_vars:
        raise ValueError("oracle_dist vars missing")
    loss = tf.reduce_mean(policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(ORACLE_LR, name="oracle_dist_adam")
    train_op = opt.minimize(loss, var_list=adapt)
    sess.run(tf.compat.v1.variables_initializer(opt.variables()))
    order = np.arange(n)
    epoch_losses = []
    for epoch in range(n_epochs):
        rng.shuffle(order)
        losses = []
        for start in range(0, n, BC_BATCH):
            sl = order[start : start + BC_BATCH]
            fd = oracle_feed(policy, obs[sl], dist_ids[sl], n_tok, shift=shift[sl], acts=acts[sl])
            _, batch_loss = sess.run([train_op, loss], feed_dict=fd)
            losses.append(float(batch_loss))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        epoch_losses.append(row)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == n_epochs:
            print("oracle_epoch %d/%d loss=%.4f" % (epoch + 1, n_epochs, row["loss"]))
    if save_ckpt:
        ckpt_path = Path(save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_variables(str(ckpt_path), sess=sess)
        print("oracle_ckpt %s" % ckpt_path)
    rollout = greedy_oracle_eval(sess, env, policy, cache_path=expert_cache, split_name="train")
    stats = {
        "expert": expert_stats,
        "epochs": n_epochs,
        "lr": float(ORACLE_LR),
        "n_loaded": int(n_loaded),
        "n_frozen_encoder": int(len(frozen)),
        "n_adapt": int(len(adapt)),
        "n_oracle_vars": int(len(oracle_vars)),
        "load_ckpt": str(load_ckpt) if load_ckpt else None,
        "epoch_losses": epoch_losses,
        "identity": identity_row,
        "greedy_rollout": rollout,
        "encoder_frozen": True,
        "ppo": False,
        "pair_search_at_eval": False,
        "paper_result": False,
    }
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "oracle_train.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )
    print(
        "oracle_train T=%.1f expert=%.1f ref=%.1f mix=%.3f/%.3f/%.3f"
        % (
            rollout["greedy_T_mean"],
            rollout["expert_T_mean"],
            ORACLE_TRAIN_REF_T,
            rollout["greedy_local_frac"],
            rollout["greedy_mec_frac"],
            rollout["greedy_v2v_frac"],
        )
    )
    return stats
