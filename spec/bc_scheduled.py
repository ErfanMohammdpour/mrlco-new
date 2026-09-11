"""Scheduled-sampling BC from π_BC. Diagnostic only. Not the frozen 3500 primary.

Teacher-force mix with the model's own greedy prefix. Encoder frozen.
Tests exposure vs OOD fat/density. No PPO.
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
    greedy_rollout_eval,
)

SS_EPOCHS = 40
SS_EPS_START = 0.8
SS_EPS_END = 0.3
SS_PROBE = 256


def ss_eps(epoch, n_epochs, start=SS_EPS_START, end=SS_EPS_END):
    n_epochs = int(n_epochs)
    epoch = int(epoch)
    if n_epochs <= 1:
        return float(end)
    t = float(epoch) / float(n_epochs - 1)
    return float(start) + (float(end) - float(start)) * t


def mix_decoder_inputs(gold_shift, pred_shift, eps, rng):
    gold_shift = np.asarray(gold_shift, dtype=np.int32)
    pred_shift = np.asarray(pred_shift, dtype=np.int32)
    if gold_shift.shape != pred_shift.shape:
        raise ValueError("shift shape mismatch %s vs %s" % (gold_shift.shape, pred_shift.shape))
    use_gold = rng.rand(*gold_shift.shape) < float(eps)
    mixed = np.where(use_gold, gold_shift, pred_shift).astype(np.int32)
    mixed[:, 0] = 0
    return mixed


def greedy_token_acc(sess, policy, obs, acts, idx):
    idx = np.asarray(idx, dtype=np.int32)
    n_tok = int(acts.shape[1])
    hits = 0
    n = 0
    for start in range(0, idx.shape[0], BC_BATCH):
        sl = idx[start : start + BC_BATCH]
        fl = np.full((sl.shape[0],), n_tok, dtype=np.int32)
        pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict={policy.obs: obs[sl], policy.decoder_full_length: fl},
        )
        pred, _ = align_greedy_pred(pred, n_tok)
        hits += int(np.sum(pred == acts[sl]))
        n += int(acts[sl].size)
    return float(hits) / float(max(n, 1))


def run_scheduled_bc(
    sess,
    env,
    core_policy,
    rng,
    run_dir=None,
    epochs=SS_EPOCHS,
    load_ckpt=None,
    save_ckpt=None,
    cache_path=None,
):
    import tensorflow as tf
    from meta_algos.variable_io import restore_trainable, snapshot_trainable

    n_epochs = int(epochs)
    if n_epochs != SS_EPOCHS:
        raise ValueError("scheduled BC epochs must be %d, got %s" % (SS_EPOCHS, n_epochs))
    obs, acts, stats, extras = collect_expert_dataset(env, cache_path=cache_path)
    n, n_tok = acts.shape
    gold_shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    if load_ckpt:
        core_policy.load_variables(str(load_ckpt), sess=sess)
        stats["load_ckpt"] = str(load_ckpt)
        print("bc_ss_load_ckpt %s" % load_ckpt)

    loss = tf.reduce_mean(core_policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(BC_LR, name="bc_ss_adam")
    frozen, adapt = encoder_frozen_adapt_vars(core_policy.get_trainable_variables())
    train_op = opt.minimize(loss, var_list=adapt)
    sess.run(tf.compat.v1.variables_initializer(opt.variables()))
    stats["encoder_frozen"] = True
    stats["n_adapt"] = int(len(adapt))
    stats["n_frozen"] = int(len(frozen))
    stats["ss_epochs"] = n_epochs
    stats["ss_eps_start"] = float(SS_EPS_START)
    stats["ss_eps_end"] = float(SS_EPS_END)
    stats["ppo"] = False

    probe = rng.choice(n, size=min(SS_PROBE, n), replace=False)
    acc0 = greedy_token_acc(sess, core_policy, obs, acts, probe)
    stats["greedy_token_acc_before"] = float(acc0)
    print("bc_ss_before greedy_token_acc=%.3f frozen=%d adapt=%d" % (acc0, len(frozen), len(adapt)))

    order = np.arange(n)
    epoch_losses = []
    epoch_eps = []
    epoch_acc = []
    best_acc = -1.0
    best_snap = None
    best_epoch = 0
    for epoch in range(n_epochs):
        eps = ss_eps(epoch, n_epochs)
        rng.shuffle(order)
        batch_losses = []
        for start in range(0, n, BC_BATCH):
            sl = order[start : start + BC_BATCH]
            fl = np.full((len(sl),), n_tok, dtype=np.int32)
            pred = sess.run(
                core_policy.network.greedy_decoder_prediction,
                feed_dict={
                    core_policy.obs: obs[sl],
                    core_policy.decoder_full_length: fl,
                },
            )
            pred, _ = align_greedy_pred(pred, n_tok)
            pred_shift = np.concatenate(
                [np.zeros((pred.shape[0], 1), dtype=np.int32), pred[:, :-1]], axis=1
            )
            mixed = mix_decoder_inputs(gold_shift[sl], pred_shift, eps, rng)
            _, lv = sess.run(
                [train_op, loss],
                feed_dict={
                    core_policy.obs: obs[sl],
                    core_policy.decoder_inputs: mixed,
                    core_policy.decoder_targets: acts[sl],
                    core_policy.decoder_full_length: fl,
                },
            )
            batch_losses.append(float(lv))
        cur = float(np.mean(batch_losses))
        acc = greedy_token_acc(sess, core_policy, obs, acts, probe)
        epoch_losses.append(cur)
        epoch_eps.append(float(eps))
        epoch_acc.append(float(acc))
        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_snap = snapshot_trainable(core_policy, sess=sess)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
            print(
                "bc_ss_epoch %d/%d eps=%.3f loss=%.4f greedy_token_acc=%.3f best=%.3f@%d"
                % (epoch + 1, n_epochs, eps, cur, acc, best_acc, best_epoch)
            )
    if best_snap is not None:
        restore_trainable(core_policy, best_snap, sess=sess)
        print("bc_ss_restore_best epoch=%d greedy_token_acc=%.3f" % (best_epoch, best_acc))

    stats["epoch_losses"] = epoch_losses
    stats["epoch_eps"] = epoch_eps
    stats["epoch_greedy_token_acc"] = epoch_acc
    stats["best_greedy_token_acc"] = float(best_acc)
    stats["best_epoch"] = int(best_epoch)
    rollout = greedy_rollout_eval(sess, env, core_policy, obs, acts, extras)
    stats["greedy_rollout"] = rollout
    print(
        "bc_ss_train T=%.1f expert=%.1f mec=%.1f token_vs_ex=%.3f mix L/M/V=%.3f/%.3f/%.3f"
        % (
            rollout["greedy_T_mean"],
            rollout["expert_T_mean"],
            rollout["all_mec_T_mean"],
            rollout["greedy_token_acc_vs_expert"],
            rollout["greedy_local_frac"],
            rollout["greedy_mec_frac"],
            rollout["greedy_v2v_frac"],
        )
    )
    if save_ckpt:
        ckpt_path = Path(save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        core_policy.save_variables(str(ckpt_path), sess=sess)
        stats["bc_ckpt"] = str(ckpt_path)
        print("bc_ss_ckpt %s" % ckpt_path)
    if run_dir is not None:
        path = Path(run_dir) / "bc_scheduled.json"
        path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        (Path(run_dir) / "greedy_rollout.json").write_text(
            json.dumps(rollout, indent=2, sort_keys=True) + "\n"
        )
    return stats
