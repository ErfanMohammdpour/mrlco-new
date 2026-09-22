"""Greedy-from-MEC behavior cloning. Diagnostic only. Not the frozen 3500 primary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BC_EPOCHS = 8
BC_ONLY_EPOCHS = 40
BC_CONTINUE_MAX_EPOCHS = 80
BC_CONTINUE_PATIENCE = 10
BC_CONTINUE_MIN_DELTA = 0.005
BC_BATCH = 32
BC_LR = 5e-4
BC_MAX_PASSES = 2
GREEDY_PAD_ACTION = 1  # leftover tokens stay MEC if TF still truncates


def collect_reachability(env):
    from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag
    from env.mec_offloaing_envs.scheduler.encoder_obs import reachability_mask

    rows = []
    for graphs in env.task_graphs_batchs:
        for tg in graphs:
            dag = to_canonical_dag(tg)
            order = [int(tid) for tid in tg.prioritize_sequence]
            rows.append(reachability_mask(dag, order))
    return np.stack(rows, axis=0).astype(np.float32)


def policy_feed(core_policy, obs, fl, shift=None, acts=None, reach=None):
    fd = {
        core_policy.obs: obs,
        core_policy.decoder_full_length: fl,
    }
    if shift is not None:
        fd[core_policy.decoder_inputs] = shift
    if acts is not None:
        fd[core_policy.decoder_targets] = acts
    ph = getattr(core_policy, "reachability_mask", None)
    if ph is not None and reach is not None:
        fd[ph] = reach
    # ⑥b feasibility shield: no-op unless MARGO_MASK_MODE enables it, in which
    # case every decode runs on the same masked distribution the PPO update uses.
    from policies.meta_seq2seq_policy import feasibility_feed

    fd.update(feasibility_feed(core_policy, obs))
    return fd


def collect_expert_dataset(env, cache_path=None):
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    n_expected = sum(len(graphs) for graphs in env.task_graphs_batchs)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.is_file():
        blob = np.load(str(cache_path), allow_pickle=False)
        obs = np.asarray(blob["obs"], dtype=np.float32)
        acts = np.asarray(blob["acts"], dtype=np.int32)
        t_ex = np.asarray(blob["t_expert"], dtype=np.float64)
        t_m = np.asarray(blob["t_mec"], dtype=np.float64)
        n_non = np.asarray(blob["n_non"], dtype=np.int32)
        if obs.shape[0] != n_expected or acts.shape[0] != n_expected:
            print("expert cache size mismatch; rebuilding")
        else:
            print("expert cache hit %s n=%d" % (cache_path, int(obs.shape[0])))
            counts = np.bincount(acts.reshape(-1), minlength=3).astype(np.float64)
            counts = counts / max(float(counts.sum()), 1.0)
            stats = {
                "n_graphs": int(obs.shape[0]),
                "n_nonmec_mean": float(np.mean(n_non)),
                "n_nonmec_p50": float(np.median(n_non)),
                "frac_n_nonmec_le3": float(np.mean(n_non <= 3)),
                "expert_local_frac": float(counts[0]),
                "expert_mec_frac": float(counts[1]),
                "expert_v2v_frac": float(counts[2]),
                "expert_T_mean": float(np.mean(t_ex)),
                "all_mec_T_mean": float(np.mean(t_m)),
                "frac_expert_beats_mec": float(np.mean(t_ex + 1e-12 < t_m)),
                "bc_epochs": BC_EPOCHS,
                "bc_batch": BC_BATCH,
                "bc_max_passes": BC_MAX_PASSES,
                "expert_cache": str(cache_path),
            }
            extras = {"t_expert": t_ex, "t_mec": t_m, "n_non": n_non}
            if "dist_id" in blob.files:
                extras["dist_id"] = np.asarray(blob["dist_id"], dtype=np.int32)
            return obs, acts, stats, extras

    obs_rows = []
    act_rows = []
    n_non = []
    t_expert = []
    t_mec = []
    dist_ids = []
    n_graphs = 0
    dist_id_list = list(getattr(env, "distribution_ids", []) or [])
    for dist_i, graphs in enumerate(env.task_graphs_batchs):
        enc = np.asarray(env.encoder_batchs[dist_i])
        dist_id = int(dist_id_list[dist_i]) if dist_i < len(dist_id_list) else dist_i
        for g_i, tg in enumerate(graphs):
            plan, result = greedy_from_mec_plan(
                tg, env.scheduler_resources, max_passes=BC_MAX_PASSES
            )
            actions = [int(a) for _, a in plan]
            if len(actions) != int(tg.task_number):
                raise ValueError("expert plan length mismatch")
            order = [int(tid) for tid in tg.prioritize_sequence]
            mec_res, _, _ = schedule_via_adapter(
                tg, list(zip(order, [1] * len(actions))), env.scheduler_resources
            )
            obs_rows.append(np.asarray(enc[g_i], dtype=np.float32))
            act_rows.append(np.asarray(actions, dtype=np.int32))
            n_non.append(sum(1 for a in actions if a != 1))
            t_expert.append(float(result.makespan_seconds))
            t_mec.append(float(mec_res.makespan_seconds))
            dist_ids.append(dist_id)
            n_graphs += 1
    obs = np.stack(obs_rows, axis=0)
    acts = np.stack(act_rows, axis=0)
    counts = np.bincount(acts.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    t_ex = np.asarray(t_expert, dtype=np.float64)
    t_m = np.asarray(t_mec, dtype=np.float64)
    stats = {
        "n_graphs": int(n_graphs),
        "n_nonmec_mean": float(np.mean(n_non)),
        "n_nonmec_p50": float(np.median(n_non)),
        "frac_n_nonmec_le3": float(np.mean(np.asarray(n_non) <= 3)),
        "expert_local_frac": float(counts[0]),
        "expert_mec_frac": float(counts[1]),
        "expert_v2v_frac": float(counts[2]),
        "expert_T_mean": float(np.mean(t_ex)),
        "all_mec_T_mean": float(np.mean(t_m)),
        "frac_expert_beats_mec": float(np.mean(t_ex + 1e-12 < t_m)),
        "bc_epochs": BC_EPOCHS,
        "bc_batch": BC_BATCH,
        "bc_max_passes": BC_MAX_PASSES,
    }
    extras = {
        "t_expert": t_ex,
        "t_mec": t_m,
        "n_non": np.asarray(n_non, dtype=np.int32),
        "dist_id": np.asarray(dist_ids, dtype=np.int32),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            obs=obs,
            acts=acts,
            t_expert=t_ex,
            t_mec=t_m,
            n_non=extras["n_non"],
            dist_id=extras["dist_id"],
        )
        stats["expert_cache"] = str(cache_path)
        print("expert cache write %s" % cache_path)
    return obs, acts, stats, extras


def align_greedy_pred(pred, n_tok, pad_value=GREEDY_PAD_ACTION):
    """Force greedy sample_id to (batch, n_tok). TF may stop early on end_token."""
    pred = np.asarray(pred, dtype=np.int32)
    if pred.ndim != 2:
        raise ValueError("greedy pred rank %d, want 2" % pred.ndim)
    n_tok = int(n_tok)
    t = int(pred.shape[1])
    if t == n_tok:
        return pred, False
    if t > n_tok:
        return pred[:, :n_tok], True
    pad = np.full((pred.shape[0], n_tok - t), int(pad_value), dtype=np.int32)
    return np.concatenate([pred, pad], axis=1), True


def greedy_rollout_eval(sess, env, core_policy, obs, expert_acts, extras):
    """Argmax greedy decode then schedule. Not teacher-forced token acc."""
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    n, n_tok = expert_acts.shape
    preds = []
    n_trunc = 0
    reach_all = extras.get("reach") if extras is not None else None
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        reach = None if reach_all is None else reach_all[sl]
        greedy_pred = sess.run(
            core_policy.network.greedy_decoder_prediction,
            feed_dict=policy_feed(core_policy, obs[sl], fl, reach=reach),
        )
        aligned, trunc = align_greedy_pred(greedy_pred, n_tok)
        n_trunc += int(trunc)
        preds.append(aligned)
    pred = np.concatenate(preds, axis=0)
    if n_trunc:
        print("greedy_decode_align_batches %d" % n_trunc)
    if pred.shape != expert_acts.shape:
        raise ValueError("greedy decode shape %s != expert %s" % (pred.shape, expert_acts.shape))

    t_policy = []
    n_non = []
    idx = 0
    for graphs in env.task_graphs_batchs:
        for tg in graphs:
            actions = [int(a) for a in pred[idx].tolist()]
            order = [int(tid) for tid in tg.prioritize_sequence]
            if len(actions) != len(order):
                raise ValueError("greedy plan length mismatch")
            result, _, _ = schedule_via_adapter(
                tg, list(zip(order, actions)), env.scheduler_resources
            )
            t_policy.append(float(result.makespan_seconds))
            n_non.append(sum(1 for a in actions if a != 1))
            idx += 1
    if idx != n:
        raise ValueError("greedy eval graph count %d != dataset %d" % (idx, n))

    t_p = np.asarray(t_policy, dtype=np.float64)
    t_ex = np.asarray(extras["t_expert"], dtype=np.float64)
    t_m = np.asarray(extras["t_mec"], dtype=np.float64)
    counts = np.bincount(pred.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non_a = np.asarray(n_non, dtype=np.float64)
    out = {
        "n_graphs": int(n),
        "greedy_token_acc_vs_expert": float(np.mean(pred == expert_acts)),
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
    }
    if extras.get("dist_id") is not None:
        did = np.asarray(extras["dist_id"]).reshape(-1)
        per = {}
        for d in np.unique(did):
            m = did == d
            per[str(int(d))] = {
                "n_graphs": int(np.sum(m)),
                "greedy_T_mean": float(np.mean(t_p[m])),
                "expert_T_mean": float(np.mean(t_ex[m])),
                "all_mec_T_mean": float(np.mean(t_m[m])),
                "greedy_over_expert": float(np.mean(t_p[m]) / max(float(np.mean(t_ex[m])), 1e-12)),
                "greedy_over_mec": float(np.mean(t_p[m]) / max(float(np.mean(t_m[m])), 1e-12)),
                "frac_greedy_beats_mec": float(np.mean(t_p[m] + 1e-12 < t_m[m])),
                "n_nonmec_p50": float(np.median(n_non_a[m])),
            }
        out["per_distribution"] = per
    return out


def eval_loaded_policy_on_env(sess, env, core_policy, cache_path=None, split_name=""):
    """Greedy decode + env T. No BC train. Diagnostic unseen check."""
    obs, acts, expert_stats, extras = collect_expert_dataset(env, cache_path=cache_path)
    if getattr(core_policy, "reachability_mask", None) is not None and extras.get("reach") is None:
        extras["reach"] = collect_reachability(env)
    rollout = greedy_rollout_eval(sess, env, core_policy, obs, acts, extras)
    print(
        "bc_unseen split=%s n=%d T=%.1f expert=%.1f mec=%.1f over_ex=%.3f over_mec=%.3f "
        "beat_mec=%.3f token=%.3f mix L/M/V=%.3f/%.3f/%.3f n_non_p50=%.1f"
        % (
            split_name or "?",
            rollout["n_graphs"],
            rollout["greedy_T_mean"],
            rollout["expert_T_mean"],
            rollout["all_mec_T_mean"],
            rollout["greedy_over_expert"],
            rollout["greedy_over_mec"],
            rollout["frac_greedy_beats_mec"],
            rollout["greedy_token_acc_vs_expert"],
            rollout["greedy_local_frac"],
            rollout["greedy_mec_frac"],
            rollout["greedy_v2v_frac"],
            rollout["n_nonmec_p50"],
        )
    )
    return {
        "split": split_name,
        "expert": {
            "n_graphs": expert_stats["n_graphs"],
            "T_mean": expert_stats["expert_T_mean"],
            "all_mec_T_mean": expert_stats["all_mec_T_mean"],
            "mix_L_M_V": [
                expert_stats["expert_local_frac"],
                expert_stats["expert_mec_frac"],
                expert_stats["expert_v2v_frac"],
            ],
            "n_nonmec_p50": expert_stats["n_nonmec_p50"],
            "frac_expert_beats_mec": expert_stats["frac_expert_beats_mec"],
        },
        "greedy": rollout,
    }


def unseen_ok_for_kl(split_row):
    """Law-not-lookup gate. Fail = do not KL-refine a memorized train table."""
    g = split_row["greedy"]
    return bool(
        g["frac_greedy_beats_mec"] >= 0.5
        and g["greedy_over_mec"] < 1.0
        and g["greedy_mec_frac"] < 0.95
        and g["n_nonmec_p50"] >= 2.0
    )


def run_bc_greedy_mec(
    sess,
    env,
    core_policy,
    rng,
    run_dir=None,
    epochs=None,
    greedy_env_eval=False,
    save_ckpt=None,
    load_ckpt=None,
    cache_path=None,
    early_stop_patience=None,
    early_stop_min_delta=None,
    max_graphs=None,
):
    """Teacher-force core_policy on greedy-from-MEC labels. Init BC Adam slots."""
    import tensorflow as tf

    n_epochs = BC_EPOCHS if epochs is None else int(epochs)
    if n_epochs < 1:
        raise ValueError("epochs must be positive")
    obs, acts, stats, extras = collect_expert_dataset(env, cache_path=cache_path)
    n, n_tok = acts.shape
    if getattr(core_policy, "reachability_mask", None) is not None and extras.get("reach") is None:
        extras["reach"] = collect_reachability(env)
        if extras["reach"].shape[0] != n:
            raise ValueError("reachability n=%d != graphs %d" % (extras["reach"].shape[0], n))
    if max_graphs is not None:
        nkeep = min(int(max_graphs), n)
        obs = obs[:nkeep]
        acts = acts[:nkeep]
        for key in list(extras.keys()):
            val = extras[key]
            if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0 and int(val.shape[0]) == n:
                extras[key] = val[:nkeep]
        n = nkeep
        stats["n_graphs"] = int(n)
        stats["smoke_max_graphs"] = int(max_graphs)
    reach_all = extras.get("reach")
    shift = np.concatenate(
        [np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1
    )
    if load_ckpt:
        core_policy.load_variables(str(load_ckpt), sess=sess)
        stats["load_ckpt"] = str(load_ckpt)
        print("bc_load_ckpt %s" % load_ckpt)
    loss = tf.reduce_mean(core_policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(BC_LR, name="bc_greedy_adam")
    train_op = opt.minimize(loss, var_list=core_policy.get_trainable_variables())
    sess.run(tf.compat.v1.variables_initializer(opt.variables()))

    stats["bc_epochs"] = n_epochs
    order = np.arange(n)
    epoch_losses = []
    best_loss = float("inf")
    stall = 0
    stopped_at = n_epochs
    patience = None if early_stop_patience is None else int(early_stop_patience)
    min_delta = 0.0 if early_stop_min_delta is None else float(early_stop_min_delta)
    for epoch in range(n_epochs):
        rng.shuffle(order)
        batch_losses = []
        for start in range(0, n, BC_BATCH):
            sl = order[start : start + BC_BATCH]
            fl = np.full((len(sl),), n_tok, dtype=np.int32)
            reach = None if reach_all is None else reach_all[sl]
            _, lv = sess.run(
                [train_op, loss],
                feed_dict=policy_feed(
                    core_policy, obs[sl], fl, shift=shift[sl], acts=acts[sl], reach=reach
                ),
            )
            batch_losses.append(float(lv))
        epoch_losses.append(float(np.mean(batch_losses)))
        cur = epoch_losses[-1]
        if best_loss - cur >= min_delta:
            best_loss = cur
            stall = 0
        else:
            stall += 1
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
            print("bc_epoch %d/%d loss=%.4f stall=%d" % (epoch + 1, n_epochs, cur, stall))
        if patience is not None and stall >= patience:
            stopped_at = epoch + 1
            print("bc_early_stop epoch=%d best_loss=%.4f" % (stopped_at, best_loss))
            break
    stats["bc_epochs_ran"] = int(stopped_at)
    stats["bc_early_stop"] = bool(patience is not None and stopped_at < n_epochs)
    stats["bc_best_loss"] = float(best_loss) if best_loss < float("inf") else None

    teacher_hits = 0
    greedy_hits = 0
    n_tok_total = 0
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        reach = None if reach_all is None else reach_all[sl]
        teacher_pred, greedy_pred = sess.run(
            [
                core_policy.network.decoder_prediction,
                core_policy.network.greedy_decoder_prediction,
            ],
            feed_dict=policy_feed(
                core_policy, obs[sl], fl, shift=shift[sl], acts=acts[sl], reach=reach
            ),
        )
        greedy_pred, _ = align_greedy_pred(greedy_pred, n_tok)
        teacher_hits += int(np.sum(np.asarray(teacher_pred) == acts[sl]))
        greedy_hits += int(np.sum(greedy_pred == acts[sl]))
        n_tok_total += int(acts[sl].size)
    stats["token_acc"] = float(teacher_hits) / float(max(n_tok_total, 1))
    stats["greedy_decode_acc"] = float(greedy_hits) / float(max(n_tok_total, 1))
    stats["epoch_losses"] = epoch_losses
    if save_ckpt:
        ckpt_path = Path(save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        core_policy.save_variables(str(ckpt_path), sess=sess)
        stats["bc_ckpt"] = str(ckpt_path)
        print("bc_ckpt %s" % ckpt_path)
    if greedy_env_eval:
        rollout = greedy_rollout_eval(sess, env, core_policy, obs, acts, extras)
        stats["greedy_rollout"] = rollout
        print(
            "bc_greedy_rollout T=%.1f expert=%.1f mec=%.1f token_vs_ex=%.3f "
            "n_non_p50=%.1f le3=%.3f mix L/M/V=%.3f/%.3f/%.3f"
            % (
                rollout["greedy_T_mean"],
                rollout["expert_T_mean"],
                rollout["all_mec_T_mean"],
                rollout["greedy_token_acc_vs_expert"],
                rollout["n_nonmec_p50"],
                rollout["frac_n_nonmec_le3"],
                rollout["greedy_local_frac"],
                rollout["greedy_mec_frac"],
                rollout["greedy_v2v_frac"],
            )
        )
    if run_dir is not None:
        path = Path(run_dir) / "bc_pretrain.json"
        path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        if stats.get("greedy_rollout"):
            (Path(run_dir) / "greedy_rollout.json").write_text(
                json.dumps(stats["greedy_rollout"], indent=2, sort_keys=True) + "\n"
            )
    print(
        "bc_greedy_mec graphs=%d token_acc=%.3f greedy_acc=%.3f expert_MEC=%.3f "
        "n_non_p50=%.1f T_ex=%.1f T_mec=%.1f beat_mec=%.3f"
        % (
            stats["n_graphs"],
            stats["token_acc"],
            stats["greedy_decode_acc"],
            stats["expert_mec_frac"],
            stats["n_nonmec_p50"],
            stats["expert_T_mean"],
            stats["all_mec_T_mean"],
            stats["frac_expert_beats_mec"],
        )
    )
    return stats
