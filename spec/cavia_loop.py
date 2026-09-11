"""STATUS: baseline/ablation only (ADR-007)

CAVIA-on-z inner loop + greedy query eval. Diagnostic. Not the frozen 3500 primary.
Phase-1 default: encoder/decoder/FiLM kernel frozen, Adam updates only `cavia_z`.
Energy is an on/off cost switch; joules are always logged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import GREEDY_PAD_ACTION, align_greedy_pred
from spec.cavia_objective import (
    CAVIA_IDENTITY_T_VAL_HI,
    CAVIA_IDENTITY_T_VAL_LO,
    CAVIA_INNER_LR,
    CAVIA_INNER_STEPS,
    CAVIA_Z_DIM,
    CaviaObjective,
    centered_advantages,
    classify_cavia_verdict,
)
from spec.split_loader import support_query_tasks, validation_distribution_ids, meta_test_distribution_ids

EVAL_BATCH = 32
K_REPORT = (0, 5, 10, 20)


def _env_index(env, dist_id):
    ids = [int(x) for x in env.distribution_ids]
    dist_id = int(dist_id)
    if dist_id not in ids:
        raise ValueError("dist %s not in env %s" % (dist_id, ids))
    return ids.index(dist_id)


def _slice_graphs(env, dist_id, graph_indices):
    di = _env_index(env, dist_id)
    idx = [int(i) for i in graph_indices]
    graphs = env.task_graphs_batchs[di]
    enc = np.asarray(env.encoder_batchs[di], dtype=np.float32)
    tgs = [graphs[i] for i in idx]
    obs = enc[np.asarray(idx, dtype=np.int32)]
    return tgs, obs


def _schedule_batch(tgs, actions, resources, objective, refs_cache, cache_prefix):
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter
    from env.mec_offloaing_envs.scheduler.energy_api import compute_reference_ranges

    actions = np.asarray(actions, dtype=np.int32)
    costs = []
    ts = []
    es = []
    for i, tg in enumerate(tgs):
        order = [int(tid) for tid in tg.prioritize_sequence]
        acts = [int(a) for a in actions[i].tolist()]
        if len(acts) != len(order):
            raise ValueError("plan length %d != order %d" % (len(acts), len(order)))
        if any(a not in (0, 1, 2) for a in acts):
            raise ValueError("invalid plan actions %s" % acts)
        result, _, _ = schedule_via_adapter(tg, list(zip(order, acts)), resources)
        refs = None
        if objective.use_energy:
            key = (cache_prefix, i)
            if key not in refs_cache:
                refs_cache[key] = compute_reference_ranges(tg, resources)
            refs = refs_cache[key]
        cost, t, e = objective.cost(
            result.makespan_seconds, result.energy.total_mobile_joules, refs
        )
        costs.append(cost)
        ts.append(t)
        es.append(e)
    return (
        np.asarray(costs, dtype=np.float64),
        np.asarray(ts, dtype=np.float64),
        np.asarray(es, dtype=np.float64),
    )


def _mix_and_nnon(actions):
    acts = np.asarray(actions, dtype=np.int32)
    flat = acts.reshape(-1)
    counts = np.bincount(flat, minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non = np.sum(acts != 1, axis=-1).astype(np.float64)
    return {
        "local_frac": float(counts[0]),
        "mec_frac": float(counts[1]),
        "v2v_frac": float(counts[2]),
        "n_nonmec_mean": float(np.mean(n_non)),
        "n_nonmec_p50": float(np.median(n_non)),
    }


def greedy_actions(sess, policy, obs):
    n = int(obs.shape[0])
    n_tok = int(obs.shape[1])
    rows = []
    for start in range(0, n, EVAL_BATCH):
        sl = slice(start, min(start + EVAL_BATCH, n))
        fl = np.full((obs[sl].shape[0],), n_tok, dtype=np.int32)
        pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict={policy.obs: obs[sl], policy.decoder_full_length: fl},
        )
        aligned, _trunc = align_greedy_pred(pred, n_tok, pad_value=GREEDY_PAD_ACTION)
        rows.append(aligned)
    return np.concatenate(rows, axis=0)


def sample_actions_and_neglogp(sess, policy, obs):
    n = int(obs.shape[0])
    n_tok = int(obs.shape[1])
    fl = np.full((n,), n_tok, dtype=np.int32)
    acts, neglogp = sess.run(
        [policy.network.sample_decoder_prediction, policy.network.sample_neglogp],
        feed_dict={policy.obs: obs, policy.decoder_full_length: fl},
    )
    acts = np.asarray(acts, dtype=np.int32)
    if acts.shape[1] != n_tok:
        acts, _trunc = align_greedy_pred(acts, n_tok, pad_value=GREEDY_PAD_ACTION)
        # sampled neglogp length must match; if TF truncated, resample is safer
        if acts.shape[1] != n_tok or np.asarray(neglogp).shape[1] != n_tok:
            raise ValueError("sample decode length mismatch acts%s neglogp%s want %d" % (
                np.asarray(acts).shape, np.asarray(neglogp).shape, n_tok
            ))
    return acts, np.asarray(neglogp, dtype=np.float32)


def build_cavia_train_ops(policy, lr=CAVIA_INNER_LR):
    import tensorflow as tf

    if policy.network.cavia_z is None:
        raise ValueError("CAVIA train ops require enable_cavia=True")
    adv_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name="cavia_adv")
    # Teacher-force the sampled plan so logπ matches the scheduled actions.
    logp_sum = -tf.reduce_sum(policy.network.neglogp(), axis=1)
    loss = tf.reduce_mean(tf.stop_gradient(adv_ph) * logp_sum)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=float(lr), name="cavia_z_adam")
    train = opt.minimize(loss, var_list=policy.cavia_trainable_variables())
    return {"adv": adv_ph, "loss": loss, "train": train, "opt": opt, "logp_sum": logp_sum}


def reset_cavia_opt(sess, ops):
    import tensorflow as tf

    slot = ops["opt"].variables()
    if slot:
        sess.run(tf.compat.v1.variables_initializer(slot))


def cavia_inner_steps(
    sess,
    policy,
    ops,
    tgs,
    obs,
    resources,
    objective,
    refs_cache,
    cache_prefix,
    n_steps=CAVIA_INNER_STEPS,
    log_ks=K_REPORT,
):
    import tensorflow as tf

    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    policy.reset_cavia_z(sess=sess)
    slot = ops["opt"].variables()
    if slot:
        sess.run(tf.compat.v1.variables_initializer(slot))
    trace = []
    wanted = set(int(k) for k in log_ks)
    if 0 in wanted:
        g_acts = greedy_actions(sess, policy, obs)
        costs, ts, es = _schedule_batch(tgs, g_acts, resources, objective, refs_cache, cache_prefix + "|q0")
        row = {"k": 0, "cost_mean": float(np.mean(costs)), "T_mean": float(np.mean(ts)), "E_mean": float(np.mean(es))}
        row.update(_mix_and_nnon(g_acts))
        trace.append(row)
    for step in range(1, n_steps + 1):
        acts, _nlp = sample_actions_and_neglogp(sess, policy, obs)
        costs, ts, es = _schedule_batch(
            tgs, acts, resources, objective, refs_cache, cache_prefix + "|s%d" % step
        )
        adv = centered_advantages(costs)
        n_tok = int(obs.shape[1])
        n = int(obs.shape[0])
        fl = np.full((n,), n_tok, dtype=np.int32)
        shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
        _, loss_v, z_v = sess.run(
            [ops["train"], ops["loss"], policy.network.cavia_z],
            feed_dict={
                policy.obs: obs,
                policy.decoder_inputs: shift,
                policy.decoder_targets: acts,
                policy.decoder_full_length: fl,
                ops["adv"]: adv,
            },
        )
        if step in wanted:
            g_acts = greedy_actions(sess, policy, obs)
            gc, gt, ge = _schedule_batch(
                tgs, g_acts, resources, objective, refs_cache, cache_prefix + "|g%d" % step
            )
            row = {
                "k": int(step),
                "cost_mean": float(np.mean(gc)),
                "T_mean": float(np.mean(gt)),
                "E_mean": float(np.mean(ge)),
                "inner_loss": float(loss_v),
                "z_l2": float(np.linalg.norm(np.asarray(z_v, dtype=np.float64))),
                "sample_T_mean": float(np.mean(ts)),
                "sample_T_std": float(np.std(ts)),
                "adv_std": float(np.std(adv)),
            }
            row.update(_mix_and_nnon(g_acts))
            trace.append(row)
    z_final = sess.run(policy.network.cavia_z)
    return trace, np.asarray(z_final, dtype=np.float64)


def eval_split_greedy(sess, policy, env, objective, dist_ids, role, refs_cache):
    """Full-split greedy (all graphs in env for those dist ids)."""
    all_t = []
    all_e = []
    all_c = []
    all_acts = []
    per = {}
    for dist_id in dist_ids:
        di = _env_index(env, dist_id)
        graphs = env.task_graphs_batchs[di]
        enc = np.asarray(env.encoder_batchs[di], dtype=np.float32)
        tgs = list(graphs)
        obs = enc
        acts = greedy_actions(sess, policy, obs)
        costs, ts, es = _schedule_batch(
            tgs, acts, env.scheduler_resources, objective, refs_cache, "%s|d%s" % (role, dist_id)
        )
        mix = _mix_and_nnon(acts)
        per[str(int(dist_id))] = {
            "n_graphs": int(len(tgs)),
            "cost_mean": float(np.mean(costs)),
            "T_mean": float(np.mean(ts)),
            "E_mean": float(np.mean(es)),
            **mix,
        }
        all_t.append(ts)
        all_e.append(es)
        all_c.append(costs)
        all_acts.append(acts)
    tcat = np.concatenate(all_t)
    ecat = np.concatenate(all_e)
    ccat = np.concatenate(all_c)
    acat = np.concatenate(all_acts, axis=0)
    out = {
        "split": role,
        "n_graphs": int(tcat.size),
        "cost_mean": float(np.mean(ccat)),
        "T_mean": float(np.mean(tcat)),
        "E_mean": float(np.mean(ecat)),
        "per_distribution": per,
    }
    out.update(_mix_and_nnon(acat))
    print(
        "cavia_greedy split=%s n=%d cost=%.3f T=%.1f E=%.1f mix L/M/V=%.3f/%.3f/%.3f n_non_p50=%.1f"
        % (
            role,
            out["n_graphs"],
            out["cost_mean"],
            out["T_mean"],
            out["E_mean"],
            out["local_frac"],
            out["mec_frac"],
            out["v2v_frac"],
            out["n_nonmec_p50"],
        )
    )
    return out


def eval_query_after_adapt(sess, policy, env, dist_id, objective, refs_cache):
    support, query = support_query_tasks(_env_index(env, dist_id), dist_id)
    tgs, obs = _slice_graphs(env, dist_id, query["graph_indices"])
    acts = greedy_actions(sess, policy, obs)
    costs, ts, es = _schedule_batch(
        tgs, acts, env.scheduler_resources, objective, refs_cache, "query|d%s" % dist_id
    )
    mix = _mix_and_nnon(acts)
    return {
        "n_graphs": int(len(tgs)),
        "cost_mean": float(np.mean(costs)),
        "T_mean": float(np.mean(ts)),
        "E_mean": float(np.mean(es)),
        **mix,
    }


def run_cavia_phase1(
    sess,
    policy,
    ops,
    val_env,
    test_env,
    objective,
    n_steps=CAVIA_INNER_STEPS,
    run_dir=None,
    identity_lo=None,
    identity_hi=None,
):
    if int(getattr(policy.network, "cavia_z_dim", CAVIA_Z_DIM)) != CAVIA_Z_DIM:
        raise ValueError("cavia_z_dim must be %d" % CAVIA_Z_DIM)
    n_steps = int(n_steps)
    log_ks = tuple(sorted(set(int(x) for x in K_REPORT) | {0, n_steps}))
    ident_lo = CAVIA_IDENTITY_T_VAL_LO if identity_lo is None else float(identity_lo)
    ident_hi = CAVIA_IDENTITY_T_VAL_HI if identity_hi is None else float(identity_hi)
    refs_cache = {}
    policy.reset_cavia_z(sess=sess)
    val_k0 = eval_split_greedy(
        sess, policy, val_env, objective, validation_distribution_ids(), "validation_k0", refs_cache
    )
    test_k0 = eval_split_greedy(
        sess, policy, test_env, objective, meta_test_distribution_ids(), "meta_test_k0", refs_cache
    )
    ident_ok = ident_lo <= float(val_k0["T_mean"]) <= ident_hi
    print(
        "cavia_identity val_full_T0=%.1f window=[%.0f,%.0f] %s"
        % (
            val_k0["T_mean"],
            ident_lo,
            ident_hi,
            "PASS" if ident_ok else "FAIL",
        )
    )
    if not ident_ok:
        payload = {
            "paper_result": False,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_frozen": True,
            "cavia_z_dim": CAVIA_Z_DIM,
            "inner_steps": 0,
            "use_energy": bool(objective.use_energy),
            "latency_weight": float(objective.latency_weight),
            "energy_weight": float(objective.energy_weight),
            "validation_k0_full": val_k0,
            "meta_test_k0_full": test_k0,
            "validation": None,
            "meta_test": None,
            "verdict": "identity_fail",
            "note": "z=0 greedy T outside BC window; inner CAVIA skipped; not 3500",
        }
        print("cavia_phase1 verdict=identity_fail val_full_T0=%.1f" % val_k0["T_mean"])
        if run_dir is not None:
            path = Path(run_dir) / "cavia_eval.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    def _adapt_env(env, dist_ids, tag):
        per = {}
        t0s = []
        t20s = []
        c0s = []
        c20s = []
        loc20s = []
        for dist_id in dist_ids:
            support, _query = support_query_tasks(_env_index(env, dist_id), dist_id)
            tgs, obs = _slice_graphs(env, dist_id, support["graph_indices"])
            if len(tgs) != 20:
                raise ValueError("support must be 20 graphs, got %d" % len(tgs))
            policy.reset_cavia_z(sess=sess)
            q_k0 = eval_query_after_adapt(sess, policy, env, dist_id, objective, refs_cache)
            trace, z_final = cavia_inner_steps(
                sess,
                policy,
                ops,
                tgs,
                obs,
                env.scheduler_resources,
                objective,
                refs_cache,
                "%s|d%s" % (tag, dist_id),
                n_steps=n_steps,
                log_ks=log_ks,
            )
            q_k20 = eval_query_after_adapt(sess, policy, env, dist_id, objective, refs_cache)
            per[str(int(dist_id))] = {
                "support_trace": trace,
                "query_k0": q_k0,
                "query_k20": q_k20,
                "z_l2": float(np.linalg.norm(z_final)),
            }
            t0s.append(q_k0["T_mean"])
            t20s.append(q_k20["T_mean"])
            c0s.append(q_k0["cost_mean"])
            c20s.append(q_k20["cost_mean"])
            loc20s.append(q_k20["local_frac"])
            print(
                "cavia_dist tag=%s dist=%s queryT k0=%.1f k20=%.1f local20=%.3f z_l2=%.3f"
                % (tag, dist_id, q_k0["T_mean"], q_k20["T_mean"], q_k20["local_frac"], float(np.linalg.norm(z_final)))
            )
        t0 = float(np.mean(t0s))
        t20 = float(np.mean(t20s))
        loc20 = float(np.mean(loc20s))
        return {
            "query_T_k0": t0,
            "query_T_k20": t20,
            "query_cost_k0": float(np.mean(c0s)),
            "query_cost_k20": float(np.mean(c20s)),
            "query_local_k20": loc20,
            "per_distribution": per,
        }

    val_adapt = _adapt_env(val_env, validation_distribution_ids(), "validation")
    test_adapt = _adapt_env(test_env, meta_test_distribution_ids(), "meta_test")
    verdict = classify_cavia_verdict(
        val_adapt["query_T_k0"],
        val_adapt["query_T_k20"],
        val_adapt["query_local_k20"],
        identity_t=val_k0["T_mean"],
        identity_lo=ident_lo,
        identity_hi=ident_hi,
    )
    payload = {
        "paper_result": False,
        "ppo": False,
        "encoder_frozen": True,
        "decoder_frozen": True,
        "cavia_z_dim": CAVIA_Z_DIM,
        "inner_steps": int(n_steps),
        "use_energy": bool(objective.use_energy),
        "latency_weight": float(objective.latency_weight),
        "energy_weight": float(objective.energy_weight),
        "validation_k0_full": val_k0,
        "meta_test_k0_full": test_k0,
        "validation": val_adapt,
        "meta_test": test_adapt,
        "verdict": verdict,
        "note": "CAVIA-on-z; theta frozen; schedule is physics; not 3500",
    }
    print(
        "cavia_phase1 verdict=%s val_full_T0=%.1f queryT k0=%.1f k20=%.1f"
        % (verdict, val_k0["T_mean"], val_adapt["query_T_k0"], val_adapt["query_T_k20"])
    )
    if run_dir is not None:
        path = Path(run_dir) / "cavia_eval.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
