"""EAS-style support adaptation of a small φ subset. Phase 3. No meta-test.

STATUS: method path (ADR-007 replacement). paper_result=false until Phase 6.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from spec.bc_greedy_mec import policy_feed
from spec.cavia_loop import _schedule_batch, greedy_actions, sample_actions_and_neglogp
from spec.split_loader import support_query_indices

ROOT = Path(__file__).resolve().parent.parent
CKPT_MEANAGG_MEAN = (
    ROOT
    / "runs"
    / "phase4"
    / "margo_v0.3_diag_encoder"
    / "meanagg_mean"
    / "seed_0"
    / "ckpt"
    / "bc_core.ckpt"
)
ADAPT_SUBSETS = ("lastlayer", "film", "emb")
LOSS_TYPES = ("pg_il", "pg", "il", "ce2opt")
LOG_ITERS = (0, 5, 10, 20, 50, 100)
DEFAULT_K = 16
DEFAULT_N_ADAPT = 100
DEFAULT_LR = 1e-3
DEFAULT_LAMBDA_IL = 1.0
SMOKE_N_ADAPT = 5
SMOKE_K = 4
# Phase 1 meanagg+mean seed0 greedy val (ADR-010). Schedule/batch noise ~0.3s.
IDENTITY_T_TOL = 0.5
IDENTITY_VAL_T_REF = 577.017539613315


def load_policy_partial(policy, ckpt_path, sess):
    """Restore overlapping vars from joblib ckpt; leave new φ vars at init."""
    import os

    import joblib
    import tensorflow as tf

    loaded = joblib.load(os.path.expanduser(str(ckpt_path)))
    restores = []
    missing = []
    if isinstance(loaded, list):
        variables = policy.get_variables()
        if len(loaded) != len(variables):
            raise ValueError(
                "list ckpt len %d != n_vars %d; use dict ckpt for partial EAS load"
                % (len(loaded), len(variables))
            )
        for d, v in zip(loaded, variables):
            restores.append(v.assign(d))
    else:
        for v in policy.get_variables():
            if v.name in loaded:
                restores.append(v.assign(loaded[v.name]))
            else:
                missing.append(v.name)
    if restores:
        sess.run(restores)
    return missing


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ints(xs):
    h = hashlib.sha256()
    h.update(np.asarray(xs, dtype=np.int32).tobytes())
    return h.hexdigest()


def advantages_pomo(ts, t_mec):
    """Per-graph shared baseline. A = -(T - mean_j T) / T_mec. mean_j A == 0."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError("ts rank %d want 2 [G,k]" % ts.ndim)
    t_mec = np.asarray(t_mec, dtype=np.float64).reshape(-1)
    if t_mec.shape[0] != ts.shape[0]:
        raise ValueError("t_mec len %d != G %d" % (t_mec.shape[0], ts.shape[0]))
    b = np.mean(ts, axis=1, keepdims=True)
    denom = np.maximum(t_mec.reshape(-1, 1), 1e-6)
    adv = -(ts - b) / denom
    return adv


def assert_adv_zero_mean(adv, atol=1e-6):
    m = np.mean(adv, axis=1)
    if np.max(np.abs(m)) > atol:
        raise ValueError("advantage mean_j not zero: max|mean|=%s" % float(np.max(np.abs(m))))


def filter_adapt_vars(policy, subset):
    """Return trainable Variables that constitute φ for the chosen subset."""
    subset = str(subset)
    all_vars = list(policy.get_trainable_variables())
    if subset == "lastlayer":
        chosen = [v for v in all_vars if "output_projection" in v.name]
    elif subset == "film":
        chosen = [
            v
            for v in all_vars
            if ("cavia_z" in v.name) or ("cavia_film" in v.name)
        ]
    elif subset == "emb":
        chosen = [v for v in all_vars if "eas_emb_delta" in v.name]
    elif subset == "full":
        chosen = list(all_vars)
    else:
        raise ValueError("adapt_subset %r not in %s|full" % (subset, ADAPT_SUBSETS))
    if not chosen:
        raise ValueError("adapt_subset %r matched 0 variables" % subset)
    return chosen


def n_params_vars(vars_):
    n = 0
    for v in vars_:
        shape = v.get_shape().as_list()
        prod = 1
        for d in shape:
            prod *= int(d or 1)
        n += prod
    return int(n)


def collect_mec_T(tgs, resources, refs_cache, cache_prefix):
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    out = []
    for i, tg in enumerate(tgs):
        key = (cache_prefix, i, "mec")
        if key not in refs_cache:
            order = [int(tid) for tid in tg.prioritize_sequence]
            mec, _, _ = schedule_via_adapter(
                tg, list(zip(order, [1] * len(order))), resources
            )
            refs_cache[key] = float(mec.makespan_seconds)
        out.append(refs_cache[key])
    return np.asarray(out, dtype=np.float64)


def _mix_nnon(actions):
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


def slice_dist(env, dist_id, indices):
    ids = [int(x) for x in env.distribution_ids]
    dist_id = int(dist_id)
    if dist_id not in ids:
        raise ValueError("dist %s not in env %s" % (dist_id, ids))
    di = ids.index(dist_id)
    idx = [int(i) for i in indices]
    graphs = env.task_graphs_batchs[di]
    enc = np.asarray(env.encoder_batchs[di], dtype=np.float32)
    tgs = [graphs[i] for i in idx]
    obs = enc[np.asarray(idx, dtype=np.int32)]
    return tgs, obs


def load_twopt_support_acts(cache_path, dist_id, support_idx):
    blob = np.load(str(cache_path))
    did = np.asarray(blob["dist_id"], dtype=np.int32)
    acts = np.asarray(blob["acts"], dtype=np.int32)
    mask = did == int(dist_id)
    dist_acts = acts[mask]
    idx = np.asarray(support_idx, dtype=np.int32)
    if int(np.max(idx)) >= dist_acts.shape[0]:
        raise ValueError(
            "support idx max %d >= dist rows %d" % (int(np.max(idx)), dist_acts.shape[0])
        )
    return dist_acts[idx]


class LatencyOnlyObjective(object):
    use_energy = False

    def cost(self, makespan_seconds, energy_joules, refs):
        t = float(makespan_seconds)
        return t, t, float(energy_joules or 0.0)


def eval_greedy_T(sess, policy, tgs, obs, resources, refs_cache, cache_prefix):
    acts = greedy_actions(sess, policy, obs)
    if int(acts.max()) > 2:
        raise ValueError("greedy token >2")
    _costs, ts, _es = _schedule_batch(
        tgs, acts, resources, LatencyOnlyObjective(), refs_cache, cache_prefix
    )
    return float(np.mean(ts)), acts, ts


def sample_k_and_neglogp(sess, policy, obs, k):
    """Sample k plans with matching token neglogp sums. [B,k,L], [B,k]."""
    k = int(k)
    plans = []
    nlls = []
    for _ in range(k):
        acts, tok_nll = sample_actions_and_neglogp(sess, policy, obs)
        plans.append(acts)
        nlls.append(np.sum(np.asarray(tok_nll, dtype=np.float64), axis=-1))
    return np.stack(plans, axis=1), np.stack(nlls, axis=1)


def build_eas_train_ops(policy, adapt_vars, lr=DEFAULT_LR):
    """PG and IL share train decoder; combined step = grads_pg + λ·grads_il."""
    import tensorflow as tf

    adv_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name="eas_adv")
    logp_sum = -tf.reduce_sum(policy.network.neglogp(), axis=1)
    loss_pg = -tf.reduce_mean(tf.stop_gradient(adv_ph) * logp_sum)
    loss_il = -tf.reduce_mean(logp_sum)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=float(lr), name="eas_adam")
    grads_pg = tf.gradients(loss_pg, adapt_vars)
    grads_il = tf.gradients(loss_il, adapt_vars)
    # Full-θ: disconnected vars → None; embedding/lookup → IndexedSlices.
    # Densify everything so sess.run always returns float arrays.
    def _dense(g, v):
        if g is None:
            return tf.zeros_like(v)
        if isinstance(g, tf.IndexedSlices):
            return tf.convert_to_tensor(g)
        return g

    grads_pg = [_dense(g, v) for g, v in zip(grads_pg, adapt_vars)]
    grads_il = [_dense(g, v) for g, v in zip(grads_il, adapt_vars)]
    # Placeholders to apply pre-summed numpy grads (one Adam step on L_PG+λ L_IL).
    grad_phs = [
        tf.compat.v1.placeholder(tf.float32, shape=v.get_shape(), name="eas_g_%d" % i)
        for i, v in enumerate(adapt_vars)
    ]
    apply = opt.apply_gradients(list(zip(grad_phs, adapt_vars)))
    return {
        "adv": adv_ph,
        "loss_pg": loss_pg,
        "loss_il": loss_il,
        "logp_sum": logp_sum,
        "grads_pg": grads_pg,
        "grads_il": grads_il,
        "grad_phs": grad_phs,
        "apply": apply,
        "opt": opt,
        "adapt_vars": adapt_vars,
    }


def apply_eas_step(sess, ops, fd_pg, fd_il, lambda_il, loss_type):
    """One Adam step. loss_type: pg_il | pg | il."""
    import numpy as np

    def _np(g, ph):
        if g is None:
            shape = [int(d) for d in ph.get_shape().as_list()]
            return np.zeros(shape, dtype=np.float32)
        # IndexedSlices / nested → flat dense float32
        arr = np.asarray(g, dtype=np.float32)
        return arr

    loss_type = str(loss_type)
    lam = float(lambda_il)
    if loss_type == "pg":
        g_pg, l_pg = sess.run([ops["grads_pg"], ops["loss_pg"]], feed_dict=fd_pg)
        g_pg = [_np(g, ph) for g, ph in zip(g_pg, ops["grad_phs"])]
        combined = g_pg
        l_il = 0.0
    elif loss_type == "il":
        g_il, l_il = sess.run([ops["grads_il"], ops["loss_il"]], feed_dict=fd_il)
        g_il = [_np(g, ph) for g, ph in zip(g_il, ops["grad_phs"])]
        combined = g_il
        l_pg = 0.0
    elif loss_type in ("pg_il", "ce2opt"):
        if loss_type == "ce2opt":
            g_il, l_il = sess.run([ops["grads_il"], ops["loss_il"]], feed_dict=fd_il)
            g_il = [_np(g, ph) for g, ph in zip(g_il, ops["grad_phs"])]
            combined = g_il
            l_pg = 0.0
        else:
            g_pg, l_pg = sess.run([ops["grads_pg"], ops["loss_pg"]], feed_dict=fd_pg)
            g_il, l_il = sess.run([ops["grads_il"], ops["loss_il"]], feed_dict=fd_il)
            combined = []
            for a, b, ph in zip(g_pg, g_il, ops["grad_phs"]):
                a = _np(a, ph)
                b = _np(b, ph)
                combined.append(a + lam * b)
    else:
        raise ValueError("loss_type %r" % loss_type)
    fd_apply = {ph: np.asarray(g, dtype=np.float32) for ph, g in zip(ops["grad_phs"], combined)}
    sess.run(ops["apply"], feed_dict=fd_apply)
    return float(l_pg), float(l_il)


def phi_l2(sess, adapt_vars, init_vals):
    import numpy as np

    cur = sess.run(adapt_vars)
    s = 0.0
    for a, b in zip(cur, init_vals):
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        s += float(np.sum(d * d))
    return float(np.sqrt(s))


def reset_adapt_state(sess, policy, adapt_vars, opt, subset, init_vals):
    import tensorflow as tf

    # restore φ to identity snapshot
    assigns = []
    for v, arr in zip(adapt_vars, init_vals):
        assigns.append(v.assign(arr))
    if assigns:
        sess.run(assigns)
    if subset == "film" and policy.network.cavia_z is not None:
        sess.run(policy.network.cavia_z.initializer)
    slot = opt.variables()
    if slot:
        sess.run(tf.compat.v1.variables_initializer(slot))


def snapshot_vars(sess, vars_):
    return sess.run(vars_)


def _teacher_feed(policy, ops, obs, acts, adv=None):
    acts = np.asarray(acts, dtype=np.int32)
    shift = np.zeros_like(acts)
    shift[:, 1:] = acts[:, :-1]
    fl = np.full((obs.shape[0],), obs.shape[1], dtype=np.int32)
    fd = policy_feed(policy, obs, fl, shift=shift, acts=acts)
    if adv is None:
        adv = np.zeros((obs.shape[0],), dtype=np.float32)
    fd[ops["adv"]] = np.asarray(adv, dtype=np.float32)
    return fd


def run_eas_on_dist(
    sess,
    policy,
    ops,
    adapt_vars,
    env,
    dist_id,
    resources,
    subset,
    loss_type,
    lambda_il,
    n_adapt,
    k,
    seed,
    init_vals,
    log_iters=LOG_ITERS,
    twopt_acts=None,
):
    support_idx, query_idx = support_query_indices(dist_id)
    if set(support_idx) & set(query_idx):
        raise ValueError("support/query overlap dist=%s" % dist_id)
    support_sha = sha256_ints(support_idx)
    query_sha = sha256_ints(query_idx)

    reset_adapt_state(sess, policy, adapt_vars, ops["opt"], subset, init_vals)

    refs = {}
    prefix = "d%d" % int(dist_id)
    s_tgs, s_obs = slice_dist(env, dist_id, support_idx)
    q_tgs, q_obs = slice_dist(env, dist_id, query_idx)
    t_mec = collect_mec_T(s_tgs, resources, refs, prefix)

    # best_d[g] init = greedy
    g_T, g_acts, g_ts = eval_greedy_T(
        sess, policy, s_tgs, s_obs, resources, refs, prefix + "|g0"
    )
    best_acts = np.asarray(g_acts, dtype=np.int32).copy()
    best_T = np.asarray(g_ts, dtype=np.float64).copy()
    best_T0_mean = float(np.mean(best_T))

    q0_T, q0_acts, _ = eval_greedy_T(
        sess, policy, q_tgs, q_obs, resources, refs, prefix + "|q0"
    )
    curve = []
    schedule_counts = []
    last_losses = {"L_PG": None, "L_IL": None}

    def _log_row(it):
        qT, qacts, _ = eval_greedy_T(
            sess, policy, q_tgs, q_obs, resources, refs, prefix + "|qit%d" % it
        )
        mix = _mix_nnon(qacts)
        row = {
            "iter": int(it),
            "query_greedy_T": float(qT),
            "support_best_T_mean": float(np.mean(best_T)),
            "support_greedy_T0": float(g_T),
            "phi_l2": phi_l2(sess, adapt_vars, init_vals),
            "L_PG": last_losses["L_PG"],
            "L_IL": last_losses["L_IL"],
            **mix,
        }
        curve.append(row)
        return row

    _log_row(0)

    loss_type = str(loss_type)
    for it in range(1, int(n_adapt) + 1):
        prev_best_mean = float(np.mean(best_T))
        if loss_type == "ce2opt":
            if twopt_acts is None:
                raise ValueError("ce2opt needs twopt_acts [G,L]")
            schedule_counts.append(0)  # no sample schedules in update
            fd_il = _teacher_feed(policy, ops, s_obs, twopt_acts)
            fd_pg = fd_il  # unused
            l_pg, l_il = apply_eas_step(sess, ops, fd_pg, fd_il, lambda_il, "ce2opt")
        else:
            plans, _nll = sample_k_and_neglogp(sess, policy, s_obs, k)
            G, K, L = plans.shape
            flat_plans = plans.reshape(G * K, L)
            flat_tgs = []
            for i in range(G):
                flat_tgs.extend([s_tgs[i]] * K)
            schedule_counts.append(int(G * K))
            _c, ts_flat, _e = _schedule_batch(
                flat_tgs,
                flat_plans,
                resources,
                LatencyOnlyObjective(),
                refs,
                prefix + "|it%d" % it,
            )
            ts = ts_flat.reshape(G, K)
            min_j = np.argmin(ts, axis=1)
            min_T = ts[np.arange(G), min_j]
            improved = min_T + 1e-12 < best_T
            for i in np.where(improved)[0]:
                best_T[i] = min_T[i]
                best_acts[i] = plans[i, min_j[i]]
            if float(np.mean(best_T)) > prev_best_mean + 1e-9:
                raise ValueError("best_T increased — bookkeeping bug")

            adv = advantages_pomo(ts, t_mec)
            assert_adv_zero_mean(adv)
            flat_adv = adv.reshape(-1).astype(np.float32)
            flat_obs = np.repeat(s_obs, K, axis=0)
            fd_pg = _teacher_feed(policy, ops, flat_obs, flat_plans, adv=flat_adv)
            fd_il = _teacher_feed(policy, ops, s_obs, best_acts)
            l_pg, l_il = apply_eas_step(sess, ops, fd_pg, fd_il, lambda_il, loss_type)

        last_losses["L_PG"] = float(l_pg)
        last_losses["L_IL"] = float(l_il)
        if not np.isfinite(l_pg) or not np.isfinite(l_il):
            raise ValueError("non-finite loss at it=%d L_PG=%s L_IL=%s" % (it, l_pg, l_il))

        if it in log_iters:
            _log_row(it)

    # final query
    qT, qacts, _ = eval_greedy_T(
        sess, policy, q_tgs, q_obs, resources, refs, prefix + "|qfinal"
    )
    mix = _mix_nnon(qacts)
    if loss_type != "ce2opt":
        for c in schedule_counts:
            if c != len(support_idx) * int(k):
                raise ValueError("schedule count %d != 20*k=%d" % (c, 20 * int(k)))

    return {
        "dist_id": int(dist_id),
        "support_n": len(support_idx),
        "query_n": len(query_idx),
        "support_sha256": support_sha,
        "query_sha256": query_sha,
        "query_greedy_T0": float(q0_T),
        "query_greedy_T": float(qT),
        "support_best_T0": float(best_T0_mean),
        "support_best_T_final": float(np.mean(best_T)),
        "curve": curve,
        "mix": mix,
        "schedule_per_iter": int(len(support_idx) * int(k)) if loss_type != "ce2opt" else 0,
        "n_adapt": int(n_adapt),
        "k": int(k),
        "phi_l2_final": phi_l2(sess, adapt_vars, init_vals),
    }


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
