"""Per-instance EAS at test time. Path B after ADR-011. Not meta. paper_result=false."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec.cavia_loop import _schedule_batch, greedy_actions, sample_actions_and_neglogp
from spec.eas_adapt import (
    CKPT_MEANAGG_MEAN,
    LatencyOnlyObjective,
    _mix_nnon,
    _teacher_feed,
    advantages_pomo,
    apply_eas_step,
    assert_adv_zero_mean,
    build_eas_train_ops,
    collect_mec_T,
    dump_json,
    eval_greedy_T,
    filter_adapt_vars,
    load_policy_partial,
    n_params_vars,
    phi_l2,
    reset_adapt_state,
    sha256_file,
    snapshot_vars,
)

ROOT = Path(__file__).resolve().parent.parent

# Budget B = N_adapt * k. Match best-of-B.
BUDGET_PRESETS = {
    8: {"n_adapt": 2, "k": 4},
    32: {"n_adapt": 2, "k": 16},
    64: {"n_adapt": 4, "k": 16},
}


def budget_nk(budget):
    budget = int(budget)
    if budget not in BUDGET_PRESETS:
        raise ValueError("budget %s not in %s" % (budget, sorted(BUDGET_PRESETS)))
    p = BUDGET_PRESETS[budget]
    if int(p["n_adapt"]) * int(p["k"]) != budget:
        raise ValueError("preset math broken for budget %s" % budget)
    return int(p["n_adapt"]), int(p["k"])


def iter_val_graphs(env, max_graphs=None):
    """Yield (dist_id, local_idx, tg, obs[1,L,D]) over validation env order."""
    ids = [int(x) for x in env.distribution_ids]
    n_out = 0
    for di, dist_id in enumerate(ids):
        graphs = env.task_graphs_batchs[di]
        enc = np.asarray(env.encoder_batchs[di], dtype=np.float32)
        for li in range(len(graphs)):
            if max_graphs is not None and n_out >= int(max_graphs):
                return
            yield int(dist_id), int(li), graphs[li], enc[li : li + 1]
            n_out += 1


def run_eas_one_graph(
    sess,
    policy,
    ops,
    adapt_vars,
    init_vals,
    tg,
    obs,
    resources,
    subset,
    loss_type,
    lambda_il,
    n_adapt,
    k,
    cache_prefix,
):
    """Adapt φ on one graph. Return best T among all N*k samples + greedy after."""
    reset_adapt_state(sess, policy, adapt_vars, ops["opt"], subset, init_vals)
    refs = {}
    t_mec = collect_mec_T([tg], resources, refs, cache_prefix)
    # init best = greedy
    g_T, g_acts, _ = eval_greedy_T(
        sess, policy, [tg], obs, resources, refs, cache_prefix + "|g0"
    )
    best_T = float(g_T)
    best_acts = np.asarray(g_acts, dtype=np.int32).copy()
    n_sched = 0
    all_sample_T = []

    for it in range(1, int(n_adapt) + 1):
        plans = []
        for _ in range(int(k)):
            a, _ = sample_actions_and_neglogp(sess, policy, obs)
            plans.append(np.asarray(a, dtype=np.int32).reshape(-1))
        plans = np.stack(plans, axis=0)  # [k,L]
        flat_tgs = [tg] * int(k)
        _c, ts_flat, _e = _schedule_batch(
            flat_tgs,
            plans,
            resources,
            LatencyOnlyObjective(),
            refs,
            cache_prefix + "|it%d" % it,
        )
        n_sched += int(k)
        ts = ts_flat.reshape(1, k)
        all_sample_T.extend([float(x) for x in ts_flat])
        min_j = int(np.argmin(ts[0]))
        if float(ts[0, min_j]) + 1e-12 < best_T:
            best_T = float(ts[0, min_j])
            best_acts = plans[min_j].copy()

        adv = advantages_pomo(ts, t_mec)
        assert_adv_zero_mean(adv)
        flat_adv = adv.reshape(-1).astype(np.float32)
        flat_obs = np.repeat(obs, k, axis=0)
        fd_pg = _teacher_feed(policy, ops, flat_obs, plans, adv=flat_adv)
        fd_il = _teacher_feed(policy, ops, obs, best_acts.reshape(1, -1))
        l_pg, l_il = apply_eas_step(sess, ops, fd_pg, fd_il, lambda_il, loss_type)
        if not np.isfinite(l_pg) or not np.isfinite(l_il):
            raise ValueError("non-finite loss graph=%s it=%d" % (cache_prefix, it))

    # greedy after φ*
    q_T, q_acts, _ = eval_greedy_T(
        sess, policy, [tg], obs, resources, refs, cache_prefix + "|g*"
    )
    n_sched_final_greedy = 1
    mix = _mix_nnon(q_acts)
    return {
        "T_greedy0": float(g_T),
        "T_greedy_after": float(q_T),
        "T_best_among_samples": float(best_T),
        "T_best_samples_only": float(min(all_sample_T)) if all_sample_T else float(best_T),
        "n_schedule_adapt": int(n_sched),
        "n_schedule_greedy_final": int(n_sched_final_greedy),
        "phi_l2": phi_l2(sess, adapt_vars, init_vals),
        "mix_after": mix,
        "L_PG_last": float(l_pg),
        "L_IL_last": float(l_il),
    }


def run_bok_one_graph(sess, policy, tg, obs, resources, budget, cache_prefix):
    """Frozen best-of-budget: slot0 greedy + (budget-1) samples."""
    refs = {}
    g_acts = np.asarray(greedy_actions(sess, policy, obs), dtype=np.int32).reshape(-1)
    plans = [g_acts]
    for _ in range(max(0, int(budget) - 1)):
        a, _ = sample_actions_and_neglogp(sess, policy, obs)
        plans.append(np.asarray(a, dtype=np.int32).reshape(-1))
    plans = np.stack(plans, axis=0)  # [B,L]
    if plans.ndim != 2:
        raise ValueError("bok plans rank %d want 2" % plans.ndim)
    flat_tgs = [tg] * int(budget)
    _c, ts, _e = _schedule_batch(
        flat_tgs,
        plans,
        resources,
        LatencyOnlyObjective(),
        refs,
        cache_prefix + "|bok",
    )
    return {
        "T_greedy": float(ts[0]),
        "T_best": float(np.min(ts)),
        "n_schedule": int(budget),
    }


def summarize_rows(rows, budget):
    t0 = np.asarray([r["eas"]["T_greedy0"] for r in rows], dtype=np.float64)
    t_best = np.asarray([r["eas"]["T_best_among_samples"] for r in rows], dtype=np.float64)
    t_gstar = np.asarray([r["eas"]["T_greedy_after"] for r in rows], dtype=np.float64)
    t_bok = np.asarray([r["bok"]["T_best"] for r in rows], dtype=np.float64)
    t_bok_g = np.asarray([r["bok"]["T_greedy"] for r in rows], dtype=np.float64)
    return {
        "n_graphs": len(rows),
        "budget": int(budget),
        "T_greedy0_mean": float(np.mean(t0)),
        "T_eas_best_mean": float(np.mean(t_best)),
        "T_eas_greedy_after_mean": float(np.mean(t_gstar)),
        "T_bok_best_mean": float(np.mean(t_bok)),
        "T_bok_greedy_mean": float(np.mean(t_bok_g)),
        "delta_eas_minus_bok": float(np.mean(t_best) - np.mean(t_bok)),
        "frac_eas_beats_bok": float(np.mean(t_best + 1e-12 < t_bok)),
        "gate_eas_better_by_5s": bool(float(np.mean(t_bok) - np.mean(t_best)) >= 5.0),
    }
