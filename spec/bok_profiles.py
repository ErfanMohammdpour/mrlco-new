"""Phase 4 best-of-k on BC-profiles ckpt + obs v2. No training. paper_result=false."""

from __future__ import annotations

import time

import numpy as np

from spec.best_of_k import (
    DEFAULT_TEMPERATURE,
    assert_monotone_k,
    crosscheck_twopt_score,
    prefix_scores,
    sample_plans,
    score_plans,
    summarize_k,
)
from spec.hamming2_probe import _gv_path
from spec.phase4_campaign import (
    DIAG_BOK_PROFILES_METHOD_ID,
    diag_bc_profiles_run_dir,
    diag_bok_profiles_run_dir,
)
from spec.resource_profiles import role_profile_ids

FROZEN_PROFILE_ID = "frozen_7_5_10"
EVAL_PROFILE_IDS = (FROZEN_PROFILE_ID, "p_5_5_10", "p_9_5_10")
BC_SEED0_FROZEN_GREEDY_T = 493.82
CKPT_MATCH_SLACK_S = 8.0
SAMPLING_HELP_S = 10.0


def bc_profiles_ckpt(seed):
    return diag_bc_profiles_run_dir(seed) / "ckpt" / "bc_core.ckpt"


def eval_profile_ids():
    ids = list(EVAL_PROFILE_IDS)
    leak = set(ids) & set(role_profile_ids("meta_test_heldout"))
    if leak:
        raise ValueError("bok_profiles eval leaked into meta-test: %s" % leak)
    return ids


def graphs_from_chunk(chunk):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

    cluster = chunk["cluster"]
    tgs = []
    n = int(chunk["obs"].shape[0])
    for i in range(n):
        gv = _gv_path(int(chunk["dist_id"][i]), int(chunk["graph_idx"][i]))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        tgs.append(tg)
    return tgs


def eval_bok_on_chunk(
    sess,
    policy,
    chunk,
    k_max,
    ks,
    seed,
    split_name,
    temperature=DEFAULT_TEMPERATURE,
):
    tgs = graphs_from_chunk(chunk)
    obs = np.asarray(chunk["obs"], dtype=np.float32)
    dist_ids = np.asarray(chunk["dist_id"], dtype=np.int32)
    expert_T = np.asarray(chunk["t_expert"], dtype=np.float64)
    t0 = time.time()
    plans = sample_plans(sess, policy, obs, k=int(k_max), temperature=temperature)
    sample_s = time.time() - t0
    if int(plans.max()) > 2:
        raise ValueError("end_token leaked: max token %s" % int(plans.max()))
    if plans.shape[-1] != int(obs.shape[1]):
        raise ValueError("plan length %s != obs %s" % (plans.shape, obs.shape))
    t1 = time.time()
    scores = score_plans(
        tgs, plans, chunk["resources"], refs_cache={}, cache_prefix=split_name
    )
    score_s = time.time() - t1
    wall_s = sample_s + score_s
    assert_monotone_k(scores, ks=tuple(k for k in ks if k <= int(k_max)))
    greedy_T = float(np.mean(scores[:, 0]))
    best1 = float(np.mean(np.min(prefix_scores(scores, 1), axis=1)))
    if abs(best1 - greedy_T) > 0.1:
        raise ValueError("T_best_1=%.4f != T_greedy=%.4f (sample0 != greedy)" % (best1, greedy_T))
    pid = chunk["profile_id"][0]
    if isinstance(pid, bytes):
        pid = pid.decode("utf-8")
    rng = np.random.RandomState(int(seed) + 17)
    checked = crosscheck_twopt_score(tgs, plans, scores, chunk["resources"], rng)
    rows = {}
    for k in ks:
        if k > int(k_max):
            continue
        rows[str(k)] = summarize_k(
            plans,
            scores,
            k,
            expert_T=expert_T,
            dist_ids=dist_ids,
            wall_s=wall_s * (float(k) / float(k_max)),
        )
    return {
        "split": split_name,
        "profile_id": str(pid),
        "n_graphs": int(len(tgs)),
        "k_max": int(k_max),
        "temperature": float(temperature),
        "T_greedy": greedy_T,
        "T_best_1": best1,
        "expert_T_mean": float(np.mean(expert_T)),
        "sample_s": float(sample_s),
        "score_s": float(score_s),
        "wall_clock_s": float(wall_s),
        "seconds_per_graph": float(wall_s) / max(float(len(tgs)), 1.0),
        "crosscheck_graphs": checked,
        "by_k": rows,
        "evals_per_graph": int(k_max),
    }


assert DIAG_BOK_PROFILES_METHOD_ID == "margo_v0.3_bok_profiles"
assert diag_bok_profiles_run_dir(0).name == "seed_0"
assert FROZEN_PROFILE_ID in EVAL_PROFILE_IDS
