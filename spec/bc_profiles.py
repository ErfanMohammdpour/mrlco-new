"""Phase 4 BC on per-profile 2-opt labels + obs v2. paper_result=false."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

# Obs v2 must be active before policy/env imports in the train driver.
os.environ.setdefault("MARGO_OBS_VERSION", "v2")

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
from env.mec_offloaing_envs.scheduler import encoder_obs as eo
from env.mec_offloaing_envs.scheduler.encoder_obs import (
    encode_task_graph,
    set_obs_version,
)
from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter
from spec.bc_greedy_mec import (
    BC_BATCH,
    BC_CONTINUE_MIN_DELTA,
    BC_CONTINUE_PATIENCE,
    BC_LR,
    GREEDY_PAD_ACTION,
    policy_feed,
)
from spec.expert_profiles import profile_cache_path
from spec.hamming2_probe import _gv_path
from spec.phase4_campaign import (
    DIAG_BC_PROFILES_METHOD_ID,
    DIAG_ENC_MAX_EPOCHS,
    DIAG_ENC_SMOKE_EPOCHS,
    DIAG_ENC_SMOKE_GRAPHS,
    META_TRAIN_IDS,
    RUNS_ROOT,
    VALIDATION_IDS,
    diag_bc_profiles_run_dir,
    provenance_template,
    require_gpu_permission,
)
from spec.resource_profiles import (
    resource_config_for_profile,
    resources_cluster_for_profile,
    role_profile_ids,
)
from spec.split_loader import meta_train_distribution_ids, validation_distribution_ids

COMBINED_TRAIN = Path(RUNS_ROOT) / "bc_profiles_train_obs_v2.npz"
PHASE1_MEANAGG_MEAN_VAL_T = 577.02  # ADR-010 seed0 identity / Phase1 winner val greedy
CONTINUITY_SLACK_S = 10.0


def continuity_no_regression(greedy_t, phase1_t=PHASE1_MEANAGG_MEAN_VAL_T, slack=CONTINUITY_SLACK_S):
    """One-sided: extra features must not *worsen* frozen val vs Phase 1.

    Two-sided |Δ|≤10 rejected a real improvement (seed0 T=493.8 vs 577.0).
    """
    t = float(greedy_t)
    ref = float(phase1_t)
    delta = t - ref
    return {
        "continuity_delta_vs_phase1": float(delta),
        "continuity_pass": bool(t <= ref + float(slack)),
        "continuity_rule": "no_regression_T_le_phase1_plus_%.0f" % float(slack),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dist_list(split: str):
    if split == "meta_train":
        return list(meta_train_distribution_ids())
    if split == "validation":
        return list(validation_distribution_ids())
    raise ValueError(split)


def build_profile_obs_acts(
    split: str,
    profile_id: str,
    max_graphs=None,
):
    """Encode graphs under profile resources; pair with 2-opt acts from cache."""
    set_obs_version("v2")
    if eo.FEATURE_DIM != 15 or eo.PACKED_DIM != 54:
        raise RuntimeError("need obs v2: dim=%s packed=%s" % (eo.FEATURE_DIM, eo.PACKED_DIM))
    cache = profile_cache_path(split, profile_id)
    if not cache.is_file():
        raise FileNotFoundError("profile cache missing: %s" % cache)
    blob = np.load(str(cache), allow_pickle=False)
    acts_all = np.asarray(blob["acts"], dtype=np.int32)
    dist_all = np.asarray(blob["dist_id"], dtype=np.int32)
    gidx_all = np.asarray(blob["graph_idx"], dtype=np.int32)
    t_ex = np.asarray(blob["t_expert"], dtype=np.float64)
    t_mec = np.asarray(blob["t_mec"], dtype=np.float64)
    n_non = np.asarray(blob["n_non"], dtype=np.int32)
    n = int(acts_all.shape[0])
    if max_graphs is not None:
        n = min(n, int(max_graphs))
    cluster = resources_cluster_for_profile(profile_id)
    resources = resource_config_for_profile(profile_id)
    obs_rows = []
    for i in range(n):
        dist_id = int(dist_all[i])
        graph_idx = int(gidx_all[i])
        gv = _gv_path(dist_id, graph_idx)
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        order = [int(tid) for tid in tg.prioritize_sequence]
        packed = encode_task_graph(tg, decoder_order=order, resource_cluster=cluster)
        if packed.shape != (20, 54):
            raise ValueError("packed shape %s != (20,54)" % (packed.shape,))
        obs_rows.append(packed.astype(np.float32))
    obs = np.stack(obs_rows, axis=0)
    return {
        "obs": obs,
        "acts": acts_all[:n],
        "t_expert": t_ex[:n],
        "t_mec": t_mec[:n],
        "n_non": n_non[:n],
        "dist_id": dist_all[:n],
        "graph_idx": gidx_all[:n],
        "profile_id": np.asarray([profile_id] * n),
        "resources": resources,
        "cluster": cluster,
        "cache": cache,
        "cache_sha256": _sha256_file(cache),
    }


def build_combined_train_cache(out_path=COMBINED_TRAIN, max_graphs_per_profile=None):
    """Stack all meta_train profiles into one BC cache."""
    set_obs_version("v2")
    profiles = list(role_profile_ids("meta_train"))
    chunks = []
    for pid in profiles:
        print("bc_profiles build_obs start %s" % pid)
        chunk = build_profile_obs_acts(
            "meta_train", pid, max_graphs=max_graphs_per_profile
        )
        chunks.append(chunk)
        print(
            "bc_profiles build_obs done %s n=%d packed=%s"
            % (pid, chunk["obs"].shape[0], chunk["obs"].shape[1:])
        )
    obs = np.concatenate([c["obs"] for c in chunks], axis=0)
    acts = np.concatenate([c["acts"] for c in chunks], axis=0)
    payload = dict(
        obs=obs,
        acts=acts,
        t_expert=np.concatenate([c["t_expert"] for c in chunks]),
        t_mec=np.concatenate([c["t_mec"] for c in chunks]),
        n_non=np.concatenate([c["n_non"] for c in chunks]),
        dist_id=np.concatenate([c["dist_id"] for c in chunks]),
        graph_idx=np.concatenate([c["graph_idx"] for c in chunks]),
        profile_id=np.concatenate([c["profile_id"] for c in chunks]),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **payload)
    meta = {
        "path": str(out_path),
        "n_graphs": int(obs.shape[0]),
        "obs_shape": list(obs.shape),
        "profiles": profiles,
        "sha256": _sha256_file(out_path),
        "per_profile_sha": {c["profile_id"][0]: c["cache_sha256"] for c in chunks},
        "paper_result": False,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("bc_profiles combined n=%d shape=%s sha=%s" % (obs.shape[0], obs.shape, meta["sha256"][:16]))
    return out_path, meta


def _score_plans(obs_chunk, acts, resources, dist_id, graph_idx, cluster):
    """Greedy-decode already done; score action rows under profile resources."""
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter

    ts = []
    for i in range(acts.shape[0]):
        gv = _gv_path(int(dist_id[i]), int(graph_idx[i]))
        tg = OffloadingTaskGraph(str(gv))
        tg.prioritize_tasks(cluster)
        order = [int(tid) for tid in tg.prioritize_sequence]
        plan = list(zip(order, [int(a) for a in acts[i]]))
        res, _, _ = schedule_via_adapter(tg, plan, resources)
        ts.append(float(res.makespan_seconds))
    return float(np.mean(ts))


def greedy_decode_and_score(sess, policy, chunk):
    """Greedy decode on chunk obs; score with chunk resources."""
    from spec.bc_greedy_mec import align_greedy_pred

    obs = chunk["obs"]
    acts_ex = chunk["acts"]
    n, n_tok = acts_ex.shape
    preds = []
    for start in range(0, n, BC_BATCH):
        sl = slice(start, min(start + BC_BATCH, n))
        batch = obs[sl]
        fl = np.full((batch.shape[0],), n_tok, dtype=np.int32)
        greedy_pred = sess.run(
            policy.network.greedy_decoder_prediction,
            feed_dict=policy_feed(policy, batch, fl),
        )
        aligned, _trunc = align_greedy_pred(greedy_pred, n_tok)
        preds.append(aligned)
    pred = np.concatenate(preds, axis=0)
    token_acc = float(np.mean(pred == acts_ex))
    t_mean = _score_plans(
        obs,
        pred,
        chunk["resources"],
        chunk["dist_id"],
        chunk["graph_idx"],
        chunk["cluster"],
    )
    counts = np.bincount(pred.reshape(-1), minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    return {
        "greedy_T_mean": float(t_mean),
        "greedy_token_acc_vs_expert": float(token_acc),
        "expert_T_mean": float(np.mean(chunk["t_expert"])),
        "mix_local": float(counts[0]),
        "mix_mec": float(counts[1]),
        "mix_v2v": float(counts[2]),
        "n_graphs": int(n),
        "pred_acts": pred,
    }


def run_bc_ce_on_arrays(sess, policy, obs, acts, rng, epochs, early_stop_patience, early_stop_min_delta, max_graphs=None):
    """CE teacher forcing on prebuilt obs/acts. Same recipe as Phase 1."""
    import tensorflow as tf

    n, n_tok = acts.shape
    if max_graphs is not None:
        n = min(n, int(max_graphs))
        obs = obs[:n]
        acts = acts[:n]
    shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    loss = tf.reduce_mean(policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(BC_LR, name="bc_profiles_adam")
    train_op = opt.minimize(loss, var_list=policy.get_trainable_variables())
    sess.run(tf.compat.v1.variables_initializer(opt.variables()))
    order = np.arange(n)
    best_loss = float("inf")
    stall = 0
    epoch_losses = []
    stopped_at = int(epochs)
    for epoch in range(int(epochs)):
        rng.shuffle(order)
        batch_losses = []
        for start in range(0, n, BC_BATCH):
            sl = order[start : start + BC_BATCH]
            fl = np.full((len(sl),), n_tok, dtype=np.int32)
            _, lv = sess.run(
                [train_op, loss],
                feed_dict=policy_feed(policy, obs[sl], fl, shift=shift[sl], acts=acts[sl]),
            )
            batch_losses.append(float(lv))
        cur = float(np.mean(batch_losses))
        epoch_losses.append(cur)
        if best_loss - cur >= float(early_stop_min_delta):
            best_loss = cur
            stall = 0
        else:
            stall += 1
        print("bc_profiles epoch=%d loss=%.5f best=%.5f stall=%d" % (epoch, cur, best_loss, stall))
        if early_stop_patience is not None and stall >= int(early_stop_patience):
            stopped_at = epoch + 1
            break
    return {
        "bc_epochs_ran": int(stopped_at),
        "bc_best_loss": float(best_loss),
        "bc_epoch_losses": epoch_losses,
        "n_graphs": int(n),
    }
