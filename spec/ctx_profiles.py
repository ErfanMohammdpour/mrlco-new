"""Phase 4 PEARL-style context-only baseline. paper_result=false. No meta-test."""

from __future__ import annotations

import numpy as np

from spec.bc_greedy_mec import policy_feed
from spec.phase4_campaign import (
    DIAG_CTX_PROFILES_METHOD_ID,
    diag_bc_profiles_run_dir,
    diag_ctx_profiles_run_dir,
)
from spec.resource_profiles import role_profile_ids

GATE_DT_S = 10.0
CTX_Z_DIM = 32
TRAIN_SUPPORT_GIDX = tuple(range(20))
TRAIN_QUERY_GIDX = tuple(range(20, 100))
DUMMY_OBS = np.zeros((1, 20, 54), dtype=np.float32)
DUMMY_ACTS = np.zeros((1, 20), dtype=np.int32)
DUMMY_T = np.zeros((1,), dtype=np.float32)


def _as_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def bc_profiles_ckpt(seed):
    return diag_bc_profiles_run_dir(seed) / "ckpt" / "bc_core.ckpt"


def heldout_profile_ids():
    ids = list(role_profile_ids("validation_heldout"))
    leak = set(ids) & set(role_profile_ids("meta_test_heldout"))
    if leak:
        raise ValueError("held-out val profile leaked into meta-test: %s" % leak)
    return ids


def mask_task(dist_arr, prof_arr, gidx_arr, profile_id, dist_id, gidx_keep):
    prof = np.asarray([_as_str(x) for x in prof_arr])
    dist = np.asarray(dist_arr, dtype=np.int32)
    gidx = np.asarray(gidx_arr, dtype=np.int32)
    keep = np.asarray(list(gidx_keep), dtype=np.int32)
    return (prof == str(profile_id)) & (dist == int(dist_id)) & np.isin(gidx, keep)


def ctx_feed(policy, obs, fl, ctx_obs, ctx_acts, ctx_t, shift=None, acts=None, ctx_zero=False):
    fd = policy_feed(policy, obs, fl, shift=shift, acts=acts)
    if getattr(policy, "ctx_obs", None) is None:
        return fd
    if ctx_zero:
        fd[policy.ctx_obs] = DUMMY_OBS
        fd[policy.ctx_acts] = DUMMY_ACTS
        fd[policy.ctx_t] = DUMMY_T
        fd[policy.ctx_zero] = True
    else:
        fd[policy.ctx_obs] = ctx_obs
        fd[policy.ctx_acts] = ctx_acts
        fd[policy.ctx_t] = np.asarray(ctx_t, dtype=np.float32)
        fd[policy.ctx_zero] = False
    return fd


assert DIAG_CTX_PROFILES_METHOD_ID == "margo_v0.3_ctx_profiles"
assert diag_ctx_profiles_run_dir(0).name == "seed_0"
