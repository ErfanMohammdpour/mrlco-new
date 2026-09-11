#!/usr/bin/env python3
"""Phase 4 campaign plan. Stdlib only. Does not import TensorFlow. Does not train."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CAMPAIGN_YAML = SPEC / "phase4_campaign.yaml"
RUNS_ROOT = ROOT / "runs" / "phase4"

METHOD_ID = "margo_v0.1_primary"
PROBE_METHOD_ID = "margo_v0.1_learning_probe"
PARALLEL_PROBE_METHOD_ID = "margo_v0.1_parallel_probe"
DIAG_METHOD_ID = "margo_v0.1_diag_1k"
DIAG200_METHOD_ID = "margo_v0.1_diag_200"
DIAG500_METHOD_ID = "margo_v0.1_diag_500_parallel"
DIAG_LAT_METHOD_ID = "margo_v0.1_diag_latency_tmec"
DIAG_POMO_METHOD_ID = "margo_v0.1_diag_pomo_tmec"
DIAG_BC_METHOD_ID = "margo_v0.1_diag_bc_greedy_tmec"
DIAG_BCONLY_METHOD_ID = "margo_v0.1_diag_bc_only_eval"
DIAG_BCCONT_METHOD_ID = "margo_v0.1_diag_bc_continue"
DIAG_KLPPO_METHOD_ID = "margo_v0.1_diag_kl_bc_ppo"
DIAG_BCUNSEEN_METHOD_ID = "margo_v0.1_diag_bc_unseen"
DIAG_BCFEW_METHOD_ID = "margo_v0.1_diag_bc_fewshot"
DIAG_BCSS_METHOD_ID = "margo_v0.1_diag_bc_scheduled"
DIAG_BCSS_EPOCHS = 40
DIAG_H2_METHOD_ID = "margo_v0.1_diag_hamming2_expert"
DIAG_H2_N_GRAPHS = 200
DIAG_MOTIF_METHOD_ID = "margo_v0.1_diag_motif_expert"
DIAG_MOTIF_N_TRAIN = 200
DIAG_TWOPT_METHOD_ID = "margo_v0.1_diag_2opt_expert"
DIAG_TWOPT_MAX_H2_ROUNDS = 8
DIAG_BC2OPT_METHOD_ID = "margo_v0.1_diag_bc_2opt"
DIAG_PAIR_METHOD_ID = "margo_v0.1_diag_pair_head"
DIAG_PAIR_TOP_K = 20
DIAG_PAIR_ORACLE_N_VAL = 100
DIAG_PAIR_KSWEEP_METHOD_ID = "margo_v0.1_diag_pair_ksweep"
DIAG_PAIR_K_SWEEP = (10, 20, 50)
DIAG_PAIR_RANKER_METHOD_ID = "margo_v0.1_diag_pair_ranker"
DIAG_PAIR_RANKER_K = (10, 20)
DIAG_PAIR_SEQ_METHOD_ID = "margo_v0.1_diag_pair_seq"
DIAG_PAIR_SEQ_BESTIMP_K = (20, 50)
DIAG_CAVIA_METHOD_ID = "margo_v0.2_diag_cavia_frozen"
DIAG_CAVIA_ENERGY_METHOD_ID = "margo_v0.2_diag_cavia_energy"
DIAG_CAVIA_STRONG_METHOD_ID = "margo_v0.2_diag_cavia_strong"
DIAG_CAVIA_Z_DIM = 32
DIAG_CAVIA_INNER_STEPS = 20
DIAG_CAVIA_INNER_LR = 5.0e-4
DIAG_CAVIA_STRONG_INNER_STEPS = 50
DIAG_CAVIA_STRONG_INNER_LR = 1.0e-2
DIAG_PAIRSUP_METHOD_ID = "margo_v0.2_diag_pairsup"
DIAG_PAIRSUP_EPOCHS = 40
DIAG_PAIRSUP_LAMBDA = 0.5
DIAG_PAIRSUP_A_VAL_T = 575.0
DIAG_PAIRSUP_A_TEST_T = 557.0
DIAG_CAVIA_BCCONT_METHOD_ID = "margo_v0.2_diag_cavia_bccont"
DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO = 568.0
DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI = 582.0
DIAG_PAIRFRAC_METHOD_ID = "margo_v0.2_diag_pairfrac_mec"
DIAG_PAIRFRAC_N_GRAPHS = 200
DIAG_REWRITE_METHOD_ID = "margo_v0.2_diag_rewrite_mec"
DIAG_REWRITE_EPOCHS = 40
DIAG_REWRITE_LAMBDA = 0.5
DIAG_REWRITE_K = (1, 2, 3)
DIAG_REWRITE_A_VAL_T = 575.0
DIAG_REWRITE_A_TEST_T = 557.0
DIAG_ORACLE_DIST_METHOD_ID = "margo_v0.2_diag_oracle_dist"
DIAG_ORACLE_DIST_EPOCHS = 40
DIAG_ORACLE_DIST_Z_DIM = 32
DIAG_ORACLE_DIST_N_DIST = 26
DIAG_ORACLE_DIST_A_VAL_T = 575.0
DIAG_ORACLE_DIST_A_TEST_T = 557.0
DIAG_ORACLE_DIST_TRAIN_REF_T = 464.0
DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO = 568.0
DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI = 582.0
DIAG_BINARY_LAT_METHOD_ID = "margo_v0.2_diag_binary_lat"
DIAG_BINARY_LAT_ITERS = 50
DIAG_BINARY_LAT_VOCAB = 2
DIAG_ENC_METHOD_ID = "margo_v0.3_diag_encoder"
DIAG_ENC_TYPES = ("meanagg", "gatv2", "dagformer")
DIAG_ENC_READOUTS = ("triple", "mean", "max", "attn", "zero")
DIAG_ENC_MAX_EPOCHS = 120
DIAG_ENC_SMOKE_EPOCHS = 2
DIAG_ENC_SMOKE_GRAPHS = 64
DIAG_BOK_METHOD_ID = "margo_v0.3_diag_bestofk"
DIAG_BOK_K_SWEEP = (1, 4, 8, 16, 32, 64)
DIAG_BOK_TEMPS = (0.7, 1.0, 1.3)
DIAG_BOK_K_TEMP = 32
DIAG_BOK_K_MAX = 64
DIAG_BOK_SMOKE_GRAPHS = 8
DIAG_BOK_SMOKE_K = 4
DIAG_BOK_SANITY_GRAPHS = 200
DIAG_EAS_METHOD_ID = "margo_v0.3_diag_eas"
DIAG_EAS_SUBSETS = ("lastlayer", "film", "emb", "full")
DIAG_EAS_LOSSES = ("pg_il", "pg", "il", "ce2opt")
DIAG_EAS_DEFAULT_SUBSET = "lastlayer"
DIAG_EAS_DEFAULT_LOSS = "pg_il"
DIAG_EAS_DEFAULT_LAMBDA_IL = 1.0
DIAG_EAS_N_ADAPT = 100
DIAG_EAS_K = 16
DIAG_EAS_LR = 1e-3
DIAG_EAS_SMOKE_N_ADAPT = 5
DIAG_EAS_SMOKE_K = 4
DIAG_EAS_SMOKE_DIST = 2
DIAG_EAS_INST_METHOD_ID = "margo_v0.3_diag_eas_inst"
DIAG_EAS_INST_DEFAULT_SUBSET = "lastlayer"
DIAG_EAS_INST_DEFAULT_LOSS = "pg"
DIAG_EAS_INST_BUDGETS = (32, 64)
DIAG_EAS_INST_SMOKE_BUDGET = 8
DIAG_EAS_INST_SMOKE_GRAPHS = 8
DIAG_EAS_INST_LR = 1e-3
DIAG_EAS_INST_LAMBDA_IL = 1.0
DIAG_EXPERT_PROFILES_METHOD_ID = "margo_v0.3_expert_profiles"
DIAG_BC_PROFILES_METHOD_ID = "margo_v0.3_bc_profiles"
DIAG_EAS_PROFILES_METHOD_ID = "margo_v0.3_eas_profiles"
DIAG_CTX_PROFILES_METHOD_ID = "margo_v0.3_ctx_profiles"
DIAG_BOK_PROFILES_METHOD_ID = "margo_v0.3_bok_profiles"
DIAG_EXPERT_ENERGY_METHOD_ID = "margo_v0.3_expert_energy"
DIAG_BC_ENERGY_METHOD_ID = "margo_v0.3_bc_energy"
DIAG_EAS_ENERGY_METHOD_ID = "margo_v0.3_eas_energy"
DIAG_CTX_Z_DIM = 32
DIAG_CTX_MAX_EPOCHS = 40
DIAG_CTX_SMOKE_EPOCHS = 2
DIAG_CTX_SMOKE_TASKS = 2
PARENT_FREEZE = "phase3-freeze-v0.1"
PARENT_SHA = "0c776924b49da6c66c511c12a8cde70be732e25d"
SEEDS = (0, 1, 2, 3, 4)
K_REPORT = (0, 3)
OUTER_ITERS = 3500
PROBE_ITERS = 5
DIAG_ITERS = 1000
DIAG200_ITERS = 200
DIAG500_ITERS = 500
DIAG_LAT_ITERS = 50
DIAG_POMO_ITERS = 50
DIAG_BC_ITERS = 50
DIAG_BCONLY_EPOCHS = 40
DIAG_BCCONT_MAX_EPOCHS = 80
DIAG_KLPPO_ITERS = 50
DIAG_KLPPO_WARMUP_ITERS = 5
DIAG_KLPPO_KL_COEF = 0.1
META_TEST_IDS = (7, 12, 14, 20, 23)
VALIDATION_IDS = (2, 6, 10, 16, 17)
META_TRAIN_IDS = (1, 3, 4, 5, 8, 9, 11, 13, 15, 18, 19, 21, 22, 24, 25)
GPU_ENV = "MARGO_ALLOW_GPU"
GPU_FLAG = "--i-allow-gpu"


class GpuPermissionError(RuntimeError):
    """Raised when a train/GPU launch is requested without the frozen locks."""


def require_gpu_permission(allow_gpu_flag, environ=None):
    environ = os.environ if environ is None else environ
    if not allow_gpu_flag:
        raise GpuPermissionError(
            "Phase 4 GPU train blocked: need CLI %s after explicit human approval" % GPU_FLAG
        )
    if environ.get(GPU_ENV) != "1":
        raise GpuPermissionError(
            "Phase 4 GPU train blocked: env %s must be 1" % GPU_ENV
        )


def git_head_sha(root=ROOT):
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def git_dirty(root=ROOT):
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return True
    if proc.returncode != 0:
        return True
    return bool(proc.stdout.strip())


def smoke_run_dir(seed=0, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / "gpu_smoke" / ("seed_%d" % seed)


def seed_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / METHOD_ID / ("seed_%d" % seed)


def probe_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / PROBE_METHOD_ID / ("seed_%d" % seed)


def parallel_probe_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / PARALLEL_PROBE_METHOD_ID / ("seed_%d" % seed)


def diag_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_METHOD_ID / ("seed_%d" % seed)


def diag200_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG200_METHOD_ID / ("seed_%d" % seed)


def diag500_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG500_METHOD_ID / ("seed_%d" % seed)


def diag_lat_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_LAT_METHOD_ID / ("seed_%d" % seed)


def diag_pomo_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_POMO_METHOD_ID / ("seed_%d" % seed)


def diag_bc_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BC_METHOD_ID / ("seed_%d" % seed)


def diag_bconly_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BCONLY_METHOD_ID / ("seed_%d" % seed)


def diag_bccont_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BCCONT_METHOD_ID / ("seed_%d" % seed)


def diag_klppo_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_KLPPO_METHOD_ID / ("seed_%d" % seed)


def diag_bcunseen_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BCUNSEEN_METHOD_ID / ("seed_%d" % seed)


def diag_bcfew_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BCFEW_METHOD_ID / ("seed_%d" % seed)


def diag_bcss_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BCSS_METHOD_ID / ("seed_%d" % seed)


def diag_h2_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_H2_METHOD_ID / ("seed_%d" % seed)


def diag_motif_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_MOTIF_METHOD_ID / ("seed_%d" % seed)


def diag_twopt_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_TWOPT_METHOD_ID / ("seed_%d" % seed)


def diag_bc2opt_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BC2OPT_METHOD_ID / ("seed_%d" % seed)


def diag_pair_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIR_METHOD_ID / ("seed_%d" % seed)


def diag_pair_ksweep_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIR_KSWEEP_METHOD_ID / ("seed_%d" % seed)


def diag_pair_ranker_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIR_RANKER_METHOD_ID / ("seed_%d" % seed)


def diag_pair_seq_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIR_SEQ_METHOD_ID / ("seed_%d" % seed)


def diag_cavia_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_CAVIA_METHOD_ID / ("seed_%d" % seed)


def diag_cavia_energy_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_CAVIA_ENERGY_METHOD_ID / ("seed_%d" % seed)


def diag_cavia_strong_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_CAVIA_STRONG_METHOD_ID / ("seed_%d" % seed)


def diag_pairsup_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIRSUP_METHOD_ID / ("seed_%d" % seed)


def diag_cavia_bccont_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_CAVIA_BCCONT_METHOD_ID / ("seed_%d" % seed)


def diag_pairfrac_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_PAIRFRAC_METHOD_ID / ("seed_%d" % seed)


def diag_rewrite_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_REWRITE_METHOD_ID / ("seed_%d" % seed)


def diag_oracle_dist_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_ORACLE_DIST_METHOD_ID / ("seed_%d" % seed)


def diag_binary_lat_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BINARY_LAT_METHOD_ID / ("seed_%d" % seed)


def diag_encoder_run_dir(seed, encoder_type, readout_type, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    encoder_type = str(encoder_type)
    readout_type = str(readout_type)
    if encoder_type not in DIAG_ENC_TYPES:
        raise ValueError("encoder_type %r not in %s" % (encoder_type, DIAG_ENC_TYPES))
    if readout_type not in DIAG_ENC_READOUTS:
        raise ValueError("readout_type %r not in %s" % (readout_type, DIAG_ENC_READOUTS))
    return (
        Path(runs_root)
        / DIAG_ENC_METHOD_ID
        / ("%s_%s" % (encoder_type, readout_type))
        / ("seed_%d" % seed)
    )


def diag_bestofk_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BOK_METHOD_ID / ("seed_%d" % seed)


def diag_eas_run_dir(seed, subset, loss, runs_root=RUNS_ROOT):
    seed = int(seed)
    subset = str(subset)
    loss = str(loss)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    if subset not in DIAG_EAS_SUBSETS:
        raise ValueError("adapt_subset %r not in %s" % (subset, DIAG_EAS_SUBSETS))
    if loss not in DIAG_EAS_LOSSES:
        raise ValueError("loss %r not in %s" % (loss, DIAG_EAS_LOSSES))
    return (
        Path(runs_root)
        / DIAG_EAS_METHOD_ID
        / ("%s_%s" % (subset, loss))
        / ("seed_%d" % seed)
    )


def diag_eas_inst_run_dir(seed, subset, loss, budget, runs_root=RUNS_ROOT):
    seed = int(seed)
    subset = str(subset)
    loss = str(loss)
    budget = int(budget)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    if subset not in DIAG_EAS_SUBSETS:
        raise ValueError("adapt_subset %r not in %s" % (subset, DIAG_EAS_SUBSETS))
    if loss not in DIAG_EAS_LOSSES:
        raise ValueError("loss %r not in %s" % (loss, DIAG_EAS_LOSSES))
    return (
        Path(runs_root)
        / DIAG_EAS_INST_METHOD_ID
        / ("%s_%s_b%d" % (subset, loss, budget))
        / ("seed_%d" % seed)
    )


def diag_expert_profiles_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_EXPERT_PROFILES_METHOD_ID / ("seed_%d" % seed)


def diag_bc_profiles_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BC_PROFILES_METHOD_ID / ("seed_%d" % seed)


def diag_eas_profiles_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_EAS_PROFILES_METHOD_ID / ("seed_%d" % seed)


def diag_ctx_profiles_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_CTX_PROFILES_METHOD_ID / ("seed_%d" % seed)


def diag_bok_profiles_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BOK_PROFILES_METHOD_ID / ("seed_%d" % seed)


def diag_expert_energy_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_EXPERT_ENERGY_METHOD_ID / ("seed_%d" % seed)


def diag_bc_energy_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_BC_ENERGY_METHOD_ID / ("seed_%d" % seed)


def diag_eas_energy_run_dir(seed, runs_root=RUNS_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("seed %s is not in frozen campaign seeds %s" % (seed, list(SEEDS)))
    return Path(runs_root) / DIAG_EAS_ENERGY_METHOD_ID / ("seed_%d" % seed)


def eval_json_path(run_dir, dist_id, k_steps):
    return Path(run_dir) / "eval" / "meta_test" / ("dist_%d_k%d.json" % (int(dist_id), int(k_steps)))


def campaign_jobs():
    jobs = []
    for seed in SEEDS:
        run_dir = seed_run_dir(seed)
        jobs.append(
            {
                "kind": "train",
                "method_id": METHOD_ID,
                "seed": seed,
                "outer_iterations": OUTER_ITERS,
                "run_dir": str(run_dir),
                "ckpt_dir": str(run_dir / "ckpt"),
                "log_dir": str(run_dir / "logs"),
                "gpu_required": True,
            }
        )
        for dist_id in META_TEST_IDS:
            for k_steps in K_REPORT:
                jobs.append(
                    {
                        "kind": "meta_test_eval",
                        "method_id": METHOD_ID,
                        "seed": seed,
                        "distribution_id": dist_id,
                        "k_steps": k_steps,
                        "support_graphs": 20,
                        "query_graphs": 80,
                        "artifact": str(eval_json_path(run_dir, dist_id, k_steps)),
                    }
                )
    return jobs


def provenance_template(seed):
    from spec.eval_protocol import protocol_log_kvs
    from spec.split_loader import split_version

    kvs = protocol_log_kvs(seed=seed, k_steps=3, outer_update_count=OUTER_ITERS)
    kvs.update(
        {
            "method_id": METHOD_ID,
            "parent_freeze": PARENT_FREEZE,
            "parent_sha": PARENT_SHA,
            "code_sha": git_head_sha(),
            "code_dirty": git_dirty(),
            "paper_result": False,
            "gpu_requested": False,
            "gpu_finished": False,
            "split_version": split_version(),
            "meta_train_distribution_ids": list(META_TRAIN_IDS),
            "validation_distribution_ids": list(VALIDATION_IDS),
            "meta_test_distribution_ids": list(META_TEST_IDS),
            "outer_iterations": OUTER_ITERS,
            "k_report": list(K_REPORT),
            "latency_weight": 0.5,
            "energy_weight": 0.5,
            "units_time": "seconds",
            "units_energy": "joules",
            "optimization_claim": False,
            "eval_checkpoint": "meta_model_best_val.ckpt",
        }
    )
    return kvs


def write_plan(path=None):
    plan = {
        "phase": 4,
        "status": "IN PROGRESS",
        "parent_freeze": PARENT_FREEZE,
        "parent_sha": PARENT_SHA,
        "method_id": METHOD_ID,
        "seeds": list(SEEDS),
        "meta_test_distribution_ids": list(META_TEST_IDS),
        "validation_distribution_ids": list(VALIDATION_IDS),
        "jobs": campaign_jobs(),
        "gpu": {
            "default": "forbidden",
            "require_env": "%s=1" % GPU_ENV,
            "require_cli_flag": GPU_FLAG,
            "require_chat_approval": True,
        },
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Phase 4 campaign (plan by default; GPU locked)")
    parser.add_argument("--plan", action="store_true", help="print campaign plan JSON")
    parser.add_argument("--write-plan", action="store_true", help="write runs/phase4/plan.json")
    parser.add_argument("--execute-train", action="store_true", help="launch 3500-iter train (GPU locked)")
    parser.add_argument("--gpu-smoke", action="store_true", help="1 outer-iter GPU stack test; not a paper result")
    parser.add_argument(
        "--learning-probe",
        action="store_true",
        help="5 outer-iter audit dump; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-1k",
        action="store_true",
        help="1000 outer-iter diagnostic with audit; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-200",
        action="store_true",
        help="200 outer-iter diagnostic with audit; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--parallel-probe",
        action="store_true",
        help="5-iter spawn-parallel env probe; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-500",
        action="store_true",
        help="500-iter spawn-parallel diagnostic; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-latency-tmec",
        action="store_true",
        help="50-iter latency-only R=-dT/T_allMEC diagnostic; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-pomo-tmec",
        action="store_true",
        help="50-iter POMO-advantage + elite-select latency diagnostic; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-bc-greedy-tmec",
        action="store_true",
        help="50-iter BC greedy-from-MEC then latency PPO; not a paper result; does not replace 3500",
    )
    parser.add_argument(
        "--diagnostic-bc-only-eval",
        action="store_true",
        help="BC greedy-from-MEC only: 40 epochs + greedy rollout T + ckpt; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-continue",
        action="store_true",
        help="Resume BC from bc_only ckpt until CE plateau; greedy rollout; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-kl-bc-ppo",
        action="store_true",
        help="Load bc_continue ckpt; critic warmup then KL(π||π_BC) latency PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-unseen",
        action="store_true",
        help="Greedy-decode bc_continue ckpt on val+meta-test; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-fewshot",
        action="store_true",
        help="Few-shot from bc_continue: k0 vs k3 PPO vs encoder-frozen CE on support; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-scheduled",
        action="store_true",
        help="Scheduled-sampling BC from bc_continue then unseen greedy; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-hamming2-expert",
        action="store_true",
        help="CPU Hamming-2 from greedy_from_mec on 200 train graphs; no GPU; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-motif-expert",
        action="store_true",
        help="CPU motif audit of greedy_from_mec (CC/homophily/critical path); no GPU; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-2opt-expert",
        action="store_true",
        help="CPU iterative 2-opt teacher from greedy_from_mec; no GPU; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-2opt",
        action="store_true",
        help="BC on 2-opt expert labels from bc_continue, then unseen greedy; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pair-head",
        action="store_true",
        help="Frozen-encoder 9-way pair/motif head then top-K x 9 schedule refine; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pair-ksweep",
        action="store_true",
        help="k=10/20/50 pair refine vs same-split motif oracle; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pair-ranker",
        action="store_true",
        help="ΔT pair ranker vs CE vs true-gain rank at k=10/20; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pair-seq",
        action="store_true",
        help="CE scan vs multipass vs bestimp sequential pair refine; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-cavia",
        action="store_true",
        help="CAVIA-on-z from bc_2opt; cost=T; energy logged; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-cavia-energy",
        action="store_true",
        help="CAVIA-on-z from bc_2opt; cost=0.5 L_norm + 0.5 E_norm; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-cavia-strong",
        action="store_true",
        help="CAVIA-on-z T-only with lr=1e-2 and 50 inner steps; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pairsup",
        action="store_true",
        help="BC greedy_from_mec + λ joint motif CE; greedy eval only; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-cavia-bccont",
        action="store_true",
        help="CAVIA-on-z T-only from bc_continue ckpt; 20 inner steps; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-pairfrac-mec",
        action="store_true",
        help="CPU motif pair frac_pos from all-MEC on 200 train graphs; no GPU; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-rewrite-mec",
        action="store_true",
        help="BC greedy_from_mec + λ all-MEC joint CE then greedy+K neural apply; no search; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-oracle-dist",
        action="store_true",
        help="Oracle dist_id embed at every decoder step, Graph2Seq frozen, from bc_continue; no PPO; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-binary-lat",
        action="store_true",
        help="50-iter binary no-V2V latency PPO; energy off; parallel env executor; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-encoder",
        action="store_true",
        help="v0.3 encoder ablation BC-2opt; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bestofk",
        action="store_true",
        help="v0.3 best-of-k neural inference; no training; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-eas",
        action="store_true",
        help="v0.3 EAS-on-φ support adaptation; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-eas-inst",
        action="store_true",
        help="v0.3 per-instance EAS at test time vs best-of-k; not meta; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-expert-profiles",
        action="store_true",
        help="v0.3 per-profile 2-opt teachers; CPU; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bc-profiles",
        action="store_true",
        help="v0.3 BC on profile teachers + obs v2; GPU; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-eas-profiles",
        action="store_true",
        help="v0.3 EAS lastlayer+pg_il on held-out resource profiles; GPU; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-ctx-profiles",
        action="store_true",
        help="v0.3 PEARL-style context-only FiLM on held-out resource profiles; GPU; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-bok-profiles",
        action="store_true",
        help="v0.3 best-of-k on BC-profiles ckpt + obs v2; no training; not a paper result",
    )
    parser.add_argument(
        "--diagnostic-expert-energy",
        action="store_true",
        help="v0.3 per-λ 2-opt teachers (J_λ); CPU; not a paper result",
    )
    parser.add_argument(
        "--expert-split",
        type=str,
        default="meta_train",
        choices=("meta_train", "validation"),
        help="graph split for expert-profiles (validation = held-out+frozen profiles)",
    )
    parser.add_argument("--encoder-type", default="meanagg", help="meanagg|gatv2|dagformer")
    parser.add_argument("--readout-type", default="triple", help="triple|mean|max|attn|zero")
    parser.add_argument(
        "--adapt-subset",
        default="lastlayer",
        help="lastlayer|film|emb|full",
    )
    parser.add_argument(
        "--loss",
        default="pg_il",
        dest="eas_loss",
        help="pg_il|pg|il|ce2opt",
    )
    parser.add_argument("--lambda-il", type=float, default=1.0)
    parser.add_argument("--n-adapt", type=int, default=None)
    parser.add_argument("--eas-k", type=int, default=None, help="EAS sample k (default 16)")
    parser.add_argument(
        "--eas-budget",
        type=int,
        default=None,
        help="per-instance EAS schedule budget B=N*k (8|32|64)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="dry-run: encoder 2 epochs / bestofk n=8 k=4 / eas 1 dist N=5 k=4 / eas-inst n=8 B=8 / expert-profiles 3×32 / eas-profiles 1 profile 1 dist / ctx-profiles 2 tasks 2 epochs / bok-profiles n=8 k=4 frozen / expert-energy frozen×8×5λ",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(GPU_FLAG, dest="allow_gpu", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    gpu_jobs = [
        args.execute_train,
        args.gpu_smoke,
        args.learning_probe,
        args.diagnostic_1k,
        args.diagnostic_200,
        args.parallel_probe,
        args.diagnostic_500,
        args.diagnostic_latency_tmec,
        args.diagnostic_pomo_tmec,
        args.diagnostic_bc_greedy_tmec,
        args.diagnostic_bc_only_eval,
        args.diagnostic_bc_continue,
        args.diagnostic_kl_bc_ppo,
        args.diagnostic_bc_unseen,
        args.diagnostic_bc_fewshot,
        args.diagnostic_bc_scheduled,
        args.diagnostic_bc_2opt,
        args.diagnostic_pair_head,
        args.diagnostic_pair_ksweep,
        args.diagnostic_pair_ranker,
        args.diagnostic_pair_seq,
        args.diagnostic_cavia,
        args.diagnostic_cavia_energy,
        args.diagnostic_cavia_strong,
        args.diagnostic_pairsup,
        args.diagnostic_cavia_bccont,
        args.diagnostic_rewrite_mec,
        args.diagnostic_oracle_dist,
        args.diagnostic_binary_lat,
        args.diagnostic_encoder,
        args.diagnostic_bestofk,
        args.diagnostic_eas,
        args.diagnostic_eas_inst,
        args.diagnostic_bc_profiles,
        args.diagnostic_eas_profiles,
        args.diagnostic_ctx_profiles,
        args.diagnostic_bok_profiles,
    ]
    cpu_jobs = [
        args.diagnostic_hamming2_expert,
        args.diagnostic_motif_expert,
        args.diagnostic_2opt_expert,
        args.diagnostic_pairfrac_mec,
        args.diagnostic_expert_profiles,
        args.diagnostic_expert_energy,
    ]
    if sum(bool(x) for x in gpu_jobs) + sum(bool(x) for x in cpu_jobs) > 1:
        raise ValueError(
            "pass only one of --execute-train --gpu-smoke --learning-probe --diagnostic-1k --diagnostic-200 --parallel-probe --diagnostic-500 --diagnostic-latency-tmec --diagnostic-pomo-tmec --diagnostic-bc-greedy-tmec --diagnostic-bc-only-eval --diagnostic-bc-continue --diagnostic-kl-bc-ppo --diagnostic-bc-unseen --diagnostic-bc-fewshot --diagnostic-bc-scheduled --diagnostic-bc-2opt --diagnostic-pair-head --diagnostic-pair-ksweep --diagnostic-pair-ranker --diagnostic-pair-seq --diagnostic-cavia --diagnostic-cavia-energy --diagnostic-cavia-strong --diagnostic-pairsup --diagnostic-cavia-bccont --diagnostic-rewrite-mec --diagnostic-oracle-dist --diagnostic-binary-lat --diagnostic-encoder --diagnostic-bestofk --diagnostic-eas --diagnostic-eas-inst --diagnostic-bc-profiles --diagnostic-eas-profiles --diagnostic-ctx-profiles --diagnostic-bok-profiles --diagnostic-expert-profiles --diagnostic-expert-energy --diagnostic-hamming2-expert --diagnostic-motif-expert --diagnostic-2opt-expert --diagnostic-pairfrac-mec"
        )
    if args.diagnostic_ctx_profiles:
        if args.seed is None:
            raise ValueError("Phase 4 ctx-profiles needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_ctx_profiles

        run_diagnostic_ctx_profiles(seed=args.seed, allow_gpu=args.allow_gpu, smoke=bool(args.smoke))
        return 0
    if args.diagnostic_bok_profiles:
        if args.seed is None:
            raise ValueError("Phase 4 bok-profiles needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_bok_profiles

        run_diagnostic_bok_profiles(seed=args.seed, allow_gpu=args.allow_gpu, smoke=bool(args.smoke))
        return 0
    if args.diagnostic_eas_profiles:
        if args.seed is None:
            raise ValueError("Phase 4 eas-profiles needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_eas_profiles

        run_diagnostic_eas_profiles(seed=args.seed, allow_gpu=args.allow_gpu, smoke=bool(args.smoke))
        return 0
    if args.diagnostic_bc_profiles:
        if args.seed is None:
            raise ValueError("Phase 4 bc-profiles needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_bc_profiles

        run_diagnostic_bc_profiles(seed=args.seed, allow_gpu=args.allow_gpu, smoke=bool(args.smoke))
        return 0
    if args.diagnostic_expert_profiles:
        if args.seed is None:
            raise ValueError("Phase 4 expert-profiles needs one --seed from %s" % (list(SEEDS),))
        from spec.expert_profiles import run_expert_profiles

        run_expert_profiles(
            seed=args.seed,
            smoke=bool(args.smoke),
            split=str(args.expert_split),
        )
        return 0
    if args.diagnostic_expert_energy:
        if args.seed is None:
            raise ValueError("Phase 5 expert-energy needs one --seed from %s" % (list(SEEDS),))
        from spec.expert_energy import run_expert_energy

        run_expert_energy(
            seed=args.seed,
            smoke=bool(args.smoke),
            split=str(args.expert_split),
        )
        return 0
    if args.diagnostic_hamming2_expert:
        if args.seed is None:
            raise ValueError("Phase 4 Hamming-2 probe needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_hamming2_expert

        run_diagnostic_hamming2_expert(seed=args.seed)
        return 0
    if args.diagnostic_motif_expert:
        if args.seed is None:
            raise ValueError("Phase 4 motif audit needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_motif_expert

        run_diagnostic_motif_expert(seed=args.seed)
        return 0
    if args.diagnostic_2opt_expert:
        if args.seed is None:
            raise ValueError("Phase 4 2-opt teacher needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_2opt_expert

        run_diagnostic_2opt_expert(seed=args.seed)
        return 0
    if args.diagnostic_pairfrac_mec:
        if args.seed is None:
            raise ValueError("Phase 4 pairfrac-mec probe needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_diagnostic_pairfrac_mec

        run_diagnostic_pairfrac_mec(seed=args.seed)
        return 0
    if any(gpu_jobs):
        require_gpu_permission(args.allow_gpu)
        if args.seed is None:
            raise ValueError("Phase 4 GPU job needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import (
            run_diagnostic_1k,
            run_diagnostic_200,
            run_diagnostic_500,
            run_diagnostic_latency_tmec,
            run_diagnostic_pomo_tmec,
            run_diagnostic_bc_greedy_tmec,
            run_diagnostic_bc_only_eval,
            run_diagnostic_bc_continue,
            run_diagnostic_kl_bc_ppo,
            run_diagnostic_bc_unseen,
            run_diagnostic_bc_fewshot,
            run_diagnostic_bc_scheduled,
            run_diagnostic_bc_2opt,
            run_diagnostic_pair_head,
            run_diagnostic_pair_ksweep,
            run_diagnostic_pair_ranker,
            run_diagnostic_pair_seq,
            run_diagnostic_cavia,
            run_diagnostic_pairsup,
            run_diagnostic_rewrite_mec,
            run_diagnostic_oracle_dist,
            run_diagnostic_binary_lat,
            run_diagnostic_encoder,
            run_diagnostic_bestofk,
            run_diagnostic_eas,
            run_diagnostic_eas_inst,
            run_gpu_smoke,
            run_learning_probe,
            run_parallel_probe,
            run_primary_seed,
        )

        if args.gpu_smoke:
            run_gpu_smoke(seed=args.seed, allow_gpu=True)
        elif args.learning_probe:
            run_learning_probe(seed=args.seed, allow_gpu=True)
        elif args.parallel_probe:
            run_parallel_probe(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_500:
            run_diagnostic_500(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_latency_tmec:
            run_diagnostic_latency_tmec(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_pomo_tmec:
            run_diagnostic_pomo_tmec(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_greedy_tmec:
            run_diagnostic_bc_greedy_tmec(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_only_eval:
            run_diagnostic_bc_only_eval(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_continue:
            run_diagnostic_bc_continue(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_kl_bc_ppo:
            run_diagnostic_kl_bc_ppo(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_unseen:
            run_diagnostic_bc_unseen(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_fewshot:
            run_diagnostic_bc_fewshot(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_scheduled:
            run_diagnostic_bc_scheduled(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_bc_2opt:
            run_diagnostic_bc_2opt(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_pair_head:
            run_diagnostic_pair_head(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_pair_ksweep:
            run_diagnostic_pair_ksweep(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_pair_ranker:
            run_diagnostic_pair_ranker(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_pair_seq:
            run_diagnostic_pair_seq(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_cavia:
            run_diagnostic_cavia(seed=args.seed, allow_gpu=True, use_energy=False)
        elif args.diagnostic_cavia_energy:
            run_diagnostic_cavia(seed=args.seed, allow_gpu=True, use_energy=True)
        elif args.diagnostic_cavia_strong:
            run_diagnostic_cavia(seed=args.seed, allow_gpu=True, use_energy=False, strong=True)
        elif args.diagnostic_pairsup:
            run_diagnostic_pairsup(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_cavia_bccont:
            run_diagnostic_cavia(seed=args.seed, allow_gpu=True, use_energy=False, from_bccont=True)
        elif args.diagnostic_rewrite_mec:
            run_diagnostic_rewrite_mec(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_oracle_dist:
            run_diagnostic_oracle_dist(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_binary_lat:
            run_diagnostic_binary_lat(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_encoder:
            run_diagnostic_encoder(
                seed=args.seed,
                allow_gpu=True,
                encoder_type=args.encoder_type,
                readout_type=args.readout_type,
                smoke=bool(args.smoke),
            )
        elif args.diagnostic_bestofk:
            run_diagnostic_bestofk(
                seed=args.seed,
                allow_gpu=True,
                smoke=bool(args.smoke),
            )
        elif args.diagnostic_eas:
            run_diagnostic_eas(
                seed=args.seed,
                allow_gpu=True,
                adapt_subset=args.adapt_subset,
                loss_type=args.eas_loss,
                lambda_il=args.lambda_il,
                n_adapt=args.n_adapt,
                k=args.eas_k,
                smoke=bool(args.smoke),
            )
        elif args.diagnostic_eas_inst:
            run_diagnostic_eas_inst(
                seed=args.seed,
                allow_gpu=True,
                adapt_subset=args.adapt_subset,
                loss_type=("pg" if args.eas_loss == "pg_il" else args.eas_loss),
                lambda_il=args.lambda_il,
                budget=args.eas_budget,
                smoke=bool(args.smoke),
            )
        elif args.diagnostic_1k:
            run_diagnostic_1k(seed=args.seed, allow_gpu=True)
        elif args.diagnostic_200:
            run_diagnostic_200(seed=args.seed, allow_gpu=True)
        else:
            run_primary_seed(seed=args.seed, allow_gpu=True)
        return 0
    plan = write_plan(RUNS_ROOT / "plan.json" if args.write_plan else None)
    if args.plan or args.write_plan or argv is None:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GpuPermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
