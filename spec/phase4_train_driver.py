"""Phase 4 primary-run driver. Imports TensorFlow. Do not import from numpy tests.

GPU launch requires spec.phase4_campaign.require_gpu_permission plus human chat approval.

Primary v0.1 train remains 3500. Probe/diag paths are not paper results and do not
rewrite the frozen budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from spec.phase4_campaign import (
    DIAG200_ITERS,
    DIAG500_ITERS,
    DIAG_ITERS,
    DIAG_LAT_ITERS,
    DIAG_BC_ITERS,
    DIAG_BCONLY_EPOCHS,
    DIAG_BCCONT_MAX_EPOCHS,
    DIAG_KLPPO_ITERS,
    DIAG_KLPPO_KL_COEF,
    DIAG_KLPPO_WARMUP_ITERS,
    DIAG_POMO_ITERS,
    DIAG_BCSS_EPOCHS,
    METHOD_ID,
    OUTER_ITERS,
    PARALLEL_PROBE_METHOD_ID,
    PROBE_ITERS,
    diag200_run_dir,
    diag500_run_dir,
    diag_bc_run_dir,
    diag_bconly_run_dir,
    diag_bccont_run_dir,
    diag_klppo_run_dir,
    diag_bcunseen_run_dir,
    diag_bcfew_run_dir,
    diag_bcss_run_dir,
    diag_h2_run_dir,
    diag_motif_run_dir,
    diag_twopt_run_dir,
    diag_bc2opt_run_dir,
    diag_pair_run_dir,
    diag_pair_ksweep_run_dir,
    diag_pair_ranker_run_dir,
    diag_pair_seq_run_dir,
    DIAG_CAVIA_INNER_STEPS,
    DIAG_CAVIA_INNER_LR,
    DIAG_CAVIA_METHOD_ID,
    DIAG_CAVIA_ENERGY_METHOD_ID,
    DIAG_CAVIA_STRONG_METHOD_ID,
    DIAG_CAVIA_STRONG_INNER_STEPS,
    DIAG_CAVIA_STRONG_INNER_LR,
    DIAG_CAVIA_Z_DIM,
    DIAG_CAVIA_BCCONT_METHOD_ID,
    DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO,
    DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI,
    DIAG_PAIRFRAC_METHOD_ID,
    DIAG_PAIRFRAC_N_GRAPHS,
    DIAG_REWRITE_METHOD_ID,
    DIAG_REWRITE_EPOCHS,
    DIAG_REWRITE_LAMBDA,
    DIAG_REWRITE_K,
    DIAG_REWRITE_A_VAL_T,
    DIAG_REWRITE_A_TEST_T,
    DIAG_ORACLE_DIST_METHOD_ID,
    DIAG_ORACLE_DIST_EPOCHS,
    DIAG_ORACLE_DIST_Z_DIM,
    DIAG_ORACLE_DIST_N_DIST,
    DIAG_ORACLE_DIST_A_VAL_T,
    DIAG_ORACLE_DIST_A_TEST_T,
    DIAG_ORACLE_DIST_TRAIN_REF_T,
    DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO,
    DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI,
    DIAG_BINARY_LAT_METHOD_ID,
    DIAG_BINARY_LAT_ITERS,
    DIAG_BINARY_LAT_VOCAB,
    DIAG_ENC_METHOD_ID,
    DIAG_ENC_MAX_EPOCHS,
    DIAG_ENC_SMOKE_EPOCHS,
    DIAG_ENC_SMOKE_GRAPHS,
    DIAG_ENC_TYPES,
    DIAG_ENC_READOUTS,
    DIAG_BOK_METHOD_ID,
    DIAG_BOK_K_SWEEP,
    DIAG_BOK_TEMPS,
    DIAG_BOK_K_TEMP,
    DIAG_BOK_K_MAX,
    DIAG_BOK_SMOKE_GRAPHS,
    DIAG_BOK_SMOKE_K,
    DIAG_BOK_SANITY_GRAPHS,
    DIAG_EAS_METHOD_ID,
    DIAG_EAS_SUBSETS,
    DIAG_EAS_LOSSES,
    DIAG_EAS_DEFAULT_SUBSET,
    DIAG_EAS_DEFAULT_LOSS,
    DIAG_EAS_DEFAULT_LAMBDA_IL,
    DIAG_EAS_N_ADAPT,
    DIAG_EAS_K,
    DIAG_EAS_LR,
    DIAG_EAS_SMOKE_N_ADAPT,
    DIAG_EAS_SMOKE_K,
    DIAG_EAS_SMOKE_DIST,
    DIAG_EAS_INST_METHOD_ID,
    DIAG_EAS_INST_DEFAULT_SUBSET,
    DIAG_EAS_INST_DEFAULT_LOSS,
    DIAG_EAS_INST_BUDGETS,
    DIAG_EAS_INST_SMOKE_BUDGET,
    DIAG_EAS_INST_SMOKE_GRAPHS,
    DIAG_EAS_INST_LR,
    DIAG_EAS_INST_LAMBDA_IL,
    DIAG_PAIRSUP_METHOD_ID,
    DIAG_PAIRSUP_EPOCHS,
    DIAG_PAIRSUP_LAMBDA,
    DIAG_PAIRSUP_A_VAL_T,
    DIAG_PAIRSUP_A_TEST_T,
    diag_cavia_run_dir,
    diag_cavia_energy_run_dir,
    diag_cavia_strong_run_dir,
    diag_cavia_bccont_run_dir,
    diag_pairfrac_run_dir,
    diag_rewrite_run_dir,
    diag_oracle_dist_run_dir,
    diag_binary_lat_run_dir,
    diag_encoder_run_dir,
    diag_bestofk_run_dir,
    diag_eas_run_dir,
    diag_eas_inst_run_dir,
    diag_pairsup_run_dir,
    diag_lat_run_dir,
    diag_pomo_run_dir,
    diag_run_dir,
    parallel_probe_run_dir,
    probe_run_dir,
    provenance_template,
    require_gpu_permission,
    seed_run_dir,
    smoke_run_dir,
)
from spec.train_audit import TrainAuditWriter

SMOKE_ITERS = 1


def _write_payload(run_dir, payload):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval" / "meta_test").mkdir(parents=True, exist_ok=True)
    (run_dir / "audit").mkdir(exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (run_dir / "provenance.json").write_text(text)
    (run_dir / "config.resolved.json").write_text(text)


def _train(seed, n_itr, run_dir, audit=False, print_action_choices=False, parallel=False,
           reward_mode="publication", learning_mode="publication",
           vocab_size=3, use_energy=True):
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    import tensorflow as tf

    # Constrained V2V/energy budgets: set MARGO_CONSTRAINTS=/path/to/constraints.yaml.
    # Unset / "off" -> constraint_spec is None and every training path is unchanged.
    from spec.constraints_config import constraints_from_env

    constraint_spec, constraint_dual_lr = constraints_from_env()
    if constraint_spec is not None:
        print(
            "[constraints] %s  budgets=%s  dual_lr=%s"
            % (run_dir, list(constraint_spec.active_names), constraint_dual_lr)
        )
        payload_path = run_dir / "config.resolved.json"
        if payload_path.exists():
            payload = json.loads(payload_path.read_text())
            payload["constraints"] = constraint_spec.as_dict()
            payload["constraint_dual_lr"] = float(constraint_dual_lr)
            text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            payload_path.write_text(text)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    writer = TrainAuditWriter(run_dir) if audit else None
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=int(n_itr),
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=writer,
        print_action_choices=print_action_choices,
        parallel=bool(parallel),
        reward_mode=reward_mode,
        learning_mode=learning_mode,
        vocab_size=int(vocab_size),
        use_energy=bool(use_energy),
        constraints=constraint_spec,
        constraint_dual_lr=constraint_dual_lr,
    )
    bc_stats = None
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if learning_mode == "bc_greedy_mec":
            import numpy as np
            from spec.bc_greedy_mec import run_bc_greedy_mec

            bc_stats = run_bc_greedy_mec(
                sess,
                trainer.env,
                trainer.policy.core_policy,
                np.random.RandomState(int(seed)),
                run_dir,
            )
        algo.sync_task_policies_from_core()
        trainer.train()
    return bc_stats


def run_primary_seed(seed, allow_gpu, n_itr=OUTER_ITERS):
    require_gpu_permission(allow_gpu)
    if int(n_itr) != OUTER_ITERS:
        raise ValueError("v0.1 primary train n_itr must be %d, got %s" % (OUTER_ITERS, n_itr))
    run_dir = seed_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": OUTER_ITERS,
            "run_dir": str(run_dir),
        }
    )
    _write_payload(run_dir, payload)
    _train(seed, OUTER_ITERS, run_dir, audit=False)
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_gpu_smoke(seed, allow_gpu):
    """One outer iteration. Not a paper result. Not the 3500 primary run."""
    require_gpu_permission(allow_gpu)
    run_dir = smoke_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_gpu_smoke",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": SMOKE_ITERS,
            "outer_update_count": SMOKE_ITERS,
            "run_dir": str(run_dir),
            "note": "GPU stack smoke; do not cite as evaluation",
        }
    )
    _write_payload(run_dir, payload)
    _train(seed, SMOKE_ITERS, run_dir, audit=False)
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def _run_named(seed, allow_gpu, n_itr, method_id, run_dir, note, parallel=False,
               reward_mode="publication", learning_mode="publication",
               vocab_size=3, use_energy=True):
    require_gpu_permission(allow_gpu)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": method_id,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": int(n_itr),
            "outer_update_count": int(n_itr),
            "run_dir": str(run_dir),
            "audit": True,
            "parallel_env": bool(parallel),
            "reward_mode": reward_mode,
            "learning_mode": learning_mode,
            "vocab_size": int(vocab_size),
            "use_energy": bool(use_energy),
            "end_token": int(vocab_size),
            "greedy_actions": [0, 1] if int(vocab_size) == 2 else [0, 1, 2],
            "note": note,
        }
    )
    if reward_mode == "latency_over_all_mec":
        payload["energy_weight"] = 0.0
        payload["latency_weight"] = 1.0
        payload["latency_ref"] = "l_mec"
        payload["entropy_coefficient"] = 0.0
    if learning_mode == "pomo_elite":
        payload["support_select"] = "elite"
        payload["pomo_shared_baseline"] = True
        payload["entropy_coefficient"] = 0.0
    if learning_mode == "bc_greedy_mec":
        payload["support_select"] = "random"
        payload["bc_expert"] = "greedy_from_mec"
        payload["bc_epochs"] = 8
        payload["entropy_coefficient"] = 0.0
    _write_payload(run_dir, payload)
    bc_stats = _train(
        seed,
        n_itr,
        run_dir,
        audit=True,
        print_action_choices=True,
        parallel=bool(parallel),
        reward_mode=reward_mode,
        learning_mode=learning_mode,
        vocab_size=int(vocab_size),
        use_energy=bool(use_energy),
    )
    if bc_stats is not None:
        payload["bc_pretrain"] = bc_stats
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_learning_probe(seed, allow_gpu, n_itr=PROBE_ITERS):
    """Few outer iters with full stage audit. Not a paper result. Does not replace 3500."""
    if int(n_itr) != PROBE_ITERS:
        raise ValueError("learning probe n_itr must be %d, got %s" % (PROBE_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        PROBE_ITERS,
        "margo_v0.1_learning_probe",
        probe_run_dir(seed),
        "learning probe with per-stage audit; do not cite as evaluation; frozen primary remains 3500",
    )


def run_diagnostic_1k(seed, allow_gpu, n_itr=DIAG_ITERS):
    """1000-iter diagnostic with audit. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG_ITERS:
        raise ValueError("diagnostic n_itr must be %d, got %s" % (DIAG_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG_ITERS,
        "margo_v0.1_diag_1k",
        diag_run_dir(seed),
        "1000-iter diagnostic with audit; not the frozen 3500 primary; paper_result=false",
    )


def run_diagnostic_200(seed, allow_gpu, n_itr=DIAG200_ITERS):
    """200-iter diagnostic with audit. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG200_ITERS:
        raise ValueError("diagnostic-200 n_itr must be %d, got %s" % (DIAG200_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG200_ITERS,
        "margo_v0.1_diag_200",
        diag200_run_dir(seed),
        "200-iter diagnostic with audit; not the frozen 3500 primary; paper_result=false",
    )


def run_parallel_probe(seed, allow_gpu, n_itr=PROBE_ITERS):
    """5-iter spawn-parallel env speed probe. Not a paper result. Does not replace 3500."""
    if int(n_itr) != PROBE_ITERS:
        raise ValueError("parallel probe n_itr must be %d, got %s" % (PROBE_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        PROBE_ITERS,
        "margo_v0.1_parallel_probe",
        parallel_probe_run_dir(seed),
        "spawn-parallel env 5-iter speed probe; not a paper result; frozen primary remains 3500",
        parallel=True,
    )


def run_diagnostic_500(seed, allow_gpu, n_itr=DIAG500_ITERS):
    """500-iter spawn-parallel diagnostic. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG500_ITERS:
        raise ValueError("diagnostic-500 n_itr must be %d, got %s" % (DIAG500_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG500_ITERS,
        "margo_v0.1_diag_500_parallel",
        diag500_run_dir(seed),
        "500-iter spawn-parallel diagnostic; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        parallel=True,
    )


def run_diagnostic_latency_tmec(seed, allow_gpu, n_itr=DIAG_LAT_ITERS):
    """50-iter latency-only T/T_allMEC diagnostic. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG_LAT_ITERS:
        raise ValueError("diagnostic-latency-tmec n_itr must be %d, got %s" % (DIAG_LAT_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG_LAT_ITERS,
        "margo_v0.1_diag_latency_tmec",
        diag_lat_run_dir(seed),
        "latency-only unclipped R=-dT/T_allMEC; energy term off; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        parallel=True,
        reward_mode="latency_over_all_mec",
    )


def run_diagnostic_binary_lat(seed, allow_gpu, n_itr=DIAG_BINARY_LAT_ITERS):
    """50-iter binary no-V2V latency PPO. Energy off. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG_BINARY_LAT_ITERS:
        raise ValueError("diagnostic-binary-lat n_itr must be %d, got %s" % (DIAG_BINARY_LAT_ITERS, n_itr))
    if int(DIAG_BINARY_LAT_VOCAB) != 2:
        raise ValueError("binary-lat vocab must stay 2, got %s" % DIAG_BINARY_LAT_VOCAB)
    if DIAG_BINARY_LAT_METHOD_ID != "margo_v0.2_diag_binary_lat":
        raise ValueError("binary-lat method_id drift %s" % DIAG_BINARY_LAT_METHOD_ID)
    return _run_named(
        seed,
        allow_gpu,
        DIAG_BINARY_LAT_ITERS,
        DIAG_BINARY_LAT_METHOD_ID,
        diag_binary_lat_run_dir(seed),
        "binary vocab=2 no V2V; energy off; R=-dT/T_allMEC; parallel=True env executor only; current Graph2Seq+LSTM; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        parallel=True,
        reward_mode="latency_over_all_mec",
        vocab_size=DIAG_BINARY_LAT_VOCAB,
        use_energy=False,
    )


def run_diagnostic_pomo_tmec(seed, allow_gpu, n_itr=DIAG_POMO_ITERS):
    """50-iter POMO + elite-select latency diagnostic. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG_POMO_ITERS:
        raise ValueError("diagnostic-pomo-tmec n_itr must be %d, got %s" % (DIAG_POMO_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG_POMO_ITERS,
        "margo_v0.1_diag_pomo_tmec",
        diag_pomo_run_dir(seed),
        "POMO per-graph baseline + elite lowest-T select; latency-only R=-dT/T_allMEC; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        parallel=True,
        reward_mode="latency_over_all_mec",
        learning_mode="pomo_elite",
    )


def run_diagnostic_bc_greedy_tmec(seed, allow_gpu, n_itr=DIAG_BC_ITERS):
    """50-iter BC greedy-from-MEC then latency PPO. Not a paper result. Does not rewrite frozen 3500."""
    if int(n_itr) != DIAG_BC_ITERS:
        raise ValueError("diagnostic-bc-greedy-tmec n_itr must be %d, got %s" % (DIAG_BC_ITERS, n_itr))
    return _run_named(
        seed,
        allow_gpu,
        DIAG_BC_ITERS,
        "margo_v0.1_diag_bc_greedy_tmec",
        diag_bc_run_dir(seed),
        "BC greedy-from-MEC then latency-only R=-dT/T_allMEC PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        parallel=True,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )


def run_diagnostic_bc_only_eval(seed, allow_gpu, n_epochs=DIAG_BCONLY_EPOCHS):
    """BC greedy-from-MEC only: more epochs, greedy env T, save ckpt. No PPO. Not a paper result."""
    from spec.bc_greedy_mec import BC_ONLY_EPOCHS

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != DIAG_BCONLY_EPOCHS:
        raise ValueError("diagnostic-bc-only-eval n_epochs must be %d, got %s" % (DIAG_BCONLY_EPOCHS, n_epochs))
    if int(n_epochs) != BC_ONLY_EPOCHS:
        raise ValueError("BC_ONLY_EPOCHS mismatch: campaign %s spec %s" % (n_epochs, BC_ONLY_EPOCHS))
    run_dir = diag_bconly_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_only_eval",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "bc_epochs": DIAG_BCONLY_EPOCHS,
            "bc_expert": "greedy_from_mec",
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "BC-only greedy-from-MEC + greedy rollout T + bc_core.ckpt; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    import numpy as np
    import tensorflow as tf
    from spec.bc_greedy_mec import run_bc_greedy_mec

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bc_stats = run_bc_greedy_mec(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed)),
            run_dir,
            epochs=DIAG_BCONLY_EPOCHS,
            greedy_env_eval=True,
            save_ckpt=ckpt_path,
        )
    payload["gpu_finished"] = True
    payload["bc_pretrain"] = bc_stats
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_bc_continue(seed, allow_gpu, n_epochs=DIAG_BCCONT_MAX_EPOCHS):
    """Resume BC from bc_only ckpt until CE plateau. No PPO. Not a paper result."""
    from spec.bc_greedy_mec import (
        BC_CONTINUE_MAX_EPOCHS,
        BC_CONTINUE_MIN_DELTA,
        BC_CONTINUE_PATIENCE,
        run_bc_greedy_mec,
    )
    from spec.phase4_campaign import RUNS_ROOT

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != DIAG_BCCONT_MAX_EPOCHS:
        raise ValueError("diagnostic-bc-continue n_epochs must be %d, got %s" % (DIAG_BCCONT_MAX_EPOCHS, n_epochs))
    if int(n_epochs) != BC_CONTINUE_MAX_EPOCHS:
        raise ValueError("BC_CONTINUE_MAX_EPOCHS mismatch: campaign %s spec %s" % (n_epochs, BC_CONTINUE_MAX_EPOCHS))
    src_ckpt = diag_bconly_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_only ckpt missing: %s" % src_ckpt)
    run_dir = diag_bccont_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_continue",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "bc_epochs_max": DIAG_BCCONT_MAX_EPOCHS,
            "bc_early_stop_patience": BC_CONTINUE_PATIENCE,
            "bc_early_stop_min_delta": BC_CONTINUE_MIN_DELTA,
            "load_ckpt": str(src_ckpt),
            "bc_expert": "greedy_from_mec",
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "resume BC from bc_only ckpt until CE plateau + greedy rollout T; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    cache_path = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bc_stats = run_bc_greedy_mec(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 40),
            run_dir,
            epochs=DIAG_BCCONT_MAX_EPOCHS,
            greedy_env_eval=True,
            save_ckpt=ckpt_path,
            load_ckpt=src_ckpt,
            cache_path=cache_path,
            early_stop_patience=BC_CONTINUE_PATIENCE,
            early_stop_min_delta=BC_CONTINUE_MIN_DELTA,
        )
    payload["gpu_finished"] = True
    payload["bc_pretrain"] = bc_stats
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_kl_bc_ppo(seed, allow_gpu, n_itr=DIAG_KLPPO_ITERS):
    """Load frozen π_BC, critic warmup, then KL(π||π_BC) latency PPO. Not a paper result."""
    from spec.kl_bc_anchor import load_named_policy_from_core_ckpt

    require_gpu_permission(allow_gpu)
    if int(n_itr) != DIAG_KLPPO_ITERS:
        raise ValueError("diagnostic-kl-bc-ppo n_itr must be %d, got %s" % (DIAG_KLPPO_ITERS, n_itr))
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    run_dir = diag_klppo_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_kl_bc_ppo",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": DIAG_KLPPO_ITERS,
            "outer_update_count": DIAG_KLPPO_ITERS,
            "ppo": True,
            "kl_to": "frozen_pi_bc",
            "bc_kl_coef": DIAG_KLPPO_KL_COEF,
            "critic_warmup_iters": DIAG_KLPPO_WARMUP_ITERS,
            "load_ckpt": str(src_ckpt),
            "learning_mode": "kl_bc_ppo",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "entropy_coefficient": 0.0,
            "support_select": "random",
            "parallel_env": True,
            "audit": True,
            "run_dir": str(run_dir),
            "note": "load bc_continue ckpt; 5-iter qvalue-only warmup then KL(π||frozen π_BC) latency PPO; not unconstrained PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    writer = TrainAuditWriter(run_dir)
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=DIAG_KLPPO_ITERS,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=writer,
        print_action_choices=True,
        parallel=True,
        reward_mode="latency_over_all_mec",
        learning_mode="kl_bc_ppo",
        bc_kl_coef=DIAG_KLPPO_KL_COEF,
        critic_warmup_iters=DIAG_KLPPO_WARMUP_ITERS,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        n_restored = load_named_policy_from_core_ckpt(trainer.bc_policy, src_ckpt, sess)
        payload["bc_frozen_vars_loaded"] = int(n_restored)
        algo.sync_task_policies_from_core()
        print("kl_bc_ppo_load_ckpt %s frozen_vars=%d" % (src_ckpt, n_restored))
        trainer.train()
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_bc_fewshot(seed, allow_gpu):
    """Few-shot from bc_continue: k0 vs k3 PPO vs encoder-frozen CE. Not a paper result."""
    from spec.bc_fewshot import (
        CE_LR,
        CE_STEPS,
        build_scratch_held_out_evaluator,
        encoder_frozen_ce_ops,
        evaluate_split,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    run_dir = diag_bcfew_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_fewshot",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": True,
            "load_ckpt": str(src_ckpt),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "inner_ce_steps": CE_STEPS,
            "inner_ce_lr": CE_LR,
            "encoder_frozen": True,
            "adaptations": ["k0", "k3_ppo", "k3_ce"],
            "run_dir": str(run_dir),
            "note": "few-shot from bc_continue ckpt on held-out val+meta-test; k0 vs inner PPO vs encoder-frozen CE on greedy-from-MEC support tokens; not unconstrained train PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    val_eval = trainer.held_out_evaluator
    test_eval = build_scratch_held_out_evaluator(
        test_env,
        trainer.policy.core_policy,
        "metatest_policy",
        int(seed) + 2,
    )
    val_ce = encoder_frozen_ce_ops(val_eval.policy, "ce_val")
    test_ce = encoder_frozen_ce_ops(test_eval.policy, "ce_test")
    payload["ce_val_n_adapt"] = int(val_ce["n_adapt"])
    payload["ce_val_n_frozen"] = int(val_ce["n_frozen"])
    payload["ce_test_n_adapt"] = int(test_ce["n_adapt"])
    payload["ce_test_n_frozen"] = int(test_ce["n_frozen"])
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print(
            "bc_fewshot_load_ckpt %s val_adapt=%d val_frozen=%d test_adapt=%d test_frozen=%d"
            % (
                src_ckpt,
                val_ce["n_adapt"],
                val_ce["n_frozen"],
                test_ce["n_adapt"],
                test_ce["n_frozen"],
            )
        )
        val_row = evaluate_split(val_eval, sess, val_ce, "validation")
        test_row = evaluate_split(test_eval, sess, test_ce, "meta_test")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "fewshot_eval.json").write_text(
        json.dumps({"validation": val_row, "meta_test": test_row}, indent=2, sort_keys=True) + "\n"
    )
    print(
        "bc_fewshot_verdict val_ppo_destroyed=%s val_ce_helps=%s test_ppo_destroyed=%s test_ce_helps=%s"
        % (
            val_row["ppo_destroyed"],
            val_row["ce_helps"],
            test_row["ppo_destroyed"],
            test_row["ce_helps"],
        )
    )
    return run_dir


def run_diagnostic_bc_scheduled(seed, allow_gpu, n_epochs=DIAG_BCSS_EPOCHS):
    """Scheduled-sampling BC from bc_continue, then unseen greedy. No PPO. Not a paper result."""
    from spec.bc_greedy_mec import eval_loaded_policy_on_env
    from spec.bc_scheduled import SS_EPOCHS, run_scheduled_bc
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != DIAG_BCSS_EPOCHS:
        raise ValueError("diagnostic-bc-scheduled n_epochs must be %d, got %s" % (DIAG_BCSS_EPOCHS, n_epochs))
    if int(n_epochs) != SS_EPOCHS:
        raise ValueError("SS_EPOCHS mismatch: campaign %s spec %s" % (n_epochs, SS_EPOCHS))
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    run_dir = diag_bcss_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_scheduled",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "ss_epochs": DIAG_BCSS_EPOCHS,
            "encoder_frozen": True,
            "load_ckpt": str(src_ckpt),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "scheduled-sampling BC from bc_continue ckpt, encoder frozen, then greedy decode train+val+meta-test; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from spec.phase4_campaign import RUNS_ROOT
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        ss_stats = run_scheduled_bc(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 50),
            run_dir=run_dir,
            epochs=DIAG_BCSS_EPOCHS,
            load_ckpt=src_ckpt,
            save_ckpt=ckpt_path,
            cache_path=train_cache,
        )
        print("bc_ss_eval_unseen")
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    payload["train"] = ss_stats.get("greedy_rollout")
    payload["ss"] = {
        "best_epoch": ss_stats.get("best_epoch"),
        "best_greedy_token_acc": ss_stats.get("best_greedy_token_acc"),
        "greedy_token_acc_before": ss_stats.get("greedy_token_acc_before"),
        "n_adapt": ss_stats.get("n_adapt"),
        "n_frozen": ss_stats.get("n_frozen"),
    }
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "scheduled_eval.json").write_text(
        json.dumps(
            {
                "train": ss_stats.get("greedy_rollout"),
                "validation": val_row,
                "meta_test": test_row,
                "ss": payload["ss"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    vg = val_row["greedy"]
    tg = test_row["greedy"]
    print(
        "bc_ss_verdict train_T=%.1f val_T=%.1f test_T=%.1f val_token=%.3f test_token=%.3f"
        % (
            ss_stats["greedy_rollout"]["greedy_T_mean"],
            vg["greedy_T_mean"],
            tg["greedy_T_mean"],
            vg["greedy_token_acc_vs_expert"],
            tg["greedy_token_acc_vs_expert"],
        )
    )
    return run_dir


def run_diagnostic_hamming2_expert(seed):
    """CPU Hamming-2 from greedy_from_mec. No GPU. No PPO. Not a paper result. Frozen primary remains 3500."""
    from spec.hamming2_probe import H2_N_GRAPHS, run_hamming2_probe
    from spec.phase4_campaign import DIAG_H2_N_GRAPHS

    if int(H2_N_GRAPHS) != int(DIAG_H2_N_GRAPHS):
        raise ValueError("H2_N_GRAPHS mismatch campaign %s probe %s" % (DIAG_H2_N_GRAPHS, H2_N_GRAPHS))
    run_dir = diag_h2_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_hamming2_expert",
            "paper_result": False,
            "gpu_requested": False,
            "gpu_finished": False,
            "ppo": False,
            "n_graphs": DIAG_H2_N_GRAPHS,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "run_dir": str(run_dir),
            "note": "CPU Hamming-2 from greedy_from_mec expert; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    _, stats = run_hamming2_probe(seed=seed, n_graphs=DIAG_H2_N_GRAPHS, run_dir=run_dir)
    payload["hamming2"] = {
        "verdict": stats.get("verdict"),
        "greedy_T_mean": stats.get("greedy_T_mean"),
        "h2_T_mean": stats.get("h2_T_mean"),
        "mec_T_mean": stats.get("mec_T_mean"),
        "mean_delta_h2": stats.get("mean_delta_h2"),
        "frac_h2_improved": stats.get("frac_h2_improved"),
        "frac_h1_improved": stats.get("frac_h1_improved"),
    }
    payload["gpu_finished"] = False
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_pairfrac_mec(seed):
    """CPU motif frac_pos from all-MEC. No GPU. No PPO. Not a paper result. Frozen primary remains 3500."""
    from spec.pair_frac import PAIRFRAC_N_GRAPHS, run_pairfrac_mec
    from spec.phase4_campaign import DIAG_PAIRFRAC_N_GRAPHS

    if int(PAIRFRAC_N_GRAPHS) != int(DIAG_PAIRFRAC_N_GRAPHS):
        raise ValueError(
            "PAIRFRAC_N_GRAPHS mismatch campaign %s probe %s" % (DIAG_PAIRFRAC_N_GRAPHS, PAIRFRAC_N_GRAPHS)
        )
    if DIAG_PAIRFRAC_METHOD_ID != "margo_v0.2_diag_pairfrac_mec":
        raise ValueError("pairfrac method_id drift %s" % DIAG_PAIRFRAC_METHOD_ID)
    run_dir = diag_pairfrac_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.2_diag_pairfrac_mec",
            "paper_result": False,
            "gpu_requested": False,
            "gpu_finished": False,
            "ppo": False,
            "n_graphs": DIAG_PAIRFRAC_N_GRAPHS,
            "start": "all_mec",
            "outer_iterations": 0,
            "outer_update_count": 0,
            "run_dir": str(run_dir),
            "note": "CPU motif pair frac_pos from all-MEC; no train; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    _, stats = run_pairfrac_mec(seed=seed, n_graphs=DIAG_PAIRFRAC_N_GRAPHS, run_dir=run_dir)
    payload["pairfrac"] = {
        "verdict": stats.get("verdict"),
        "frac_pos": stats.get("frac_pos"),
        "n_pos": stats.get("n_pos"),
        "n_pairs": stats.get("n_pairs"),
        "n_eval": stats.get("n_eval"),
        "mean_delta_pos": stats.get("mean_delta_pos"),
        "frac_graphs_with_pos": stats.get("frac_graphs_with_pos"),
        "clone_ref_frac_pos": stats.get("clone_ref_frac_pos"),
        "proceed_rewrite": stats.get("proceed_rewrite"),
        "mec_T_mean": stats.get("mec_T_mean"),
    }
    payload["gpu_finished"] = False
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_rewrite_mec(seed, allow_gpu, n_epochs=DIAG_REWRITE_EPOCHS):
    """BC greedy_from_mec + λ all-MEC joint CE, greedy+K neural apply. No search. Not 3500."""
    from spec.rewrite_mec import (
        REWRITE_A_TEST_T,
        REWRITE_A_VAL_T,
        REWRITE_EPOCHS,
        REWRITE_K,
        REWRITE_LAMBDA,
        classify_rewrite_verdict,
        eval_rewrite_on_env,
        run_rewrite_train,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != int(DIAG_REWRITE_EPOCHS):
        raise ValueError("rewrite n_epochs must be %d, got %s" % (DIAG_REWRITE_EPOCHS, n_epochs))
    if int(REWRITE_EPOCHS) != int(DIAG_REWRITE_EPOCHS):
        raise ValueError("REWRITE_EPOCHS mismatch campaign %s spec %s" % (DIAG_REWRITE_EPOCHS, REWRITE_EPOCHS))
    if abs(float(REWRITE_LAMBDA) - float(DIAG_REWRITE_LAMBDA)) > 1e-12:
        raise ValueError("REWRITE_LAMBDA mismatch")
    if abs(float(REWRITE_A_VAL_T) - float(DIAG_REWRITE_A_VAL_T)) > 1e-12:
        raise ValueError("REWRITE_A_VAL_T mismatch")
    if abs(float(REWRITE_A_TEST_T) - float(DIAG_REWRITE_A_TEST_T)) > 1e-12:
        raise ValueError("REWRITE_A_TEST_T mismatch")
    if tuple(REWRITE_K) != tuple(DIAG_REWRITE_K):
        raise ValueError("REWRITE_K mismatch")
    if DIAG_REWRITE_METHOD_ID != "margo_v0.2_diag_rewrite_mec":
        raise ValueError("rewrite method_id drift %s" % DIAG_REWRITE_METHOD_ID)
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    from spec.phase4_campaign import RUNS_ROOT

    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    joint_cache = Path(RUNS_ROOT) / "rewrite_joint_from_allmec_train.npz"
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("greedy_from_mec expert cache missing: %s" % p)
    run_dir = diag_rewrite_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_REWRITE_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "pair_search_at_eval": False,
            "bc_expert": "greedy_from_mec",
            "joint_start": "all_mec",
            "load_ckpt": str(src_ckpt),
            "lam": float(DIAG_REWRITE_LAMBDA),
            "bc_epochs": int(DIAG_REWRITE_EPOCHS),
            "rewrite_k": list(DIAG_REWRITE_K),
            "baseline_a": "margo_v0.1_diag_bc_unseen",
            "baseline_a_val_T": float(DIAG_REWRITE_A_VAL_T),
            "baseline_a_test_T": float(DIAG_REWRITE_A_TEST_T),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "rewrite: L_BC greedy_from_mec + λ L_joint all-MEC; greedy+K neural apply; no schedule search; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_stats = run_rewrite_train(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 91),
            run_dir,
            load_ckpt=src_ckpt,
            expert_cache=train_cache,
            joint_cache=joint_cache,
            epochs=DIAG_REWRITE_EPOCHS,
            lam=DIAG_REWRITE_LAMBDA,
            save_ckpt=ckpt_path,
        )
        print("rewrite_eval_unseen")
        val_row = eval_rewrite_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
            k_max=max(DIAG_REWRITE_K),
        )
        test_row = eval_rewrite_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
            k_max=max(DIAG_REWRITE_K),
        )
    vg0 = val_row["greedy"]
    verdict = classify_rewrite_verdict(val_row["best_T"], vg0["greedy_local_frac"])
    out = {
        "verdict": verdict,
        "baseline_a_val_T": float(DIAG_REWRITE_A_VAL_T),
        "baseline_a_test_T": float(DIAG_REWRITE_A_TEST_T),
        "validation": val_row,
        "meta_test": test_row,
        "train": train_stats.get("greedy_rollout"),
        "labels": train_stats.get("labels"),
        "lam": float(DIAG_REWRITE_LAMBDA),
        "rewrite_k": list(DIAG_REWRITE_K),
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
        "joint_start": "all_mec",
    }
    payload["train"] = train_stats.get("greedy_rollout")
    payload["labels"] = train_stats.get("labels")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "rewrite_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "rewrite_verdict=%s val_bestT=%.1f k=%d A=%.1f local0=%.3f test_bestT=%.1f A_test=%.1f frac_pos=%.3f"
        % (
            verdict,
            val_row["best_T"],
            val_row["best_k"],
            DIAG_REWRITE_A_VAL_T,
            vg0["greedy_local_frac"],
            test_row["best_T"],
            DIAG_REWRITE_A_TEST_T,
            float((train_stats.get("labels") or {}).get("frac_pos") or 0.0),
        )
    )
    return run_dir


def _oracle_envs():
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from spec.split_loader import meta_train_graph_prefixes

    val_env, test_env = _cavia_held_out_envs()
    train_env = OffloadingEnvironment(
        resource_cluster=val_env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=meta_train_graph_prefixes(),
        time_major=False,
    )
    return train_env, val_env, test_env


def run_diagnostic_oracle_dist(seed, allow_gpu, n_epochs=DIAG_ORACLE_DIST_EPOCHS):
    """True dist_id embed every decoder step, encoder frozen. No search. Not 3500."""
    from spec.oracle_dist import (
        ORACLE_A_TEST_T,
        ORACLE_A_VAL_T,
        ORACLE_EPOCHS,
        ORACLE_IDENTITY_T_VAL_HI,
        ORACLE_IDENTITY_T_VAL_LO,
        ORACLE_N_DIST,
        ORACLE_TRAIN_REF_T,
        ORACLE_Z_DIM,
        classify_oracle_verdict,
        greedy_oracle_eval,
        is_oracle_dist_var_name,
        run_oracle_train,
    )
    from policies.meta_seq2seq_policy import Seq2SeqPolicy

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != int(DIAG_ORACLE_DIST_EPOCHS):
        raise ValueError("oracle n_epochs must be %d, got %s" % (DIAG_ORACLE_DIST_EPOCHS, n_epochs))
    if int(ORACLE_EPOCHS) != int(DIAG_ORACLE_DIST_EPOCHS):
        raise ValueError("ORACLE_EPOCHS mismatch campaign %s spec %s" % (DIAG_ORACLE_DIST_EPOCHS, ORACLE_EPOCHS))
    if int(ORACLE_Z_DIM) != int(DIAG_ORACLE_DIST_Z_DIM):
        raise ValueError("ORACLE_Z_DIM mismatch")
    if int(ORACLE_N_DIST) != int(DIAG_ORACLE_DIST_N_DIST):
        raise ValueError("ORACLE_N_DIST mismatch")
    if abs(float(ORACLE_A_VAL_T) - float(DIAG_ORACLE_DIST_A_VAL_T)) > 1e-12:
        raise ValueError("ORACLE_A_VAL_T mismatch")
    if abs(float(ORACLE_A_TEST_T) - float(DIAG_ORACLE_DIST_A_TEST_T)) > 1e-12:
        raise ValueError("ORACLE_A_TEST_T mismatch")
    if abs(float(ORACLE_TRAIN_REF_T) - float(DIAG_ORACLE_DIST_TRAIN_REF_T)) > 1e-12:
        raise ValueError("ORACLE_TRAIN_REF_T mismatch")
    if abs(float(ORACLE_IDENTITY_T_VAL_LO) - float(DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO)) > 1e-12:
        raise ValueError("oracle identity lo drift")
    if abs(float(ORACLE_IDENTITY_T_VAL_HI) - float(DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI)) > 1e-12:
        raise ValueError("oracle identity hi drift")
    if DIAG_ORACLE_DIST_METHOD_ID != "margo_v0.2_diag_oracle_dist":
        raise ValueError("oracle method_id drift %s" % DIAG_ORACLE_DIST_METHOD_ID)
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    from spec.phase4_campaign import RUNS_ROOT

    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("greedy_from_mec expert cache missing: %s" % p)
    run_dir = diag_oracle_dist_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_ORACLE_DIST_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "pair_search_at_eval": False,
            "encoder_frozen": True,
            "oracle_dist": True,
            "load_ckpt": str(src_ckpt),
            "bc_epochs": int(DIAG_ORACLE_DIST_EPOCHS),
            "oracle_z_dim": int(DIAG_ORACLE_DIST_Z_DIM),
            "baseline_a": "margo_v0.1_diag_bc_unseen",
            "baseline_a_val_T": float(DIAG_ORACLE_DIST_A_VAL_T),
            "baseline_a_test_T": float(DIAG_ORACLE_DIST_A_TEST_T),
            "train_ref_T": float(DIAG_ORACLE_DIST_TRAIN_REF_T),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "oracle dist_id embed every decoder step; Graph2Seq frozen; greedy eval; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    train_env, val_env, test_env = _oracle_envs()
    policy = Seq2SeqPolicy(
        obs_dim=train_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="oracle_policy",
        enable_oracle_dist=True,
        oracle_n_dist=int(DIAG_ORACLE_DIST_N_DIST),
        oracle_z_dim=int(DIAG_ORACLE_DIST_Z_DIM),
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        from spec.oracle_dist import load_oracle_from_bc_ckpt

        n_loaded = load_oracle_from_bc_ckpt(policy, src_ckpt, sess)
        n_expect = len([v for v in policy.get_variables() if not is_oracle_dist_var_name(v.name)])
        if int(n_loaded) != int(n_expect):
            raise ValueError("oracle ckpt loaded %s vars, want %s" % (n_loaded, n_expect))
        print("oracle_identity_eval")
        ident = greedy_oracle_eval(
            sess, val_env, policy, cache_path=val_cache, split_name="identity_validation"
        )
        ident_t = float(ident["greedy_T_mean"])
        if ident_t < float(DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO) or ident_t > float(
            DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI
        ):
            raise ValueError(
                "oracle identity val T=%.1f outside [%.1f, %.1f]"
                % (
                    ident_t,
                    DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO,
                    DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI,
                )
            )
        print("oracle_identity_ok T=%.1f" % ident_t)
        import numpy as np

        train_stats = run_oracle_train(
            sess,
            train_env,
            policy,
            np.random.RandomState(int(seed) + 93),
            run_dir,
            load_ckpt=None,
            expert_cache=train_cache,
            epochs=DIAG_ORACLE_DIST_EPOCHS,
            save_ckpt=ckpt_path,
            identity_row=ident,
        )
        print("oracle_eval_unseen")
        val_row = greedy_oracle_eval(
            sess, val_env, policy, cache_path=val_cache, split_name="validation"
        )
        test_row = greedy_oracle_eval(
            sess, test_env, policy, cache_path=test_cache, split_name="meta_test"
        )
    train_t = float((train_stats.get("greedy_rollout") or {}).get("greedy_T_mean") or 0.0)
    verdict = classify_oracle_verdict(
        val_row["greedy_T_mean"], val_row["greedy_local_frac"], train_t=train_t
    )
    out = {
        "verdict": verdict,
        "baseline_a_val_T": float(DIAG_ORACLE_DIST_A_VAL_T),
        "baseline_a_test_T": float(DIAG_ORACLE_DIST_A_TEST_T),
        "train_ref_T": float(DIAG_ORACLE_DIST_TRAIN_REF_T),
        "identity_val_T": ident_t,
        "validation": val_row,
        "meta_test": test_row,
        "train": train_stats.get("greedy_rollout"),
        "encoder_frozen": True,
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
    }
    payload["train"] = train_stats.get("greedy_rollout")
    payload["identity"] = ident
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "oracle_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "oracle_verdict=%s valT=%.1f A=%.1f local=%.3f testT=%.1f A_test=%.1f trainT=%.1f identT=%.1f"
        % (
            verdict,
            val_row["greedy_T_mean"],
            DIAG_ORACLE_DIST_A_VAL_T,
            val_row["greedy_local_frac"],
            test_row["greedy_T_mean"],
            DIAG_ORACLE_DIST_A_TEST_T,
            train_t,
            ident_t,
        )
    )
    return run_dir


def run_diagnostic_motif_expert(seed):
    """CPU motif audit of greedy_from_mec. No GPU. No PPO. Not a paper result. Frozen primary remains 3500."""
    from spec.motif_audit import MOTIF_N_TRAIN, run_motif_audit
    from spec.phase4_campaign import DIAG_MOTIF_N_TRAIN

    if int(MOTIF_N_TRAIN) != int(DIAG_MOTIF_N_TRAIN):
        raise ValueError(
            "MOTIF_N_TRAIN mismatch campaign %s probe %s" % (DIAG_MOTIF_N_TRAIN, MOTIF_N_TRAIN)
        )
    run_dir = diag_motif_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_motif_expert",
            "paper_result": False,
            "gpu_requested": False,
            "gpu_finished": False,
            "ppo": False,
            "n_train_graphs": DIAG_MOTIF_N_TRAIN,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "run_dir": str(run_dir),
            "note": "CPU motif audit of greedy_from_mec expert; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    _, stats = run_motif_audit(seed=seed, run_dir=run_dir)
    payload["motif"] = {
        "train_verdict": stats.get("train_verdict"),
        "val_verdict": stats.get("val_verdict"),
        "metatest_verdict": stats.get("metatest_verdict"),
        "architecture_hint": stats.get("architecture_hint"),
        "train_n_non_mean": stats.get("splits", {}).get("meta_train", {}).get("n_non_mean"),
        "train_n_cc_mean": stats.get("splits", {}).get("meta_train", {}).get("n_cc_mean"),
        "train_local_nbr_frac_mean": stats.get("splits", {}).get("meta_train", {}).get("local_nbr_frac_mean"),
        "train_edge_homophily_mean": stats.get("splits", {}).get("meta_train", {}).get("edge_homophily_mean"),
    }
    payload["gpu_finished"] = False
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_2opt_expert(seed):
    """CPU iterative 2-opt from greedy_from_mec. No GPU. No PPO. Not a paper result. Frozen primary remains 3500."""
    from spec.twopt_expert import MAX_H2_ROUNDS, run_twopt_expert
    from spec.phase4_campaign import DIAG_TWOPT_MAX_H2_ROUNDS

    if int(MAX_H2_ROUNDS) != int(DIAG_TWOPT_MAX_H2_ROUNDS):
        raise ValueError(
            "MAX_H2_ROUNDS mismatch campaign %s probe %s" % (DIAG_TWOPT_MAX_H2_ROUNDS, MAX_H2_ROUNDS)
        )
    run_dir = diag_twopt_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_2opt_expert",
            "paper_result": False,
            "gpu_requested": False,
            "gpu_finished": False,
            "ppo": False,
            "max_h2_rounds": DIAG_TWOPT_MAX_H2_ROUNDS,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "run_dir": str(run_dir),
            "note": "CPU iterative 2-opt teacher from greedy_from_mec; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    _, stats = run_twopt_expert(seed=seed, run_dir=run_dir)
    train = stats.get("splits", {}).get("meta_train", {})
    payload["twopt"] = {
        "train_greedy_T_mean": train.get("greedy_T_mean"),
        "train_twopt_T_mean": train.get("twopt_T_mean"),
        "train_mean_delta_twopt": train.get("mean_delta_twopt"),
        "train_mean_extra_vs_first_h2": train.get("mean_extra_vs_first_h2"),
        "train_h2_moves_mean": train.get("h2_moves_mean"),
        "caches": stats.get("caches"),
    }
    payload["gpu_finished"] = False
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_bc_2opt(seed, allow_gpu, n_epochs=DIAG_BCCONT_MAX_EPOCHS):
    """BC on 2-opt expert labels from bc_continue, then unseen greedy. No PPO. Frozen primary remains 3500."""
    from spec.bc_greedy_mec import (
        BC_CONTINUE_MAX_EPOCHS,
        BC_CONTINUE_MIN_DELTA,
        BC_CONTINUE_PATIENCE,
        eval_loaded_policy_on_env,
        run_bc_greedy_mec,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != DIAG_BCCONT_MAX_EPOCHS:
        raise ValueError("diagnostic-bc-2opt n_epochs must be %d, got %s" % (DIAG_BCCONT_MAX_EPOCHS, n_epochs))
    if int(n_epochs) != BC_CONTINUE_MAX_EPOCHS:
        raise ValueError("BC_CONTINUE_MAX_EPOCHS mismatch: campaign %s spec %s" % (n_epochs, BC_CONTINUE_MAX_EPOCHS))
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    test_cache = CACHE_TWOPT["meta_test"]
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_bc2opt_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_2opt",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "bc_epochs_max": DIAG_BCCONT_MAX_EPOCHS,
            "bc_early_stop_patience": BC_CONTINUE_PATIENCE,
            "bc_early_stop_min_delta": BC_CONTINUE_MIN_DELTA,
            "load_ckpt": str(src_ckpt),
            "bc_expert": "iterative_2opt",
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "train_cache": str(train_cache),
            "run_dir": str(run_dir),
            "note": "BC on 2-opt expert labels from bc_continue ckpt, then greedy decode train+val+meta-test; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bc_stats = run_bc_greedy_mec(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 60),
            run_dir,
            epochs=DIAG_BCCONT_MAX_EPOCHS,
            greedy_env_eval=True,
            save_ckpt=ckpt_path,
            load_ckpt=src_ckpt,
            cache_path=train_cache,
            early_stop_patience=BC_CONTINUE_PATIENCE,
            early_stop_min_delta=BC_CONTINUE_MIN_DELTA,
        )
        print("bc_2opt_eval_unseen")
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    payload["bc_pretrain"] = bc_stats
    payload["train"] = bc_stats.get("greedy_rollout")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "bc2opt_eval.json").write_text(
        json.dumps(
            {
                "train": bc_stats.get("greedy_rollout"),
                "validation": val_row,
                "meta_test": test_row,
                "bc": {
                    "bc_epochs_ran": bc_stats.get("bc_epochs_ran"),
                    "bc_best_loss": bc_stats.get("bc_best_loss"),
                    "token_acc": bc_stats.get("token_acc"),
                    "greedy_decode_acc": bc_stats.get("greedy_decode_acc"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tr = bc_stats.get("greedy_rollout") or {}
    vg = val_row["greedy"]
    tg = test_row["greedy"]
    print(
        "bc_2opt_verdict train_T=%.1f expert=%.1f val_T=%.1f val_ex=%.1f test_T=%.1f test_ex=%.1f val_token=%.3f test_token=%.3f"
        % (
            tr.get("greedy_T_mean", float("nan")),
            tr.get("expert_T_mean", float("nan")),
            vg["greedy_T_mean"],
            vg["expert_T_mean"],
            tg["greedy_T_mean"],
            tg["expert_T_mean"],
            vg["greedy_token_acc_vs_expert"],
            tg["greedy_token_acc_vs_expert"],
        )
    )
    return run_dir


def run_diagnostic_pair_head(seed, allow_gpu):
    """Frozen encoder pair/motif head + top-K x 9 refine. No PPO. Frozen primary remains 3500."""
    from spec.pair_head import (
        ORACLE_N_VAL,
        PAIR_TOP_K,
        run_pair_head,
    )
    from spec.phase4_campaign import DIAG_PAIR_ORACLE_N_VAL, DIAG_PAIR_TOP_K
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if int(PAIR_TOP_K) != int(DIAG_PAIR_TOP_K):
        raise ValueError("PAIR_TOP_K mismatch campaign %s spec %s" % (DIAG_PAIR_TOP_K, PAIR_TOP_K))
    if int(ORACLE_N_VAL) != int(DIAG_PAIR_ORACLE_N_VAL):
        raise ValueError("ORACLE_N_VAL mismatch campaign %s spec %s" % (DIAG_PAIR_ORACLE_N_VAL, ORACLE_N_VAL))
    src_ckpt = diag_bc2opt_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_2opt ckpt missing: %s" % src_ckpt)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    test_cache = CACHE_TWOPT["meta_test"]
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_pair_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_pair_head",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_replaced": False,
            "load_ckpt": str(src_ckpt),
            "bc_expert": "iterative_2opt",
            "learning_mode": "pair_head_aux",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "top_k": DIAG_PAIR_TOP_K,
            "oracle_n_val": DIAG_PAIR_ORACLE_N_VAL,
            "train_cache": str(train_cache),
            "run_dir": str(run_dir),
            "note": "auxiliary 9-way pair/motif head on frozen bc_2opt encoder; LSTM decoder unchanged; top-K x 9 schedule refine; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print("pair_load_ckpt %s" % src_ckpt)
        stats = run_pair_head(
            sess,
            trainer.env,
            trainer.held_out_evaluator.env,
            test_env,
            trainer.policy.core_policy,
            train_cache,
            val_cache,
            test_cache,
            seed,
            run_dir,
        )
    payload["pair"] = {
        "verdict": stats.get("verdict"),
        "metrics": stats.get("metrics"),
        "refine": stats.get("refine"),
        "oracle_val": stats.get("oracle_val"),
    }
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_pair_ksweep(seed, allow_gpu):
    """k=10/20/50 pair refine vs same-split motif oracle. No PPO. Frozen primary remains 3500."""
    from spec.pair_head import PAIR_K_SWEEP, run_pair_ksweep
    from spec.phase4_campaign import DIAG_PAIR_K_SWEEP
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if tuple(int(x) for x in PAIR_K_SWEEP) != tuple(int(x) for x in DIAG_PAIR_K_SWEEP):
        raise ValueError("PAIR_K_SWEEP mismatch campaign %s spec %s" % (DIAG_PAIR_K_SWEEP, PAIR_K_SWEEP))
    src_ckpt = diag_bc2opt_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_2opt ckpt missing: %s" % src_ckpt)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    test_cache = CACHE_TWOPT["meta_test"]
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_pair_ksweep_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_pair_ksweep",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_replaced": False,
            "load_ckpt": str(src_ckpt),
            "k_sweep": list(DIAG_PAIR_K_SWEEP),
            "learning_mode": "pair_ksweep",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "train_cache": str(train_cache),
            "run_dir": str(run_dir),
            "note": "k=10/20/50 vs motif oracle on the same val graphs; LSTM decoder unchanged; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print("pair_ksweep_load_ckpt %s" % src_ckpt)
        stats = run_pair_ksweep(
            sess,
            trainer.env,
            trainer.held_out_evaluator.env,
            test_env,
            trainer.policy.core_policy,
            train_cache,
            val_cache,
            test_cache,
            seed,
            run_dir,
        )
    payload["ksweep"] = {
        "curve": stats.get("curve"),
        "oracle_val": stats.get("oracle_val"),
        "k_sweep": stats.get("k_sweep"),
    }
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_pair_ranker(seed, allow_gpu):
    """ΔT pair ranker vs CE vs true-gain rank at k=10/20. No PPO. Frozen primary remains 3500."""
    from spec.pair_ranker import PAIR_RANKER_K, run_pair_ranker
    from spec.phase4_campaign import DIAG_PAIR_RANKER_K
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if tuple(int(x) for x in PAIR_RANKER_K) != tuple(int(x) for x in DIAG_PAIR_RANKER_K):
        raise ValueError("PAIR_RANKER_K mismatch campaign %s spec %s" % (DIAG_PAIR_RANKER_K, PAIR_RANKER_K))
    src_ckpt = diag_bc2opt_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_2opt ckpt missing: %s" % src_ckpt)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    test_cache = CACHE_TWOPT["meta_test"]
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_pair_ranker_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_pair_ranker",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_replaced": False,
            "load_ckpt": str(src_ckpt),
            "k_sweep": list(DIAG_PAIR_RANKER_K),
            "learning_mode": "pair_delta_ranker",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "train_cache": str(train_cache),
            "run_dir": str(run_dir),
            "note": "ΔT ridge ranker vs CE 1-P_BC vs true one-shot gain ranking at k=10/20; LSTM decoder unchanged; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print("pair_ranker_load_ckpt %s" % src_ckpt)
        stats = run_pair_ranker(
            sess,
            trainer.env,
            trainer.held_out_evaluator.env,
            test_env,
            trainer.policy.core_policy,
            train_cache,
            val_cache,
            test_cache,
            seed,
            run_dir,
        )
    payload["ranker"] = {
        "verdict": stats.get("verdict"),
        "rank_metrics": stats.get("rank_metrics"),
        "refine": stats.get("refine"),
        "gain_stats": stats.get("gain_stats"),
        "k_sweep": stats.get("k_sweep"),
    }
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_pair_seq(seed, allow_gpu):
    """CE scan vs multipass vs bestimp sequential pair refine. No PPO. Frozen primary remains 3500."""
    from spec.pair_seq import SEQ_BESTIMP_K, run_pair_seq
    from spec.phase4_campaign import DIAG_PAIR_SEQ_BESTIMP_K
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if tuple(int(x) for x in SEQ_BESTIMP_K) != tuple(int(x) for x in DIAG_PAIR_SEQ_BESTIMP_K):
        raise ValueError(
            "SEQ_BESTIMP_K mismatch campaign %s spec %s" % (DIAG_PAIR_SEQ_BESTIMP_K, SEQ_BESTIMP_K)
        )
    src_ckpt = diag_bc2opt_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_2opt ckpt missing: %s" % src_ckpt)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    test_cache = CACHE_TWOPT["meta_test"]
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_pair_seq_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_pair_seq",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_replaced": False,
            "load_ckpt": str(src_ckpt),
            "bestimp_k": list(DIAG_PAIR_SEQ_BESTIMP_K),
            "learning_mode": "pair_seq_refine",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "train_cache": str(train_cache),
            "run_dir": str(run_dir),
            "note": "CE scan vs multipass vs bestimp on frozen top-k; LSTM decoder unchanged; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print("pair_seq_load_ckpt %s" % src_ckpt)
        stats = run_pair_seq(
            sess,
            trainer.env,
            trainer.held_out_evaluator.env,
            test_env,
            trainer.policy.core_policy,
            train_cache,
            val_cache,
            test_cache,
            seed,
            run_dir,
        )
    payload["seq"] = {
        "verdict": stats.get("verdict"),
        "methods": stats.get("methods"),
        "oracle_ref_val": stats.get("oracle_ref_val"),
    }
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def _cavia_held_out_envs():
    """Val+meta-test envs. Physics cluster matches frozen stack. No train graphs. No PPO."""
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment, Resources
    from spec.split_loader import (
        assert_held_out_prefixes,
        meta_test_graph_prefixes,
        validation_graph_prefixes,
    )

    energy_config = {
        "use_energy": True,
        "reward_mode": "latency_over_all_mec",
        "energy_weight": 0.5,
        "latency_weight": 0.5,
        "rho": 1.0,
        "f_l": 1.0,
        "zeta": 2.0,
        "ptx": 0.1,
        "prx": 0.05,
        "ptx_v2v": 0.06,
        "prx_v2v": 0.03,
        "rho_v2v": 0.7,
        "f_v2v": 1.0,
        "normalize_energy": True,
    }
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        v2v_process_capable=(1.0 * 1024 * 1024),
        v2v_bandwidth=5.0,
        use_energy=True,
        energy_config=energy_config,
    )
    val_paths = validation_graph_prefixes()
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(val_paths, "validation")
    assert_held_out_prefixes(test_paths, "meta_test")
    val_env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=val_paths,
        time_major=False,
    )
    test_env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    return val_env, test_env


def run_diagnostic_cavia(seed, allow_gpu, use_energy=False, strong=False, from_bccont=False):
    """CAVIA-on-z. Default ckpt bc_2opt. from_bccont uses bc_continue. Theta frozen. Not 3500."""
    from spec.cavia_loop import build_cavia_train_ops, run_cavia_phase1
    from spec.cavia_objective import (
        CAVIA_BCCONT_IDENTITY_T_VAL_HI,
        CAVIA_BCCONT_IDENTITY_T_VAL_LO,
        CAVIA_INNER_LR,
        CAVIA_INNER_STEPS,
        CAVIA_STRONG_INNER_LR,
        CAVIA_STRONG_INNER_STEPS,
        CAVIA_Z_DIM,
        CaviaObjective,
        is_cavia_var_name,
    )
    from spec.kl_bc_anchor import load_named_policy_from_core_ckpt
    from policies.meta_seq2seq_policy import Seq2SeqPolicy

    require_gpu_permission(allow_gpu)
    if int(CAVIA_Z_DIM) != int(DIAG_CAVIA_Z_DIM):
        raise ValueError("CAVIA_Z_DIM mismatch campaign %s spec %s" % (DIAG_CAVIA_Z_DIM, CAVIA_Z_DIM))
    strong = bool(strong)
    use_energy = bool(use_energy)
    from_bccont = bool(from_bccont)
    if strong and use_energy:
        raise ValueError("cavia-strong is T-only")
    if from_bccont and (strong or use_energy):
        raise ValueError("cavia-bccont is T-only 20-step")
    ident_lo = None
    ident_hi = None
    if from_bccont:
        src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
        if not src_ckpt.is_file():
            raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
        if abs(float(CAVIA_INNER_LR) - float(DIAG_CAVIA_INNER_LR)) > 1e-12:
            raise ValueError("CAVIA_INNER_LR mismatch campaign %s spec %s" % (DIAG_CAVIA_INNER_LR, CAVIA_INNER_LR))
        if int(CAVIA_INNER_STEPS) != int(DIAG_CAVIA_INNER_STEPS):
            raise ValueError(
                "CAVIA_INNER_STEPS mismatch campaign %s spec %s" % (DIAG_CAVIA_INNER_STEPS, CAVIA_INNER_STEPS)
            )
        if abs(float(CAVIA_BCCONT_IDENTITY_T_VAL_LO) - float(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO)) > 1e-12:
            raise ValueError("bccont identity lo drift")
        if abs(float(CAVIA_BCCONT_IDENTITY_T_VAL_HI) - float(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI)) > 1e-12:
            raise ValueError("bccont identity hi drift")
        inner_lr = float(DIAG_CAVIA_INNER_LR)
        n_steps = int(DIAG_CAVIA_INNER_STEPS)
        method_id = "margo_v0.2_diag_cavia_bccont"
        run_dir = diag_cavia_bccont_run_dir(seed)
        if method_id != DIAG_CAVIA_BCCONT_METHOD_ID:
            raise ValueError("bccont method_id drift %s" % DIAG_CAVIA_BCCONT_METHOD_ID)
        ident_lo = float(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO)
        ident_hi = float(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI)
        objective = CaviaObjective(use_energy=False)
    else:
        src_ckpt = diag_bc2opt_run_dir(seed) / "ckpt" / "bc_core.ckpt"
        if not src_ckpt.is_file():
            raise FileNotFoundError("bc_2opt ckpt missing: %s" % src_ckpt)
        objective = CaviaObjective(use_energy=use_energy)
        if strong:
            if abs(float(CAVIA_STRONG_INNER_LR) - float(DIAG_CAVIA_STRONG_INNER_LR)) > 1e-12:
                raise ValueError(
                    "CAVIA_STRONG_INNER_LR mismatch campaign %s spec %s"
                    % (DIAG_CAVIA_STRONG_INNER_LR, CAVIA_STRONG_INNER_LR)
                )
            if int(CAVIA_STRONG_INNER_STEPS) != int(DIAG_CAVIA_STRONG_INNER_STEPS):
                raise ValueError(
                    "CAVIA_STRONG_INNER_STEPS mismatch campaign %s spec %s"
                    % (DIAG_CAVIA_STRONG_INNER_STEPS, CAVIA_STRONG_INNER_STEPS)
                )
            inner_lr = float(DIAG_CAVIA_STRONG_INNER_LR)
            n_steps = int(DIAG_CAVIA_STRONG_INNER_STEPS)
            method_id = "margo_v0.2_diag_cavia_strong"
            run_dir = diag_cavia_strong_run_dir(seed)
            if method_id != DIAG_CAVIA_STRONG_METHOD_ID:
                raise ValueError("strong method_id drift %s" % DIAG_CAVIA_STRONG_METHOD_ID)
        else:
            if abs(float(CAVIA_INNER_LR) - float(DIAG_CAVIA_INNER_LR)) > 1e-12:
                raise ValueError("CAVIA_INNER_LR mismatch campaign %s spec %s" % (DIAG_CAVIA_INNER_LR, CAVIA_INNER_LR))
            if int(CAVIA_INNER_STEPS) != int(DIAG_CAVIA_INNER_STEPS):
                raise ValueError(
                    "CAVIA_INNER_STEPS mismatch campaign %s spec %s" % (DIAG_CAVIA_INNER_STEPS, CAVIA_INNER_STEPS)
                )
            inner_lr = float(DIAG_CAVIA_INNER_LR)
            n_steps = int(DIAG_CAVIA_INNER_STEPS)
            if use_energy:
                method_id = "margo_v0.2_diag_cavia_energy"
                run_dir = diag_cavia_energy_run_dir(seed)
                if method_id != DIAG_CAVIA_ENERGY_METHOD_ID:
                    raise ValueError("energy method_id drift %s" % DIAG_CAVIA_ENERGY_METHOD_ID)
            else:
                method_id = "margo_v0.2_diag_cavia_frozen"
                run_dir = diag_cavia_run_dir(seed)
                if method_id != DIAG_CAVIA_METHOD_ID:
                    raise ValueError("frozen method_id drift %s" % DIAG_CAVIA_METHOD_ID)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": method_id,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "encoder_frozen": True,
            "decoder_frozen": True,
            "cavia_film_kernel_frozen": True,
            "cavia_z_dim": int(DIAG_CAVIA_Z_DIM),
            "inner_steps": n_steps,
            "inner_lr": inner_lr,
            "use_energy": use_energy,
            "load_ckpt": str(src_ckpt),
            "learning_mode": "cavia_z",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": float(objective.energy_weight) if use_energy else 0.0,
            "latency_weight": float(objective.latency_weight) if use_energy else 1.0,
            "from_bccont": from_bccont,
            "run_dir": str(run_dir),
            "note": "CAVIA-on-z; theta frozen; schedule is physics; energy is cost switch; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    val_env, test_env = _cavia_held_out_envs()
    policy = Seq2SeqPolicy(
        obs_dim=val_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="cavia_policy",
        enable_cavia=True,
        cavia_z_dim=int(DIAG_CAVIA_Z_DIM),
    )
    ops = build_cavia_train_ops(policy, lr=inner_lr)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_loaded = load_named_policy_from_core_ckpt(policy, src_ckpt, sess, src_scope="core_policy")
        n_expect = len([v for v in policy.get_variables() if not is_cavia_var_name(v.name)])
        if int(n_loaded) != int(n_expect):
            raise ValueError("cavia ckpt loaded %s vars, want %s" % (n_loaded, n_expect))
        z0 = sess.run(policy.network.cavia_z)
        z_l2 = float((z0 ** 2).sum() ** 0.5)
        if z_l2 > 1e-6:
            raise ValueError("cavia_z not zero after load, l2=%s" % z_l2)
        print("cavia_load_ckpt n_vars=%d z_l2=%.6f %s" % (n_loaded, z_l2, src_ckpt))
        stats = run_cavia_phase1(
            sess,
            policy,
            ops,
            val_env,
            test_env,
            objective,
            n_steps=n_steps,
            run_dir=run_dir,
            identity_lo=ident_lo,
            identity_hi=ident_hi,
        )
    val_row = stats.get("validation") or {}
    test_row = stats.get("meta_test") or {}
    payload["cavia"] = {
        "verdict": stats.get("verdict"),
        "use_energy": stats.get("use_energy"),
        "validation_k0_full": stats.get("validation_k0_full"),
        "meta_test_k0_full": stats.get("meta_test_k0_full"),
        "validation": {
            "query_T_k0": val_row.get("query_T_k0"),
            "query_T_k20": val_row.get("query_T_k20"),
            "query_cost_k0": val_row.get("query_cost_k0"),
            "query_cost_k20": val_row.get("query_cost_k20"),
            "query_local_k20": val_row.get("query_local_k20"),
        },
        "meta_test": {
            "query_T_k0": test_row.get("query_T_k0"),
            "query_T_k20": test_row.get("query_T_k20"),
            "query_cost_k0": test_row.get("query_cost_k0"),
            "query_cost_k20": test_row.get("query_cost_k20"),
            "query_local_k20": test_row.get("query_local_k20"),
        },
    }
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir


def run_diagnostic_pairsup(seed, allow_gpu, n_epochs=DIAG_PAIRSUP_EPOCHS):
    """BC greedy_from_mec + λ motif-joint CE from bc_continue. Greedy eval only. Not 3500."""
    from spec.bc_greedy_mec import eval_loaded_policy_on_env
    from spec.pair_sup import (
        PAIRSUP_A_TEST_T,
        PAIRSUP_A_VAL_T,
        PAIRSUP_EPOCHS,
        PAIRSUP_LAMBDA,
        classify_pairsup_verdict,
        run_pairsup_train,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("pairsup n_epochs must be %d, got %s" % (DIAG_PAIRSUP_EPOCHS, n_epochs))
    if int(PAIRSUP_EPOCHS) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("PAIRSUP_EPOCHS mismatch campaign %s spec %s" % (DIAG_PAIRSUP_EPOCHS, PAIRSUP_EPOCHS))
    if abs(float(PAIRSUP_LAMBDA) - float(DIAG_PAIRSUP_LAMBDA)) > 1e-12:
        raise ValueError("PAIRSUP_LAMBDA mismatch campaign %s spec %s" % (DIAG_PAIRSUP_LAMBDA, PAIRSUP_LAMBDA))
    if abs(float(PAIRSUP_A_VAL_T) - float(DIAG_PAIRSUP_A_VAL_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_VAL_T mismatch")
    if abs(float(PAIRSUP_A_TEST_T) - float(DIAG_PAIRSUP_A_TEST_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_TEST_T mismatch")
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    from spec.phase4_campaign import RUNS_ROOT

    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    joint_cache = Path(RUNS_ROOT) / "pairsup_joint_from_mec_train.npz"
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("greedy_from_mec expert cache missing: %s" % p)
    run_dir = diag_pairsup_run_dir(seed)
    if run_dir.name != "seed_%d" % int(seed):
        raise ValueError("pairsup run dir drift")
    if DIAG_PAIRSUP_METHOD_ID != "margo_v0.2_diag_pairsup":
        raise ValueError("pairsup method_id drift %s" % DIAG_PAIRSUP_METHOD_ID)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_PAIRSUP_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "pair_search_at_eval": False,
            "bc_expert": "greedy_from_mec",
            "load_ckpt": str(src_ckpt),
            "lam": float(DIAG_PAIRSUP_LAMBDA),
            "bc_epochs": int(DIAG_PAIRSUP_EPOCHS),
            "baseline_a": "margo_v0.1_diag_bc_unseen",
            "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
            "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "B vs A pair-sup: L_BC greedy_from_mec + λ L_joint motif counterfactual; init bc_continue not bc_2opt; greedy eval only; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_stats = run_pairsup_train(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 77),
            run_dir,
            load_ckpt=src_ckpt,
            expert_cache=train_cache,
            joint_cache=joint_cache,
            epochs=DIAG_PAIRSUP_EPOCHS,
            lam=DIAG_PAIRSUP_LAMBDA,
            save_ckpt=ckpt_path,
        )
        print("pairsup_eval_unseen")
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    vg = val_row["greedy"]
    tg = test_row["greedy"]
    verdict = classify_pairsup_verdict(vg["greedy_T_mean"], vg["greedy_local_frac"])
    out = {
        "verdict": verdict,
        "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
        "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
        "validation": val_row,
        "meta_test": test_row,
        "train": train_stats.get("greedy_rollout"),
        "labels": train_stats.get("labels"),
        "lam": float(DIAG_PAIRSUP_LAMBDA),
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
    }
    payload["train"] = train_stats.get("greedy_rollout")
    payload["labels"] = train_stats.get("labels")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "pairsup_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "pairsup_verdict=%s val_T=%.1f A=%.1f local=%.3f test_T=%.1f A_test=%.1f frac_pos=%.3f"
        % (
            verdict,
            vg["greedy_T_mean"],
            DIAG_PAIRSUP_A_VAL_T,
            vg["greedy_local_frac"],
            tg["greedy_T_mean"],
            DIAG_PAIRSUP_A_TEST_T,
            float((train_stats.get("labels") or {}).get("frac_pos") or 0.0),
        )
    )
    return run_dir


def run_diagnostic_pairsup(seed, allow_gpu, n_epochs=DIAG_PAIRSUP_EPOCHS):
    """BC greedy_from_mec + λ motif-joint CE from bc_continue. Greedy eval only. Not 3500."""
    from spec.bc_greedy_mec import eval_loaded_policy_on_env
    from spec.pair_sup import (
        PAIRSUP_A_TEST_T,
        PAIRSUP_A_VAL_T,
        PAIRSUP_EPOCHS,
        PAIRSUP_LAMBDA,
        classify_pairsup_verdict,
        run_pairsup_train,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("pairsup n_epochs must be %d, got %s" % (DIAG_PAIRSUP_EPOCHS, n_epochs))
    if int(PAIRSUP_EPOCHS) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("PAIRSUP_EPOCHS mismatch campaign %s spec %s" % (DIAG_PAIRSUP_EPOCHS, PAIRSUP_EPOCHS))
    if abs(float(PAIRSUP_LAMBDA) - float(DIAG_PAIRSUP_LAMBDA)) > 1e-12:
        raise ValueError("PAIRSUP_LAMBDA mismatch campaign %s spec %s" % (DIAG_PAIRSUP_LAMBDA, PAIRSUP_LAMBDA))
    if abs(float(PAIRSUP_A_VAL_T) - float(DIAG_PAIRSUP_A_VAL_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_VAL_T mismatch")
    if abs(float(PAIRSUP_A_TEST_T) - float(DIAG_PAIRSUP_A_TEST_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_TEST_T mismatch")
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    from spec.phase4_campaign import RUNS_ROOT

    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    joint_cache = Path(RUNS_ROOT) / "pairsup_joint_from_mec_train.npz"
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("greedy_from_mec expert cache missing: %s" % p)
    run_dir = diag_pairsup_run_dir(seed)
    if run_dir.name != "seed_%d" % int(seed):
        raise ValueError("pairsup run dir drift")
    if DIAG_PAIRSUP_METHOD_ID != "margo_v0.2_diag_pairsup":
        raise ValueError("pairsup method_id drift %s" % DIAG_PAIRSUP_METHOD_ID)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_PAIRSUP_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "pair_search_at_eval": False,
            "bc_expert": "greedy_from_mec",
            "load_ckpt": str(src_ckpt),
            "lam": float(DIAG_PAIRSUP_LAMBDA),
            "bc_epochs": int(DIAG_PAIRSUP_EPOCHS),
            "baseline_a": "margo_v0.1_diag_bc_unseen",
            "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
            "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "B vs A pair-sup: L_BC greedy_from_mec + λ L_joint motif counterfactual; init bc_continue not bc_2opt; greedy eval only; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_stats = run_pairsup_train(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 77),
            run_dir,
            load_ckpt=src_ckpt,
            expert_cache=train_cache,
            joint_cache=joint_cache,
            epochs=DIAG_PAIRSUP_EPOCHS,
            lam=DIAG_PAIRSUP_LAMBDA,
            save_ckpt=ckpt_path,
        )
        print("pairsup_eval_unseen")
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    vg = val_row["greedy"]
    tg = test_row["greedy"]
    verdict = classify_pairsup_verdict(vg["greedy_T_mean"], vg["greedy_local_frac"])
    out = {
        "verdict": verdict,
        "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
        "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
        "validation": val_row,
        "meta_test": test_row,
        "train": train_stats.get("greedy_rollout"),
        "labels": train_stats.get("labels"),
        "lam": float(DIAG_PAIRSUP_LAMBDA),
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
    }
    payload["train"] = train_stats.get("greedy_rollout")
    payload["labels"] = train_stats.get("labels")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "pairsup_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "pairsup_verdict=%s val_T=%.1f A=%.1f local=%.3f test_T=%.1f A_test=%.1f frac_pos=%.3f"
        % (
            verdict,
            vg["greedy_T_mean"],
            DIAG_PAIRSUP_A_VAL_T,
            vg["greedy_local_frac"],
            tg["greedy_T_mean"],
            DIAG_PAIRSUP_A_TEST_T,
            float((train_stats.get("labels") or {}).get("frac_pos") or 0.0),
        )
    )
    return run_dir


def run_diagnostic_pairsup(seed, allow_gpu, n_epochs=DIAG_PAIRSUP_EPOCHS):
    """BC greedy_from_mec + λ motif-joint CE from bc_continue. Greedy eval only. Not 3500."""
    from spec.bc_greedy_mec import eval_loaded_policy_on_env
    from spec.pair_sup import (
        PAIRSUP_A_TEST_T,
        PAIRSUP_A_VAL_T,
        PAIRSUP_EPOCHS,
        PAIRSUP_LAMBDA,
        classify_pairsup_verdict,
        run_pairsup_train,
    )
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    if int(n_epochs) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("pairsup n_epochs must be %d, got %s" % (DIAG_PAIRSUP_EPOCHS, n_epochs))
    if int(PAIRSUP_EPOCHS) != int(DIAG_PAIRSUP_EPOCHS):
        raise ValueError("PAIRSUP_EPOCHS mismatch campaign %s spec %s" % (DIAG_PAIRSUP_EPOCHS, PAIRSUP_EPOCHS))
    if abs(float(PAIRSUP_LAMBDA) - float(DIAG_PAIRSUP_LAMBDA)) > 1e-12:
        raise ValueError("PAIRSUP_LAMBDA mismatch campaign %s spec %s" % (DIAG_PAIRSUP_LAMBDA, PAIRSUP_LAMBDA))
    if abs(float(PAIRSUP_A_VAL_T) - float(DIAG_PAIRSUP_A_VAL_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_VAL_T mismatch")
    if abs(float(PAIRSUP_A_TEST_T) - float(DIAG_PAIRSUP_A_TEST_T)) > 1e-12:
        raise ValueError("PAIRSUP_A_TEST_T mismatch")
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    from spec.phase4_campaign import RUNS_ROOT

    train_cache = Path(RUNS_ROOT) / "expert_greedy_mec_train.npz"
    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    joint_cache = Path(RUNS_ROOT) / "pairsup_joint_from_mec_train.npz"
    for p in (train_cache, val_cache, test_cache):
        if not p.is_file():
            raise FileNotFoundError("greedy_from_mec expert cache missing: %s" % p)
    run_dir = diag_pairsup_run_dir(seed)
    if run_dir.name != "seed_%d" % int(seed):
        raise ValueError("pairsup run dir drift")
    if DIAG_PAIRSUP_METHOD_ID != "margo_v0.2_diag_pairsup":
        raise ValueError("pairsup method_id drift %s" % DIAG_PAIRSUP_METHOD_ID)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_PAIRSUP_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "pair_search_at_eval": False,
            "bc_expert": "greedy_from_mec",
            "load_ckpt": str(src_ckpt),
            "lam": float(DIAG_PAIRSUP_LAMBDA),
            "bc_epochs": int(DIAG_PAIRSUP_EPOCHS),
            "baseline_a": "margo_v0.1_diag_bc_unseen",
            "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
            "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "B vs A pair-sup: L_BC greedy_from_mec + λ L_joint motif counterfactual; init bc_continue not bc_2opt; greedy eval only; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_stats = run_pairsup_train(
            sess,
            trainer.env,
            trainer.policy.core_policy,
            np.random.RandomState(int(seed) + 77),
            run_dir,
            load_ckpt=src_ckpt,
            expert_cache=train_cache,
            joint_cache=joint_cache,
            epochs=DIAG_PAIRSUP_EPOCHS,
            lam=DIAG_PAIRSUP_LAMBDA,
            save_ckpt=ckpt_path,
        )
        print("pairsup_eval_unseen")
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    vg = val_row["greedy"]
    tg = test_row["greedy"]
    verdict = classify_pairsup_verdict(vg["greedy_T_mean"], vg["greedy_local_frac"])
    out = {
        "verdict": verdict,
        "baseline_a_val_T": float(DIAG_PAIRSUP_A_VAL_T),
        "baseline_a_test_T": float(DIAG_PAIRSUP_A_TEST_T),
        "validation": val_row,
        "meta_test": test_row,
        "train": train_stats.get("greedy_rollout"),
        "labels": train_stats.get("labels"),
        "lam": float(DIAG_PAIRSUP_LAMBDA),
        "pair_search_at_eval": False,
        "ppo": False,
        "paper_result": False,
    }
    payload["train"] = train_stats.get("greedy_rollout")
    payload["labels"] = train_stats.get("labels")
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "pairsup_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "pairsup_verdict=%s val_T=%.1f A=%.1f local=%.3f test_T=%.1f A_test=%.1f frac_pos=%.3f"
        % (
            verdict,
            vg["greedy_T_mean"],
            DIAG_PAIRSUP_A_VAL_T,
            vg["greedy_local_frac"],
            tg["greedy_T_mean"],
            DIAG_PAIRSUP_A_TEST_T,
            float((train_stats.get("labels") or {}).get("frac_pos") or 0.0),
        )
    )
    return run_dir


def _unseen_ok_row(split_row):
    from spec.bc_greedy_mec import unseen_ok_for_kl

    return unseen_ok_for_kl(split_row)


def run_diagnostic_bc_unseen(seed, allow_gpu):
    """Greedy-decode bc_continue ckpt on validation and meta-test. No PPO. Not a paper result."""
    from spec.bc_greedy_mec import eval_loaded_policy_on_env
    from spec.split_loader import assert_held_out_prefixes, meta_test_graph_prefixes

    require_gpu_permission(allow_gpu)
    src_ckpt = diag_bccont_run_dir(seed) / "ckpt" / "bc_core.ckpt"
    if not src_ckpt.is_file():
        raise FileNotFoundError("bc_continue ckpt missing: %s" % src_ckpt)
    run_dir = diag_bcunseen_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": "margo_v0.1_diag_bc_unseen",
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "outer_iterations": 0,
            "outer_update_count": 0,
            "ppo": False,
            "load_ckpt": str(src_ckpt),
            "learning_mode": "bc_greedy_mec",
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.0,
            "latency_weight": 1.0,
            "run_dir": str(run_dir),
            "note": "greedy decode bc_continue ckpt on held-out val+meta-test graphs; law vs memorization; no PPO; not the frozen 3500 primary; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=1,
        ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_over_all_mec",
        learning_mode="bc_greedy_mec",
    )
    del algo
    test_paths = meta_test_graph_prefixes()
    assert_held_out_prefixes(test_paths, "meta_test")
    test_env = OffloadingEnvironment(
        resource_cluster=trainer.env.resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=test_paths,
        time_major=False,
    )
    from spec.phase4_campaign import RUNS_ROOT

    val_cache = Path(RUNS_ROOT) / "expert_greedy_mec_validation.npz"
    test_cache = Path(RUNS_ROOT) / "expert_greedy_mec_metatest.npz"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(src_ckpt), sess=sess)
        print("bc_unseen_load_ckpt %s" % src_ckpt)
        val_row = eval_loaded_policy_on_env(
            sess,
            trainer.held_out_evaluator.env,
            trainer.policy.core_policy,
            cache_path=val_cache,
            split_name="validation",
        )
        test_row = eval_loaded_policy_on_env(
            sess,
            test_env,
            trainer.policy.core_policy,
            cache_path=test_cache,
            split_name="meta_test",
        )
    val_ok = _unseen_ok_row(val_row)
    test_ok = _unseen_ok_row(test_row)
    verdict = {
        "validation_ok_for_kl": val_ok,
        "meta_test_ok_for_kl": test_ok,
        "launch_kl_ppo": bool(val_ok and test_ok),
        "rule": "frac_beats_mec>=0.5 and greedy_over_mec<1 and mec_frac<0.95 and n_non_p50>=2",
    }
    payload["splits"] = {"validation": val_row, "meta_test": test_row}
    payload["unseen_verdict"] = verdict
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    (run_dir / "unseen_eval.json").write_text(
        json.dumps({"validation": val_row, "meta_test": test_row, "verdict": verdict}, indent=2, sort_keys=True)
        + "\n"
    )
    print(
        "bc_unseen_verdict val_ok=%s test_ok=%s launch_kl=%s"
        % (val_ok, test_ok, verdict["launch_kl_ppo"])
    )
    return run_dir


def _encoder_train_val_envs():
    """Train + validation only. Does not open meta-test graphs."""
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment, Resources
    from spec.split_loader import (
        assert_held_out_prefixes,
        meta_train_graph_prefixes,
        validation_graph_prefixes,
    )

    energy_config = {
        "use_energy": True,
        "reward_mode": "latency_over_all_mec",
        "energy_weight": 0.5,
        "latency_weight": 0.5,
        "rho": 1.0,
        "f_l": 1.0,
        "zeta": 2.0,
        "ptx": 0.1,
        "prx": 0.05,
        "ptx_v2v": 0.06,
        "prx_v2v": 0.03,
        "rho_v2v": 0.7,
        "f_v2v": 1.0,
        "normalize_energy": True,
    }
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        v2v_process_capable=(1.0 * 1024 * 1024),
        v2v_bandwidth=5.0,
        use_energy=True,
        energy_config=energy_config,
    )
    train_paths = meta_train_graph_prefixes()
    val_paths = validation_graph_prefixes()
    assert_held_out_prefixes(val_paths, "validation")
    train_env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=train_paths,
        time_major=False,
    )
    val_env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=val_paths,
        time_major=False,
    )
    return train_env, val_env


def _count_encoder_params(policy):
    import numpy as np

    n = 0
    for var in policy.get_trainable_variables():
        if "/encoder/" in var.name or var.name.startswith("encoder/"):
            shape = var.get_shape().as_list()
            prod = 1
            for d in shape:
                prod *= int(d or 1)
            n += prod
    return int(n)


def run_diagnostic_encoder(
    seed,
    allow_gpu,
    encoder_type="meanagg",
    readout_type="triple",
    smoke=False,
):
    """BC on 2-opt labels with a swapped encoder. Validation only. No PPO. paper_result=false."""
    import hashlib
    import time

    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from spec.bc_greedy_mec import (
        BC_CONTINUE_MIN_DELTA,
        BC_CONTINUE_PATIENCE,
        eval_loaded_policy_on_env,
        policy_feed,
        run_bc_greedy_mec,
    )
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    encoder_type = str(encoder_type)
    readout_type = str(readout_type)
    if encoder_type not in DIAG_ENC_TYPES:
        raise ValueError("encoder_type %r" % encoder_type)
    if readout_type not in DIAG_ENC_READOUTS:
        raise ValueError("readout_type %r" % readout_type)
    if DIAG_ENC_METHOD_ID != "margo_v0.3_diag_encoder":
        raise ValueError("encoder method_id drift")
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    for p in (train_cache, val_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    h = hashlib.sha256()
    with open(str(train_cache), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    label_sha = h.hexdigest()
    run_dir = diag_encoder_run_dir(seed, encoder_type, readout_type)
    epochs = DIAG_ENC_SMOKE_EPOCHS if smoke else DIAG_ENC_MAX_EPOCHS
    max_graphs = DIAG_ENC_SMOKE_GRAPHS if smoke else None
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_ENC_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "encoder_type": encoder_type,
            "readout_type": readout_type,
            "bc_expert": "iterative_2opt",
            "label_cache": str(train_cache),
            "label_cache_sha256": label_sha,
            "bc_epochs_max": int(epochs),
            "bc_early_stop_patience": BC_CONTINUE_PATIENCE,
            "bc_early_stop_min_delta": BC_CONTINUE_MIN_DELTA,
            "smoke": bool(smoke),
            "evals_per_graph": 1,
            "run_dir": str(run_dir),
            "note": "v0.3 encoder ablation BC-2opt; validation only; no meta-test; no PPO; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    train_env, val_env = _encoder_train_val_envs()
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    policy = Seq2SeqPolicy(
        obs_dim=train_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type=encoder_type,
        readout_type=readout_type,
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_params = _count_encoder_params(policy)
        payload["n_params"] = int(n_params)
        probe_obs = np.asarray(train_env.encoder_batchs[0][:2], dtype=np.float32)
        fl = np.full((probe_obs.shape[0],), probe_obs.shape[1], dtype=np.int32)
        reach = None
        if policy.reachability_mask is not None:
            from spec.bc_greedy_mec import collect_reachability

            reach_all = collect_reachability(train_env)
            reach = reach_all[: probe_obs.shape[0]]
        enc_out = sess.run(
            policy.network.encoder_outputs,
            feed_dict=policy_feed(policy, probe_obs, fl, reach=reach),
        )
        enc_out = np.asarray(enc_out)
        if list(enc_out.shape[1:]) != [20, 256]:
            raise ValueError("encoder_outputs shape %s != [B,20,256]" % (enc_out.shape,))
        if not np.isfinite(enc_out).all():
            raise ValueError("encoder_outputs has NaN/Inf")
        payload["encoder_probe_shape"] = [int(x) for x in enc_out.shape]
        payload["encoder_probe_finite"] = True
        print(
            "encoder_probe type=%s readout=%s shape=%s n_params=%d finite=1"
            % (encoder_type, readout_type, enc_out.shape, n_params)
        )
        bc_stats = run_bc_greedy_mec(
            sess,
            train_env,
            policy,
            np.random.RandomState(int(seed) + 70),
            run_dir,
            epochs=epochs,
            greedy_env_eval=not smoke,
            save_ckpt=ckpt_path,
            load_ckpt=None,
            cache_path=train_cache,
            early_stop_patience=None if smoke else BC_CONTINUE_PATIENCE,
            early_stop_min_delta=BC_CONTINUE_MIN_DELTA,
            max_graphs=max_graphs,
        )
        val_row = None
        if not smoke:
            print("encoder_eval_validation")
            val_row = eval_loaded_policy_on_env(
                sess,
                val_env,
                policy,
                cache_path=val_cache,
                split_name="validation",
            )
    minutes = (time.time() - t0) / 60.0
    payload["bc_pretrain"] = bc_stats
    payload["train"] = bc_stats.get("greedy_rollout")
    payload["splits"] = {"validation": val_row} if val_row is not None else {}
    payload["gpu_finished"] = True
    payload["wall_clock_s"] = float(minutes * 60.0)
    payload["minutes"] = float(minutes)
    _write_payload(run_dir, payload)
    out = {
        "encoder_type": encoder_type,
        "readout_type": readout_type,
        "seed": int(seed),
        "n_params": int(n_params),
        "smoke": bool(smoke),
        "evals_per_graph": 1,
        "paper_result": False,
        "train": bc_stats.get("greedy_rollout"),
        "validation": None if val_row is None else val_row.get("greedy"),
        "bc": {
            "bc_epochs_ran": bc_stats.get("bc_epochs_ran"),
            "bc_best_loss": bc_stats.get("bc_best_loss"),
            "token_acc": bc_stats.get("token_acc"),
            "greedy_decode_acc": bc_stats.get("greedy_decode_acc"),
            "n_graphs": bc_stats.get("n_graphs"),
        },
        "minutes": float(minutes),
        "label_cache_sha256": label_sha,
    }
    (run_dir / "encoder_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    tr = bc_stats.get("greedy_rollout") or {}
    vg = (val_row or {}).get("greedy") or {}
    print(
        "encoder_verdict type=%s readout=%s seed=%s train_T=%s val_T=%s token_tr=%s token_val=%s n_params=%d min=%.1f"
        % (
            encoder_type,
            readout_type,
            seed,
            tr.get("greedy_T_mean"),
            vg.get("greedy_T_mean"),
            bc_stats.get("token_acc"),
            vg.get("greedy_token_acc_vs_expert"),
            n_params,
            minutes,
        )
    )
    return run_dir


def _bok_encoder_hparams(ckpt_path):
    prov_path = Path(ckpt_path).resolve().parents[1] / "provenance.json"
    encoder_type = "meanagg"
    readout_type = "triple"
    if prov_path.is_file():
        row = json.loads(prov_path.read_text())
        encoder_type = str(row.get("encoder_type") or encoder_type)
        readout_type = str(row.get("readout_type") or readout_type)
    return encoder_type, readout_type


def run_diagnostic_bestofk(seed, allow_gpu, smoke=False, ckpt=None):
    """Frozen-policy best-of-k sampling. No training. Validation + train. No meta-test."""
    import time

    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from spec.best_of_k import (
        PAIR_SEQ_HEURISTIC,
        dump_json,
        eval_best_of_k,
        resolve_bok_ckpt,
        sha256_file,
    )
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if DIAG_BOK_METHOD_ID != "margo_v0.3_diag_bestofk":
        raise ValueError("bestofk method_id drift")
    ckpt_path = resolve_bok_ckpt(ckpt)
    ckpt_sha = sha256_file(ckpt_path)
    encoder_type, readout_type = _bok_encoder_hparams(ckpt_path)
    train_cache = CACHE_TWOPT["meta_train"]
    val_cache = CACHE_TWOPT["validation"]
    for p in (train_cache, val_cache):
        if not p.is_file():
            raise FileNotFoundError("2-opt expert cache missing: %s" % p)
    run_dir = diag_bestofk_run_dir(seed)
    k_max = DIAG_BOK_SMOKE_K if smoke else DIAG_BOK_K_MAX
    max_train = DIAG_BOK_SMOKE_GRAPHS if smoke else None
    max_val = DIAG_BOK_SMOKE_GRAPHS if smoke else None
    ks = tuple(k for k in DIAG_BOK_K_SWEEP if k <= int(k_max))
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_BOK_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "training": False,
            "encoder_type": encoder_type,
            "readout_type": readout_type,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "k_sweep": list(ks),
            "k_max": int(k_max),
            "temperature_default": 1.0,
            "temperature_sweep": list(DIAG_BOK_TEMPS) if not smoke else [1.0],
            "temperature_sweep_k": int(DIAG_BOK_K_TEMP),
            "top_p": None,
            "smoke": bool(smoke),
            "evals_per_graph": int(k_max),
            "pair_seq_heuristic_reference": PAIR_SEQ_HEURISTIC,
            "run_dir": str(run_dir),
            "note": "v0.3 best-of-k neural inference; no training; validation only for gate; no meta-test; no PPO; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    tf.compat.v1.set_random_seed(int(seed))
    np.random.seed(int(seed))
    train_env, val_env = _encoder_train_val_envs()
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(int(seed))
    policy = Seq2SeqPolicy(
        obs_dim=train_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type=encoder_type,
        readout_type=readout_type,
    )
    out = {
        "method_id": DIAG_BOK_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "encoder_type": encoder_type,
        "readout_type": readout_type,
        "smoke": bool(smoke),
        "paper_result": False,
        "k_sweep": list(ks),
        "pair_seq_heuristic_reference": PAIR_SEQ_HEURISTIC,
    }
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        policy.load_variables(str(ckpt_path), sess=sess)
        print("bestofk_load_ckpt %s sha=%s encoder=%s" % (ckpt_path, ckpt_sha[:16], encoder_type))
        if not smoke:
            print("bestofk_sanity_train n=%d" % DIAG_BOK_SANITY_GRAPHS)
            sanity = eval_best_of_k(
                sess,
                policy,
                train_env,
                train_env.scheduler_resources,
                k_max=1,
                temperature=1.0,
                max_graphs=DIAG_BOK_SANITY_GRAPHS,
                expert_cache=train_cache,
                ks=(1,),
                seed=seed,
                split_name="train_sanity",
            )
            out["train_sanity"] = sanity
            print(
                "bestofk_sanity T_best_1=%.4f T_greedy=%.4f n=%d"
                % (sanity["T_best_1"], sanity["T_greedy"], sanity["n_graphs"])
            )
        print("bestofk_eval_train n=%s k_max=%d" % (max_train or 1500, k_max))
        train_row = eval_best_of_k(
            sess,
            policy,
            train_env,
            train_env.scheduler_resources,
            k_max=k_max,
            temperature=1.0,
            max_graphs=max_train,
            expert_cache=train_cache,
            ks=ks,
            seed=seed,
            split_name="train",
        )
        out["train"] = train_row
        print(
            "bestofk_train T_greedy=%.1f T_best_%d=%.1f n=%d min=%.1f"
            % (
                train_row["T_greedy"],
                k_max,
                train_row["by_k"][str(k_max)]["T_best_k"],
                train_row["n_graphs"],
                (time.time() - t0) / 60.0,
            )
        )
        print("bestofk_eval_validation n=%s k_max=%d" % (max_val or 500, k_max))
        val_row = eval_best_of_k(
            sess,
            policy,
            val_env,
            val_env.scheduler_resources,
            k_max=k_max,
            temperature=1.0,
            max_graphs=max_val,
            expert_cache=val_cache,
            ks=ks,
            seed=seed,
            split_name="validation",
        )
        out["validation"] = val_row
        temp_rows = {}
        if not smoke:
            for temp in DIAG_BOK_TEMPS:
                if abs(float(temp) - 1.0) < 1e-12:
                    temp_rows[str(temp)] = val_row["by_k"].get(str(DIAG_BOK_K_TEMP))
                    continue
                print("bestofk_temp_sweep t=%.1f k=%d validation" % (temp, DIAG_BOK_K_TEMP))
                trow = eval_best_of_k(
                    sess,
                    policy,
                    val_env,
                    val_env.scheduler_resources,
                    k_max=DIAG_BOK_K_TEMP,
                    temperature=float(temp),
                    max_graphs=max_val,
                    expert_cache=val_cache,
                    ks=(1, DIAG_BOK_K_TEMP),
                    seed=seed,
                    split_name="validation_t%s" % temp,
                )
                temp_rows[str(temp)] = trow["by_k"][str(DIAG_BOK_K_TEMP)]
        out["temp_sweep_k32_validation"] = temp_rows
    minutes = (time.time() - t0) / 60.0
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["wall_clock_s"] = float(minutes * 60.0)
    _write_payload(run_dir, payload)
    out["minutes"] = float(minutes)
    val_by = val_row["by_k"]
    t32 = (val_by.get("32") or {}).get("T_best_k")
    t64 = (val_by.get("64") or val_by.get(str(k_max)) or {}).get("T_best_k")
    t1 = val_row["T_greedy"]
    delta64 = None if t64 is None else float(t1) - float(t64)
    out["gate"] = {
        "T_greedy_val": t1,
        "T_best_32_val": t32,
        "T_best_64_val": t64,
        "delta_best64_over_greedy": delta64,
        "target_T_best_32_le": 510.0,
        "tail_claim_transfer": None if delta64 is None else bool(delta64 >= 20.0),
        "dist10": ((val_by.get(str(k_max)) or {}).get("per_distribution") or {}).get("10"),
    }
    dump_json(run_dir / "bestofk_eval.json", out)
    print(
        "bestofk_verdict seed=%s smoke=%s val_T_greedy=%s T_best_32=%s T_best_64=%s d64=%s min=%.1f"
        % (seed, int(smoke), t1, t32, t64, delta64, minutes)
    )
    return run_dir



def run_diagnostic_eas(
    seed,
    allow_gpu,
    adapt_subset=None,
    loss_type=None,
    lambda_il=None,
    n_adapt=None,
    k=None,
    smoke=False,
    ckpt=None,
):
    """EAS-on-φ support adaptation. Validation only. No meta-test. paper_result=false."""
    import time

    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from spec.eas_adapt import (
        ADAPT_SUBSETS,
        CKPT_MEANAGG_MEAN,
        IDENTITY_T_TOL,
        IDENTITY_VAL_T_REF,
        LOSS_TYPES,
        build_eas_train_ops,
        dump_json,
        eval_greedy_T,
        filter_adapt_vars,
        load_policy_partial,
        load_twopt_support_acts,
        n_params_vars,
        run_eas_on_dist,
        sample_k_and_neglogp,
        sha256_file,
        snapshot_vars,
        LatencyOnlyObjective,
        _schedule_batch,
        _mix_nnon,
        slice_dist,
    )
    from spec.split_loader import support_query_indices, validation_distribution_ids
    from spec.twopt_expert import CACHE_TWOPT

    require_gpu_permission(allow_gpu)
    if DIAG_EAS_METHOD_ID != "margo_v0.3_diag_eas":
        raise ValueError("eas method_id drift")
    subset = str(adapt_subset or DIAG_EAS_DEFAULT_SUBSET)
    loss_type = str(loss_type or DIAG_EAS_DEFAULT_LOSS)
    if subset == "full":
        pass
    elif subset not in ADAPT_SUBSETS:
        raise ValueError("adapt_subset %r" % subset)
    if loss_type not in LOSS_TYPES:
        raise ValueError("loss %r" % loss_type)
    lambda_il = float(DIAG_EAS_DEFAULT_LAMBDA_IL if lambda_il is None else lambda_il)
    if smoke:
        n_adapt = int(DIAG_EAS_SMOKE_N_ADAPT if n_adapt is None else n_adapt)
        k = int(DIAG_EAS_SMOKE_K if k is None else k)
        dist_ids = (int(DIAG_EAS_SMOKE_DIST),)
    else:
        n_adapt = int(DIAG_EAS_N_ADAPT if n_adapt is None else n_adapt)
        k = int(DIAG_EAS_K if k is None else k)
        dist_ids = tuple(int(d) for d in validation_distribution_ids())
    ckpt_path = Path(ckpt) if ckpt else CKPT_MEANAGG_MEAN
    if not ckpt_path.is_file():
        raise FileNotFoundError("EAS ckpt missing: %s" % ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    val_cache = CACHE_TWOPT["validation"]
    if loss_type == "ce2opt" and not val_cache.is_file():
        raise FileNotFoundError("2-opt cache missing for ce2opt: %s" % val_cache)
    run_dir = diag_eas_run_dir(seed, subset, loss_type)
    enable_cavia = subset == "film"
    enable_eas_emb = subset == "emb"
    film_trainable = subset == "film"
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_EAS_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "adapt_subset": subset,
            "loss_type": loss_type,
            "lambda_il": lambda_il,
            "n_adapt": int(n_adapt),
            "k": int(k),
            "lr": float(DIAG_EAS_LR),
            "smoke": bool(smoke),
            "dist_ids": list(dist_ids),
            "run_dir": str(run_dir),
            "note": "v0.3 EAS-on-φ support adaptation; validation only; no meta-test; no PPO on full θ unless subset=full; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    tf.compat.v1.set_random_seed(int(seed))
    np.random.seed(int(seed))
    _train_env, val_env = _encoder_train_val_envs()
    del _train_env
    resources = val_env.scheduler_resources
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(int(seed))
    policy = Seq2SeqPolicy(
        obs_dim=val_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
        enable_cavia=enable_cavia,
        cavia_film_trainable=film_trainable,
        enable_eas_emb=enable_eas_emb,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        missing = load_policy_partial(policy, ckpt_path, sess)
        if missing:
            print("eas_ckpt_missing_ok_new_phi %s" % missing)
        adapt_vars = filter_adapt_vars(policy, subset)
        n_phi = n_params_vars(adapt_vars)
        print(
            "eas_trainable_audit subset=%s n_vars=%d n_params=%d names=%s"
            % (subset, len(adapt_vars), n_phi, [v.name for v in adapt_vars])
        )
        all_train = policy.get_trainable_variables()
        print("eas_all_trainable_count=%d (θ frozen except φ)" % len(all_train))
        ops = build_eas_train_ops(policy, adapt_vars, lr=DIAG_EAS_LR)
        sess.run(tf.compat.v1.variables_initializer(ops["opt"].variables()))
        init_vals = snapshot_vars(sess, adapt_vars)
        # Identity: full-dist greedy (100/dist) must match Phase-1 meanagg+mean ≈577.02
        id_ts = []
        for did in dist_ids:
            ids = list(range(100))
            q_tgs, q_obs = slice_dist(val_env, did, ids)
            refs = {}
            qT, _, _ = eval_greedy_T(
                sess, policy, q_tgs, q_obs, resources, refs, "id_d%d" % did
            )
            id_ts.append(qT)
            print("eas_identity dist=%s full_greedy_T=%.4f" % (did, qT))
        id_mean = float(np.mean(id_ts))
        if (not smoke) and abs(id_mean - IDENTITY_VAL_T_REF) > IDENTITY_T_TOL:
            raise ValueError(
                "identity fail mean=%.4f ref=%.4f tol=%.2f"
                % (id_mean, IDENTITY_VAL_T_REF, IDENTITY_T_TOL)
            )
        per_dist = []
        for did in dist_ids:
            twopt = None
            if loss_type == "ce2opt":
                sidx, _ = support_query_indices(did)
                twopt = load_twopt_support_acts(val_cache, did, sidx)
            row = run_eas_on_dist(
                sess,
                policy,
                ops,
                adapt_vars,
                val_env,
                did,
                resources,
                subset,
                loss_type,
                lambda_il,
                n_adapt,
                k,
                seed,
                init_vals,
                twopt_acts=twopt,
            )
            # best-of-32 (or smoke k) on query after φ*
            _, qidx = support_query_indices(did)
            q_tgs, q_obs = slice_dist(val_env, did, qidx)
            bok_k = 4 if smoke else 32
            plans, _ = sample_k_and_neglogp(sess, policy, q_obs, bok_k)
            # slot0 not greedy here; also score greedy separately already have
            G, K, L = plans.shape
            flat = plans.reshape(G * K, L)
            flat_tgs = []
            for i in range(G):
                flat_tgs.extend([q_tgs[i]] * K)
            refs = {}
            _c, ts_flat, _e = _schedule_batch(
                flat_tgs,
                flat,
                resources,
                LatencyOnlyObjective(),
                refs,
                "bok_d%d" % did,
            )
            ts = ts_flat.reshape(G, K)
            row["query_bestof%d_T" % bok_k] = float(np.mean(np.min(ts, axis=1)))
            row["n_trainable_params_adapted"] = int(n_phi)
            per_dist.append(row)
            print(
                "eas_dist %s T0=%.1f T*=%.1f best=%.1f phi_l2=%.4f min=%.1f"
                % (
                    did,
                    row["query_greedy_T0"],
                    row["query_greedy_T"],
                    row["support_best_T_final"],
                    row["phi_l2_final"],
                    (time.time() - t0) / 60.0,
                )
            )
    minutes = (time.time() - t0) / 60.0
    q_final = float(np.mean([r["query_greedy_T"] for r in per_dist]))
    q0 = float(np.mean([r["query_greedy_T0"] for r in per_dist]))
    mixes = [r["mix"] for r in per_dist]
    local_m = float(np.mean([m["local_frac"] for m in mixes]))
    v2v_m = float(np.mean([m["v2v_frac"] for m in mixes]))
    nnon_p50 = float(np.mean([m["n_nonmec_p50"] for m in mixes]))
    out = {
        "method_id": DIAG_EAS_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "adapt_subset": subset,
        "loss_type": loss_type,
        "lambda_il": lambda_il,
        "n_adapt": int(n_adapt),
        "k": int(k),
        "n_trainable_params_adapted": int(n_phi),
        "identity_query_T_mean": id_mean,
        "identity_ref": IDENTITY_VAL_T_REF,
        "smoke": bool(smoke),
        "paper_result": False,
        "per_distribution": per_dist,
        "query_greedy_T0_mean": q0,
        "query_greedy_T_mean": q_final,
        "mix_mean": {
            "local_frac": local_m,
            "v2v_frac": v2v_m,
            "n_nonmec_p50": nnon_p50,
        },
        "minutes": float(minutes),
        "gate": {
            "query_greedy_T_le_520": bool(q_final <= 520.0),
            "local_band": bool(0.15 <= local_m <= 0.25),
            "v2v_band": bool(0.02 <= v2v_m <= 0.08),
            "n_non_p50_ok": bool(nnon_p50 in (4.0, 5.0, 6.0) or 4 <= nnon_p50 <= 6),
            "target_query_T": 520.0,
        },
    }
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["n_trainable_params_adapted"] = int(n_phi)
    payload["query_greedy_T_mean"] = q_final
    _write_payload(run_dir, payload)
    dump_json(run_dir / "eas_eval.json", out)
    print(
        "eas_verdict seed=%s subset=%s loss=%s smoke=%s T0=%.1f T*=%.1f gate520=%s min=%.1f"
        % (
            seed,
            subset,
            loss_type,
            int(smoke),
            q0,
            q_final,
            out["gate"]["query_greedy_T_le_520"],
            minutes,
        )
    )
    return run_dir


def run_diagnostic_eas_inst(
    seed,
    allow_gpu,
    adapt_subset=None,
    loss_type=None,
    lambda_il=None,
    budget=None,
    smoke=False,
    ckpt=None,
    max_graphs=None,
):
    """Per-instance EAS vs best-of-k under equal schedule budget. Validation only."""
    import time

    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from spec.eas_adapt import (
        CKPT_MEANAGG_MEAN,
        build_eas_train_ops,
        dump_json,
        filter_adapt_vars,
        load_policy_partial,
        n_params_vars,
        reset_adapt_state,
        sha256_file,
        snapshot_vars,
    )
    from spec.eas_instance import (
        budget_nk,
        iter_val_graphs,
        run_bok_one_graph,
        run_eas_one_graph,
        summarize_rows,
    )

    require_gpu_permission(allow_gpu)
    if DIAG_EAS_INST_METHOD_ID != "margo_v0.3_diag_eas_inst":
        raise ValueError("eas_inst method_id drift")
    subset = str(adapt_subset or DIAG_EAS_INST_DEFAULT_SUBSET)
    loss_type = str(loss_type or DIAG_EAS_INST_DEFAULT_LOSS)
    lambda_il = float(
        DIAG_EAS_INST_LAMBDA_IL if lambda_il is None else lambda_il
    )
    if smoke:
        budget = int(DIAG_EAS_INST_SMOKE_BUDGET if budget is None else budget)
        max_graphs = int(
            DIAG_EAS_INST_SMOKE_GRAPHS if max_graphs is None else max_graphs
        )
    else:
        budget = int(32 if budget is None else budget)
        max_graphs = None if max_graphs is None else int(max_graphs)
    n_adapt, k = budget_nk(budget)
    ckpt_path = Path(ckpt) if ckpt else CKPT_MEANAGG_MEAN
    if not ckpt_path.is_file():
        raise FileNotFoundError("EAS-inst ckpt missing: %s" % ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    run_dir = diag_eas_inst_run_dir(seed, subset, loss_type, budget)
    enable_cavia = subset == "film"
    enable_eas_emb = subset == "emb"
    film_trainable = subset == "film"
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_EAS_INST_METHOD_ID,
            "paper_result": False,
            "meta_adaptation": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "adapt_subset": subset,
            "loss_type": loss_type,
            "lambda_il": lambda_il,
            "budget": int(budget),
            "n_adapt": int(n_adapt),
            "k": int(k),
            "lr": float(DIAG_EAS_INST_LR),
            "smoke": bool(smoke),
            "max_graphs": max_graphs,
            "run_dir": str(run_dir),
            "note": "per-instance EAS vs best-of-k equal budget; not meta; not Phase 3; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    tf.compat.v1.set_random_seed(int(seed))
    np.random.seed(int(seed))
    _train_env, val_env = _encoder_train_val_envs()
    del _train_env
    resources = val_env.scheduler_resources
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(int(seed))
    policy = Seq2SeqPolicy(
        obs_dim=val_env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
        enable_cavia=enable_cavia,
        cavia_film_trainable=film_trainable,
        enable_eas_emb=enable_eas_emb,
    )
    rows = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        missing = load_policy_partial(policy, ckpt_path, sess)
        if missing:
            print("eas_inst_ckpt_missing_ok %s" % missing)
        adapt_vars = filter_adapt_vars(policy, subset)
        n_phi = n_params_vars(adapt_vars)
        print(
            "eas_inst_audit subset=%s n_params=%d budget=%d N=%d k=%d loss=%s"
            % (subset, n_phi, budget, n_adapt, k, loss_type)
        )
        ops = build_eas_train_ops(policy, adapt_vars, lr=DIAG_EAS_INST_LR)
        sess.run(tf.compat.v1.variables_initializer(ops["opt"].variables()))
        init_vals = snapshot_vars(sess, adapt_vars)
        for gi, (dist_id, li, tg, obs) in enumerate(
            iter_val_graphs(val_env, max_graphs=max_graphs)
        ):
            prefix = "d%d_i%d" % (dist_id, li)
            # Frozen bok needs identity φ; previous graph may have moved φ.
            reset_adapt_state(
                sess, policy, adapt_vars, ops["opt"], subset, init_vals
            )
            np.random.seed(int(seed) * 1000003 + gi)
            tf.compat.v1.set_random_seed(int(seed) * 1000003 + gi)
            bok = run_bok_one_graph(
                sess, policy, tg, obs, resources, budget, prefix + "|bok"
            )
            eas = run_eas_one_graph(
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
                prefix + "|eas",
            )
            if eas["n_schedule_adapt"] != int(budget):
                raise ValueError(
                    "adapt schedules %d != budget %d" % (eas["n_schedule_adapt"], budget)
                )
            row = {
                "dist_id": int(dist_id),
                "local_idx": int(li),
                "bok": bok,
                "eas": eas,
            }
            rows.append(row)
            if (gi + 1) % 25 == 0 or gi < 3:
                print(
                    "eas_inst g=%d/%s d=%s bok=%.1f eas=%.1f g*=%.1f min=%.1f"
                    % (
                        gi + 1,
                        max_graphs or 500,
                        dist_id,
                        bok["T_best"],
                        eas["T_best_among_samples"],
                        eas["T_greedy_after"],
                        (time.time() - t0) / 60.0,
                    )
                )
    summary = summarize_rows(rows, budget)
    minutes = (time.time() - t0) / 60.0
    out = {
        "method_id": DIAG_EAS_INST_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "adapt_subset": subset,
        "loss_type": loss_type,
        "budget": int(budget),
        "n_adapt": int(n_adapt),
        "k": int(k),
        "n_trainable_params_adapted": int(n_phi),
        "smoke": bool(smoke),
        "paper_result": False,
        "meta_adaptation": False,
        "summary": summary,
        "per_graph": rows if smoke or len(rows) <= 32 else rows[:: max(1, len(rows) // 50)],
        "n_rows_stored": len(rows) if smoke or len(rows) <= 32 else len(rows[:: max(1, len(rows) // 50)]),
        "n_rows_total": len(rows),
        "minutes": float(minutes),
        "gate": {
            "eas_beats_bok_by_5s": summary["gate_eas_better_by_5s"],
            "delta_eas_minus_bok": summary["delta_eas_minus_bok"],
            "frac_eas_beats_bok": summary["frac_eas_beats_bok"],
        },
    }
    # Always dump full thin metrics without full plans
    thin = []
    for r in rows:
        thin.append(
            {
                "dist_id": r["dist_id"],
                "local_idx": r["local_idx"],
                "bok_best": r["bok"]["T_best"],
                "bok_greedy": r["bok"]["T_greedy"],
                "eas_best": r["eas"]["T_best_among_samples"],
                "eas_greedy0": r["eas"]["T_greedy0"],
                "eas_greedy_after": r["eas"]["T_greedy_after"],
                "phi_l2": r["eas"]["phi_l2"],
            }
        )
    out["per_graph_thin"] = thin
    out.pop("per_graph", None)
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["summary"] = summary
    _write_payload(run_dir, payload)
    dump_json(run_dir / "eas_inst_eval.json", out)
    print(
        "eas_inst_verdict seed=%s subset=%s loss=%s B=%d smoke=%s bok=%.1f eas=%.1f d=%.1f gate5=%s min=%.1f"
        % (
            seed,
            subset,
            loss_type,
            budget,
            int(smoke),
            summary["T_bok_best_mean"],
            summary["T_eas_best_mean"],
            summary["delta_eas_minus_bok"],
            summary["gate_eas_better_by_5s"],
            minutes,
        )
    )
    return run_dir

# Phase 4 resource-profile axis method ids (isolated from primary 3500).
# Drivers land later; strings must stay greppable for phase4_gate.
_PHASE4_PROFILE_METHOD_IDS = (
    "margo_v0.3_expert_profiles",
    "margo_v0.3_bc_profiles",
    "margo_v0.3_eas_profiles",
    "margo_v0.3_ctx_profiles",
    "margo_v0.3_bok_profiles",
)
assert all(isinstance(x, str) and x.startswith("margo_v0.3_") for x in _PHASE4_PROFILE_METHOD_IDS)
_PHASE5_ENERGY_METHOD_IDS = (
    "margo_v0.3_expert_energy",
    "margo_v0.3_bc_energy",
    "margo_v0.3_eas_energy",
)
assert all(isinstance(x, str) and x.startswith("margo_v0.3_") for x in _PHASE5_ENERGY_METHOD_IDS)


def run_diagnostic_bc_profiles(seed, allow_gpu, smoke=False):
    """BC on 195 profile tasks, obs v2, meanagg+mean. Continuity + held-out val. No PPO."""
    import os
    import time

    os.environ["MARGO_OBS_VERSION"] = "v2"
    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.encoder_obs import set_obs_version
    from spec.bc_profiles import (
        COMBINED_TRAIN,
        PHASE1_MEANAGG_MEAN_VAL_T,
        build_combined_train_cache,
        build_profile_obs_acts,
        continuity_no_regression,
        greedy_decode_and_score,
        run_bc_ce_on_arrays,
    )
    from spec.phase4_campaign import (
        DIAG_BC_PROFILES_METHOD_ID,
        DIAG_ENC_MAX_EPOCHS,
        DIAG_ENC_SMOKE_EPOCHS,
        DIAG_ENC_SMOKE_GRAPHS,
        diag_bc_profiles_run_dir,
    )
    from spec.bc_greedy_mec import BC_CONTINUE_MIN_DELTA, BC_CONTINUE_PATIENCE
    from spec.resource_profiles import role_profile_ids

    require_gpu_permission(allow_gpu)
    set_obs_version("v2")
    if eo.PACKED_DIM != 54:
        raise RuntimeError("PACKED_DIM=%s want 54" % eo.PACKED_DIM)
    if DIAG_BC_PROFILES_METHOD_ID != "margo_v0.3_bc_profiles":
        raise ValueError("bc_profiles method_id drift")

    run_dir = diag_bc_profiles_run_dir(seed)
    epochs = DIAG_ENC_SMOKE_EPOCHS if smoke else DIAG_ENC_MAX_EPOCHS
    max_per = (DIAG_ENC_SMOKE_GRAPHS // 3) if smoke else None  # ~21/profile smoke
    if smoke:
        max_per = 32
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_BC_PROFILES_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "obs_version": "v2",
            "packed_dim": 54,
            "bc_epochs_max": int(epochs),
            "smoke": bool(smoke),
            "run_dir": str(run_dir),
            "note": "Phase4 BC profiles obs v2 meanagg+mean; validation only; no meta-test; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])

    t0 = time.time()
    train_path = COMBINED_TRAIN
    if smoke or (not train_path.is_file()):
        train_path, train_meta = build_combined_train_cache(
            out_path=run_dir / ("bc_profiles_train_obs_v2%s.npz" % ("_smoke" if smoke else "")),
            max_graphs_per_profile=max_per,
        )
    else:
        train_meta = json.loads(train_path.with_suffix(".json").read_text())
    blob = np.load(str(train_path), allow_pickle=False)
    obs = np.asarray(blob["obs"], dtype=np.float32)
    acts = np.asarray(blob["acts"], dtype=np.int32)
    if obs.shape[-1] != 54:
        raise ValueError("train obs last dim %s != 54" % obs.shape[-1])

    tf.compat.v1.reset_default_graph()
    policy = Seq2SeqPolicy(
        obs_dim=54,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
    )
    ckpt_path = run_dir / "ckpt" / "bc_core.ckpt"
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_params = _count_encoder_params(policy)
        payload["n_params"] = int(n_params)
        bc_stats = run_bc_ce_on_arrays(
            sess,
            policy,
            obs,
            acts,
            np.random.RandomState(int(seed) + 80),
            epochs=epochs,
            early_stop_patience=None if smoke else BC_CONTINUE_PATIENCE,
            early_stop_min_delta=BC_CONTINUE_MIN_DELTA,
            max_graphs=None,
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_variables(str(ckpt_path), sess=sess)
        evals = {}
        if not smoke:
            # continuity: frozen val
            frozen_chunk = build_profile_obs_acts("validation", "frozen_7_5_10")
            fr = greedy_decode_and_score(sess, policy, frozen_chunk)
            fr.pop("pred_acts", None)
            fr.update(continuity_no_regression(fr["greedy_T_mean"], PHASE1_MEANAGG_MEAN_VAL_T))
            evals["validation_frozen"] = fr
            print(
                "bc_profiles_continuity T=%.1f phase1=%.1f d=%.1f pass=%s"
                % (
                    fr["greedy_T_mean"],
                    PHASE1_MEANAGG_MEAN_VAL_T,
                    fr["continuity_delta_vs_phase1"],
                    fr["continuity_pass"],
                )
            )
            held = {}
            for pid in role_profile_ids("validation_heldout"):
                ch = build_profile_obs_acts("validation", pid)
                row = greedy_decode_and_score(sess, policy, ch)
                row.pop("pred_acts", None)
                held[pid] = row
                print(
                    "bc_profiles_heldout %s T=%.1f expert=%.1f token=%.3f mix_mec=%.3f"
                    % (pid, row["greedy_T_mean"], row["expert_T_mean"], row["greedy_token_acc_vs_expert"], row["mix_mec"])
                )
            evals["validation_heldout"] = held
    minutes = (time.time() - t0) / 60.0
    payload["bc_pretrain"] = bc_stats
    payload["train_meta"] = train_meta
    payload["evals"] = evals
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    _write_payload(run_dir, payload)
    out = {
        "method_id": DIAG_BC_PROFILES_METHOD_ID,
        "seed": int(seed),
        "smoke": bool(smoke),
        "paper_result": False,
        "n_params": int(payload.get("n_params") or 0),
        "bc": bc_stats,
        "evals": evals,
        "minutes": float(minutes),
    }
    (run_dir / "bc_profiles_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        "bc_profiles_verdict seed=%s smoke=%s n_train=%s epochs=%s min=%.1f"
        % (seed, int(smoke), bc_stats.get("n_graphs"), bc_stats.get("bc_epochs_ran"), minutes)
    )
    return run_dir


def run_diagnostic_eas_profiles(seed, allow_gpu, smoke=False):
    """Phase-3 EAS (lastlayer + pg_il) on 10 held-out-profile val tasks. obs v2. No meta-test."""
    import os
    import time

    os.environ["MARGO_OBS_VERSION"] = "v2"
    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.encoder_obs import set_obs_version
    from spec.eas_adapt import (
        build_eas_train_ops,
        dump_json,
        filter_adapt_vars,
        load_policy_partial,
        n_params_vars,
        run_eas_on_dist,
        sample_k_and_neglogp,
        sha256_file,
        slice_dist,
        snapshot_vars,
        LatencyOnlyObjective,
        _schedule_batch,
    )
    from spec.eas_profiles import (
        GATE_DT_S,
        bc_profiles_ckpt,
        heldout_profile_ids,
        make_val_env_for_profile,
    )
    from spec.phase4_campaign import (
        DIAG_EAS_DEFAULT_LAMBDA_IL,
        DIAG_EAS_DEFAULT_LOSS,
        DIAG_EAS_DEFAULT_SUBSET,
        DIAG_EAS_K,
        DIAG_EAS_LR,
        DIAG_EAS_N_ADAPT,
        DIAG_EAS_PROFILES_METHOD_ID,
        DIAG_EAS_SMOKE_DIST,
        DIAG_EAS_SMOKE_K,
        DIAG_EAS_SMOKE_N_ADAPT,
        diag_eas_profiles_run_dir,
    )
    from spec.split_loader import support_query_indices, validation_distribution_ids
    from spec.bc_profiles import PHASE1_MEANAGG_MEAN_VAL_T, continuity_no_regression

    require_gpu_permission(allow_gpu)
    set_obs_version("v2")
    if eo.PACKED_DIM != 54:
        raise RuntimeError("PACKED_DIM=%s want 54" % eo.PACKED_DIM)
    if DIAG_EAS_PROFILES_METHOD_ID != "margo_v0.3_eas_profiles":
        raise ValueError("eas_profiles method_id drift")

    subset = DIAG_EAS_DEFAULT_SUBSET
    loss_type = DIAG_EAS_DEFAULT_LOSS
    lambda_il = float(DIAG_EAS_DEFAULT_LAMBDA_IL)
    if smoke:
        n_adapt = int(DIAG_EAS_SMOKE_N_ADAPT)
        k = int(DIAG_EAS_SMOKE_K)
        dist_ids = (int(DIAG_EAS_SMOKE_DIST),)
        profiles = heldout_profile_ids()[:1]
    else:
        n_adapt = int(DIAG_EAS_N_ADAPT)
        k = int(DIAG_EAS_K)
        dist_ids = tuple(int(d) for d in validation_distribution_ids())
        profiles = heldout_profile_ids()

    ckpt_path = bc_profiles_ckpt(seed)
    if not ckpt_path.is_file():
        raise FileNotFoundError("eas_profiles needs BC ckpt: %s" % ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    run_dir = diag_eas_profiles_run_dir(seed)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_EAS_PROFILES_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "obs_version": "v2",
            "packed_dim": 54,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "adapt_subset": subset,
            "loss_type": loss_type,
            "lambda_il": lambda_il,
            "n_adapt": int(n_adapt),
            "k": int(k),
            "lr": float(DIAG_EAS_LR),
            "smoke": bool(smoke),
            "dist_ids": list(dist_ids),
            "profiles": list(profiles),
            "run_dir": str(run_dir),
            "note": "Phase4 EAS lastlayer+pg_il on held-out resource profiles; validation only; no meta-test; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    tf.compat.v1.set_random_seed(int(seed))
    np.random.seed(int(seed))

    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(int(seed))
    policy = Seq2SeqPolicy(
        obs_dim=54,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
    )
    per_task = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        missing = load_policy_partial(policy, ckpt_path, sess)
        if missing:
            raise ValueError("eas_profiles ckpt missing vars: %s" % missing)
        adapt_vars = filter_adapt_vars(policy, subset)
        n_phi = n_params_vars(adapt_vars)
        print(
            "eas_profiles_trainable subset=%s n_vars=%d n_params=%d"
            % (subset, len(adapt_vars), n_phi)
        )
        ops = build_eas_train_ops(policy, adapt_vars, lr=DIAG_EAS_LR)
        sess.run(tf.compat.v1.variables_initializer(ops["opt"].variables()))
        init_vals = snapshot_vars(sess, adapt_vars)
        for pid in profiles:
            print("eas_profiles env_build start %s" % pid)
            val_env = make_val_env_for_profile(pid)
            resources = val_env.scheduler_resources
            print(
                "eas_profiles env_build done %s input_dim=%s dists=%s"
                % (pid, val_env.input_dim, list(val_env.distribution_ids))
            )
            for did in dist_ids:
                row = run_eas_on_dist(
                    sess,
                    policy,
                    ops,
                    adapt_vars,
                    val_env,
                    did,
                    resources,
                    subset,
                    loss_type,
                    lambda_il,
                    n_adapt,
                    k,
                    seed,
                    init_vals,
                )
                _, qidx = support_query_indices(did)
                q_tgs, q_obs = slice_dist(val_env, did, qidx)
                bok_k = 4 if smoke else 32
                plans, _ = sample_k_and_neglogp(sess, policy, q_obs, bok_k)
                G, K, L = plans.shape
                flat = plans.reshape(G * K, L)
                flat_tgs = []
                for i in range(G):
                    flat_tgs.extend([q_tgs[i]] * K)
                refs = {}
                _c, ts_flat, _e = _schedule_batch(
                    flat_tgs,
                    flat,
                    resources,
                    LatencyOnlyObjective(),
                    refs,
                    "bok_%s_d%d" % (pid, did),
                )
                ts = ts_flat.reshape(G, K)
                row["query_bestof%d_T" % bok_k] = float(np.mean(np.min(ts, axis=1)))
                row["n_trainable_params_adapted"] = int(n_phi)
                row["profile_id"] = str(pid)
                row["delta_T"] = float(row["query_greedy_T0"] - row["query_greedy_T"])
                per_task.append(row)
                print(
                    "eas_profiles %s dist=%s T0=%.1f T*=%.1f dT=%.1f best=%.1f phi_l2=%.4f min=%.1f"
                    % (
                        pid,
                        did,
                        row["query_greedy_T0"],
                        row["query_greedy_T"],
                        row["delta_T"],
                        row["support_best_T_final"],
                        row["phi_l2_final"],
                        (time.time() - t0) / 60.0,
                    )
                )
            del val_env

    minutes = (time.time() - t0) / 60.0
    q0 = float(np.mean([r["query_greedy_T0"] for r in per_task]))
    q_star = float(np.mean([r["query_greedy_T"] for r in per_task]))
    dT = float(q0 - q_star)
    mixes = [r["mix"] for r in per_task]
    local_m = float(np.mean([m["local_frac"] for m in mixes]))
    mec_m = float(np.mean([m["mec_frac"] for m in mixes]))
    v2v_m = float(np.mean([m["v2v_frac"] for m in mixes]))
    per_profile = {}
    for pid in profiles:
        rows = [r for r in per_task if r["profile_id"] == pid]
        per_profile[pid] = {
            "query_greedy_T0_mean": float(np.mean([r["query_greedy_T0"] for r in rows])),
            "query_greedy_T_mean": float(np.mean([r["query_greedy_T"] for r in rows])),
            "delta_T_mean": float(np.mean([r["delta_T"] for r in rows])),
            "n_tasks": int(len(rows)),
        }
    # One-sided continuity already measured on BC seed0 frozen val (T=493.8).
    bc_eval = Path(ckpt_path).parents[1] / "bc_profiles_eval.json"
    cont = {}
    if bc_eval.is_file():
        bc_blob = json.loads(bc_eval.read_text())
        fr = (bc_blob.get("evals") or {}).get("validation_frozen") or {}
        if "greedy_T_mean" in fr:
            cont = continuity_no_regression(fr["greedy_T_mean"], PHASE1_MEANAGG_MEAN_VAL_T)
            cont["greedy_T_mean"] = float(fr["greedy_T_mean"])
    gate_dt = bool(smoke) or bool(dT >= GATE_DT_S)
    mix_ok = bool(mec_m < 0.95 and (local_m + v2v_m) > 0.05)
    out = {
        "method_id": DIAG_EAS_PROFILES_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "adapt_subset": subset,
        "loss_type": loss_type,
        "lambda_il": lambda_il,
        "n_adapt": int(n_adapt),
        "k": int(k),
        "n_trainable_params_adapted": int(n_phi),
        "smoke": bool(smoke),
        "paper_result": False,
        "obs_version": "v2",
        "profiles": list(profiles),
        "dist_ids": list(dist_ids),
        "per_task": per_task,
        "per_profile": per_profile,
        "query_greedy_T0_mean": q0,
        "query_greedy_T_mean": q_star,
        "delta_T_mean": dT,
        "mix_mean": {
            "local_frac": local_m,
            "mec_frac": mec_m,
            "v2v_frac": v2v_m,
        },
        "continuity": cont,
        "minutes": float(minutes),
        "gate": {
            "delta_T_ge_15": bool(dT >= GATE_DT_S),
            "mix_not_collapsed": mix_ok,
            "continuity_pass": bool(cont.get("continuity_pass", False)),
            "pass": bool(gate_dt and mix_ok and (smoke or cont.get("continuity_pass", False))),
        },
    }
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["n_trainable_params_adapted"] = int(n_phi)
    payload["query_greedy_T_mean"] = q_star
    payload["delta_T_mean"] = dT
    payload["evals"] = {"per_profile": per_profile, "continuity": cont}
    _write_payload(run_dir, payload)
    dump_json(run_dir / "eas_profiles_eval.json", out)
    print(
        "eas_profiles_verdict seed=%s smoke=%s T0=%.1f T*=%.1f dT=%.1f gate=%s min=%.1f"
        % (seed, int(smoke), q0, q_star, dT, out["gate"]["pass"], minutes)
    )
    return run_dir


def run_diagnostic_ctx_profiles(seed, allow_gpu, smoke=False):
    """PEARL-style context z + FiLM, joint BC finetune from bc_profiles ckpt. No meta-test."""
    import os
    import time

    os.environ["MARGO_OBS_VERSION"] = "v2"
    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.encoder_obs import set_obs_version
    from spec.bc_profiles import (
        COMBINED_TRAIN,
        PHASE1_MEANAGG_MEAN_VAL_T,
        build_profile_obs_acts,
        continuity_no_regression,
        _score_plans,
    )
    from spec.bc_greedy_mec import BC_BATCH, BC_LR, align_greedy_pred
    from spec.ctx_profiles import (
        CTX_Z_DIM,
        GATE_DT_S,
        TRAIN_QUERY_GIDX,
        TRAIN_SUPPORT_GIDX,
        bc_profiles_ckpt,
        ctx_feed,
        heldout_profile_ids,
        mask_task,
    )
    from spec.eas_adapt import dump_json, load_policy_partial, sha256_file
    from spec.phase4_campaign import (
        DIAG_CTX_MAX_EPOCHS,
        DIAG_CTX_PROFILES_METHOD_ID,
        DIAG_CTX_SMOKE_EPOCHS,
        DIAG_CTX_SMOKE_TASKS,
        diag_ctx_profiles_run_dir,
    )
    from spec.split_loader import (
        iter_meta_tasks,
        support_query_indices,
        validation_distribution_ids,
    )

    require_gpu_permission(allow_gpu)
    set_obs_version("v2")
    if eo.PACKED_DIM != 54:
        raise RuntimeError("PACKED_DIM=%s want 54" % eo.PACKED_DIM)
    if DIAG_CTX_PROFILES_METHOD_ID != "margo_v0.3_ctx_profiles":
        raise ValueError("ctx_profiles method_id drift")

    ckpt_path = bc_profiles_ckpt(seed)
    if not ckpt_path.is_file():
        raise FileNotFoundError("ctx_profiles needs BC ckpt: %s" % ckpt_path)
    train_path = Path(ckpt_path).parents[1] / "bc_profiles_train_obs_v2.npz"
    if not train_path.is_file():
        train_path = COMBINED_TRAIN
    if not train_path.is_file():
        raise FileNotFoundError("ctx_profiles needs train cache: %s or %s" % (train_path, COMBINED_TRAIN))
    blob = np.load(str(train_path), allow_pickle=False)
    obs_all = np.asarray(blob["obs"], dtype=np.float32)
    acts_all = np.asarray(blob["acts"], dtype=np.int32)
    t_all = np.asarray(blob["t_expert"], dtype=np.float32)
    dist_all = np.asarray(blob["dist_id"], dtype=np.int32)
    gidx_all = np.asarray(blob["graph_idx"], dtype=np.int32)
    prof_all = blob["profile_id"]
    if obs_all.shape[-1] != 54:
        raise ValueError("train obs last dim %s != 54" % obs_all.shape[-1])

    tasks = [(int(d), str(p)) for d, p in iter_meta_tasks("meta_train")]
    epochs = int(DIAG_CTX_SMOKE_EPOCHS if smoke else DIAG_CTX_MAX_EPOCHS)
    if smoke:
        tasks = tasks[: int(DIAG_CTX_SMOKE_TASKS)]

    run_dir = diag_ctx_profiles_run_dir(seed)
    ckpt_sha = sha256_file(ckpt_path)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_CTX_PROFILES_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "obs_version": "v2",
            "packed_dim": 54,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "ctx_z_dim": int(CTX_Z_DIM),
            "bc_epochs_max": int(epochs),
            "smoke": bool(smoke),
            "n_train_tasks": int(len(tasks)),
            "run_dir": str(run_dir),
            "note": "Phase4 PEARL-style context-only FiLM z from support (obs,plan,T); validation only; no meta-test; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    rng = np.random.RandomState(int(seed) + 90)
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(int(seed))
    policy = Seq2SeqPolicy(
        obs_dim=54,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
        enable_context_encoder=True,
        cavia_z_dim=int(CTX_Z_DIM),
        cavia_film_trainable=True,
    )
    loss = tf.reduce_mean(policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(BC_LR, name="ctx_profiles_adam")
    train_op = opt.minimize(loss, var_list=policy.get_trainable_variables())

    def _slice_train(profile_id, dist_id, gidx_keep):
        m = mask_task(dist_all, prof_all, gidx_all, profile_id, dist_id, gidx_keep)
        return obs_all[m], acts_all[m], t_all[m]

    def _greedy_rows(sess, obs, acts, t_ex, resources, dist_id, graph_idx, cluster, ctx_o, ctx_a, ctx_t, ctx_zero):
        n, n_tok = acts.shape
        preds = []
        for start in range(0, n, BC_BATCH):
            sl = slice(start, min(start + BC_BATCH, n))
            batch = obs[sl]
            fl = np.full((batch.shape[0],), n_tok, dtype=np.int32)
            greedy_pred = sess.run(
                policy.network.greedy_decoder_prediction,
                feed_dict=ctx_feed(policy, batch, fl, ctx_o, ctx_a, ctx_t, ctx_zero=ctx_zero),
            )
            aligned, _trunc = align_greedy_pred(greedy_pred, n_tok)
            preds.append(aligned)
        pred = np.concatenate(preds, axis=0)
        t_mean = _score_plans(obs, pred, resources, dist_id, graph_idx, cluster)
        counts = np.bincount(pred.reshape(-1), minlength=3).astype(np.float64)
        counts = counts / max(float(counts.sum()), 1.0)
        return {
            "greedy_T_mean": float(t_mean),
            "greedy_token_acc_vs_expert": float(np.mean(pred == acts)),
            "expert_T_mean": float(np.mean(t_ex)),
            "mix_local": float(counts[0]),
            "mix_mec": float(counts[1]),
            "mix_v2v": float(counts[2]),
            "n_graphs": int(n),
        }

    def _eval_profile(sess, profile_id, dist_ids):
        chunk = build_profile_obs_acts("validation", profile_id)
        rows = []
        for did in dist_ids:
            sidx, qidx = support_query_indices(did)
            m_s = mask_task(chunk["dist_id"], chunk["profile_id"], chunk["graph_idx"], profile_id, did, sidx)
            m_q = mask_task(chunk["dist_id"], chunk["profile_id"], chunk["graph_idx"], profile_id, did, qidx)
            if int(np.sum(m_s)) < 1 or int(np.sum(m_q)) < 1:
                raise ValueError("empty support/query %s dist=%s" % (profile_id, did))
            ctx_o, ctx_a, ctx_t = chunk["obs"][m_s], chunk["acts"][m_s], chunk["t_expert"][m_s]
            q_obs, q_acts = chunk["obs"][m_q], chunk["acts"][m_q]
            z0 = _greedy_rows(
                sess, q_obs, q_acts, chunk["t_expert"][m_q], chunk["resources"],
                chunk["dist_id"][m_q], chunk["graph_idx"][m_q], chunk["cluster"],
                ctx_o, ctx_a, ctx_t, True,
            )
            zc = _greedy_rows(
                sess, q_obs, q_acts, chunk["t_expert"][m_q], chunk["resources"],
                chunk["dist_id"][m_q], chunk["graph_idx"][m_q], chunk["cluster"],
                ctx_o, ctx_a, ctx_t, False,
            )
            dT = float(z0["greedy_T_mean"] - zc["greedy_T_mean"])
            row = {
                "profile_id": str(profile_id),
                "dist_id": int(did),
                "query_greedy_T0": float(z0["greedy_T_mean"]),
                "query_greedy_T_ctx": float(zc["greedy_T_mean"]),
                "delta_T": dT,
                "token0": float(z0["greedy_token_acc_vs_expert"]),
                "token_ctx": float(zc["greedy_token_acc_vs_expert"]),
                "mix_mec0": float(z0["mix_mec"]),
                "mix_mec_ctx": float(zc["mix_mec"]),
                "n_support": int(np.sum(m_s)),
                "n_query": int(np.sum(m_q)),
            }
            rows.append(row)
            print(
                "ctx_profiles %s dist=%s T0=%.1f Tctx=%.1f dT=%.1f"
                % (profile_id, did, z0["greedy_T_mean"], zc["greedy_T_mean"], dT)
            )
        return rows

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        missing = load_policy_partial(policy, ckpt_path, sess)
        print("ctx_profiles_ckpt_missing_ok_new n=%d" % len(missing))
        sess.run(tf.compat.v1.variables_initializer(opt.variables()))
        epoch_losses = []
        best_loss = float("inf")
        n_tok = int(acts_all.shape[1])
        for epoch in range(int(epochs)):
            rng.shuffle(tasks)
            batch_losses = []
            for dist_id, profile_id in tasks:
                s_obs, s_acts, s_t = _slice_train(profile_id, dist_id, TRAIN_SUPPORT_GIDX)
                q_obs, q_acts, _q_t = _slice_train(profile_id, dist_id, TRAIN_QUERY_GIDX)
                if s_obs.shape[0] < 1 or q_obs.shape[0] < 1:
                    raise ValueError("empty train task %s dist=%s" % (profile_id, dist_id))
                shift = np.concatenate(
                    [np.zeros((q_acts.shape[0], 1), dtype=np.int32), q_acts[:, :-1]], axis=1
                )
                for start in range(0, q_acts.shape[0], BC_BATCH):
                    sl = slice(start, min(start + BC_BATCH, q_acts.shape[0]))
                    fl = np.full((q_obs[sl].shape[0],), n_tok, dtype=np.int32)
                    _, lv = sess.run(
                        [train_op, loss],
                        feed_dict=ctx_feed(
                            policy,
                            q_obs[sl],
                            fl,
                            s_obs,
                            s_acts,
                            s_t,
                            shift=shift[sl],
                            acts=q_acts[sl],
                            ctx_zero=False,
                        ),
                    )
                    batch_losses.append(float(lv))
            cur = float(np.mean(batch_losses))
            epoch_losses.append(cur)
            if cur < best_loss:
                best_loss = cur
            print("ctx_profiles epoch=%d loss=%.5f best=%.5f" % (epoch, cur, best_loss))
        ckpt_out = run_dir / "ckpt" / "bc_core.ckpt"
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        policy.save_variables(str(ckpt_out), sess=sess)
        dist_ids = list(validation_distribution_ids())
        profiles = heldout_profile_ids()
        if smoke:
            dist_ids = dist_ids[:1]
            profiles = profiles[:1]
        per_task = []
        for pid in profiles:
            per_task.extend(_eval_profile(sess, pid, dist_ids))
        frozen_rows = []
        if not smoke:
            frozen_rows = _eval_profile(sess, "frozen_7_5_10", dist_ids)

    minutes = (time.time() - t0) / 60.0
    dT = float(np.mean([r["delta_T"] for r in per_task]))
    t0m = float(np.mean([r["query_greedy_T0"] for r in per_task]))
    tcm = float(np.mean([r["query_greedy_T_ctx"] for r in per_task]))
    per_profile = {}
    for pid in sorted(set(r["profile_id"] for r in per_task)):
        rows = [r for r in per_task if r["profile_id"] == pid]
        per_profile[pid] = {
            "query_greedy_T0_mean": float(np.mean([r["query_greedy_T0"] for r in rows])),
            "query_greedy_T_ctx_mean": float(np.mean([r["query_greedy_T_ctx"] for r in rows])),
            "delta_T_mean": float(np.mean([r["delta_T"] for r in rows])),
            "n_tasks": int(len(rows)),
        }
    cont = {}
    if frozen_rows:
        ft0 = float(np.mean([r["query_greedy_T0"] for r in frozen_rows]))
        cont = continuity_no_regression(ft0, PHASE1_MEANAGG_MEAN_VAL_T)
        cont["greedy_T0_mean"] = ft0
        cont["greedy_T_ctx_mean"] = float(np.mean([r["query_greedy_T_ctx"] for r in frozen_rows]))
    gate_dt = bool(smoke) or bool(dT >= GATE_DT_S)
    out = {
        "method_id": DIAG_CTX_PROFILES_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "smoke": bool(smoke),
        "paper_result": False,
        "obs_version": "v2",
        "ctx_z_dim": int(CTX_Z_DIM),
        "bc": {"bc_epochs_ran": int(epochs), "bc_best_loss": float(best_loss), "bc_epoch_losses": epoch_losses, "n_tasks": int(len(tasks))},
        "per_task": per_task,
        "per_profile": per_profile,
        "frozen": frozen_rows,
        "query_greedy_T0_mean": t0m,
        "query_greedy_T_ctx_mean": tcm,
        "delta_T_mean": dT,
        "continuity": cont,
        "minutes": float(minutes),
        "gate": {
            "delta_T_ge_10": bool(dT >= GATE_DT_S),
            "continuity_pass": bool(cont.get("continuity_pass", smoke)),
            "pass": bool(gate_dt and (smoke or cont.get("continuity_pass", False))),
        },
    }
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["delta_T_mean"] = dT
    payload["evals"] = {"per_profile": per_profile, "continuity": cont}
    _write_payload(run_dir, payload)
    dump_json(run_dir / "ctx_profiles_eval.json", out)
    print(
        "ctx_profiles_verdict seed=%s smoke=%s T0=%.1f Tctx=%.1f dT=%.1f gate=%s min=%.1f"
        % (seed, int(smoke), t0m, tcm, dT, out["gate"]["pass"], minutes)
    )
    return run_dir


def run_diagnostic_bok_profiles(seed, allow_gpu, smoke=False):
    """Frozen BC-profiles policy, best-of-k on frozen + held-out val. No training. No meta-test."""
    import os
    import time

    os.environ["MARGO_OBS_VERSION"] = "v2"
    import numpy as np
    import tensorflow as tf
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.encoder_obs import set_obs_version
    from spec.bc_profiles import (
        PHASE1_MEANAGG_MEAN_VAL_T,
        build_profile_obs_acts,
        continuity_no_regression,
    )
    from spec.best_of_k import dump_json, sha256_file
    from spec.bok_profiles import (
        BC_SEED0_FROZEN_GREEDY_T,
        CKPT_MATCH_SLACK_S,
        EVAL_PROFILE_IDS,
        FROZEN_PROFILE_ID,
        SAMPLING_HELP_S,
        bc_profiles_ckpt,
        eval_bok_on_chunk,
        eval_profile_ids,
    )
    from spec.phase4_campaign import (
        DIAG_BOK_K_MAX,
        DIAG_BOK_K_SWEEP,
        DIAG_BOK_K_TEMP,
        DIAG_BOK_PROFILES_METHOD_ID,
        DIAG_BOK_SMOKE_GRAPHS,
        DIAG_BOK_SMOKE_K,
        DIAG_BOK_TEMPS,
        diag_bok_profiles_run_dir,
    )

    require_gpu_permission(allow_gpu)
    set_obs_version("v2")
    if eo.PACKED_DIM != 54:
        raise RuntimeError("PACKED_DIM=%s want 54" % eo.PACKED_DIM)
    if DIAG_BOK_PROFILES_METHOD_ID != "margo_v0.3_bok_profiles":
        raise ValueError("bok_profiles method_id drift")
    if eval_profile_ids() != list(EVAL_PROFILE_IDS):
        raise ValueError("bok_profiles eval profile drift")

    ckpt_path = bc_profiles_ckpt(seed)
    if not ckpt_path.is_file():
        raise FileNotFoundError("bok_profiles needs BC ckpt: %s" % ckpt_path)
    run_dir = diag_bok_profiles_run_dir(seed)
    k_max = DIAG_BOK_SMOKE_K if smoke else DIAG_BOK_K_MAX
    max_graphs = DIAG_BOK_SMOKE_GRAPHS if smoke else None
    ks = tuple(k for k in DIAG_BOK_K_SWEEP if k <= int(k_max))
    profiles = [FROZEN_PROFILE_ID] if smoke else list(EVAL_PROFILE_IDS)
    ckpt_sha = sha256_file(ckpt_path)
    payload = provenance_template(seed)
    payload.update(
        {
            "method_id": DIAG_BOK_PROFILES_METHOD_ID,
            "paper_result": False,
            "gpu_requested": True,
            "gpu_finished": False,
            "ppo": False,
            "training": False,
            "encoder_type": "meanagg",
            "readout_type": "mean",
            "obs_version": "v2",
            "packed_dim": 54,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "k_sweep": list(ks),
            "k_max": int(k_max),
            "temperature_default": 1.0,
            "eval_profiles": profiles,
            "smoke": bool(smoke),
            "run_dir": str(run_dir),
            "note": "Phase4 best-of-k on bc_profiles ckpt obs v2; validation only; no meta-test; no PPO; paper_result=false; frozen primary remains 3500",
        }
    )
    _write_payload(run_dir, payload)
    from utils import logger

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    tf.compat.v1.set_random_seed(int(seed))
    np.random.seed(int(seed))
    t0 = time.time()
    tf.compat.v1.reset_default_graph()
    policy = Seq2SeqPolicy(
        obs_dim=54,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name="pi",
        encoder_type="meanagg",
        readout_type="mean",
    )
    out = {
        "method_id": DIAG_BOK_PROFILES_METHOD_ID,
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": ckpt_sha,
        "encoder_type": "meanagg",
        "readout_type": "mean",
        "obs_version": "v2",
        "smoke": bool(smoke),
        "paper_result": False,
        "k_sweep": list(ks),
        "eval_profiles": profiles,
    }
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        policy.load_variables(str(ckpt_path), sess=sess)
        print(
            "bok_profiles_load_ckpt %s sha=%s packed=%s"
            % (ckpt_path, ckpt_sha[:16], eo.PACKED_DIM)
        )
        per_profile = {}
        for pid in profiles:
            print("bok_profiles_eval %s n=%s k_max=%d" % (pid, max_graphs or 500, k_max))
            chunk = build_profile_obs_acts("validation", pid, max_graphs=max_graphs)
            row = eval_bok_on_chunk(
                sess,
                policy,
                chunk,
                k_max=k_max,
                ks=ks,
                seed=seed,
                split_name="validation_%s" % pid,
            )
            per_profile[pid] = row
            print(
                "bok_profiles %s T_greedy=%.1f T_best_%d=%.1f expert=%.1f min=%.1f"
                % (
                    pid,
                    row["T_greedy"],
                    k_max,
                    (row["by_k"].get(str(k_max)) or {}).get("T_best_k"),
                    row["expert_T_mean"],
                    (time.time() - t0) / 60.0,
                )
            )
            if (not smoke) and pid == FROZEN_PROFILE_ID:
                temp_rows = {}
                for temp in DIAG_BOK_TEMPS:
                    if abs(float(temp) - 1.0) < 1e-12:
                        temp_rows[str(temp)] = row["by_k"].get(str(DIAG_BOK_K_TEMP))
                        continue
                    print("bok_profiles_temp t=%.1f k=%d %s" % (temp, DIAG_BOK_K_TEMP, pid))
                    trow = eval_bok_on_chunk(
                        sess,
                        policy,
                        chunk,
                        k_max=DIAG_BOK_K_TEMP,
                        ks=(1, DIAG_BOK_K_TEMP),
                        seed=seed,
                        split_name="validation_%s_t%s" % (pid, temp),
                        temperature=float(temp),
                    )
                    temp_rows[str(temp)] = trow["by_k"][str(DIAG_BOK_K_TEMP)]
                per_profile[pid]["temp_sweep_k32"] = temp_rows
        out["per_profile"] = per_profile

    minutes = (time.time() - t0) / 60.0
    frozen = per_profile.get(FROZEN_PROFILE_ID) or {}
    t_g = frozen.get("T_greedy")
    by_k = frozen.get("by_k") or {}
    t32 = (by_k.get("32") or {}).get("T_best_k")
    t64 = (by_k.get("64") or by_k.get(str(k_max)) or {}).get("T_best_k")
    cont = {}
    if t_g is not None:
        cont = continuity_no_regression(t_g, PHASE1_MEANAGG_MEAN_VAL_T)
    ckpt_match = None
    if t_g is not None:
        ckpt_match = abs(float(t_g) - float(BC_SEED0_FROZEN_GREEDY_T)) <= float(CKPT_MATCH_SLACK_S)
    sampling_help = None
    if t_g is not None and t32 is not None:
        sampling_help = bool((float(t_g) - float(t32)) >= float(SAMPLING_HELP_S))
    gate = {
        "T_greedy_frozen": t_g,
        "T_best_32_frozen": t32,
        "T_best_64_frozen": t64,
        "delta_best32_over_greedy": None if (t_g is None or t32 is None) else float(t_g) - float(t32),
        "continuity_pass": bool(cont.get("continuity_pass", smoke)),
        "ckpt_match_bc_seed0": ckpt_match,
        "sampling_helps_10s": sampling_help,
        "pass": bool(
            smoke
            or (
                cont.get("continuity_pass")
                and bool(ckpt_match)
                and bool(sampling_help)
            )
        ),
    }
    out["continuity"] = cont
    out["gate"] = gate
    out["minutes"] = float(minutes)
    payload["gpu_finished"] = True
    payload["minutes"] = float(minutes)
    payload["wall_clock_s"] = float(minutes * 60.0)
    _write_payload(run_dir, payload)
    dump_json(run_dir / "bok_profiles_eval.json", out)
    print(
        "bok_profiles_verdict seed=%s smoke=%s frozen_T_greedy=%s T_best_32=%s T_best_64=%s gate=%s min=%.1f"
        % (seed, int(smoke), t_g, t32, t64, gate["pass"], minutes)
    )
    return run_dir


