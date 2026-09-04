"""Phase 4 primary-run driver. Imports TensorFlow. Do not import from numpy tests.

GPU launch requires spec.phase4_campaign.require_gpu_permission plus human chat approval.
"""

from __future__ import annotations

import json

from spec.phase4_campaign import (
    METHOD_ID,
    OUTER_ITERS,
    provenance_template,
    require_gpu_permission,
    seed_run_dir,
    smoke_run_dir,
)

SMOKE_ITERS = 1


def _write_payload(run_dir, payload):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval" / "meta_test").mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (run_dir / "provenance.json").write_text(text)
    (run_dir / "config.resolved.json").write_text(text)


def _train(seed, n_itr, run_dir):
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])
    trainer, algo = build_frozen_primary_stack(
        seed=int(seed),
        n_itr=int(n_itr),
        ckpt_dir=str(run_dir / "ckpt"),
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        algo.sync_task_policies_from_core()
        trainer.train()


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
    _train(seed, OUTER_ITERS, run_dir)
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
    _train(seed, SMOKE_ITERS, run_dir)
    payload["gpu_finished"] = True
    _write_payload(run_dir, payload)
    return run_dir
