#!/usr/bin/env python3
"""Gate G: evaluate the OLD 5x500 pilot core checkpoints with the corrected evaluator.

The pilot's validation was mis-measured (no core->scratch sync, adaptation accumulated),
so this script answers the salvage question: what did the latency-only cores actually
learn? Each run gets a fresh process because the frozen TF graph can only be built once.

Label the output clearly as OLD TRAINING (constraints not active in the reward) with a
CORRECTED EVALUATOR.

CLI: python3 spec/automotive_training/salvage_eval_pilot500.py --seed N --ckpt PATH --out PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="pilot seed whose core is loaded")
    ap.add_argument("--ckpt", required=True, help="checkpoint path to load into the core")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="old_training_latency_only_corrected_evaluator")
    args = ap.parse_args()

    import tensorflow as tf

    from spec.automotive_training.automotive_primary import (
        build_automotive_primary_stack,
    )

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit("checkpoint not found: %s" % ckpt)

    trainer, algo = build_automotive_primary_stack(
        seed=0, n_itr=1, ckpt_dir="/tmp/gateG_seed%d" % args.seed)
    evaluator = trainer.held_out_evaluator
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(ckpt), sess=sess)
        algo.sync_task_policies_from_core()
        k0 = evaluator.evaluate_all(k_steps=0, sess=sess)
        k3 = evaluator.evaluate_all(k_steps=3, sess=sess)
    payload = {
        "label": args.label,
        "pilot_seed": int(args.seed),
        "checkpoint": str(ckpt),
        "k0": k0,
        "k3": k3,
        "sync_verified_max_abs_diff": float(getattr(evaluator, "last_sync_max_abs_diff", -1)),
        "core_unchanged_after_adaptation": bool(
            getattr(evaluator, "core_unchanged_after_adaptation", False)),
        "note": ("OLD TRAINING: the pilot reward was latency-only in practice (the "
                 "constraint multipliers never reached the rollout clones). Evaluation "
                 "is the corrected one (core -> scratch sync verified)."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"seed": args.seed, "ckpt": ckpt.name,
                      "k0_s": k0["query_mean_latency_seconds"],
                      "k3_s": k3["query_mean_latency_seconds"],
                      "k0_hard": k0["graph_hard_violation_rate"],
                      "k0_high_task_rate": k0["high_task_tardiness_task_rate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
