#!/usr/bin/env python3
"""Diagnostic: does deterministic decoding depend on the core weights at all?

The adaptation sweep produced bit-identical latencies for k=0..5 and for two different
checkpoints. Before trusting (or dismissing) any adaptation result we must know whether
(a) the checkpoint load reaches the evaluation, and (b) the argmax plan actually depends
on the weights. This script evaluates a RANDOM core and two loaded checkpoints and also
reports the action mix of the deterministic query plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINTS = {
    "random_core": None,
    "gatef_proof_final": "runs/automotive_mc_v1/gatef_proof/seed_0/ckpt/meta_model_final.ckpt",
    "pilot_seed0_final": "runs/automotive_mc_v1/primary_500/seed_0/ckpt/meta_model_final.ckpt",
    "pilot_seed2_final": "runs/automotive_mc_v1/primary_500/seed_2/ckpt/meta_model_final.ckpt",
}


def action_mix(paths) -> dict:
    actions = []
    for task_paths in paths.values():
        for path in task_paths:
            actions.append(np.asarray(path["actions"]).reshape(-1))
    if not actions:
        return {}
    flat = np.concatenate(actions)
    return {str(a): float((flat == a).mean()) for a in (0, 1, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/diagnose_decoding.json")
    ap.add_argument("--decoding", default="deterministic",
                    choices=["deterministic", "stochastic"])
    args = ap.parse_args()

    import tensorflow as tf

    from spec.automotive_training.automotive_primary import (
        build_automotive_primary_stack,
    )

    trainer, algo = build_automotive_primary_stack(seed=0, n_itr=1, ckpt_dir="/tmp/diag",
                                                   decoding=args.decoding)
    evaluator = trainer.held_out_evaluator
    out = {}
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for label, ckpt in CHECKPOINTS.items():
            if ckpt:
                path = Path(ckpt)
                if not path.exists():
                    out[label] = {"error": "missing %s" % path}
                    continue
                trainer.policy.core_policy.load_variables(str(path), sess=sess)
            algo.sync_task_policies_from_core()
            ev = evaluator.evaluate(0, replicates=2, decoding=args.decoding, sess=sess)
            deterministic = args.decoding == "deterministic"
            env = evaluator._paired_env(evaluator.query_graphs, 1000, 0)
            paths = evaluator._rollout_on(env, evaluator.policy, adapt_steps=0, seed=1000,
                                          deterministic=deterministic)
            out[label] = {
                "decoding": args.decoding,
                "k0_latency_s": ev["k0_latency_s"],
                "k0_latency_per_replicate": [r["k0_latency_s"] for r in ev["per_replicate"]],
                "action_mix": action_mix(paths),
            }
            print(json.dumps({label: out[label]}, sort_keys=True), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
