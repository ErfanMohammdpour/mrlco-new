#!/usr/bin/env python3
"""Support-adaptation sweep k in {0,1,2,3,5} on the frozen paired panel.

Answers the question the 5x500 pilot could not: does support adaptation actually help?
Every k is evaluated with the SAME paired realizations (frozen protocol), deterministic
query decoding, and against the candidate oracle on the same realizations, so the
comparison is paired within a replicate.

CLI: python3 spec/automotive_training/adaptation_sweep.py --ckpt PATH --out PATH
                                                          [--replicates N] [--label TEXT]
"""

from __future__ import annotations

from spec.automotive_training.v2.compat import fmean  # noqa: E402
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.eval_protocol import (  # noqa: E402
    ADAPTATION_K_STEPS,
    PRIMARY_DECODING,
    PROTOCOL_ID,
    SELECT_REPLICATES,
    protocol_sha,
    realization_seeds,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=SELECT_REPLICATES)
    ap.add_argument("--label", default="adaptation_sweep")
    ap.add_argument("--decoding", default=PRIMARY_DECODING,
                    choices=["deterministic", "stochastic"])
    args = ap.parse_args()

    import tensorflow as tf

    from spec.automotive_training.automotive_primary import (
        build_automotive_primary_stack,
    )

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit("checkpoint not found: %s" % ckpt)
    seeds = realization_seeds(args.replicates)

    trainer, algo = build_automotive_primary_stack(
        seed=0, n_itr=1, ckpt_dir="/tmp/adapt_sweep", decoding=args.decoding)
    evaluator = trainer.held_out_evaluator
    results = {}
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        trainer.policy.core_policy.load_variables(str(ckpt), sess=sess)
        algo.sync_task_policies_from_core()
        for k in ADAPTATION_K_STEPS:
            out = evaluator.evaluate(int(k), replicates=args.replicates,
                                     decoding=args.decoding, sess=sess)
            key = "k%d" % k
            results[key] = {
                "k_steps": int(k),
                "latency_s": (out["k0_latency_s"] if k == 0 else out["k3_latency_s"]),
                "latency_std_s": (out.get("k0_latency_s_std") if k == 0
                                  else out.get("k3_latency_s_std")),
                "hard_rate": (out["k0_hard_rate"] if k == 0 else out["k3_hard_rate"]),
                "mc_violations": out["mc_policy_violation_count"],
                "regret_s": out.get("k0_regret_s" if k == 0 else "k3_regret_s"),
                "per_replicate_latency": [r["k0_latency_s"] if k == 0 else r["k3_latency_s"]
                                          for r in out["per_replicate"]],
                "paired_k3_minus_k0_mean_s": out.get("paired_k3_minus_k0_mean_s"),
            }
            print(json.dumps({"label": args.label, "ckpt": ckpt.name, key: results[key]["latency_s"],
                              "regret_s": results[key]["regret_s"]}, sort_keys=True))
    # paired deltas against k=0 within the same replicate
    base = results["k0"]["per_replicate_latency"]
    summary = {}
    for key, block in results.items():
        if key == "k0":
            continue
        deltas = [a - b for a, b in zip(block["per_replicate_latency"], base)]
        summary[key] = {
            "mean_delta_vs_k0_s": fmean(deltas),
            "better_replicates": sum(1 for d in deltas if d < 0),
            "replicates": len(deltas),
        }
    payload = {
        "label": args.label,
        "checkpoint": str(ckpt),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha": protocol_sha(),
        "decoding": args.decoding,
        "replicates": int(args.replicates),
        "realization_seeds": list(seeds),
        "results": results,
        "paired_summary_vs_k0": summary,
        "note": ("deterministic query decoding; support adaptation is PPO with the frozen "
                 "inner settings; every k sees the same realizations in a replicate"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print("written", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
