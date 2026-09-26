#!/usr/bin/env python3
"""P1 read-only checkpoint evaluation on the frozen held-out split.

Loads a final checkpoint into the fresh policy and runs the SAME held-out
protocol (k=0 and k=3) with the SAME scheduler config, seed and decoder order.
It never calls `trainer.train()`, never touches the optimizer state of the
source checkpoint and never overwrites a checkpoint. The k=3 adaptation runs the
standard inner PPO update on the held-out evaluator's SCRATCH policy (that is the
protocol's definition); the loaded source weights are only read.

Usage (on Kish):
  python -m spec.evaluate_checkpoint \
      --checkpoint true_init --checkpoint final_itr24=/path/ckpt.ckpt \
      --json reports/v0.3-audit/pilot/extension_40/checkpoint_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REQUIRED_LATENCIES = ("query_mean_latency", "query_all_mec_latency", "query_greedy_latency")


def configure_obs_env() -> dict:
    """Encoder/policy modules read these at import; must be set first.

    The pilot contract is obs v3 + mask off + constraints off; without v3 the
    encoder raises `resource_vec only valid for obs v2/v3`.
    """
    os.environ["MARGO_OBS_VERSION"] = "v3"
    os.environ["MARGO_MASK_MODE"] = "off"
    os.environ["MARGO_CONSTRAINTS"] = "off"
    return {
        "MARGO_OBS_VERSION": os.environ["MARGO_OBS_VERSION"],
        "MARGO_MASK_MODE": os.environ["MARGO_MASK_MODE"],
        "MARGO_CONSTRAINTS": os.environ["MARGO_CONSTRAINTS"],
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(k0: dict, k3: dict) -> dict:
    """Pure extraction of the reported fields from the two held-out evaluations."""
    out = {"k0": {}, "k3": {}, "gaps": {}}
    for name, data in (("k0", k0), ("k3", k3)):
        for key in data:
            if key in ("per_distribution", "validation_per_graph_plans",
                       "validation_plan_identities"):
                continue
            value = data[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name][key] = float(value)
    p0 = k0.get("query_mean_latency")
    p3 = k3.get("query_mean_latency")
    for label, k in (("k0", k0), ("k3", k3)):
        mec = k.get("query_all_mec_latency")
        gr = k.get("query_greedy_latency")
        p = k.get("query_mean_latency")
        if p is not None and mec is not None:
            out["gaps"]["%s_gap_to_all_mec" % label] = float(p) - float(mec)
        if p is not None and gr is not None:
            out["gaps"]["%s_gap_to_greedy" % label] = float(p) - float(gr)
    out["k3_better_than_k0"] = (p3 is not None and p0 is not None and float(p3) < float(p0))
    return out


def _build_stack():
    from env.mec_offloaing_envs.scheduler.primary_config import (
        resolved_primary_scheduler_config,
    )
    from meta_trainer import build_frozen_primary_stack

    return build_frozen_primary_stack(
        seed=0,
        n_itr=1,
        ckpt_dir="/tmp/checkpoint_eval_ckpt",
        audit_writer=None,
        print_action_choices=False,
        parallel=False,
        reward_mode="latency_only",
        learning_mode="publication",
        vocab_size=3,
        use_energy=True,
        constraints=None,
        constraint_dual_lr=None,
        objective_mode="log_only",
        objective_spec={"latency_ref": "all_ue", "energy_budget_j": 1e12},
        scheduler_config=resolved_primary_scheduler_config(),
        strict_scheduler_config=True,
    )


def _snapshot(policy, sess):
    import numpy as np

    values = sess.run(policy.get_trainable_variables())
    return [np.asarray(v, dtype=np.float64) for v in values]


def _changed(before, after) -> bool:
    import numpy as np

    return any(
        not np.allclose(a, b, rtol=0.0, atol=0.0) for a, b in zip(before, after)
    )


def main(argv=None) -> int:
    import numpy as np
    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="LABEL=PATH, or the bare word true_init",
    )
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--one", default=None,
        help="evaluate exactly one LABEL=PATH (or true_init) in this process; "
             "each label should run as its own process for full graph isolation",
    )
    parser.add_argument("--fingerprint", default="")
    args = parser.parse_args(argv)
    if args.one:
        args.checkpoint = [args.one]
    results = {"obs_env": configure_obs_env(), "checkpoints": {}}

    trainer, algo = _build_stack()
    held = trainer.held_out_evaluator
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        algo.sync_task_policies_from_core()
        core = trainer.policy.core_policy
        # Determinism probe BEFORE any evaluation/adaptation: two fresh k0 runs in
        # one session with no state advanced between them. The probe after the
        # label loop is recorded separately as post-adaptation because k3
        # adaptation legitimately moves the scratch policy and the RNG.
        fresh_a = held.evaluate_all(k_steps=0, sess=sess)["query_mean_latency"]
        fresh_b = held.evaluate_all(k_steps=0, sess=sess)["query_mean_latency"]
        results["deterministic_k0_fresh"] = bool(
            np.isclose(fresh_a, fresh_b, rtol=0.0, atol=1e-9)
        )
        results["deterministic_k0_fresh_values"] = [float(fresh_a), float(fresh_b)]
        for spec in args.checkpoint:
            if spec == "true_init":
                label, path = "true_init", None
                changed = None
            elif "=" in spec:
                label, path = spec.split("=", 1)
            else:
                raise SystemExit("use LABEL=PATH or true_init, got %r" % spec)
            before = _snapshot(core, sess) if path else None
            if path:
                core.load_variables(path, sess)
            after = _snapshot(core, sess) if path else None
            k0 = held.evaluate_all(k_steps=0, sess=sess)
            k3 = held.evaluate_all(k_steps=3, sess=sess)
            entry = summarize(k0, k3)
            entry["weights_changed"] = None if path is None else bool(_changed(before, after))
            entry["checkpoint_path"] = path
            entry["checkpoint_sha256"] = None if path is None else sha256_file(path)
            entry["scheduler_config_sha256"] = resolved_config_sha256(
                trainer.env.scheduler_resources
            )
            results["checkpoints"][label] = entry
        # post-adaptation probe (expected to differ: k3 moved the scratch policy)
        a = held.evaluate_all(k_steps=0, sess=sess)["query_mean_latency"]
        b = held.evaluate_all(k_steps=0, sess=sess)["query_mean_latency"]
        results["deterministic_k0_post_adaptation"] = bool(
            np.isclose(a, b, rtol=0.0, atol=1e-9)
        )
        results["deterministic_k0_post_adaptation_values"] = [float(a), float(b)]
        results["fingerprint_arg"] = args.fingerprint
        results["schema"] = "checkpoint_eval_v1"
        results["training_code_sha"] = os.environ.get("MARGO_TRAINING_CODE_SHA", "")
        results["evaluation_code_sha"] = os.environ.get("MARGO_EVAL_CODE_SHA", "")
        results["obs_env"] = configure_obs_env()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results["checkpoints"], indent=1, sort_keys=True)[:2500])
    print("deterministic_k0_fresh:", results["deterministic_k0_fresh"])
    print("deterministic_k0_post_adaptation:", results["deterministic_k0_post_adaptation"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
