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
CAMPAIGN_YAML = SPEC / "phase4_campaign.yaml"
RUNS_ROOT = ROOT / "runs" / "phase4"

METHOD_ID = "margo_v0.1_primary"
PARENT_FREEZE = "phase3-freeze-v0.1"
PARENT_SHA = "0c776924b49da6c66c511c12a8cde70be732e25d"
SEEDS = (0, 1, 2, 3, 4)
K_REPORT = (0, 3)
OUTER_ITERS = 3500
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
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def git_dirty(root=ROOT):
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root),
        text=True,
        capture_output=True,
    )
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(GPU_FLAG, dest="allow_gpu", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.execute_train and args.gpu_smoke:
        raise ValueError("do not pass --execute-train and --gpu-smoke together")
    if args.execute_train or args.gpu_smoke:
        require_gpu_permission(args.allow_gpu)
        if args.seed is None:
            raise ValueError("Phase 4 GPU job needs one --seed from %s" % (list(SEEDS),))
        from spec.phase4_train_driver import run_gpu_smoke, run_primary_seed

        if args.gpu_smoke:
            run_gpu_smoke(seed=args.seed, allow_gpu=True)
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
