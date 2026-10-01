#!/usr/bin/env python3
"""Entry point for the ONE real-GPU automotive primary smoke (Gate 20 Stage B).

    python spec/automotive_gpu_smoke.py --seed 0 --i-allow-gpu --gpu 0 --iters 1

It is the real primary chain:
    phase4_train_driver.run_automotive_gpu_smoke -> _train ->
    meta_trainer.build_frozen_primary_stack(dataset="automotive_mc_v1") ->
    Trainer.train()
with production structural budgets (meta_batch 10, 20 support trajectories of 20
tokens, 3 inner PPO steps) and exactly `--iters` outer iterations. It never touches
meta_test. NOT the long run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i-allow-gpu", dest="allow_gpu", action="store_true")
    ap.add_argument("--iters", type=int, default=None,
                    help="outer iterations (smoke default 1, --long default 3500)")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--cpu", action="store_true", help="CPU-only plumbing smoke")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--run-kind", default=None,
                    help="output root under runs/automotive_mc_v1/ (default: primary)")
    ap.add_argument("--long", action="store_true",
                    help="launch the frozen 3500-iteration automotive primary run (NOT executed here)")
    ap.add_argument("--plumbing", action="store_true",
                    help="Stage A label (canonical budgets; MRLCO forbids shrinking them)")
    args = ap.parse_args()

    os.environ.setdefault("MARGO_OBS_VERSION", "automotive_mc_obs_v1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from spec.phase4_train_driver import (
        automotive_run_dir,
        run_automotive_gpu_smoke,
        run_automotive_primary_seed,
    )

    started = time.time()
    if args.long:
        if not args.allow_gpu:
            raise SystemExit("--long requires --i-allow-gpu")
        n_itr = 3500 if args.iters is None else int(args.iters)
        run_kind = args.run_kind or "primary"
        run_dir = run_automotive_primary_seed(args.seed, True, n_itr=n_itr,
                                              dataset_dir=args.dataset_dir,
                                              run_kind=run_kind)
        print(json.dumps({"long_run_dir": str(run_dir), "seed": int(args.seed),
                          "outer_iterations": n_itr, "run_kind": run_kind},
                         indent=2, sort_keys=True))
        return 0
    smoke_iters = 1 if args.iters is None else int(args.iters)
    run_dir = run_automotive_gpu_smoke(args.seed, args.allow_gpu, n_itr=smoke_iters,
                                      dataset_dir=args.dataset_dir)
    wall = time.time() - started
    summary = {
        "run_dir": str(run_dir),
        "wall_seconds": wall,
        "outer_iterations": int(smoke_iters),
        "seed": int(args.seed),
        "cpu_only": bool(args.cpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "obs_version": os.environ.get("MARGO_OBS_VERSION"),
        "plumbing": bool(args.plumbing),
    }
    report_path = run_dir / "automotive_gpu_smoke_entry.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
