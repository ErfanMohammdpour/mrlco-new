#!/usr/bin/env python3
"""Constrained-MDP smoke test on real graphs (CPU, numpy only).

Exercises exactly the machinery the trainer uses — `telescoping_token_rewards`
with a `ConstraintSpec` + `ConstraintController` — but with a *synthetic* policy
family so it runs on the laptop (no TF):

    plan_mec      = all-MEC plan                      (scalar-objective optimum)
    plan_balanced = deterministic 3-tier spread plan (MEC 50% / local 35% / V2V 15%)
    pi_p          = with probability p follow plan_mec, else plan_balanced

Why these two: measured on real graphs the all-MEC plan is the argmin of BOTH
latency and mobile energy (MEC compute is out of scope, ADR-001) and it takes
100% of the MEC slot — so it violates the MEC-share budget while the balanced
plan is feasible.  Dual ascent therefore has to lift lambda_mec until the proxy
policy stops taking the all-MEC shortcut, which is exactly the mechanism the
real constrained run relies on.

Usage:
    python spec/constraint_dual_smoke.py --graphs 6 --dist 1 --iters 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ConstraintController,
    ResourceConfig,
    compute_reference_ranges,
    compute_scoped_reference_ranges,
    pure_location_plan,
    telescoping_token_rewards,
)
from env.mec_offloaing_envs.scheduler.energy_scope import SCOPE_SYSTEM  # noqa: E402

DEFAULT_DATA = ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0
P_GRID = [round(0.1 * i, 1) for i in range(0, 11)]
# 3-tier spread: first 35% of the HEFT order local, next 50% MEC, last 15% V2V.
BALANCED_LOCAL = 0.35
BALANCED_MEC = 0.50


class _FrozenCluster:
    def __init__(self):
        self.mobile_process_capable = 1.0 * 1024 * 1024
        self.mec_process_capable = 10.0 * 1024 * 1024
        self.v2v_process_capable = 1.0 * 1024 * 1024
        self.bandwidth_up = 7.0
        self.bandwidth_dl = 7.0
        self.v2v_bandwidth = 5.0

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * MBPS_TO_BPS)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * MBPS_TO_BPS)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * MBPS_TO_BPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dual-lr", type=float, default=None)
    ap.add_argument("--config", type=str, default=str(ROOT / "spec" / "constraints.yaml"))
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    from spec.constraints_config import load_constraints

    spec, yaml_lr = load_constraints(args.config)
    if spec is None:
        print("constraints.yaml mode != lagrangian; nothing to smoke-test")
        return 2
    ctrl = ConstraintController(spec=spec, dual_lr=args.dual_lr or yaml_lr)

    folder = DEFAULT_DATA / f"offload_random20_{args.dist}"
    files = sorted(folder.glob("random.20.*.gv"))[: args.graphs]
    if not files:
        print(f"no graphs in {folder}", file=sys.stderr)
        return 2

    resources = ResourceConfig.from_frozen_yaml()
    cluster = _FrozenCluster()
    graphs = []
    for path in files:
        tg = OffloadingTaskGraph(str(path))
        tg.prioritize_tasks(cluster)
        refs = compute_scoped_reference_ranges(tg, resources, energy_scope=SCOPE_SYSTEM)
        order = [int(t) for t in tg.prioritize_sequence]
        n = len(order)
        n_local = int(round(BALANCED_LOCAL * n))
        n_mec = int(round(BALANCED_MEC * n))
        balanced = [
            (tid, 0 if k < n_local else (1 if k < n_local + n_mec else 2))
            for k, tid in enumerate(order)
        ]
        graphs.append({
            "name": path.name,
            "tg": tg,
            "refs": refs,
            "mec": pure_location_plan(order, 1),
            "balanced": balanced,
        })

    def rollout(p: float, duals, observe: bool):
        """Deterministic batch: probability p of the all-MEC plan."""
        rng = random.Random(0)
        rows, costs = [], []
        for g in graphs:
            plan = g["mec"] if rng.random() < p else g["balanced"]
            out = telescoping_token_rewards(
                g["tg"], plan, resources, constraints=spec, duals=duals
            )
            rows.append({
                "return": sum(out.rewards),
                "T": out.final_makespan,
                "E": out.final_energy,
                "violation": out.constraint_costs.total_violation,
                "costs": out.constraint_costs,
            })
            costs.append(out.constraint_costs)
        if observe:
            for c in costs:
                ctrl.observe(c)
        return {
            "p_mec": p,
            "return": statistics.mean(r["return"] for r in rows),
            "T": statistics.mean(r["T"] for r in rows),
            "E": statistics.mean(r["E"] for r in rows),
            "violation": statistics.mean(r["violation"] for r in rows),
            "mec_share": statistics.mean(
                r["costs"].as_dict().get("mec_task_fraction_raw", 0.0) for r in rows
            ),
        }

    print("=" * 78)
    print(
        f"CONSTRAINED-MDP SMOKE — dist offload_random20_{args.dist}, n={len(graphs)} graphs"
    )
    print(f"constraints: {list(spec.active_names)}   dual_lr={ctrl.dual_lr}")
    print("=" * 78)

    zero = [0.0] * len(spec.active_names)
    base = rollout(1.0, zero, observe=False)
    print(
        f"UNCONSTRAINED optimum (all-MEC): T={base['T']:.2f}s E={base['E']:.2f}J "
        f"mec_share={base['mec_share']:.2f}  violation={base['violation']:.4f}"
    )
    feas = rollout(0.0, zero, observe=False)
    print(
        f"feasible 3-tier plan:           T={feas['T']:.2f}s E={feas['E']:.2f}J "
        f"mec_share={feas['mec_share']:.2f}  violation={feas['violation']:.4f}"
    )
    print()

    history = []
    for it in range(args.iters):
        table = [rollout(p, ctrl.lambdas, observe=False) for p in P_GRID]
        best = max(table, key=lambda row: row["return"])
        # observe ONLY the trajectories the current policy actually takes
        chosen = rollout(best["p_mec"], ctrl.lambdas, observe=True)
        diag = ctrl.dual_step()
        lambdas = {
            k.split("/")[-1]: v for k, v in diag.items() if k.startswith("constraint/lambda_")
        }
        history.append({**chosen, "iter": it, "lambdas": lambdas})
        if it % 4 == 0 or it == args.iters - 1:
            print(
                f"it {it:3d}  p(all-MEC)={best['p_mec']:.1f}  return={best['return']:+.3f}  "
                f"T={best['T']:7.2f}s  E={best['E']:7.2f}J  "
                f"mec_share={chosen['mec_share']:.2f}  violation={chosen['violation']:.4f}"
            )
            print(f"        lambdas={ {k: round(v, 4) for k, v in lambdas.items()} }")

    tolerated = [h for h in history if h["violation"] <= 1e-6]
    print()
    if tolerated:
        first = tolerated[0]
        print(
            f"first feasible iteration: {first['iter']}  p(all-MEC)={first['p_mec']:.1f}  "
            f"T={first['T']:.2f}s  E={first['E']:.2f}J  "
            f"(unconstrained optimum was T={base['T']:.2f}s / E={base['E']:.2f}J → "
            f"{first['T'] - base['T']:+.2f}s, {first['E'] - base['E']:+.2f}J for feasibility)"
        )
    else:
        print("no iteration became feasible — raise dual_lr or relax a budget")

    if args.json:
        Path(args.json).write_text(json.dumps(history, indent=2, default=float))
        print(f"history -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
