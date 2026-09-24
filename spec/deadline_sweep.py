"""CPU-only deadline-regime sweeper and gate report.

    python -m spec.deadline_sweep --sweep --limit 12 \
        --kappas 1.0,1.1,1.25,1.5 --alphas 1.0,0.75,0.5 \
        --out reports/deadline_sweep.json

    python -m spec.deadline_sweep --materialize --regime medium_hard \
        --kappa 1.1 --alpha 0.5 --out-dir spec/deadline_regimes

No training, no GPU. For every (split, kappa, alpha) it builds the regime with the
real witness pipeline and reports the gate numbers per split: witness feasibility,
the pre-guard mask breakdown, WHICH action gets closed, the action mix of the
witnesses, and the depth / sink / criticality split. `infeasible_labelled` is
built with allow_infeasible and is marked non-trainable.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler import deadline_gates as gates  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler.deadline_regime import (  # noqa: E402
    DeadlineRegimeError,
    build_regime_with_witness,
    graph_key,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402
from env.mec_offloaing_envs.scheduler.static_bounds import static_action_bounds  # noqa: E402
from env.mec_offloaing_envs.scheduler.witness import find_fastest_plan  # noqa: E402

REGIME_TABLE = {
    "none": dict(kappa=1.0, alpha=1.0, dtype="none", trainable=True),
    "loose_hard": dict(kappa=1.50, alpha=1.00, dtype="hard", trainable=True),
    "medium_hard": dict(kappa=1.10, alpha=0.50, dtype="hard", trainable=True),
    "tight_hard": dict(kappa=1.02, alpha=0.25, dtype="hard", trainable=True),
    "infeasible_labelled": dict(kappa=0.80, alpha=0.25, dtype="hard", trainable=False),
    "soft_mix": dict(kappa=1.10, alpha=0.50, dtype="soft", trainable=True),
    "firm_mix": dict(kappa=1.10, alpha=0.50, dtype="firm", trainable=True),
}


class PrioritizeCluster:
    """The HEFT-priority surface `OffloadingTaskGraph.prioritize_tasks` needs.

    Mirrors `Resources` in offloading_env.py (same formulas) without importing it,
    so the CPU sweep does not need gym: the sweep is pure data work.
    """

    _MB = 1024.0 * 1024.0 / 8.0

    def __init__(self, resources: ResourceConfig):
        self.mobile_process_capable = float(resources.ue_cpu_bytes_per_second)
        self.mec_process_capable = float(resources.mec_cpu_bytes_per_second)
        self.v2v_process_capable = float(resources.helper_cpu_bytes_per_second)
        self.bandwidth_up = float(resources.mec_uplink_bytes_per_second) / self._MB
        self.bandwidth_dl = float(resources.mec_downlink_bytes_per_second) / self._MB
        self.v2v_bandwidth = float(resources.v2v_bytes_per_second) / self._MB

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * self._MB)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * self._MB)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * self._MB)


def _dataset_root():
    return ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"


def split_distributions():
    from spec.split_loader import (
        meta_test_distribution_ids,
        meta_train_distribution_ids,
        validation_distribution_ids,
    )

    return {
        "meta_train": list(meta_train_distribution_ids()),
        "validation": list(validation_distribution_ids()),
        "meta_test": list(meta_test_distribution_ids()),
    }


def collect_graphs(distribution_id, limit, cluster):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

    prefix = _dataset_root() / ("offload_random20_%d" % int(distribution_id))
    paths = sorted(prefix.glob("*.gv"))
    if limit:
        paths = paths[: int(limit)]
    out = []
    for path in paths:
        graph = OffloadingTaskGraph(str(path))
        graph.prioritize_tasks(cluster)
        out.append(path)
    return out


def _depths(dag, order):
    preds = dag.predecessors()
    depth: dict[int, int] = {}
    for tid in order:
        parents = [int(e.src_task_id) for e in preds[tid]]
        depth[tid] = 0 if not parents else 1 + max(depth[p] for p in parents)
    return depth


def evaluate_regime(regime_name, cfg, splits, resources, cluster, limit, witness_kwargs):
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

    per_split: dict[str, dict] = {}
    started = time.time()
    for split_name, dist_ids in splits.items():
        pairs = []
        for dist in dist_ids:
            for path in collect_graphs(dist, limit, cluster):
                graph = OffloadingTaskGraph(str(path))
                graph.prioritize_tasks(cluster)
                pairs.append((dist, str(path), graph))
        try:
            built, excluded = build_regime_with_witness(
                pairs,
                regime=regime_name,
            kappa=cfg["kappa"],
            alpha=cfg["alpha"],
            deadline_type=cfg["dtype"],
            resources=resources,
            cycles_per_bit=300.0,
                allow_infeasible=not cfg["trainable"],
                witness_kwargs=witness_kwargs,
            )
        except DeadlineRegimeError as exc:
            per_split[split_name] = {
                "graphs": 0, "graphs_total": len(pairs), "witness_rate": 0.0,
                "tokens": 0, "active_rate": 0.0, "forced_rate": 0.0,
                "all_invalid_rate": 0.0, "mec_closed_rate": 0.0,
                "ue_closed_rate": 0.0, "helper_closed_rate": 0.0,
                "graphs_with_mec_closure": 0, "witness_mixed_rate": 0.0,
                "error": str(exc), "gate": None,
            }
            continue

        # per-graph breakdown, recomputed from the stamped deadlines
        breakdowns = {}
        witness_found = {}
        witness_mixed = {}
        for key, entry in built.graphs.items():
            path = None
            graph = None
            for dist, p, g in pairs:
                if graph_key(dist, p) == key:
                    path, graph = p, g
                    break
            if graph is None:
                continue
            dag = to_canonical_dag(graph)
            order = [int(t) for t in graph.prioritize_sequence]
            bounds = static_action_bounds(dag, order, resources, cycles_per_bit=300.0)
            deadlines = [] if not entry.tasks else [entry.tasks[tid].deadline_s for tid in order]
            depth_map = _depths(dag, order)
            depth_by_pos = {i: depth_map[tid] for i, tid in enumerate(order)}
            crit_by_pos = {
                i: (entry.tasks[tid].criticality_class if entry.tasks else "unknown")
                for i, tid in enumerate(order)
            }
            if deadlines:
                breakdowns[key] = gates.breakdown_for_graph(
                    bounds, deadlines, depths=depth_by_pos, criticality=crit_by_pos
                )
            witness = entry.witness or {}
            witness_found[key] = bool(witness.get("found"))
            witness_mixed[key] = len(set(witness.get("actions", []) or [])) > 1

        # a few fully-worked examples: bounds, deadline, witness action, actual
        examples = []
        for key in list(breakdowns)[:2]:
            entry = built.graphs[key]
            witness = entry.witness or {}
            for dist, p, g in pairs:
                if graph_key(dist, p) != key:
                    continue
                dag = to_canonical_dag(g)
                order = [int(t) for t in g.prioritize_sequence]
                bounds = static_action_bounds(dag, order, resources, cycles_per_bit=300.0)
                actions = list(witness.get("actions") or [])
                table = witness.get("per_task") or {}
                rows = []
                for i, tid in enumerate(order):
                    row = table.get(str(tid), {})
                    rows.append({
                        "task_id": tid,
                        "ready_lb_mec": round(bounds.ready_lb[i][1], 6),
                        "ready_lb_ue": round(bounds.ready_lb[i][0], 6),
                        "ready_lb_helper": round(bounds.ready_lb[i][2], 6),
                        "deadline_s": round(entry.tasks[tid].deadline_s, 6),
                        "witness_action": actions[i] if i < len(actions) else None,
                        "actual_ready_s": row.get("all_consumers_ready_s"),
                        "slack_s": row.get("slack_s"),
                    })
                examples.append({
                    "graph": key,
                    "task_count": len(order),
                    "witness_makespan_s": witness.get("makespan_s"),
                    "witness_method": witness.get("method"),
                    "tasks": rows,
                })
                break
        summary = gates.gate_summary(breakdowns, witness_found, witness_mixed, excluded)
        summary["gate"] = gates.passes_trainable_gate(summary) if cfg["trainable"] else None
        summary["examples"] = examples
        per_split[split_name] = summary

    return {
        "regime": regime_name,
        "config": {k: v for k, v in cfg.items()},
        "limit_per_distribution": limit,
        "elapsed_s": round(time.time() - started, 2),
        "splits": per_split,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--regime", default="")
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--kappas", default="1.0,1.1,1.25,1.5")
    parser.add_argument("--alphas", default="1.0,0.75,0.5")
    parser.add_argument("--regimes", default="")
    parser.add_argument("--splits", default="meta_train,validation,meta_test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", default="reports/deadline_sweep.json")
    parser.add_argument("--out-dir", default="spec/deadline_regimes")
    parser.add_argument("--model", default="legacy")
    args = parser.parse_args(argv)

    from spec.mask_sanity import configure_env

    configure_env("off")  # obs v3, no mask, no constraints: pure data work
    resources = ResourceConfig.from_frozen_yaml(Path("spec/frozen_experiment.yaml"), model=args.model)
    cluster = PrioritizeCluster(resources)
    splits_all = split_distributions()
    wanted = [s for s in args.splits.split(",") if s]
    splits = {k: v for k, v in splits_all.items() if k in wanted}
    witness_kwargs = dict(max_passes=4, max_pair_rounds=2, max_pairs=40)

    if args.materialize:
        if args.regime not in REGIME_TABLE:
            raise SystemExit("unknown regime %r" % args.regime)
        cfg = dict(REGIME_TABLE[args.regime])
        if args.kappa is not None:
            cfg["kappa"] = args.kappa
        if args.alpha is not None:
            cfg["alpha"] = args.alpha
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

        written = []
        for split_name, dist_ids in splits_all.items():
            if split_name not in ("meta_train", "validation", "meta_test"):
                continue
            pairs = []
            for dist in dist_ids:
                for path in collect_graphs(dist, 0, cluster):   # full split
                    graph = OffloadingTaskGraph(str(path))
                    graph.prioritize_tasks(cluster)
                    pairs.append((dist, str(path), graph))
            built, excluded = build_regime_with_witness(
                pairs, regime=args.regime, kappa=cfg["kappa"], alpha=cfg["alpha"],
                deadline_type=cfg["dtype"], resources=resources,
                cycles_per_bit=300.0, allow_infeasible=not cfg["trainable"],
                witness_kwargs=witness_kwargs,
            )
            target = Path(args.out_dir) / ("%s_%s.json" % (args.regime, split_name))
            built.to_json(target)
            written.append({
                "split": split_name,
                "path": str(target),
                "graphs": len(built.graphs),
                "excluded": len(excluded),
            })
            print("wrote %s (%d graphs, %d excluded)" % (target, len(built.graphs), len(excluded)))
        print(json.dumps({"materialized": written}, indent=2))
        return 0

    if not args.sweep:
        parser.error("pass --sweep or --materialize")

    report = {"model": args.model, "limit_per_distribution": args.limit, "regimes": []}
    regime_names = [r for r in args.regimes.split(",") if r] or ["loose_hard", "medium_hard", "tight_hard"]
    for name in regime_names:
        if name == "none":
            continue
        for kappa in [float(x) for x in args.kappas.split(",") if x]:
            for alpha in [float(x) for x in args.alphas.split(",") if x]:
                cfg = dict(REGIME_TABLE.get(name, dict(dtype="hard", trainable=True)))
                cfg.update(kappa=kappa, alpha=alpha)
                entry = evaluate_regime(name, cfg, splits, resources, cluster, args.limit, witness_kwargs)
                report["regimes"].append(entry)
                mt = entry["splits"]["meta_train"]
                print("%-14s k=%.2f a=%.2f | witness=%.3f active=%.3f forced=%.3f allinv=%.4f "
                      "mec_closed=%.4f ue_closed=%.3f helper_closed=%.3f mixed=%.3f pass=%s" % (
                          name, kappa, alpha, mt["witness_rate"], mt["active_rate"],
                          mt["forced_rate"], mt["all_invalid_rate"], mt["mec_closed_rate"],
                          mt["ue_closed_rate"], mt["helper_closed_rate"],
                          mt["witness_mixed_rate"], (mt.get("gate") or {}).get("passes")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print("sweep written to %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
