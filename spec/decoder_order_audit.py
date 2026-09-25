#!/usr/bin/env python3
"""Part D — decoder-order / ranking audit (diagnostic only).

`prioritize_tasks` and every other production module are untouched. This file
only *reads* the production rank, the production scheduler and the frozen
graphs, then compares five decoder orders on the SAME frozen graphs.

Methodology (the user's contract, stated once and applied everywhere)
--------------------------------------------------------------------
1. **Order is not assignment.** The fixed-plan experiment schedules EVERY order
   with ONE frozen per-task action mapping (`FIXED_ACTION[tid] = tid % 3`), so
   only insertion order / queue interaction can explain a makespan delta. The
   greedy experiment then searches the assignment separately, per order.
2. **Nothing is tuned.** The audited orders are pure functions of the graph and
   the resolved scheduler config. No order is used to define another.
3. **Static bound is queue-blind.** `static_bound_s` is the longest path over
   compute + contention-free transfer durations (no calendar, no queueing). It
   is an admissible lower bound on the real makespan and is order-invariant by
   construction; only the makespan moves, so the reported error is a queue cost.

Orders
------
1. `legacy_current`   — the CURRENT production rank, i.e. the value returned by
   `OffloadingTaskGraph.prioritize_tasks(resource_cluster)` with the legacy cost
   cluster reconstructed from the resolved `ResourceConfig`. Formula:
       w_i = min(t_local, t_mec, t_v2v)          (per-task best single action,
                                                  cost model of the legacy env)
       rank_i = w_i + max_{j in succ(i)} rank_j
       order = descending rank  (the production `np.argsort(rank)[::-1]`)
2. `stable_topo`      — deterministic topological order, Kahn with smallest
   task id first. No cost model at all; the "do nothing clever" baseline.
3. `heft_upward`      — standard HEFT upward rank with MEAN costs:
       w_i  = mean over {UE, MEC, HELPER} of compute_workload / rate_l
       c_ij = mean over the 3x3 location pairs of the edge transfer duration
       rank_u(i) = w_i + max_{j in succ(i)} (c_ij + rank_u(j))
       sinks  add the mean UE-return transfer of their output (the terminal
       result must reach UE, so it is on the exit path).
   Kahn with `rank_u` (descending) keeps the sequence topological.
4. `canonical_aware`  — canonical scheduler-aware rank. A canonical probe run
   (`schedule` with `stable_topo` and the fixed assignment) is executed once per
   graph; the rank is the probe's observed task completion time, so it is
   derived from canonical durations, hop rates and the real resource calendars.
   Kahn with descending observed finish favours late-completing (critical-path)
   tasks first.
5. `deadline_criticality` — diagnostic-only urgency rank. Kahn priority tuple,
   highest first:
       (has_deadline, -slack, deadline_type_rank, criticality_rank, upward_rank)
   with `slack = deadline_s - ef_lb` and `ef_lb` the contention-free earliest
   finish under the min-duration action. On deadline-free graphs it degenerates
   to `(deadline_type, criticality, upward_rank)`.

Graphs
------
The repo's frozen `daggen` graphs under
`env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/` load
through `OffloadingTaskGraph` + `to_canonical_dag` WITHOUT TensorFlow, so they
are the primary set. Five deterministic synthetic `CanonicalDAG`s (sparse,
medium, dense, narrow-deep, wide-shallow, several carrying deadlines) are added
so the density/depth buckets are populated and the deadline order is exercised.

Usage: python3 -m spec.decoder_order_audit [--json PATH] [--graphs a,b,c]
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    Location,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.routes import route  # noqa: E402

SCHEMA = "rank_audit_v1"
FROZEN_GRAPH_DIR = (
    ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20" / "offload_random20_1"
)
FROZEN_GRAPH_FILES = (
    "random.20.0.gv",
    "random.20.1.gv",
    "random.20.2.gv",
    "random.20.3.gv",
    "random.20.4.gv",
)
LOCATIONS = (Location.UE, Location.MEC, Location.HELPER)
ACTION_LOCATIONS = {0: Location.UE, 1: Location.MEC, 2: Location.HELPER}
CPU_RESOURCE = {
    Location.UE: "UE_CPU",
    Location.MEC: "MEC_CPU",
    Location.HELPER: "HELPER_CPU",
}
DEADLINE_TYPE_RANK = {"none": 0, "soft": 1, "firm": 2, "hard": 3}
CRITICALITY_RANK = {"low": 0, "medium": 1, "high": 2}
EPS = 1e-9

ORDER_NAMES = (
    "legacy_current",
    "stable_topo",
    "heft_upward",
    "canonical_aware",
    "deadline_criticality",
)

ORDER_DEFINITIONS = (
    {
        "name": "legacy_current",
        "label": "current production legacy cost-model rank",
        "kind": "production",
        "formula": "w_i=min(t_local,t_mec,t_v2v); rank_i=w_i+max_succ rank; descending",
    },
    {
        "name": "stable_topo",
        "label": "deterministic topological order (smallest task id first)",
        "kind": "structural",
        "formula": "Kahn ready-queue ordered by ascending task id",
    },
    {
        "name": "heft_upward",
        "label": "standard HEFT upward rank (mean compute + mean communication to exit)",
        "kind": "heuristic",
        "formula": "rank_u(i)=mean_l(comp_i,l)+max_succ(mean_xfer_ij+rank_u(j)); sinks add mean UE return",
    },
    {
        "name": "canonical_aware",
        "label": "canonical scheduler-aware rank (observed probe completion time)",
        "kind": "scheduler_aware",
        "formula": "probe schedule(dag, stable_topo, fixed_actions); Kahn priority = descending observed finish",
    },
    {
        "name": "deadline_criticality",
        "label": "deadline / criticality urgency rank (diagnostic only)",
        "kind": "diagnostic",
        "formula": "Kahn priority=(has_deadline,-slack,deadline_type,criticality,upward_rank)",
    },
)


def fixed_action(task_id: int) -> int:
    """The ONE frozen per-task assignment shared by every order.

    Round-robin over UE/MEC/HELPER so all six calendars carry load. It is a
    function of the task id only: it cannot move with the decoder order.
    """
    return int(task_id) % 3


def fixed_actions_for(dag: CanonicalDAG) -> dict[int, int]:
    return {int(tid): fixed_action(tid) for tid in dag.tasks}


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GraphEntry:
    name: str
    source: str  # "frozen" | "synthetic"
    dag: CanonicalDAG
    path: str | None = None


class _LCG:
    """Tiny deterministic generator: no `random` module, no version drift."""

    def __init__(self, seed: int) -> None:
        self._x = int(seed) & 0x7FFFFFFF

    def __iter__(self):
        return self

    def __next__(self) -> int:
        self._x = (1103515245 * self._x + 12345) & 0x7FFFFFFF
        return self._x


def _shuffle(n: int, seed: int) -> list[int]:
    """Deterministic relabeling perm: `mapping[natural_id] = task_id`."""
    rng = _LCG(seed)
    mapping = list(range(n))
    for i in range(n - 1, 0, -1):
        j = next(rng) % (i + 1)
        mapping[i], mapping[j] = mapping[j], mapping[i]
    return mapping


def _syn_tasks(n: int, mapping: Sequence[int]) -> list[CanonicalTask]:
    """Non-monotone sizes on a shuffled id space.

    The relabeling means the id order is deliberately NOT the natural processing
    order, so `stable_topo` (smallest id first) is a genuinely different order
    from every cost-model rank instead of collapsing onto it.
    """
    rng = _LCG(20240607)
    tasks = []
    for natural in range(n):
        r1 = next(rng)
        r2 = next(rng)
        tasks.append(
            CanonicalTask(
                task_id=int(mapping[natural]),
                compute_workload_bytes=250_000 + 37_000 * (r1 % 40),
                task_output_bytes=120_000 + 11_000 * (r2 % 25),
                external_input_bytes=(150_000 if natural == 0 else 0),
            )
        )
    return tasks


def _stamp_deadlines(dag: CanonicalDAG, resources: ResourceConfig) -> CanonicalDAG:
    """Relative mixed-criticality deadlines, tight enough to carry signal.

    `deadline_s = factor * ef_lb(task)` with the contention-free earliest finish
    under the min-duration action, so factor < 1 is already infeasible and the
    urgency rank has real content. Non-monotone factors keep the diagnostic order
    from collapsing onto the id order.
    """
    from dataclasses import replace

    ef = _min_contention_free_finish(dag, resources)
    rng = _LCG(987654321)
    factors = (1.20, 0.70, 1.80, 0.90, 2.50)
    dtypes = ("hard", "firm", "soft", "none", "soft")
    crits = ("high", "medium", "low", "medium", "high")
    tasks = []
    for tid in sorted(dag.tasks):
        task = dag.tasks[tid]
        k = next(rng) % 5
        dtype = dtypes[k]
        deadline = float(factors[k] * ef[tid]) if dtype != "none" else None
        tasks.append(
            replace(
                task,
                deadline_s=deadline,
                deadline_type=dtype,
                criticality_class=crits[k],
                tardiness_weight=1.0,
            )
        )
    edges = [(e.src_task_id, e.dst_task_id, e.edge_output_bytes) for e in dag.edges]
    return CanonicalDAG.from_records(tasks, edges)


def synthetic_graphs(resources: ResourceConfig) -> "OrderedDict[str, GraphEntry]":
    """Deterministic CanonicalDAGs covering sparse/medium/dense/deep/shallow.

    Each is built in a natural order, then relabeled by a fixed permutation so
    that the id order is not the processing order. Every graph admits more than
    one topological order (no pure chains), so the five orders are genuinely
    comparable.
    """
    out: "OrderedDict[str, GraphEntry]" = OrderedDict()

    def build(name: str, n: int, natural_edges: list[tuple[int, int, int]], seed: int) -> None:
        mapping = _shuffle(n, seed)
        edges = [(mapping[s], mapping[d], w) for s, d, w in natural_edges]
        dag = CanonicalDAG.from_records(_syn_tasks(n, mapping), edges)
        out[name] = GraphEntry(name, "synthetic", _stamp_deadlines(dag, resources))

    # sparse: backbone with one leaf per backbone node (frontier width 2)
    n = 14
    backbone = list(range(7))
    edges = [(backbone[i], backbone[i + 1], 70_000 + 1_000 * i) for i in range(6)]
    edges += [(backbone[i], 7 + i, 70_000 + 1_000 * i) for i in range(7)]
    build("syn_sparse", n, edges, seed=11)

    # medium: 4 partial-bipartite layers of width 3
    n = 12
    rng = _LCG(4242)
    edges = []
    for layer in range(3):
        src = [3 * layer + k for k in range(3)]
        dst = [3 * (layer + 1) + k for k in range(3)]
        for s in src:
            for d in dst:
                if next(rng) % 3 != 0:
                    edges.append((s, d, 80_000 + 2_000 * s))
    build("syn_medium", n, edges, seed=23)

    # dense: full bipartite layers (width 4) plus cross-layer shortcuts
    n = 12
    L = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    edges = []
    for layer in range(2):
        for s in L[layer]:
            for d in L[layer + 1]:
                edges.append((s, d, 60_000))
    for s in L[0]:
        for d in L[2]:
            edges.append((s, d, 60_000))
    build("syn_dense", n, edges, seed=37)

    # narrow deep: two parallel chains merging at a single sink, depth 8
    n = 16
    chain_a = list(range(0, 8))
    chain_b = list(range(8, 15))
    edges = [(chain_a[k], chain_a[k + 1], 90_000) for k in range(len(chain_a) - 1)]
    edges += [(chain_b[k], chain_b[k + 1], 90_000) for k in range(len(chain_b) - 1)]
    edges += [(7, 15, 90_000), (14, 15, 90_000)]
    build("syn_narrow_deep", n, edges, seed=53)

    # wide shallow: one root fans out to 10 parallel tasks feeding one sink
    n = 12
    edges = [(0, j, 75_000) for j in range(1, n - 1)]
    edges += [(j, n - 1, 75_000) for j in range(1, n - 1)]
    build("syn_wide_shallow", n, edges, seed=71)
    return out


def frozen_graphs() -> "OrderedDict[str, GraphEntry]":
    """Load the repo's frozen daggen graphs. Empty when unavailable."""
    out: "OrderedDict[str, GraphEntry]" = OrderedDict()
    if not FROZEN_GRAPH_DIR.is_dir():
        return out
    try:
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
        from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag
    except Exception:  # pragma: no cover - optional parser dependency
        return out
    for fname in FROZEN_GRAPH_FILES:
        path = FROZEN_GRAPH_DIR / fname
        if not path.is_file():
            continue
        try:
            dag = to_canonical_dag(OffloadingTaskGraph(str(path)))
        except Exception:  # pragma: no cover - optional parser dependency
            continue
        name = "frozen_" + fname.replace("random.", "").replace(".gv", "")
        out[name] = GraphEntry(name, "frozen", dag, path=str(path))
    return out


def build_graphs(resources: ResourceConfig | None = None) -> "OrderedDict[str, GraphEntry]":
    resources = resources or resolved_primary_scheduler_config()
    graphs: "OrderedDict[str, GraphEntry]" = OrderedDict()
    for name, entry in frozen_graphs().items():
        graphs[name] = entry
    for name, entry in synthetic_graphs(resources).items():
        graphs[name] = entry
    return graphs


# --------------------------------------------------------------------------- #
# structural helpers
# --------------------------------------------------------------------------- #
def topological_sequence(dag: CanonicalDAG) -> list[int]:
    preds = dag.predecessors()
    succs = dag.successors()
    indeg = {tid: len(preds[tid]) for tid in dag.tasks}
    heap = [tid for tid, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    out: list[int] = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    if len(out) != len(dag.tasks):
        raise ValueError("graph is not a DAG")
    return out


def kahn_by_score(dag: CanonicalDAG, score: Mapping[int, Sequence[float]]) -> list[int]:
    """Kahn topological order; among ready tasks take the largest score tuple.

    Score tuples are compared element-wise (Python tuple order); the task id is
    appended as the final tie-break so the result is fully deterministic.
    """
    preds = dag.predecessors()
    succs = dag.successors()
    indeg = {tid: len(preds[tid]) for tid in dag.tasks}
    heap: list[tuple[tuple[float, ...], int]] = []
    for tid, d in indeg.items():
        if d == 0:
            key = tuple(-float(x) for x in score[tid]) + (float(tid),)
            heapq.heappush(heap, (key, int(tid)))
    out: list[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                key = tuple(-float(x) for x in score[v]) + (float(v),)
                heapq.heappush(heap, (key, int(v)))
    if len(out) != len(dag.tasks):
        raise ValueError("cyclic graph")
    return out


def is_topological(dag: CanonicalDAG, order: Sequence[int]) -> bool:
    if sorted(int(t) for t in order) != sorted(int(t) for t in dag.tasks):
        return False
    rank = {int(t): i for i, t in enumerate(order)}
    return all(rank[int(e.src_task_id)] < rank[int(e.dst_task_id)] for e in dag.edges)


def graph_depth(dag: CanonicalDAG) -> int:
    preds = dag.predecessors()
    depth = {tid: 0 for tid in dag.tasks}
    for tid in topological_sequence(dag):
        for edge in preds[tid]:
            depth[tid] = max(depth[tid], depth[edge.src_task_id] + 1)
    return max(depth.values()) if depth else 0


def graph_density(dag: CanonicalDAG) -> float:
    n = len(dag.tasks)
    if n < 2:
        return 0.0
    return len(dag.edges) / (n * (n - 1) / 2.0)


# --------------------------------------------------------------------------- #
# rank / order constructions
# --------------------------------------------------------------------------- #
class _LegacyTask:
    __slots__ = ("processing_data_size", "transmission_data_size")

    def __init__(self, processing_data_size: int, transmission_data_size: int) -> None:
        self.processing_data_size = processing_data_size
        self.transmission_data_size = transmission_data_size


class _LegacyGraphView:
    """Duck-typed view so the PRODUCTION `prioritize_tasks` can be called as-is.

    Production ranks by list position, so the canonical task ids must be exactly
    `0..n-1` (true for both the adapter and the synthetic builder).
    """

    def __init__(self, dag: CanonicalDAG) -> None:
        ids = sorted(int(t) for t in dag.tasks)
        if ids != list(range(len(ids))):
            raise ValueError("legacy view requires canonical task ids 0..n-1")
        self.task_number = len(ids)
        self.task_list = [
            _LegacyTask(
                int(dag.tasks[tid].compute_workload_bytes),
                int(dag.tasks[tid].task_output_bytes),
            )
            for tid in ids
        ]
        self.succ_task_sets: list[set[int]] = [set() for _ in ids]
        for edge in dag.edges:
            self.succ_task_sets[int(edge.src_task_id)].add(int(edge.dst_task_id))
        self.prioritize_sequence: list[int] = []


def _legacy_cluster(resources: ResourceConfig) -> Any:
    """Legacy cost cluster, built from the resolved canonical rates.

    Prefers the production `Resources` object (with a minimal gym stub, because
    `offloading_env` imports gym at module import time on this CPU host); falls
    back to an in-file shim that reproduces the four cost formulas exactly.
    """
    mbps = 1024.0 * 1024.0 / 8.0
    mec = float(resources.mec_cpu_bytes_per_second)
    mobile = float(resources.ue_cpu_bytes_per_second)
    v2v = float(resources.helper_cpu_bytes_per_second)
    up = float(resources.mec_uplink_bytes_per_second) / mbps
    dl = float(resources.mec_downlink_bytes_per_second) / mbps
    v2v_bw = float(resources.v2v_bytes_per_second) / mbps

    for name in ("gym", "gym.core"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules.get("gym.core"), "Env"):
        sys.modules["gym.core"].Env = type("Env", (), {})
    try:
        from env.mec_offloaing_envs.offloading_env import Resources

        return Resources(
            mec_process_capable=mec,
            mobile_process_capable=mobile,
            bandwidth_up=up,
            bandwidth_dl=dl,
            v2v_process_capable=v2v,
            v2v_bandwidth=v2v_bw,
        )
    except Exception:  # pragma: no cover - gym/TF-free fallback
        return _LegacyCostCluster(
            mec, mobile, v2v, up * mbps, dl * mbps, v2v_bw * mbps
        )


class _LegacyCostCluster:
    """Exact copy of the legacy `Resources` cost formulas (bytes/s rates)."""

    def __init__(self, mec, mobile, v2v, up_bps, dl_bps, v2v_bps) -> None:
        self.mec_process_capable = float(mec)
        self.mobile_process_capable = float(mobile)
        self.v2v_process_capable = float(v2v)
        self._up = float(up_bps)
        self._dl = float(dl_bps)
        self._v2v = float(v2v_bps)

    def up_transmission_cost(self, data):
        return data / self._up

    def dl_transmission_cost(self, data):
        return data / self._dl

    def v2v_transmission_cost(self, data):
        return data / self._v2v


def _legacy_w(view: _LegacyGraphView, cluster: Any) -> list[float]:
    w = []
    for task in view.task_list:
        t_local = task.processing_data_size / cluster.mobile_process_capable
        t_mec = (
            cluster.up_transmission_cost(task.processing_data_size)
            + task.processing_data_size / cluster.mec_process_capable
            + cluster.dl_transmission_cost(task.transmission_data_size)
        )
        t_v2v = (
            cluster.v2v_transmission_cost(task.processing_data_size)
            + task.processing_data_size / cluster.v2v_process_capable
            + cluster.v2v_transmission_cost(task.transmission_data_size)
        )
        w.append(min(t_local, t_mec, t_v2v))
    return w


def _legacy_rank_values(dag: CanonicalDAG, cluster: Any) -> dict[int, float]:
    view = _LegacyGraphView(dag)
    w = _legacy_w(view, cluster)
    rank: dict[int, float] = {}
    for tid in reversed(topological_sequence(dag)):
        succ = sorted(view.succ_task_sets[tid])
        base = w[tid]
        if succ:
            base += max(rank[j] for j in succ)
        rank[tid] = base
    return rank


def legacy_order(dag: CanonicalDAG, resources: ResourceConfig) -> tuple[list[int], dict[int, float]]:
    """Order #1: the production `prioritize_tasks` output (verbatim)."""
    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

    view = _LegacyGraphView(dag)
    cluster = _legacy_cluster(resources)
    produced = OffloadingTaskGraph.prioritize_tasks(view, cluster)  # production code
    order = [int(t) for t in produced]
    ranks = _legacy_rank_values(dag, cluster)
    return order, ranks


def heft_upward_rank(dag: CanonicalDAG, resources: ResourceConfig) -> dict[int, float]:
    def mean_compute(tid: int) -> float:
        task = dag.tasks[tid]
        return sum(
            task.compute_workload_bytes / resources.cpu_rate_for_task(loc, task)
            for loc in LOCATIONS
        ) / len(LOCATIONS)

    def mean_return(tid: int) -> float:
        nbytes = int(dag.tasks[tid].task_output_bytes)
        return sum(_transfer_duration(nbytes, loc, Location.UE, resources) for loc in LOCATIONS) / len(
            LOCATIONS
        )

    succs = dag.successors()
    edge_c: dict[tuple[int, int], float] = {}
    for edge in dag.edges:
        edge_c[(int(edge.src_task_id), int(edge.dst_task_id))] = _mean_transfer(
            int(edge.edge_output_bytes), resources
        )
    rank: dict[int, float] = {}
    for tid in reversed(topological_sequence(dag)):
        base = mean_compute(tid)
        outs = succs[tid]
        if outs:
            base += max(edge_c[(tid, j)] + rank[j] for j in outs)
        else:
            base += mean_return(tid)
        rank[tid] = base
    return rank


def _transfer_duration(nbytes: int, src: Location, dst: Location, resources: ResourceConfig) -> float:
    return sum(nbytes / resources.hop_rate(hop) for hop in route(src, dst))


def _min_transfer(nbytes: int, resources: ResourceConfig) -> float:
    if nbytes <= 0:
        return 0.0
    return min(
        _transfer_duration(nbytes, src, dst, resources)
        for src in LOCATIONS
        for dst in LOCATIONS
    )


def _mean_transfer(nbytes: int, resources: ResourceConfig) -> float:
    pairs = [(src, dst) for src in LOCATIONS for dst in LOCATIONS]
    return sum(_transfer_duration(nbytes, src, dst, resources) for src, dst in pairs) / len(pairs)


def heft_order(dag: CanonicalDAG, resources: ResourceConfig) -> list[int]:
    rank = heft_upward_rank(dag, resources)
    return kahn_by_score(dag, {tid: (rank[tid],) for tid in dag.tasks})


def canonical_probe_result(dag: CanonicalDAG, resources: ResourceConfig):
    actions = fixed_actions_for(dag)
    order = topological_sequence(dag)
    return schedule(dag, order, [actions[tid] for tid in order], resources)


def canonical_order(dag: CanonicalDAG, resources: ResourceConfig) -> list[int]:
    """Order #4: Kahn priority = descending observed finish from the probe."""
    probe = canonical_probe_result(dag, resources)
    score = {tid: (float(probe.tasks[tid].finish),) for tid in dag.tasks}
    return kahn_by_score(dag, score)


def _min_contention_free_finish(dag: CanonicalDAG, resources: ResourceConfig) -> dict[int, float]:
    """Earliest finish lower bound per task under the min-duration action."""
    preds = dag.predecessors()
    finish: dict[int, float] = {}
    for tid in topological_sequence(dag):
        task = dag.tasks[tid]
        ready = 0.0
        if int(task.external_input_bytes) > 0:
            ready = max(ready, _min_transfer(int(task.external_input_bytes), resources))
        for edge in preds[tid]:
            ready = max(
                ready,
                finish[edge.src_task_id]
                + _min_transfer(int(edge.edge_output_bytes), resources),
            )
        compute = min(
            task.compute_workload_bytes / resources.cpu_rate_for_task(loc, task)
            for loc in LOCATIONS
        )
        finish[tid] = ready + compute
    return finish


def deadline_order(dag: CanonicalDAG, resources: ResourceConfig) -> list[int]:
    upward = heft_upward_rank(dag, resources)
    ef_lb = _min_contention_free_finish(dag, resources)
    score: dict[int, tuple[float, ...]] = {}
    for tid in dag.tasks:
        task = dag.tasks[tid]
        has_deadline = 1.0 if (task.deadline_s is not None and task.deadline_type != "none") else 0.0
        slack = 0.0
        if has_deadline:
            slack = float(task.deadline_s) - ef_lb[tid]
        score[tid] = (
            has_deadline,
            -slack,
            float(DEADLINE_TYPE_RANK.get(str(task.deadline_type), 0)),
            float(CRITICALITY_RANK.get(str(task.criticality_class), 1)),
            float(upward[tid]),
        )
    return kahn_by_score(dag, score)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def kendall_tau(a: Sequence[int], b: Sequence[int]) -> float:
    """Kendall tau (tau-a) between two permutations of the same ids."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    pa = {int(t): i for i, t in enumerate(a)}
    pb = {int(t): i for i, t in enumerate(b)}
    concordant = 0
    discordant = 0
    ids = sorted(pa)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            x, y = ids[i], ids[j]
            d1 = pa[x] - pa[y]
            d2 = pb[x] - pb[y]
            if d1 == 0 or d2 == 0:
                continue
            if (d1 > 0) == (d2 > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def spearman_footrule_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Normalized Spearman footrule distance in [0, 1] (permutations)."""
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0
    pa = {int(t): i for i, t in enumerate(a)}
    pb = {int(t): i for i, t in enumerate(b)}
    total = sum(abs(pa[int(t)] - pb[int(t)]) for t in a)
    return min(1.0, total / float((n * n) // 2))


def spearman_rho(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0
    pa = {int(t): i for i, t in enumerate(a)}
    pb = {int(t): i for i, t in enumerate(b)}
    d2 = sum((pa[int(t)] - pb[int(t)]) ** 2 for t in a)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def queue_utilization(result, makespan: float) -> dict[str, float]:
    """Busy fraction of the six calendars over the makespan (fixed-plan result)."""
    out = {}
    span = float(makespan)
    for name in RESOURCE_NAMES:
        busy = sum(
            max(0.0, float(iv.end) - float(iv.start))
            for iv in result.resource_intervals
            if iv.resource == name
        )
        out[name] = min(1.0, busy / span) if span > 0.0 else 0.0
    out["mean"] = sum(out[n] for n in RESOURCE_NAMES) / len(RESOURCE_NAMES)
    return out


def static_bound(
    dag: CanonicalDAG, order: Sequence[int], actions: Sequence[int], resources: ResourceConfig
) -> float:
    """Queue-blind longest path (compute + contention-free transfers).

    Admissible: the relaxed system has infinite-capacity calendars, so it can
    only be faster than the real schedule. Order-invariant by construction.
    """
    locs = {tid: ACTION_LOCATIONS[int(actions[i])] for i, tid in enumerate(order)}
    preds = dag.predecessors()
    result = schedule(dag, order, actions, resources)
    finish: dict[int, float] = {}
    for tid in result.topo_order:
        task = dag.tasks[tid]
        ready = 0.0
        ext = int(task.external_input_bytes)
        if ext > 0:
            ready = max(ready, _transfer_duration(ext, Location.UE, locs[tid], resources))
        for edge in preds[tid]:
            src = edge.src_task_id
            ready = max(
                ready,
                finish[src]
                + _transfer_duration(int(edge.edge_output_bytes), locs[src], locs[tid], resources),
            )
        finish[tid] = ready + task.compute_workload_bytes / resources.cpu_rate_for_task(locs[tid], task)
    bound = 0.0
    for tid in dag.sinks():
        bound = max(
            bound,
            finish[tid] + _transfer_duration(int(dag.tasks[tid].task_output_bytes), locs[tid], Location.UE, resources),
        )
    return bound


def greedy_assignment(dag: CanonicalDAG, order: Sequence[int], resources: ResourceConfig):
    """Order-dependent greedy assignment: at each position pick the action that
    minimizes the makespan of the completed plan with the suffix left all-UE.
    Mirrors the production `greedy.greedy_plan` tie-break (lower action id).
    """
    order = [int(t) for t in order]
    n = len(order)
    chosen: list[int] = []
    for k in range(n):
        best_metric = None
        best_action = None
        for action in (0, 1, 2):
            fill = chosen + [action] + [0] * (n - k - 1)
            metric = float(schedule(dag, order, fill, resources).makespan_seconds)
            if (
                best_metric is None
                or metric + 1e-12 < best_metric
                or (abs(metric - best_metric) <= 1e-12 and action < int(best_action))
            ):
                best_metric = metric
                best_action = action
        chosen.append(int(best_action))
    result = schedule(dag, order, chosen, resources)
    return chosen, result


def policy_action_mix() -> dict[str, Any]:
    """Action mix of a policy checkpoint, only when trivially available.

    No policy checkpoint in this tree is loadable without TensorFlow, and the
    only checkpoint present (`meta_model_inner_step1/meta_model_0.ckpt`) is a
    TF1 MAML meta-model over the legacy decoder, not the canonical action space.
    """
    return {
        "status": "not available",
        "reason": (
            "no policy checkpoint is loadable without TensorFlow; the only in-tree "
            "checkpoint (meta_model_inner_step1/meta_model_0.ckpt) is a TF1 MAML "
            "meta-model over the legacy decoder, not the canonical action space"
        ),
    }


def action_mix(actions_by_task: Mapping[int, int]) -> dict[str, int]:
    mix = {"UE": 0, "MEC": 0, "HELPER": 0}
    for action in actions_by_task.values():
        mix[ACTION_LOCATIONS[int(action)].value] += 1
    return mix


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _bucket(values: Mapping[str, float], labels: Sequence[str]) -> dict[str, str]:
    """Tercile buckets over the audited graph set, labelled low/mid/high."""
    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    out: dict[str, str] = {}
    for i, (name, _) in enumerate(ordered):
        idx = min(len(labels) - 1, (i * len(labels)) // max(1, n))
        out[name] = labels[idx]
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    graph_names: Sequence[str] | None = None,
    *,
    resources: ResourceConfig | None = None,
) -> dict[str, Any]:
    resources = resources or resolved_primary_scheduler_config()
    all_graphs = build_graphs(resources)
    if graph_names is not None:
        wanted = [str(n) for n in graph_names]
        missing = [n for n in wanted if n not in all_graphs]
        if missing:
            raise KeyError("unknown graph(s): %s" % missing)
        selected = OrderedDict((n, all_graphs[n]) for n in wanted)
    else:
        selected = all_graphs

    density_by_graph = {n: graph_density(e.dag) for n, e in selected.items()}
    depth_by_graph = {n: graph_depth(e.dag) for n, e in selected.items()}
    density_bucket = _bucket(density_by_graph, ("sparse", "medium", "dense"))
    depth_bucket = _bucket(depth_by_graph, ("shallow", "medium", "deep"))

    graph_rows = []
    for name, entry in selected.items():
        dag = entry.dag
        row = {
            "name": name,
            "source": entry.source,
            "path": entry.path,
            "n_tasks": len(dag.tasks),
            "n_edges": len(dag.edges),
            "density": density_by_graph[name],
            "depth": depth_by_graph[name],
            "density_bucket": density_bucket[name],
            "depth_bucket": depth_bucket[name],
            "sinks": [int(t) for t in dag.sinks()],
            "has_deadlines": any(
                t.deadline_s is not None and t.deadline_type != "none" for t in dag.tasks.values()
            ),
        }
        graph_rows.append(row)

    policy_mix = policy_action_mix()
    per_graph: list[dict[str, Any]] = []
    for name, entry in selected.items():
        dag = entry.dag
        fixed_actions = fixed_actions_for(dag)
        legacy_seq, legacy_ranks = legacy_order(dag, resources)
        orders = OrderedDict(
            (
                ("legacy_current", legacy_seq),
                ("stable_topo", topological_sequence(dag)),
                ("heft_upward", heft_order(dag, resources)),
                ("canonical_aware", canonical_order(dag, resources)),
                ("deadline_criticality", deadline_order(dag, resources)),
            )
        )
        assert list(orders) == list(ORDER_NAMES)
        current = orders["legacy_current"]
        for order_name, order_seq in orders.items():
            actions_in_order = [fixed_actions[tid] for tid in order_seq]
            fixed_result = schedule(dag, order_seq, actions_in_order, resources)
            makespan = float(fixed_result.makespan_seconds)
            # order-only control: the same (order, fixed actions) must reproduce
            # the makespan exactly, so a delta can only come from the order.
            repeat = float(schedule(dag, order_seq, actions_in_order, resources).makespan_seconds)
            bound = static_bound(dag, order_seq, actions_in_order, resources)
            greedy_actions, greedy_result = greedy_assignment(dag, order_seq, resources)
            util = queue_utilization(fixed_result, makespan)
            greedy_by_task = {tid: greedy_actions[i] for i, tid in enumerate(order_seq)}
            per_graph.append(
                {
                    "graph": name,
                    "order": order_name,
                    "order_sequence": [int(t) for t in order_seq],
                    "topological_valid": is_topological(dag, order_seq),
                    "permutation_valid": sorted(int(t) for t in order_seq)
                    == sorted(int(t) for t in dag.tasks),
                    "kendall_tau": kendall_tau(order_seq, current),
                    "spearman_distance": spearman_footrule_distance(order_seq, current),
                    "spearman_rho": spearman_rho(order_seq, current),
                    "legacy_rank_values": {str(t): legacy_ranks[t] for t in sorted(legacy_ranks)},
                    "fixed_plan_actions_by_task": {str(t): fixed_actions[t] for t in sorted(fixed_actions)},
                    "fixed_plan_latency_s": _finite(makespan),
                    "fixed_plan_repeat_delta_s": _finite(abs(repeat - makespan)),
                    "fixed_plan_energy_system_j": _finite(fixed_result.energy.total_system_joules),
                    "queue_utilization": util,
                    "queue_utilization_mean": util["mean"],
                    "greedy_actions_by_task": {str(t): greedy_by_task[t] for t in sorted(greedy_by_task)},
                    "greedy_action_mix": action_mix(greedy_by_task),
                    "greedy_latency_s": _finite(float(greedy_result.makespan_seconds)),
                    "greedy_energy_system_j": _finite(greedy_result.energy.total_system_joules),
                    "fixed_action_mix": action_mix({t: fixed_actions[t] for t in fixed_actions}),
                    "policy_action_mix": policy_mix,
                    "static_bound_s": _finite(bound),
                    "static_bound_error_abs_s": _finite(makespan - bound),
                    "static_bound_error_rel": _finite((makespan - bound) / makespan)
                    if makespan > 0.0
                    else None,
                    "static_bound_valid": bool(bound <= makespan + 1e-6),
                    "n_tasks": len(dag.tasks),
                    "n_edges": len(dag.edges),
                    "density": density_by_graph[name],
                    "depth": depth_by_graph[name],
                    "density_bucket": density_bucket[name],
                    "depth_bucket": depth_bucket[name],
                }
            )

    aggregate = _aggregate(per_graph, density_bucket, depth_bucket)
    checks = _checks(per_graph, graph_rows, aggregate)

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_by": "spec/decoder_order_audit.py",
        "methodology": {
            "order_vs_assignment": (
                "fixed-plan experiment uses ONE frozen per-task mapping "
                "fixed_action(tid)=tid%3 for every order; greedy assignment is "
                "searched separately per order"
            ),
            "static_bound": "queue-blind longest path (compute + contention-free transfer)",
            "production_untouched": True,
        },
        "orders": [dict(d) for d in ORDER_DEFINITIONS],
        "graphs": graph_rows,
        "per_graph": per_graph,
        "aggregate": aggregate,
        "checks": checks,
        "all_checks_pass": False,
    }
    # the serializability check must see the final evidence, so it is resolved
    # after the body exists (and before `all_checks_pass` is frozen)
    checks["json_serializable"] = evidence_serializable(body)
    body["all_checks_pass"] = bool(all(checks.values()))
    return body


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    density_bucket: Mapping[str, str],
    depth_bucket: Mapping[str, str],
) -> dict[str, Any]:
    mean_by_order = {}
    for order in ORDER_NAMES:
        sub = [r for r in rows if r["order"] == order]
        mean_by_order[order] = {
            "n_rows": len(sub),
            "mean_kendall_tau": _mean([r["kendall_tau"] for r in sub]),
            "mean_spearman_distance": _mean([r["spearman_distance"] for r in sub]),
            "mean_spearman_rho": _mean([r["spearman_rho"] for r in sub]),
            "mean_fixed_plan_latency_s": _mean([r["fixed_plan_latency_s"] for r in sub]),
            "mean_greedy_latency_s": _mean([r["greedy_latency_s"] for r in sub]),
            "mean_queue_utilization": _mean([r["queue_utilization_mean"] for r in sub]),
            "mean_static_bound_error_abs_s": _mean([r["static_bound_error_abs_s"] for r in sub]),
            "mean_static_bound_error_rel": _mean([r["static_bound_error_rel"] for r in sub]),
        }

    def best(key: str) -> str | None:
        ranked = [
            (order, mean_by_order[order][key])
            for order in ORDER_NAMES
            if mean_by_order[order][key] is not None
        ]
        if not ranked:
            return None
        return min(ranked, key=lambda kv: (kv[1], kv[0]))[0]

    def bucket_error(axis: Mapping[str, str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label in sorted(set(axis.values())):
            sub = [r for r in rows if axis[r["graph"]] == label]
            out[label] = {
                "n_rows": len(sub),
                "n_graphs": len({r["graph"] for r in sub}),
                "mean_static_bound_error_abs_s": _mean([r["static_bound_error_abs_s"] for r in sub]),
                "mean_static_bound_error_rel": _mean([r["static_bound_error_rel"] for r in sub]),
                "mean_fixed_plan_latency_s": _mean([r["fixed_plan_latency_s"] for r in sub]),
            }
        return out

    return {
        "n_graphs": len({r["graph"] for r in rows}),
        "mean_kendall_tau_by_order": {
            o: mean_by_order[o]["mean_kendall_tau"] for o in ORDER_NAMES
        },
        "mean_spearman_distance_by_order": {
            o: mean_by_order[o]["mean_spearman_distance"] for o in ORDER_NAMES
        },
        "mean_spearman_rho_by_order": {
            o: mean_by_order[o]["mean_spearman_rho"] for o in ORDER_NAMES
        },
        "mean_fixed_plan_latency_by_order": {
            o: mean_by_order[o]["mean_fixed_plan_latency_s"] for o in ORDER_NAMES
        },
        "mean_greedy_latency_by_order": {
            o: mean_by_order[o]["mean_greedy_latency_s"] for o in ORDER_NAMES
        },
        "mean_queue_utilization_by_order": {
            o: mean_by_order[o]["mean_queue_utilization"] for o in ORDER_NAMES
        },
        "mean_static_bound_error_abs_by_order": {
            o: mean_by_order[o]["mean_static_bound_error_abs_s"] for o in ORDER_NAMES
        },
        "mean_static_bound_error_rel_by_order": {
            o: mean_by_order[o]["mean_static_bound_error_rel"] for o in ORDER_NAMES
        },
        "per_order": mean_by_order,
        "best_fixed_plan_latency_order": best("mean_fixed_plan_latency_s"),
        "best_greedy_latency_order": best("mean_greedy_latency_s"),
        "static_bound_error_by_density_bucket": bucket_error(density_bucket),
        "static_bound_error_by_depth_bucket": bucket_error(depth_bucket),
    }


def _checks(
    rows: Sequence[Mapping[str, Any]],
    graph_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, bool]:
    graphs = {g["name"] for g in graph_rows}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["graph"], []).append(row)

    five_orders = all(
        sorted(r["order"] for r in grouped[g]) == sorted(ORDER_NAMES) for g in graphs
    )

    def rows_of(g: str) -> list[Mapping[str, Any]]:
        return grouped[g]

    fixed_maps_ok = True
    for g in graphs:
        first = None
        for row in rows_of(g):
            m = row["fixed_plan_actions_by_task"]
            if any(m[str(t)] != fixed_action(t) for t in m):
                fixed_maps_ok = False
            if first is None:
                first = m
            elif m != first:
                fixed_maps_ok = False

    # order-only control: identical (order, actions) must reproduce the makespan
    repeat_ok = all(
        r["fixed_plan_repeat_delta_s"] is not None
        and float(r["fixed_plan_repeat_delta_s"]) <= 1e-9
        for r in rows
    )

    bound_by_graph: dict[str, set[float]] = {}
    for row in rows:
        bound_by_graph.setdefault(row["graph"], set()).add(round(float(row["static_bound_s"]), 9))

    density_labels = {r["density_bucket"] for r in rows}
    depth_labels = {r["depth_bucket"] for r in rows}

    checks = {
        "five_orders_per_graph": bool(five_orders),
        "all_orders_topological": bool(all(r["topological_valid"] for r in rows)),
        "all_orders_permutations": bool(all(r["permutation_valid"] for r in rows)),
        "kendall_tau_in_range": bool(
            all(-1.0 - EPS <= float(r["kendall_tau"]) <= 1.0 + EPS for r in rows)
        ),
        "spearman_in_range": bool(
            all(0.0 - EPS <= float(r["spearman_distance"]) <= 1.0 + EPS for r in rows)
            and all(-1.0 - EPS <= float(r["spearman_rho"]) <= 1.0 + EPS for r in rows)
        ),
        "makespans_finite_nonnegative": bool(
            all(
                r["fixed_plan_latency_s"] is not None
                and r["fixed_plan_latency_s"] >= 0.0
                and r["greedy_latency_s"] is not None
                and r["greedy_latency_s"] >= 0.0
                for r in rows
            )
        ),
        "fixed_plan_actions_identical_across_orders": bool(fixed_maps_ok),
        "fixed_plan_deterministic": bool(repeat_ok),
        "fixed_plan_experiment_is_order_only": bool(fixed_maps_ok and repeat_ok),
        "static_bound_admissible": bool(all(r["static_bound_valid"] for r in rows)),
        "static_bound_order_invariant": bool(
            all(len(v) == 1 for v in bound_by_graph.values())
        ),
        "queue_utilization_in_range": bool(
            all(
                0.0 - EPS <= float(r["queue_utilization"][name]) <= 1.0 + EPS
                for r in rows
                for name in RESOURCE_NAMES
            )
        ),
        "policy_action_mix_reported": bool(
            all(
                r["policy_action_mix"]["status"] in ("available", "not available")
                for r in rows
            )
        ),
        "distinct_orders_present": bool(
            len({tuple(r["order_sequence"]) for r in rows}) >= 2
        ),
        "density_buckets_populated": bool(len(density_labels) >= 2),
        "depth_buckets_populated": bool(len(depth_labels) >= 2),
        "aggregate_has_all_orders": bool(
            all(o in aggregate["per_order"] for o in ORDER_NAMES)
        ),
    }
    return checks


def evidence_serializable(evidence: Mapping[str, Any]) -> bool:
    try:
        json.dumps(evidence, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part D decoder-order / ranking audit")
    default = ROOT / "reports" / "v0.3-audit" / "rank_audit" / "rank_audit_evidence.json"
    parser.add_argument("--json", default=str(default))
    parser.add_argument(
        "--graphs", default=None, help="comma-separated subset of graph names"
    )
    args = parser.parse_args(argv)

    graph_names = None
    if args.graphs:
        graph_names = [g.strip() for g in args.graphs.split(",") if g.strip()]
    # `run()` resolves `json_serializable` and `all_checks_pass` itself
    evidence = run(graph_names)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")

    aggregate = evidence["aggregate"]
    print("orders:")
    for order in ORDER_NAMES:
        print(
            "  %-20s kendall=%s spearman_d=%s latency=%.4f greedy=%.4f static_err=%s"
            % (
                order,
                _fmt(aggregate["mean_kendall_tau_by_order"][order]),
                _fmt(aggregate["mean_spearman_distance_by_order"][order]),
                aggregate["mean_fixed_plan_latency_by_order"][order] or 0.0,
                aggregate["mean_greedy_latency_by_order"][order] or 0.0,
                _fmt(aggregate["mean_static_bound_error_abs_by_order"][order]),
            )
        )
    print("best fixed-plan order:", aggregate["best_fixed_plan_latency_order"])
    print("best greedy order:", aggregate["best_greedy_latency_order"])
    print("static-bound error by density bucket:")
    for label, vals in sorted(aggregate["static_bound_error_by_density_bucket"].items()):
        print("  %-8s n=%d abs=%s rel=%s" % (label, vals["n_rows"], _fmt(vals["mean_static_bound_error_abs_s"]), _fmt(vals["mean_static_bound_error_rel"])))
    print("static-bound error by depth bucket:")
    for label, vals in sorted(aggregate["static_bound_error_by_depth_bucket"].items()):
        print("  %-8s n=%d abs=%s rel=%s" % (label, vals["n_rows"], _fmt(vals["mean_static_bound_error_abs_s"]), _fmt(vals["mean_static_bound_error_rel"])))
    failed = [k for k, v in evidence["checks"].items() if not v]
    print("checks:", "ALL PASS" if not failed else "FAILED %s" % failed)
    print("wrote", out)
    return 0 if evidence["all_checks_pass"] else 1


def _fmt(value: Any) -> str:
    return "n/a" if value is None else "%.4f" % float(value)


if __name__ == "__main__":
    raise SystemExit(main())
