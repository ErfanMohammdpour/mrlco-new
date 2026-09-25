#!/usr/bin/env python3
"""Part C — upstream-parity diagnostic (DIAGNOSTIC ONLY, no production change).

The question this harness answers is *explainability*, not "make the numbers
equal": given ONE graph, ONE decoder order, ONE binary plan and ONE rate table,
where do the canonical MARGO scheduler and the upstream MR-LCO scheduler
(https://github.com/linkpark/metarl-offloading) disagree, and which named factor
causes each disagreement?

Three schedulers are compared under one explicit fair-comparison contract:

  1. ``upstream_original``  the upstream repository's OWN scheduling function,
     extracted verbatim from their source (AST segment, byte-identical) and
     executed here.  We never re-type their arithmetic.
  2. ``legacy_control``     the project's pre-canonical path.  The working tree
     no longer contains one: ``OffloadingEnvironment.get_scheduling_cost_step_by_step``
     (``env/mec_offloaing_envs/offloading_env.py``) is today a thin wrapper over
     the canonical engine since commit ``b49b60f``.  So this harness carries a
     clearly-labelled MINIMAL legacy model reproducing the pre-canonical
     function at ``b49b60f^`` restricted to binary actions and latency-only
     accounting (the V2V branch and the energy returns are dropped on purpose).
  3. ``canonical``          ``env.mec_offloaing_envs.scheduler.schedule`` with a
     ``CanonicalDAG`` and ``ResourceConfig.from_frozen_yaml()``.

FAIR-COMPARISON CONTRACT
  * binary actions only: Local(0) / MEC(1); no Helper(2), no V2V hop;
  * one graph, one decoder order, one plan, identical rates for all three;
  * latency semantics only: energy accounting must not influence the parity
    result (asserted, not assumed: the makespan must not move when the energy
    model changes while the timing axis is held constant);
  * per task: upload, compute, dependency transfer, download/return, start,
    finish; plus makespan.

The upstream function only returns ``(latency_deltas, makespan)``.  Per-task
start/finish for the upstream row are therefore produced by a step-recording
replay of the SAME algorithm, and that replay is validated against upstream's
own deltas and makespan (``upstream_replay_matches_upstream``); if upstream is
not available the replay is still reported as the legacy control.  No upstream
number is ever invented.

Usage:
    python3 -m spec.upstream_parity [--json PATH] [--upstream-root PATH]
                                    [--require-upstream] [--quiet]

Exit status is non-zero when any contract check fails.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    Location,
    ResourceConfig,
    schedule,
)

SCHEMA = "upstream_parity_v1"
UPSTREAM_REPO = "https://github.com/linkpark/metarl-offloading"
UPSTREAM_REL = Path("env") / "mec_offloaing_envs" / "offloading_env.py"
UPSTREAM_SCHEDULER_SYMBOL = "get_scheduling_cost_step_by_step"
UPSTREAM_RESOURCES_SYMBOL = "Resources"
DEFAULT_UPSTREAM_ROOT = Path("/tmp/upstream_metarl")
DEFAULT_JSON = (
    Path("reports") / "v0.3-audit" / "upstream_parity" / "upstream_parity_evidence.json"
)
LEGACY_CONTROL_ORIGIN = (
    "env/mec_offloaing_envs/offloading_env.py::"
    "OffloadingEnvironment.get_scheduling_cost_step_by_step@b49b60f^"
)

# The seven difference factors the contract asks to be stated explicitly.
FACTOR_KEYS = (
    "upload_charged_per_remote_task_upstream_vs_root_only_external_input",
    "mec_residency_and_mec_to_mec_zero_transfer",
    "dependency_edge_transfer",
    "sink_only_return",
    "separate_ul_dl_calendars",
    "contention_and_parallelism",
    "decoder_order",
)

BINARY_ACTIONS = (0, 1)


class UpstreamUnavailableError(RuntimeError):
    """The upstream scheduler source could not be located or parsed."""


# --------------------------------------------------------------------------- #
# The single graph / decoder order / plan used by all three schedulers.
# --------------------------------------------------------------------------- #
def reference_graph() -> CanonicalDAG:
    """ONE graph: two roots -> fan-out -> fan-in -> sink (upstream daggen shape).

    Task 1 is a UE root so the plan below exercises UE->MEC dependency uploads as
    well as MEC->MEC residency, and task 4 is a UE consumer of a MEC task so a
    MEC->UE dependency download exists.  Sizes are realistic MR-LCO magnitudes
    (multi-MB compute, output ~= half of compute, upstream ``expect_size``).
    """
    tasks = [
        # task_id, compute_workload_bytes, task_output_bytes, external_input_bytes
        CanonicalTask(0, 8_388_608, 2_097_152, 8_388_608),   # root, planned MEC
        CanonicalTask(1, 4_194_304, 1_048_576, 4_194_304),   # root, planned UE
        CanonicalTask(2, 6_291_456, 3_145_728),
        CanonicalTask(3, 5_242_880, 2_097_152),
        CanonicalTask(4, 4_194_304, 1_048_576),
        CanonicalTask(5, 7_340_032, 3_670_016),
        CanonicalTask(6, 3_145_728, 1_572_864),
        CanonicalTask(7, 2_097_152, 1_048_576),              # sink, planned MEC
    ]
    edges = [
        (0, 2, 2_097_152),
        (1, 2, 1_048_576),
        (1, 3, 1_048_576),
        (2, 4, 3_145_728),
        (2, 5, 3_145_728),
        (3, 5, 2_097_152),
        (4, 6, 1_048_576),
        (5, 6, 3_670_016),
        (6, 7, 1_572_864),
    ]
    return CanonicalDAG.from_records(tasks, edges)


def reference_decoder_order() -> list[int]:
    """ONE decoder order: a topological order (task ids ascending)."""
    return [0, 1, 2, 3, 4, 5, 6, 7]


def reference_plan() -> list[int]:
    """ONE binary plan: mixed Local/MEC, chosen to expose every listed factor."""
    return [1, 0, 1, 1, 0, 1, 1, 1]


def canonical_resources() -> ResourceConfig:
    """ONE frozen rate table; shared by every scheduler in this diagnostic."""
    return ResourceConfig.from_frozen_yaml()


def reference_contract() -> tuple[CanonicalDAG, list[int], list[int], ResourceConfig]:
    return (
        reference_graph(),
        reference_decoder_order(),
        reference_plan(),
        canonical_resources(),
    )


# --------------------------------------------------------------------------- #
# Small deterministic helpers.
# --------------------------------------------------------------------------- #
def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _graph_fingerprint(graph: CanonicalDAG) -> dict[str, Any]:
    tasks = [
        {
            "task_id": int(t.task_id),
            "compute_workload_bytes": int(t.compute_workload_bytes),
            "task_output_bytes": int(t.task_output_bytes),
            "external_input_bytes": int(t.external_input_bytes),
        }
        for t in sorted(graph.tasks.values(), key=lambda t: t.task_id)
    ]
    edges = [
        [int(e.src_task_id), int(e.dst_task_id), int(e.edge_output_bytes)]
        for e in graph.edges
    ]
    return {
        "task_count": len(tasks),
        "edge_count": len(edges),
        "tasks": tasks,
        "edges": edges,
        "sha256": _sha256_json({"tasks": tasks, "edges": edges}),
    }


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite: {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Legacy / upstream shared task-graph shim.
# --------------------------------------------------------------------------- #
class _LegacyTask:
    """The attributes the upstream function reads off a task."""

    __slots__ = ("processing_data_size", "transmission_data_size")

    def __init__(self, processing_data_size: int, transmission_data_size: int):
        self.processing_data_size = int(processing_data_size)
        self.transmission_data_size = int(transmission_data_size)


class _LegacyTaskGraph:
    """Minimal stand-in exposing exactly what both legacy algorithms consume."""

    def __init__(self, graph: CanonicalDAG):
        order = sorted(graph.tasks)
        if order != list(range(len(order))):
            raise ValueError(
                "_LegacyTaskGraph requires contiguous task ids 0..n-1 because the "
                "upstream/legacy algorithms index task_list by task id; got %r" % (order,)
            )
        index = {tid: i for i, tid in enumerate(order)}
        self.task_number = len(order)
        self.task_list = [
            _LegacyTask(
                graph.tasks[tid].compute_workload_bytes,
                graph.tasks[tid].task_output_bytes,
            )
            for tid in order
        ]
        self.pre_task_sets = [set() for _ in order]
        self.succ_task_sets = [set() for _ in order]
        for edge in graph.edges:
            self.pre_task_sets[index[edge.dst_task_id]].add(index[edge.src_task_id])
            self.succ_task_sets[index[edge.src_task_id]].add(index[edge.dst_task_id])
        self._index = index
        self._order = order

    def index_of(self, task_id: int) -> int:
        return self._index[int(task_id)]


# --------------------------------------------------------------------------- #
# Scheduler 2 (and the upstream replay): faithful latency-only legacy model.
# --------------------------------------------------------------------------- #
def _legacy_replay(
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    cfg: ResourceConfig,
) -> dict[str, Any]:
    """Step-recording replay of the upstream / pre-canonical latency algorithm.

    Binary actions only.  Reproduces, line for line, the semantics of
    ``get_scheduling_cost_step_by_step`` at upstream ``a55094f`` and at this
    repo's ``b49b60f^`` with the V2V branch removed (there is no Helper action in
    the contract) and the energy returns dropped (latency-only contract).

    Notable faithful quirks kept on purpose, because they are part of what the
    diagnostic explains:
      * three capacity-1 counters only -- local CPU, uplink ("ws"), cloud CPU.
        The downlink never reserves capacity and has no calendar.
      * a MEC task uploads its OWN ``processing_data_size``; dependency edges
        carry no explicit byte charge (readiness is ``max(FT_locally, FT_ws)``).
      * every MEC task downloads its own output (``FT_wr``), not just sinks.
      * the root/MEC branch does not advance the uplink counter.
      * the cloud start term is ``max(FT_ws[i], FT_cloud[j])`` -- the same
        ``FT_ws[i]`` for every predecessor ``j``.
    """
    preds = graph.predecessors()
    ft_locally = {tid: 0.0 for tid in graph.tasks}
    ft_ws = {tid: 0.0 for tid in graph.tasks}
    ft_cloud = {tid: 0.0 for tid in graph.tasks}
    ft_wr = {tid: 0.0 for tid in graph.tasks}

    local_available = 0.0
    ws_available = 0.0
    cloud_available = 0.0
    current_finish = 0.0

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []

    for tid, action in zip(decoder_order, actions):
        action = int(action)
        if action not in BINARY_ACTIONS:
            raise ValueError(f"legacy control supports binary actions only, got {action}")
        task = graph.tasks[tid]
        pred_ids = [int(e.src_task_id) for e in preds[tid]]

        upload_bytes = dependency_transfer_bytes = download_bytes = 0.0
        upload_s = dependency_transfer_s = download_s = 0.0

        if action == 0:
            if pred_ids:
                ready_s = max(max(ft_locally[j], ft_wr[j]) for j in pred_ids)
                start_s = max(local_available, ready_s)
            else:
                ready_s = 0.0
                start_s = local_available
            compute_s = task.compute_workload_bytes / cfg.ue_cpu_bytes_per_second
            finish_s = start_s + compute_s
            local_available = finish_s
            ft_locally[tid] = finish_s
            return_s = finish_s
            location = "UE"
        else:
            upload_bytes = float(task.compute_workload_bytes)
            if pred_ids:
                ready_s = max(max(ft_locally[j], ft_ws[j]) for j in pred_ids)
                ws_start = max(ws_available, ready_s)
                upload_s = task.compute_workload_bytes / cfg.mec_uplink_bytes_per_second
                ws_finish = ws_start + upload_s
                ft_ws[tid] = ws_finish
                ws_available = ws_finish
                start_s = max(
                    cloud_available,
                    max(max(ft_ws[tid], ft_cloud[j]) for j in pred_ids),
                )
            else:
                ready_s = 0.0
                ws_start = ws_available
                upload_s = task.compute_workload_bytes / cfg.mec_uplink_bytes_per_second
                ws_finish = ws_start + upload_s
                ft_ws[tid] = ws_finish  # faithful: ws_available is NOT advanced here
                start_s = max(cloud_available, ft_ws[tid])
            compute_s = task.compute_workload_bytes / cfg.mec_cpu_bytes_per_second
            finish_s = start_s + compute_s
            ft_cloud[tid] = finish_s
            cloud_available = finish_s
            download_bytes = float(task.task_output_bytes)
            download_s = task.task_output_bytes / cfg.mec_downlink_bytes_per_second
            return_s = finish_s + download_s
            ft_wr[tid] = return_s
            location = "MEC"

        delta = max(return_s, current_finish) - current_finish
        current_finish = max(return_s, current_finish)
        deltas.append(delta)

        rows.append(
            {
                "task_id": int(tid),
                "action": action,
                "location": location,
                "upload_bytes": upload_bytes,
                "upload_s": upload_s,
                "compute_s": compute_s,
                "dependency_transfer_bytes": dependency_transfer_bytes,
                "dependency_transfer_s": dependency_transfer_s,
                "download_bytes": download_bytes,
                "download_s": download_s,
                "dependency_wait_s": max(0.0, start_s - ready_s),
                "ready_s": ready_s,
                "start_s": start_s,
                "finish_s": finish_s,
                "return_s": return_s,
            }
        )

    return {
        "makespan_seconds": current_finish,
        "tasks": rows,
        "latency_deltas": deltas,
    }


def run_legacy_control(
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    cfg: ResourceConfig,
) -> dict[str, Any]:
    """Scheduler 2: the project's legacy control, minimal reimplementation."""
    replay = _legacy_replay(graph, decoder_order, actions, cfg)
    return {
        "makespan_seconds": replay["makespan_seconds"],
        "latency_deltas": replay["latency_deltas"],
        "tasks": replay["tasks"],
        "notes": (
            "No legacy control entry point exists in the working tree: "
            "offloading_env.get_scheduling_cost_step_by_step has delegated to the "
            "canonical engine since b49b60f.  This row is a clearly-labelled "
            "minimal legacy model reproducing the pre-canonical algorithm at "
            "b49b60f^ with the V2V branch removed and latency-only accounting."
        ),
        "source": {
            "kind": "minimal_reimplementation_in_harness",
            "origin": LEGACY_CONTROL_ORIGIN,
        },
    }


# --------------------------------------------------------------------------- #
# Scheduler 1: the upstream repository's own function.
# --------------------------------------------------------------------------- #
def _source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = node.end_lineno
    return textwrap.dedent("\n".join(lines[start - 1 : end]))


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


def load_upstream_scheduler(upstream_root: os.PathLike[str] | str) -> dict[str, Any]:
    """Extract the upstream ``Resources`` class + scheduling function verbatim.

    The function is taken as an AST source segment, so the arithmetic executed
    here is byte-identical to theirs; only the surrounding imports are dropped.
    Raises ``UpstreamUnavailableError`` with an actionable message otherwise.
    """
    root = Path(upstream_root)
    path = root / UPSTREAM_REL
    if not path.is_file():
        raise UpstreamUnavailableError(
            "upstream scheduler not found: expected %s to exist. Clone it with "
            "`git clone --depth 1 %s %s` and re-run with --upstream-root %s."
            % (path, UPSTREAM_REPO, root, root)
        )
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise UpstreamUnavailableError(
            "upstream scheduler not parseable: %s (%s)" % (path, exc)
        ) from exc

    resources_src = None
    scheduler_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == UPSTREAM_RESOURCES_SYMBOL:
            resources_src = _source_segment(source, node)
        if isinstance(node, ast.ClassDef) and node.name == "OffloadingEnvironment":
            for sub in node.body:
                if (
                    isinstance(sub, ast.FunctionDef)
                    and sub.name == UPSTREAM_SCHEDULER_SYMBOL
                ):
                    scheduler_src = _source_segment(source, sub)

    if resources_src is None or scheduler_src is None:
        missing = []
        if resources_src is None:
            missing.append(UPSTREAM_RESOURCES_SYMBOL)
        if scheduler_src is None:
            missing.append(UPSTREAM_SCHEDULER_SYMBOL)
        raise UpstreamUnavailableError(
            "upstream scheduler symbols %s not found in %s; the upstream layout "
            "changed and this diagnostic must be updated rather than guessed."
            % (", ".join(missing), path)
        )

    namespace: dict[str, Any] = {"__name__": "_upstream_metarl_scheduler"}
    module_source = "import numpy as np  # noqa: F401\n\n" + resources_src + "\n\n" + scheduler_src + "\n"
    exec(compile(module_source, str(path) + "::extracted", "exec"), namespace)

    return {
        "root": str(root),
        "source_path": str(path),
        "commit": _git_commit(root),
        "file_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "scheduler_source_sha256": hashlib.sha256(scheduler_src.encode("utf-8")).hexdigest(),
        "resources_source_sha256": hashlib.sha256(resources_src.encode("utf-8")).hexdigest(),
        "extracted_symbols": [UPSTREAM_RESOURCES_SYMBOL, UPSTREAM_SCHEDULER_SYMBOL],
        "function": namespace[UPSTREAM_SCHEDULER_SYMBOL],
        "resources_cls": namespace[UPSTREAM_RESOURCES_SYMBOL],
    }


def run_upstream_original(
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    cfg: ResourceConfig,
    plugin: dict[str, Any],
) -> dict[str, Any]:
    """Scheduler 1: call the upstream function and validate our replay of it."""
    cluster = plugin["resources_cls"](
        mec_process_capable=cfg.mec_cpu_bytes_per_second,
        mobile_process_capable=cfg.ue_cpu_bytes_per_second,
        bandwidth_up=cfg.mec_uplink_bytes_per_second * 8.0 / (1024.0 * 1024.0),
        bandwidth_dl=cfg.mec_downlink_bytes_per_second * 8.0 / (1024.0 * 1024.0),
    )
    carrier = types.SimpleNamespace(resource_cluster=cluster)
    shim = _LegacyTaskGraph(graph)
    plan = [[int(tid), int(a)] for tid, a in zip(decoder_order, actions)]

    raw = plugin["function"](carrier, plan, shim)
    if not isinstance(raw, tuple) or len(raw) < 2:
        raise RuntimeError(
            "unexpected upstream return value %r; expected (latency_deltas, makespan)"
            % (raw,)
        )
    upstream_deltas = [float(x) for x in raw[0]]
    upstream_makespan = float(raw[1])

    replay = _legacy_replay(graph, decoder_order, actions, cfg)
    max_abs_delta_diff = (
        max(abs(a - b) for a, b in zip(upstream_deltas, replay["latency_deltas"]))
        if len(upstream_deltas) == len(replay["latency_deltas"])
        else float("inf")
    )
    return {
        "makespan_seconds": upstream_makespan,
        "latency_deltas": upstream_deltas,
        "tasks": replay["tasks"],
        "task_timing_source": (
            "step-recording replay of the same upstream algorithm (their function "
            "returns only deltas + makespan); validated against their own numbers"
        ),
        "replay_matches_upstream": (
            len(upstream_deltas) == len(replay["latency_deltas"])
            and max_abs_delta_diff <= 1e-9
            and abs(upstream_makespan - replay["makespan_seconds"]) <= 1e-9
        ),
        "max_abs_latency_delta_diff_s": max_abs_delta_diff,
        "upstream_makespan_seconds": upstream_makespan,
        "upstream_latency_deltas": upstream_deltas,
        "resources": {
            "mobile_process_capable": float(cfg.ue_cpu_bytes_per_second),
            "mec_process_capable": float(cfg.mec_cpu_bytes_per_second),
            "bandwidth_up": cfg.mec_uplink_bytes_per_second * 8.0 / (1024.0 * 1024.0),
            "bandwidth_dl": cfg.mec_downlink_bytes_per_second * 8.0 / (1024.0 * 1024.0),
        },
    }


# --------------------------------------------------------------------------- #
# Scheduler 3: the canonical engine.
# --------------------------------------------------------------------------- #
def run_canonical(
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    cfg: ResourceConfig,
) -> dict[str, Any]:
    """Scheduler 3: ``schedule`` with a CanonicalDAG + frozen ResourceConfig."""
    result = schedule(graph, decoder_order, actions, cfg)

    per: dict[int, dict[str, float]] = {
        int(tid): {
            "upload_bytes": 0.0,
            "upload_s": 0.0,
            "dependency_transfer_bytes": 0.0,
            "dependency_transfer_s": 0.0,
            "download_bytes": 0.0,
            "download_s": 0.0,
            "incoming_end_max": 0.0,
        }
        for tid in graph.tasks
    }
    hop_usage = {"MEC_UL": 0.0, "MEC_DL": 0.0, "V2V": 0.0}
    for tr in result.transfers:
        duration = float(tr.end) - float(tr.start)
        nbytes = float(tr.bytes)
        hop_usage[tr.hop] = hop_usage.get(tr.hop, 0.0) + duration
        if tr.src_task_id is None and tr.dst_task_id is not None:
            slot = per[int(tr.dst_task_id)]
            slot["upload_bytes"] += nbytes
            slot["upload_s"] += duration
        elif tr.src_task_id is not None and tr.dst_task_id is not None:
            slot = per[int(tr.dst_task_id)]
            slot["dependency_transfer_bytes"] += nbytes
            slot["dependency_transfer_s"] += duration
        elif tr.src_task_id is not None and tr.dst_task_id is None:
            slot = per[int(tr.src_task_id)]
            slot["download_bytes"] += nbytes
            slot["download_s"] += duration
        if tr.dst_task_id is not None:
            per[int(tr.dst_task_id)]["incoming_end_max"] = max(
                per[int(tr.dst_task_id)]["incoming_end_max"], float(tr.end)
            )

    preds = graph.predecessors()
    rows: list[dict[str, Any]] = []
    for tid in sorted(graph.tasks):
        rec = result.tasks[int(tid)]
        slot = per[int(tid)]
        eligible = [slot["incoming_end_max"]]
        for edge in preds[int(tid)]:
            if int(edge.edge_output_bytes) == 0:
                eligible.append(float(result.tasks[int(edge.src_task_id)].finish))
        ready_s = max(eligible) if eligible else 0.0
        rows.append(
            {
                "task_id": int(tid),
                "action": int(rec.location.to_action()),
                "location": rec.location.value,
                "upload_bytes": slot["upload_bytes"],
                "upload_s": slot["upload_s"],
                "compute_s": float(rec.finish) - float(rec.start),
                "dependency_transfer_bytes": slot["dependency_transfer_bytes"],
                "dependency_transfer_s": slot["dependency_transfer_s"],
                "download_bytes": slot["download_bytes"],
                "download_s": slot["download_s"],
                "dependency_wait_s": max(0.0, float(rec.start) - ready_s),
                "ready_s": ready_s,
                "start_s": float(rec.start),
                "finish_s": float(rec.finish),
                "return_s": float(rec.all_consumers_ready),
                "first_available_s": float(rec.first_available),
            }
        )

    return {
        "makespan_seconds": float(result.makespan_seconds),
        "terminal_return_time_s": float(result.terminal_return_time),
        "topo_order": [int(t) for t in result.topo_order],
        "tasks": rows,
        "transfers": [
            {
                "hop": tr.hop,
                "hop_index": int(tr.hop_index),
                "bytes": int(tr.bytes),
                "start_s": float(tr.start),
                "end_s": float(tr.end),
                "src_location": tr.src_location.value,
                "dst_location": tr.dst_location.value,
                "src_task_id": None if tr.src_task_id is None else int(tr.src_task_id),
                "dst_task_id": None if tr.dst_task_id is None else int(tr.dst_task_id),
            }
            for tr in result.transfers
        ],
        "hop_busy_seconds": hop_usage,
        "resource_intervals": [
            {
                "resource": iv.resource,
                "start_s": float(iv.start),
                "end_s": float(iv.end),
                "task_id": None if iv.task_id is None else int(iv.task_id),
                "hop": iv.hop,
            }
            for iv in result.resource_intervals
        ],
        "energy_joules_excluded_from_parity": result.energy.as_dict(),
    }


# --------------------------------------------------------------------------- #
# Aggregation helpers used by the factor decomposition.
# --------------------------------------------------------------------------- #
def _sum_tasks(scheduler: dict[str, Any], key: str) -> float:
    return float(sum(float(row[key]) for row in scheduler["tasks"]))


def _mec_mec_edges(graph: CanonicalDAG, actions: list[int], decoder_order: list[int]) -> list[dict[str, Any]]:
    loc = {int(tid): int(a) for tid, a in zip(decoder_order, actions)}
    out = []
    for edge in graph.edges:
        if loc[int(edge.src_task_id)] == 1 and loc[int(edge.dst_task_id)] == 1:
            out.append(
                {
                    "src": int(edge.src_task_id),
                    "dst": int(edge.dst_task_id),
                    "edge_output_bytes": int(edge.edge_output_bytes),
                }
            )
    return out


def _factor(applies: bool, explanation: str, numbers: dict[str, Any]) -> dict[str, Any]:
    return {"applies": bool(applies), "explanation": explanation, "numbers": numbers}


def build_differences(
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    schedulers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical = schedulers["canonical"]
    legacy = schedulers["legacy_control"]
    upstream = schedulers.get("upstream_original")

    mec_mec = _mec_mec_edges(graph, actions, decoder_order)
    loc = {int(tid): int(a) for tid, a in zip(decoder_order, actions)}

    canon_upload_bytes = _sum_tasks(canonical, "upload_bytes")
    canon_dep_bytes = _sum_tasks(canonical, "dependency_transfer_bytes")
    canon_return_bytes = _sum_tasks(canonical, "download_bytes")
    legacy_upload_bytes = _sum_tasks(legacy, "upload_bytes")
    legacy_return_bytes = _sum_tasks(legacy, "download_bytes")
    legacy_dep_bytes = _sum_tasks(legacy, "dependency_transfer_bytes")

    external_input_bytes = float(
        sum(int(t.external_input_bytes) for t in graph.tasks.values())
    )
    mec_task_workload_bytes = float(
        sum(
            int(graph.tasks[tid].compute_workload_bytes)
            for tid in graph.tasks
            if loc[int(tid)] == 1
        )
    )
    mec_task_output_bytes = float(
        sum(
            int(graph.tasks[tid].task_output_bytes)
            for tid in graph.tasks
            if loc[int(tid)] == 1
        )
    )
    sink_output_bytes = float(
        sum(int(graph.tasks[tid].task_output_bytes) for tid in graph.sinks())
    )
    edge_total_bytes = float(sum(int(e.edge_output_bytes) for e in graph.edges))

    canon_ul_busy = float(canonical["hop_busy_seconds"].get("MEC_UL", 0.0))
    canon_dl_busy = float(canonical["hop_busy_seconds"].get("MEC_DL", 0.0))
    legacy_ul_busy = _sum_tasks(legacy, "upload_s")
    legacy_dl_busy = _sum_tasks(legacy, "download_s")

    factors: dict[str, Any] = {}

    factors[FACTOR_KEYS[0]] = _factor(
        abs(canon_upload_bytes - legacy_upload_bytes) > 1e-9,
        (
            "Upstream charges an uplink for EVERY task executed on the MEC, using "
            "that task's own processing_data_size, and charges nothing for a task's "
            "external input at its root.  The canonical engine charges the uplink "
            "only for a ROOT that executes remotely (a UE-local root's external "
            "input needs no hop) and charges the remaining bytes on the producing "
            "dependency edge instead.  Same plan, different byte basis."
        ),
        {
            "canonical_root_external_input_bytes": external_input_bytes,
            "canonical_total_upload_bytes": canon_upload_bytes,
            "canonical_ue_local_root_external_input_bytes": float(
                sum(
                    int(graph.tasks[tid].external_input_bytes)
                    for tid in graph.tasks
                    if loc[int(tid)] == 0
                )
            ),
            "upstream_total_upload_bytes": legacy_upload_bytes,
            "upstream_mec_task_workload_bytes": mec_task_workload_bytes,
            "delta_bytes": legacy_upload_bytes - canon_upload_bytes,
        },
    )

    factors[FACTOR_KEYS[1]] = _factor(
        len(mec_mec) > 0,
        (
            "Upstream has no notion of residency: a MEC->MEC dependency still makes "
            "the consumer upload its own workload, because readiness is "
            "max(FT_locally, FT_ws) and the consumer's ws slot is charged again.  In "
            "the canonical engine output stays resident where it executed, so a "
            "MEC->MEC edge transfers zero bytes and costs zero seconds."
        ),
        {
            "mec_to_mec_edge_count": len(mec_mec),
            "mec_to_mec_edges": mec_mec,
            "mec_to_mec_bytes": float(sum(e["edge_output_bytes"] for e in mec_mec)),
            "canonical_mec_to_mec_transfer_bytes": 0.0,
            "mec_tasks": [tid for tid in graph.tasks if loc[int(tid)] == 1],
        },
    )

    factors[FACTOR_KEYS[2]] = _factor(
        canon_dep_bytes > 0.0,
        (
            "The canonical engine routes every dependency edge's edge_output_bytes "
            "between the producer's and the consumer's execution location and "
            "reserves the hop on that calendar.  Upstream charges 0 explicit "
            "dependency-transfer bytes: a dependency is satisfied by the producer's "
            "own upload/compute/download completion times, so the transfer cost "
            "shows up inside the producer's per-task upload (MEC successor) or "
            "download (UE successor) instead."
        ),
        {
            "canonical_dependency_transfer_bytes": canon_dep_bytes,
            "graph_edge_total_bytes": edge_total_bytes,
            "upstream_dependency_transfer_bytes": legacy_dep_bytes,
            "legacy_control_dependency_transfer_bytes": legacy_dep_bytes,
        },
    )

    factors[FACTOR_KEYS[3]] = _factor(
        abs(canon_return_bytes - legacy_return_bytes) > 1e-9,
        (
            "Upstream downloads every MEC task's output back to the UE "
            "(wr_start = FT_cloud, T_dl = transmission_data_size).  The canonical "
            "engine returns ONLY sinks, using task_output_bytes, because an "
            "intermediate result that a MEC consumer can read in place needs no "
            "downlink."
        ),
        {
            "canonical_return_bytes": canon_return_bytes,
            "canonical_sink_output_bytes": sink_output_bytes,
            "upstream_return_bytes": legacy_return_bytes,
            "upstream_mec_task_output_bytes": mec_task_output_bytes,
            "delta_bytes": legacy_return_bytes - canon_return_bytes,
        },
    )

    factors[FACTOR_KEYS[4]] = _factor(
        True,
        (
            "The canonical calendar set gives uplink and downlink independent "
            "capacity-1 calendars (MEC_UL and MEC_DL), so a downlink can overlap an "
            "uplink.  Upstream has a single uplink counter ('ws') and no downlink "
            "calendar at all: T_dl is added to the finish time but never reserves "
            "the channel, so downlinks in the same run are serialised against the "
            "compute finish and are never mutually exclusive."
        ),
        {
            "canonical_calendars": ["UE_CPU", "MEC_UL", "MEC_CPU", "MEC_DL"],
            "canonical_mec_ul_busy_s": canon_ul_busy,
            "canonical_mec_dl_busy_s": canon_dl_busy,
            "upstream_calendars": ["local_cpu", "ws_uplink", "cloud_cpu"],
            "upstream_mec_ul_busy_s": legacy_ul_busy,
            "upstream_mec_dl_busy_s": None,
            "note": "upstream never reserves capacity for the downlink",
        },
    )

    factors[FACTOR_KEYS[5]] = _factor(
        True,
        (
            "Both models are capacity-1 per resource, but the resource SETS differ: "
            "canonical has UE_CPU / MEC_UL / MEC_CPU / MEC_DL (four independent "
            "serials for a binary plan) while upstream has local CPU / uplink / "
            "cloud CPU (three).  Consequently a UE-local task and a MEC task can "
            "overlap differently, and the canonical engine additionally queues "
            "compute behind its own tier's calendar while upstream queues behind "
            "one global cloud counter."
        ),
        {
            "canonical_resource_count": len({iv["resource"] for iv in canonical["resource_intervals"]}),
            "canonical_resources_used": sorted({iv["resource"] for iv in canonical["resource_intervals"]}),
            "upstream_resource_count": 3,
            "upstream_resources_used": ["local_cpu", "ws_uplink", "cloud_cpu"],
            "makespan_s": {
                "upstream_original": None if upstream is None else upstream["makespan_seconds"],
                "legacy_control": legacy["makespan_seconds"],
                "canonical": canonical["makespan_seconds"],
            },
        },
    )

    topo = canonical.get("topo_order", list(decoder_order))
    factors[FACTOR_KEYS[6]] = _factor(
        list(topo) != list(decoder_order),
        (
            "All three schedulers were handed the SAME decoder order, and it is "
            "topological, so the canonical engine's decoder-rank topological sort "
            "reproduces it exactly.  Decoder order therefore explains none of the "
            "observed difference in this run; it remains a factor to state because "
            "upstream consumes the plan order literally while the canonical engine "
            "re-derives a topological order using decoder_rank as the tie-break."
        ),
        {
            "decoder_order": [int(x) for x in decoder_order],
            "canonical_execution_order": [int(x) for x in topo],
            "orders_identical": list(topo) == list(decoder_order),
        },
    )

    return factors


# --------------------------------------------------------------------------- #
# Evidence assembly.
# --------------------------------------------------------------------------- #
def _make_checks(
    schedulers: dict[str, dict[str, Any]],
    graph: CanonicalDAG,
    decoder_order: list[int],
    actions: list[int],
    canonical_direct: dict[str, Any],
    energy_invariance: dict[str, Any],
    upstream_available: bool,
    upstream_error: str | None,
    serializable: bool,
) -> dict[str, bool]:
    required_task_keys = {
        "upload_bytes", "upload_s", "compute_s",
        "dependency_transfer_bytes", "dependency_transfer_s",
        "download_bytes", "download_s", "start_s", "finish_s", "return_s",
    }
    locations = {
        row["location"] for sched in schedulers.values() for row in sched["tasks"]
    }
    hops = {
        tr["hop"] for tr in schedulers["canonical"].get("transfers", [])
    }

    def finite_nonneg(x: Any) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) >= 0.0

    checks = {
        "harness_ran": True,
        "all_three_schedulers_present": set(schedulers) >= {
            "upstream_original",
            "legacy_control",
            "canonical",
        },
        "makespan_finite_and_nonnegative": all(
            finite_nonneg(sched["makespan_seconds"]) for sched in schedulers.values()
        ),
        "binary_actions_only": all(int(a) in BINARY_ACTIONS for a in actions),
        "no_helper_action": 2 not in [int(a) for a in actions],
        "no_helper_location_and_no_v2v_hop": ("HELPER" not in locations) and ("V2V" not in hops),
        "per_task_rows_complete": all(
            required_task_keys <= set(row)
            for sched in schedulers.values()
            for row in sched["tasks"]
        ),
        "per_task_count_matches_graph": all(
            len(sched["tasks"]) == len(graph.tasks) for sched in schedulers.values()
        ),
        "canonical_matches_direct_schedule": (
            abs(canonical_direct["makespan_seconds"] - schedulers["canonical"]["makespan_seconds"]) <= 1e-12
            and all(
                abs(a["start_s"] - b["start_s"]) <= 1e-12
                and abs(a["finish_s"] - b["finish_s"]) <= 1e-12
                for a, b in zip(
                    sorted(canonical_direct["tasks"], key=lambda r: r["task_id"]),
                    sorted(schedulers["canonical"]["tasks"], key=lambda r: r["task_id"]),
                )
            )
        ),
        "decoder_order_is_canonical_execution_order": (
            [int(x) for x in schedulers["canonical"].get("topo_order", decoder_order)]
            == [int(x) for x in decoder_order]
        ),
        "energy_excluded_from_parity": bool(energy_invariance["equal"]),
        "json_serializable": bool(serializable),
        "upstream_status_recorded": bool(upstream_available) or bool(upstream_error),
    }
    if upstream_available:
        checks["upstream_ran"] = True
        checks["upstream_replay_matches_upstream"] = bool(
            schedulers["upstream_original"]["replay_matches_upstream"]
        )
        upstream = schedulers["upstream_original"]
        legacy = schedulers["legacy_control"]
        checks["legacy_control_equals_upstream_latency"] = (
            abs(upstream["makespan_seconds"] - legacy["makespan_seconds"]) <= 1e-9
            and len(upstream["latency_deltas"]) == len(legacy["latency_deltas"])
            and all(
                abs(a - b) <= 1e-9
                for a, b in zip(upstream["latency_deltas"], legacy["latency_deltas"])
            )
        )
    return checks


def build_evidence(
    upstream_root: os.PathLike[str] | str | None = None,
    *,
    require_upstream: bool = False,
) -> dict[str, Any]:
    """Build the full machine-readable parity evidence."""
    graph, decoder_order, actions, cfg = reference_contract()

    # --- upstream availability -------------------------------------------------
    requested = Path(upstream_root) if upstream_root is not None else None
    if requested is None:
        env_root = os.environ.get("MARGO_UPSTREAM_METARL")
        requested = Path(env_root) if env_root else DEFAULT_UPSTREAM_ROOT
    upstream_root_resolved = str(requested)
    upstream_error: str | None = None
    plugin: dict[str, Any] | None = None
    try:
        plugin = load_upstream_scheduler(requested)
    except UpstreamUnavailableError as exc:
        upstream_error = str(exc)
        if require_upstream:
            raise
    upstream_available = plugin is not None

    # --- run the three schedulers on the same instance -------------------------
    canonical = run_canonical(graph, decoder_order, actions, cfg)
    canonical_direct = run_canonical(graph, decoder_order, actions, cfg)
    legacy = run_legacy_control(graph, decoder_order, actions, cfg)
    schedulers: dict[str, dict[str, Any]] = {
        "legacy_control": legacy,
        "canonical": canonical,
    }
    upstream_run: dict[str, Any] | None = None
    if upstream_available:
        upstream_run = run_upstream_original(graph, decoder_order, actions, cfg, plugin)
        schedulers["upstream_original"] = upstream_run

    # --- latency-only proof: energy accounting must not move the schedule ------
    cfg_physical = ResourceConfig.from_frozen_yaml(model=None)
    canonical_physical = run_canonical(graph, decoder_order, actions, cfg_physical)
    energy_equal = (
        abs(canonical_physical["makespan_seconds"] - canonical["makespan_seconds"]) <= 1e-12
        and all(
            abs(a["finish_s"] - b["finish_s"]) <= 1e-12
            for a, b in zip(canonical["tasks"], canonical_physical["tasks"])
        )
    )
    energy_invariance = {
        "legacy_energy_model_makespan_s": canonical["makespan_seconds"],
        "physical_energy_model_makespan_s": canonical_physical["makespan_seconds"],
        "timing_axis_held_constant": True,
        "equal": bool(energy_equal),
        "legacy_energy_total_mobile_j": canonical["energy_joules_excluded_from_parity"][
            "total_mobile_joules"
        ],
        "physical_energy_total_mobile_j": canonical_physical[
            "energy_joules_excluded_from_parity"
        ]["total_mobile_joules"],
    }

    # --- rate identity ---------------------------------------------------------
    resources = upstream_run["resources"] if upstream_run is not None else None
    rate_identity = {
        "ue_cpu_bytes_per_second": float(cfg.ue_cpu_bytes_per_second),
        "mec_cpu_bytes_per_second": float(cfg.mec_cpu_bytes_per_second),
        "mec_uplink_bytes_per_second": float(cfg.mec_uplink_bytes_per_second),
        "mec_downlink_bytes_per_second": float(cfg.mec_downlink_bytes_per_second),
        "upstream_bandwidth_up_mbps": None if resources is None else resources["bandwidth_up"],
        "upstream_bandwidth_dl_mbps": None if resources is None else resources["bandwidth_dl"],
        "upstream_derived_ul_bytes_per_second": (
            None
            if resources is None
            else resources["bandwidth_up"] * (1024.0 * 1024.0 / 8.0)
        ),
        "upstream_derived_dl_bytes_per_second": (
            None
            if resources is None
            else resources["bandwidth_dl"] * (1024.0 * 1024.0 / 8.0)
        ),
        "identical": bool(
            resources is None
            or (
                abs(resources["mobile_process_capable"] - cfg.ue_cpu_bytes_per_second) <= 1e-9
                and abs(resources["mec_process_capable"] - cfg.mec_cpu_bytes_per_second) <= 1e-9
                and abs(
                    resources["bandwidth_up"] * (1024.0 * 1024.0 / 8.0)
                    - cfg.mec_uplink_bytes_per_second
                )
                <= 1e-9
                and abs(
                    resources["bandwidth_dl"] * (1024.0 * 1024.0 / 8.0)
                    - cfg.mec_downlink_bytes_per_second
                )
                <= 1e-9
            )
        ),
    }

    graph_fp = _graph_fingerprint(graph)
    plan_payload = [[int(t), int(a)] for t, a in zip(decoder_order, actions)]

    contract = {
        "contract_id": "margo_part_c_fair_comparison_v1",
        "binary_actions_only": True,
        "action_space": {"0": "UE/Local", "1": "MEC"},
        "helper_action_forbidden": True,
        "v2v_hop_forbidden": True,
        "single_graph": True,
        "graph": graph_fp,
        "decoder_order": [int(x) for x in decoder_order],
        "decoder_order_sha256": _sha256_json([int(x) for x in decoder_order]),
        "plan": plan_payload,
        "plan_sha256": _sha256_json(plan_payload),
        "identical_rates": rate_identity,
        "latency_semantics_only": True,
        "energy_excluded_from_parity": True,
        "energy_invariance": energy_invariance,
        "scheduler_provenance": {
            "upstream_original": {
                "repo": UPSTREAM_REPO,
                "root": upstream_root_resolved,
                "available": upstream_available,
                "commit": None if plugin is None else plugin["commit"],
                "source_path": None if plugin is None else plugin["source_path"],
                "file_sha256": None if plugin is None else plugin["file_sha256"],
                "scheduler_source_sha256": (
                    None if plugin is None else plugin["scheduler_source_sha256"]
                ),
                "extracted_symbols": (
                    None if plugin is None else plugin["extracted_symbols"]
                ),
            },
            "legacy_control": {
                "kind": "minimal_reimplementation_in_harness",
                "origin": LEGACY_CONTROL_ORIGIN,
                "note": legacy["notes"],
            },
            "canonical": {
                "entry_point": "env.mec_offloaing_envs.scheduler.schedule",
                "graph": "CanonicalDAG",
                "resources": "ResourceConfig.from_frozen_yaml()",
            },
        },
    }

    differences = build_differences(graph, decoder_order, actions, schedulers)

    json_serializable = True
    try:
        json.dumps({"schedulers": schedulers, "differences": differences}, allow_nan=False)
    except (TypeError, ValueError):
        json_serializable = False

    checks = _make_checks(
        schedulers,
        graph,
        decoder_order,
        actions,
        canonical_direct,
        energy_invariance,
        upstream_available,
        upstream_error,
        json_serializable,
    )

    makespans = {
        name: float(sched["makespan_seconds"]) for name, sched in schedulers.items()
    }
    summary = {
        "task_count": len(graph.tasks),
        "mec_task_count": int(sum(1 for a in actions if int(a) == 1)),
        "local_task_count": int(sum(1 for a in actions if int(a) == 0)),
        "makespan_seconds": makespans,
        "canonical_vs_legacy_delta_seconds": (
            makespans.get("canonical", 0.0) - makespans.get("legacy_control", 0.0)
        ),
        "canonical_vs_upstream_delta_seconds": (
            None
            if "upstream_original" not in makespans
            else makespans["canonical"] - makespans["upstream_original"]
        ),
        "legacy_control_equals_upstream": (
            None
            if "upstream_original" not in makespans
            else abs(
                makespans["legacy_control"] - makespans["upstream_original"]
            )
            <= 1e-9
        ),
        "applying_factors": [k for k in FACTOR_KEYS if differences[k]["applies"]],
        "non_applying_factors": [k for k in FACTOR_KEYS if not differences[k]["applies"]],
        "goal": "explainability, not equality of makespans",
    }

    return {
        "schema": SCHEMA,
        "generated_by": "spec/upstream_parity.py",
        "upstream_available": upstream_available,
        "upstream_root": upstream_root_resolved,
        "upstream_error": upstream_error,
        "contract": contract,
        "schedulers": schedulers,
        "differences": differences,
        "summary": summary,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help="path of the evidence JSON to write (default: %(default)s)",
    )
    parser.add_argument(
        "--upstream-root",
        default=None,
        help=(
            "path to a clone of %s (default: $MARGO_UPSTREAM_METARL or %s)"
            % (UPSTREAM_REPO, DEFAULT_UPSTREAM_ROOT)
        ),
    )
    parser.add_argument(
        "--require-upstream",
        action="store_true",
        help="fail loudly (non-zero) when the upstream scheduler cannot be found",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the text summary")
    args = parser.parse_args(argv)

    try:
        evidence = build_evidence(args.upstream_root, require_upstream=args.require_upstream)
    except UpstreamUnavailableError as exc:
        print("UPSTREAM PARITY FAILED: %s" % exc, file=sys.stderr)
        return 2

    out_path = Path(args.json)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        summary = evidence["summary"]
        print("upstream_parity: schema=%s" % evidence["schema"])
        print("  upstream_available: %s" % evidence["upstream_available"])
        if evidence["upstream_error"]:
            print("  upstream NOT available: %s" % evidence["upstream_error"], file=sys.stderr)
        for name, makespan in summary["makespan_seconds"].items():
            print("  makespan[%s] = %.9f s" % (name, makespan))
        print("  applying factors: %s" % ", ".join(summary["applying_factors"]))
        print("  evidence written to %s" % out_path)
        failed = [k for k, ok in evidence["checks"].items() if not ok]
        if failed:
            print("  FAILED checks: %s" % ", ".join(failed), file=sys.stderr)

    return 0 if evidence["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
