"""Deterministic identity for energy references (4.2a, part C).

A reference-ranges cache may not be keyed by `id(task_graph)`: that is not stable
across processes (parallel workers), not stable across equal graphs, and blind to
the parameters that actually change the numbers. This module builds a canonical
key from everything that can change a reference result.

Included, deliberately:
  * the SCHEDULING-relevant content of the graph: task ids, compute/output/
    external bytes, per-task cycles_per_bit, edges and edge bytes, and the
    decoder order (the order is part of how bounds and pure plans are built);
  * reference_mode, energy_scope, panel_max_passes and a hash of panel_extra,
    because all four change the panel or the selected boundary;
  * the resolved scheduler fingerprint, which already covers the timing/
    accounting axes, both model names, every rate and power coefficient, the
    physical tier and radio parameters and the source config hash.

Deliberately NOT included: deadline fields. Reference ranges are built from the
pure plans (all-UE/all-MEC/all-HELPER) plus an optional greedy panel, and neither
their makespan nor their energy depends on a deadline; the deadline only affects
the tardiness terms of the objective, which are not part of ReferenceRanges. If
that ever changes, the schema version must change with it.

No object ids, no temporary paths, no unstable repr.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, MutableMapping, Mapping, Sequence

from .energy_model import SCOPE_SYSTEM
from .energy_scope import ENERGY_SCOPES
from .resources import resolved_config_sha256

REFERENCE_SCHEMA_VERSION = "energy_reference_ranges_v2"
GRAPH_FINGERPRINT_VERSION = "scheduling_graph_v1"


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return number


def _canonical_graph(graph: Any) -> Any:
    """Content view of either a CanonicalDAG or a legacy OffloadingTaskGraph."""
    tasks = getattr(graph, "tasks", None)
    if isinstance(tasks, Mapping) and tasks and getattr(graph, "edges", None) is not None:
        return graph
    from .adapter import to_canonical_dag

    try:
        return to_canonical_dag(graph)
    except Exception as exc:  # legacy graph without the adapter's required attrs
        raise ValueError(
            "graph is neither a CanonicalDAG nor a legacy task graph: %r" % (exc,)
        ) from exc


def scheduling_graph_fingerprint(graph: Any, order: Sequence[int]) -> str:
    """Content hash of everything in the graph that a reference result depends on."""
    graph = _canonical_graph(graph)
    tasks = getattr(graph, "tasks", None)
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("graph must expose a non-empty `tasks` mapping")
    task_rows = []
    for tid in sorted(int(t) for t in tasks):
        task = tasks[tid]
        cycles = getattr(task, "cycles_per_bit", None)
        task_rows.append(
            {
                "task_id": tid,
                "compute_workload_bytes": int(task.compute_workload_bytes),
                "task_output_bytes": int(task.task_output_bytes),
                "external_input_bytes": int(task.external_input_bytes),
                "cycles_per_bit": None if cycles is None else _finite(cycles, "cycles_per_bit"),
            }
        )
    edges = [
        [
            int(edge.src_task_id),
            int(edge.dst_task_id),
            int(edge.edge_output_bytes),
        ]
        for edge in graph.edges
    ]
    edges.sort()
    order = [int(t) for t in order]
    if sorted(order) != sorted(task_rows[i]["task_id"] for i in range(len(task_rows))):
        raise ValueError("order must be a permutation of the graph tasks")
    blob = {
        "schema": GRAPH_FINGERPRINT_VERSION,
        "tasks": task_rows,
        "edges": edges,
        "decoder_order": order,
    }
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reference_ranges_cache_key(
    *,
    graph: Any,
    order: Sequence[int],
    reference_mode: str,
    energy_scope: str,
    resources: Any,
    panel_max_passes: int = 0,
    panel_extra: Any = None,
    schema_version: str = REFERENCE_SCHEMA_VERSION,
) -> str:
    """Canonical cache key for reference ranges. Any input change -> new key."""
    if energy_scope not in ENERGY_SCOPES:
        raise ValueError(
            "energy_scope must be one of %s, got %r" % (list(ENERGY_SCOPES), energy_scope)
        )
    if not str(reference_mode):
        raise ValueError("reference_mode is required")
    extra = panel_extra or {}
    extra_text = json.dumps(extra, sort_keys=True, separators=(",", ":"), default=str)
    blob = {
        "schema_version": schema_version,
        "graph_fingerprint": scheduling_graph_fingerprint(graph, order),
        "reference_mode": str(reference_mode),
        "energy_scope": str(energy_scope),
        "panel_max_passes": int(panel_max_passes),
        "panel_extra_sha256": hashlib.sha256(extra_text.encode("utf-8")).hexdigest(),
        "scheduler_config_sha256": resolved_config_sha256(resources),
        "energy_model": str(getattr(getattr(resources, "energy_model", None), "model", "") or ""),
        "radio_model": str(getattr(getattr(resources, "radio_model", None), "model", "") or ""),
    }
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def primary_scope_of(resources: Any) -> str:
    """The declared primary boundary, or a loud failure -- never a guess."""
    scope = str(getattr(resources, "energy_scope", "") or "")
    if scope not in ENERGY_SCOPES:
        raise ValueError(
            "resources declare no usable energy_scope (got %r); refusing to assume %r"
            % (scope, SCOPE_SYSTEM)
        )
    return scope


def get_or_build_reference_ranges(
    cache: MutableMapping[str, Any],
    *,
    graph: Any,
    order: Sequence[int],
    reference_mode: str,
    energy_scope: str,
    resources: Any,
    panel_max_passes: int = 0,
    panel_extra: Any = None,
) -> Any:
    """Cache-resolved SCOPED reference ranges, with a fail-loud hit guard.

    The key binds graph content, reference mode, boundary, panel and the resolved
    scheduler fingerprint, so a hit is exactly the object the caller asked for.
    We still re-validate the declared metadata on a hit: a mutated or poisoned
    cache must raise, never return a differently-scoped reference silently.
    """
    from .energy_api import compute_scoped_reference_ranges
    from .energy_scope import require_reference_scope

    key = reference_ranges_cache_key(
        graph=graph,
        order=order,
        reference_mode=reference_mode,
        energy_scope=energy_scope,
        resources=resources,
        panel_max_passes=panel_max_passes,
        panel_extra=panel_extra,
    )
    hit = cache.get(key)
    if hit is None:
        hit = compute_scoped_reference_ranges(
            graph,
            resources,
            energy_scope=energy_scope,
            mode=reference_mode,
            panel_extra=panel_extra,
            panel_max_passes=panel_max_passes,
        )
        cache[key] = hit
        return hit
    require_reference_scope(
        hit,
        expected_scope=energy_scope,
        expected_scheduler_config_sha256=resolved_config_sha256(resources),
    )
    return hit
