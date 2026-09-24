"""deadline_regime_v1 — versioned sidecar schema, generator (EFT/LFT) and stamper.

Why a sidecar: the `.gv` dataset is frozen and hash-pinned, and the regression
runs must stay byte-exact. `to_canonical_dag` already reads four OPTIONAL
per-task attributes (`deadline_s`, `deadline_type`, `criticality_class`,
`tardiness_weight`), and both consumers -- the v3 observation block and the
scheduling engine -- go through it on the same task objects. Stamping those
attributes once, right after the graph is parsed, is therefore the single choke
point that reaches everything, and regime `none` simply stamps nothing.

Construction (all on the admissible per-action bounds in `static_bounds.py`):

    graph_lb   = max_i min_ready_lb[i]                      (admissible makespan LB)
    D_G        = kappa * graph_lb                           (graph budget)
    EFT_i      = min_a finish_lb[i][a]                      (sinks: ready_lb basis)
    dur_i      = EFT_i - max_p EFT_p                        (min compute duration)
    LFT_i      = min(D_G, min_{j in succ(i)} (LFT_j - dur_j))   (backward pass)
    d_i        = EFT_i + alpha * (LFT_i - EFT_i)

`alpha = 0` gives every task its own optimistic finish (maximum per-task
tightness); `alpha = 1` gives it the LATEST finish that still leaves room for its
successors inside the graph budget (classic LFT). Only the sinks reach `D_G`.
When the budget is below the relaxation's own lower bound the deadlines can fall
to zero or below; they are then clamped to a small positive floor and counted in
`bounds["clamped_tasks"]`, and it is the witness step -- not this module -- that
decides the graph is infeasible.

`LFT_i < EFT_i` means the graph cannot meet this budget on the relaxation; that
is recorded, and the witness step (see `witness.py`) decides whether the graph is
really schedulable. Feasibility of a RELAXATION is never treated as feasibility.

Sink deadlines use the `ready_lb` basis (return hop included) because the engine
measures a miss on `all_consumers_ready`, never on compute finish
(`engine.py:219-237`).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace as _dc_replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .model import CRITICALITY_CLASSES, DEADLINE_TYPES, CanonicalDAG, Location
from .resources import ResourceConfig
from .static_bounds import static_action_bounds

SCHEMA_VERSION = "deadline_regime_v1"
GENERATOR_VERSION = "deadline_regime_generator_v1"
CONSTRUCTION = "eft_lft_interpolation"

# keep in sync with model.DEADLINE_TYPES; spelled out so the sidecar can be read
# without importing the scheduler
ALLOWED_DEADLINE_TYPES = tuple(DEADLINE_TYPES)


class DeadlineRegimeError(ValueError):
    """Invalid sidecar, invalid graph, or an unsafe stamp."""


# --------------------------------------------------------------------------- #
# per-task / per-graph payload
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskDeadline:
    deadline_s: float
    deadline_type: str
    criticality_class: str
    tardiness_weight: float

    def __post_init__(self) -> None:
        value = float(self.deadline_s)
        if not math.isfinite(value) or value <= 0.0:
            raise DeadlineRegimeError(
                "deadline_s must be finite and positive, got %r" % (self.deadline_s,)
            )
        if self.deadline_type not in ALLOWED_DEADLINE_TYPES:
            raise DeadlineRegimeError(
                "deadline_type must be one of %s, got %r"
                % (", ".join(ALLOWED_DEADLINE_TYPES), self.deadline_type)
            )
        if self.criticality_class not in CRITICALITY_CLASSES:
            raise DeadlineRegimeError(
                "criticality_class must be one of %s, got %r"
                % (", ".join(CRITICALITY_CLASSES), self.criticality_class)
            )
        weight = float(self.tardiness_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise DeadlineRegimeError(
                "tardiness_weight must be finite and >= 0, got %r"
                % (self.tardiness_weight,)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "deadline_s": float(self.deadline_s),
            "deadline_type": self.deadline_type,
            "criticality_class": self.criticality_class,
            "tardiness_weight": float(self.tardiness_weight),
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> "TaskDeadline":
        try:
            return cls(
                deadline_s=blob["deadline_s"],
                deadline_type=blob["deadline_type"],
                criticality_class=blob.get("criticality_class", "medium"),
                tardiness_weight=blob.get("tardiness_weight", 1.0),
            )
        except KeyError as exc:
            raise DeadlineRegimeError("missing field %s" % (exc,)) from exc


@dataclass(frozen=True)
class GraphDeadlines:
    graph_key: str
    content_sha256: str
    tasks: Mapping[int, TaskDeadline]
    bounds: Mapping[str, float] = field(default_factory=dict)
    witness: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        blob = {
            "content_sha256": self.content_sha256,
            "tasks": {str(tid): td.as_dict() for tid, td in sorted(self.tasks.items())},
            "bounds": {k: float(v) for k, v in sorted(self.bounds.items())},
        }
        if self.witness is not None:
            blob["witness"] = dict(self.witness)
        return blob

    @classmethod
    def from_dict(cls, graph_key: str, blob: Mapping[str, Any]) -> "GraphDeadlines":
        if "content_sha256" not in blob:
            raise DeadlineRegimeError("%s: missing content_sha256" % graph_key)
        raw_tasks = blob.get("tasks")
        if not isinstance(raw_tasks, Mapping) or not raw_tasks:
            raise DeadlineRegimeError("%s: tasks missing or empty" % graph_key)
        tasks: dict[int, TaskDeadline] = {}
        for key, value in raw_tasks.items():
            try:
                tid = int(key)
            except (TypeError, ValueError) as exc:
                raise DeadlineRegimeError(
                    "%s: task id %r is not an integer" % (graph_key, key)
                ) from exc
            if tid in tasks:
                raise DeadlineRegimeError("%s: duplicate task id %d" % (graph_key, tid))
            tasks[tid] = TaskDeadline.from_dict(value)
        return cls(
            graph_key=graph_key,
            content_sha256=str(blob["content_sha256"]),
            tasks=tasks,
            bounds={k: float(v) for k, v in (blob.get("bounds") or {}).items()},
            witness=blob.get("witness"),
        )


@dataclass(frozen=True)
class DeadlineRegime:
    regime: str
    kappa: float
    alpha: float
    deadline_type: str
    cycles_per_bit: float
    graphs: Mapping[str, GraphDeadlines]
    source_manifest_sha256: str = ""
    resources_sha256: str = ""
    energy_model: str = ""
    radio_model: str = ""
    energy_scope: str = ""
    seed: int = 0
    ordering: str = "task_id"
    criticality_policy: str = "depth_quantiles"
    tardiness_weight_policy: str = "class_weights"
    generator_commit: str = ""
    construction: str = CONSTRUCTION
    schema_version: str = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DeadlineRegimeError(
                "schema_version must be %s, got %r"
                % (SCHEMA_VERSION, self.schema_version)
            )
        if not str(self.regime):
            raise DeadlineRegimeError("regime name is required")
        kappa = float(self.kappa)
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise DeadlineRegimeError("kappa must be finite and positive, got %r" % (self.kappa,))
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise DeadlineRegimeError("alpha must be in [0, 1], got %r" % (self.alpha,))
        if self.deadline_type not in ALLOWED_DEADLINE_TYPES:
            raise DeadlineRegimeError(
                "deadline_type must be one of %s, got %r"
                % (", ".join(ALLOWED_DEADLINE_TYPES), self.deadline_type)
            )
        xi = float(self.cycles_per_bit)
        if not math.isfinite(xi) or xi <= 0.0:
            raise DeadlineRegimeError("cycles_per_bit must be finite and positive")
        if not self.graphs:
            raise DeadlineRegimeError("regime must contain at least one graph")

    @property
    def is_trainable(self) -> bool:
        """Only `infeasible_labelled` is barred from training."""
        return self.regime != "infeasible_labelled"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "generator_commit": self.generator_commit,
            "regime": self.regime,
            "kappa": float(self.kappa),
            "alpha": float(self.alpha),
            "deadline_type": self.deadline_type,
            "cycles_per_bit": float(self.cycles_per_bit),
            "construction": self.construction,
            "ordering": self.ordering,
            "criticality_policy": self.criticality_policy,
            "tardiness_weight_policy": self.tardiness_weight_policy,
            "seed": int(self.seed),
            "source_manifest_sha256": self.source_manifest_sha256,
            "resources_sha256": self.resources_sha256,
            "provenance": {
                "energy_model": self.energy_model,
                "radio_model": self.radio_model,
                "energy_scope": self.energy_scope,
            },
            "graphs": {k: v.as_dict() for k, v in sorted(self.graphs.items())},
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.as_dict(), indent=indent, sort_keys=True) + "\n"
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any]) -> "DeadlineRegime":
        if isinstance(source, Mapping):
            doc = dict(source)
        elif isinstance(source, str) and source.lstrip().startswith("{"):
            try:
                doc = json.loads(source)
            except ValueError as exc:
                raise DeadlineRegimeError("invalid sidecar JSON: %s" % exc) from exc
        else:
            path = Path(source)
            if not path.is_file():
                raise DeadlineRegimeError("sidecar not found: %s" % path)
            try:
                doc = json.loads(path.read_text())
            except ValueError as exc:
                raise DeadlineRegimeError("sidecar is not valid JSON: %s" % exc) from exc
        if not isinstance(doc, Mapping):
            raise DeadlineRegimeError("sidecar root must be an object")
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise DeadlineRegimeError(
                "unsupported schema_version %r (expected %s)"
                % (doc.get("schema_version"), SCHEMA_VERSION)
            )
        prov = doc.get("provenance") or {}
        graphs_blob = doc.get("graphs")
        if not isinstance(graphs_blob, Mapping) or not graphs_blob:
            raise DeadlineRegimeError("sidecar has no graphs")
        graphs = {
            str(key): GraphDeadlines.from_dict(str(key), value)
            for key, value in graphs_blob.items()
        }
        return cls(
            regime=str(doc.get("regime", "")),
            kappa=doc.get("kappa"),
            alpha=doc.get("alpha"),
            deadline_type=doc.get("deadline_type"),
            cycles_per_bit=doc.get("cycles_per_bit"),
            graphs=graphs,
            source_manifest_sha256=str(doc.get("source_manifest_sha256", "")),
            resources_sha256=str(doc.get("resources_sha256", "")),
            energy_model=str(prov.get("energy_model", "")),
            radio_model=str(prov.get("radio_model", "")),
            energy_scope=str(prov.get("energy_scope", "")),
            seed=int(doc.get("seed", 0)),
            ordering=str(doc.get("ordering", "task_id")),
            criticality_policy=str(doc.get("criticality_policy", "depth_quantiles")),
            tardiness_weight_policy=str(doc.get("tardiness_weight_policy", "class_weights")),
            generator_commit=str(doc.get("generator_commit", "")),
            construction=str(doc.get("construction", CONSTRUCTION)),
            schema_version=str(doc.get("schema_version")),
            generator_version=str(doc.get("generator_version", GENERATOR_VERSION)),
        )


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def resources_sha256(resources: ResourceConfig) -> str:
    """Hash the fields that actually change scheduling/energy, not the object id."""
    payload = {
        "ue_cpu_bytes_per_second": resources.ue_cpu_bytes_per_second,
        "mec_cpu_bytes_per_second": resources.mec_cpu_bytes_per_second,
        "helper_cpu_bytes_per_second": resources.helper_cpu_bytes_per_second,
        "mec_uplink_bytes_per_second": resources.mec_uplink_bytes_per_second,
        "mec_downlink_bytes_per_second": resources.mec_downlink_bytes_per_second,
        "v2v_bytes_per_second": resources.v2v_bytes_per_second,
        "rho_ue": resources.rho_ue,
        "f_l": resources.f_l,
        "zeta": resources.zeta,
        "rho_helper": resources.rho_helper,
        "f_v2v": resources.f_v2v,
        "ptx_mec_w": resources.ptx_mec_w,
        "prx_mec_w": resources.prx_mec_w,
        "ptx_v2v_w": resources.ptx_v2v_w,
        "prx_v2v_w": resources.prx_v2v_w,
        "energy_model": getattr(getattr(resources, "energy_model", None), "model", None),
        "radio_model": getattr(getattr(resources, "radio_model", None), "model", None),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def graph_key(distribution_id: int, path: str | Path) -> str:
    return "%d/%s" % (int(distribution_id), Path(path).name)


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def topological_order(dag: CanonicalDAG) -> list[int]:
    """Deterministic topological order (min task id first)."""
    indegree = {tid: 0 for tid in dag.tasks}
    succ: dict[int, list[int]] = {tid: [] for tid in dag.tasks}
    for edge in dag.edges:
        src, dst = int(edge.src_task_id), int(edge.dst_task_id)
        succ[src].append(dst)
        indegree[dst] += 1
    ready = sorted(tid for tid, deg in indegree.items() if deg == 0)
    out: list[int] = []
    while ready:
        tid = ready.pop(0)
        out.append(tid)
        for nxt in sorted(succ[tid]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(out) != len(dag.tasks):
        raise DeadlineRegimeError("graph is not a DAG: topological sort incomplete")
    return out


def _depth(dag: CanonicalDAG, order: Sequence[int]) -> dict[int, int]:
    preds = dag.predecessors()
    depth: dict[int, int] = {}
    for tid in order:
        parents = [int(e.src_task_id) for e in preds[tid]]
        depth[tid] = 0 if not parents else 1 + max(depth[p] for p in parents)
    return depth


def _criticality_from_depth(depths: Mapping[int, int], sinks: set[int]) -> dict[int, str]:
    """Class from structure only; the tardiness WEIGHT is a separate field."""
    values = sorted(depths.values())
    hi = values[int(0.75 * (len(values) - 1))]
    lo = values[int(0.25 * (len(values) - 1))]
    out = {}
    for tid, depth in depths.items():
        if tid in sinks or depth >= hi:
            out[tid] = "high"
        elif depth >= lo:
            out[tid] = "medium"
        else:
            out[tid] = "low"
    return out


DEFAULT_CLASS_WEIGHTS = {"high": 2.0, "medium": 1.0, "low": 0.5}


def generate_graph_deadlines(
    dag: CanonicalDAG,
    *,
    kappa: float,
    alpha: float,
    deadline_type: str,
    resources: ResourceConfig,
    cycles_per_bit: float,
    order: Sequence[int] | None = None,
    class_weights: Mapping[str, float] | None = None,
    content_sha256: str = "",
    graph_key_name: str = "",
    anchor_ready_s: Sequence[float] | None = None,
) -> GraphDeadlines:
    """EFT/LFT-based deadlines for one graph. Pure: reads the DAG, writes nothing."""
    if order is None:
        order = topological_order(dag)
    order = [int(t) for t in order]
    if set(order) != set(dag.tasks) or len(order) != len(dag.tasks):
        raise DeadlineRegimeError("order must be a permutation of the DAG tasks")

    bounds = static_action_bounds(dag, order, resources, cycles_per_bit=cycles_per_bit)
    pos = {tid: i for i, tid in enumerate(order)}

    # deadline basis: min over actions of the ready bound (sink includes return hop)
    eft = [min(bounds.ready_lb[i]) for i in range(bounds.n)]
    # minimum compute duration per task (start is the max over predecessors' eft)
    dur = []
    preds = dag.predecessors()
    for i, tid in enumerate(order):
        start = 0.0
        for edge in preds[tid]:
            start = max(start, eft[pos[int(edge.src_task_id)]])
        dur.append(max(0.0, eft[i] - start))

    graph_lb = max(eft) if eft else 0.0
    d_graph = float(kappa) * graph_lb

    succ: dict[int, list[int]] = {tid: [] for tid in dag.tasks}
    for edge in dag.edges:
        succ[int(edge.src_task_id)].append(int(edge.dst_task_id))

    lft = [0.0] * len(order)
    for i in range(len(order) - 1, -1, -1):
        tid = order[i]
        children = succ[tid]
        if not children:
            lft[i] = d_graph
            continue
        candidate = d_graph
        for child in children:
            j = pos[child]
            candidate = min(candidate, lft[j] - dur[j])
        lft[i] = candidate

    depths = _depth(dag, order)
    sinks = set(dag.sinks())
    classes = _criticality_from_depth(depths, sinks)
    weights = dict(DEFAULT_CLASS_WEIGHTS if class_weights is None else class_weights)

    # Effective deadlines are ANCHORED to a real schedule when one is supplied:
    # the contention-free relaxation cannot certify feasibility, so kappa scales
    # the witness's own ready times and alpha interpolates back towards the LB.
    # Without a witness the pure relaxation deadlines are used (calibration only).
    anchored = anchor_ready_s is not None
    if anchored:
        anchor = [float(v) for v in anchor_ready_s]
        if len(anchor) != len(order):
            raise DeadlineRegimeError(
                "anchor_ready_s length %d != task count %d" % (len(anchor), len(order))
            )
        if any((not math.isfinite(v)) or v < 0.0 for v in anchor):
            raise DeadlineRegimeError("anchor_ready_s must be finite and >= 0")

    tasks: dict[int, TaskDeadline] = {}
    floor = max(1e-9, 1e-6 * max(graph_lb, 1e-9))
    clamped = 0
    relaxed: list[float] = []
    for i, tid in enumerate(order):
        relaxed_deadline = eft[i] + float(alpha) * (lft[i] - eft[i])
        relaxed.append(relaxed_deadline)
        if anchored:
            deadline = eft[i] + float(alpha) * (float(kappa) * anchor[i] - eft[i])
        else:
            deadline = relaxed_deadline
        if not math.isfinite(deadline) or deadline < floor:
            deadline = floor
            clamped += 1
        tasks[tid] = TaskDeadline(
            deadline_s=deadline,
            deadline_type=deadline_type,
            criticality_class=classes[tid],
            tardiness_weight=float(weights[classes[tid]]),
        )

    return GraphDeadlines(
        graph_key=graph_key_name,
        content_sha256=content_sha256,
        tasks=tasks,
        bounds={
            "graph_lb": float(graph_lb),
            "d_graph": float(d_graph),
            "min_deadline": float(min(td.deadline_s for td in tasks.values())),
            "max_deadline": float(max(td.deadline_s for td in tasks.values())),
            "relaxation_infeasible_tasks": float(
                sum(1 for i in range(len(order)) if lft[i] < eft[i])
            ),
            "clamped_tasks": float(clamped),
            "deadline_floor": float(floor),
            "anchored_to_witness": 1.0 if anchored else 0.0,
            "relaxed_deadline_min": float(min(relaxed)) if relaxed else 0.0,
            "relaxed_deadline_max": float(max(relaxed)) if relaxed else 0.0,
            "witness_ready_max": float(max(anchor_ready_s)) if anchored else 0.0,
        },
    )


def build_regime(
    graphs: Iterable[tuple[int, Any, CanonicalDAG]],
    *,
    regime: str,
    kappa: float,
    alpha: float,
    deadline_type: str,
    resources: ResourceConfig,
    cycles_per_bit: float,
    seed: int = 0,
    class_weights: Mapping[str, float] | None = None,
    source_manifest_sha256: str = "",
    generator_commit: str = "",
) -> DeadlineRegime:
    """`graphs` yields (distribution_id, path, canonical_dag).

    Regime `none` produces an empty per-graph task map on purpose: it must stamp
    nothing and leave the legacy path byte-exact.
    """
    out: dict[str, GraphDeadlines] = {}
    for dist_id, path, dag in graphs:
        key = graph_key(dist_id, path)
        if regime == "none":
            entry = GraphDeadlines(
                graph_key=key,
                content_sha256=file_sha256(path),
                tasks={},
                bounds={},
            )
        else:
            entry = generate_graph_deadlines(
                dag,
                kappa=kappa,
                alpha=alpha,
                deadline_type=deadline_type,
                resources=resources,
                cycles_per_bit=cycles_per_bit,
                class_weights=class_weights,
                content_sha256=file_sha256(path),
                graph_key_name=key,
            )
        if key in out:
            raise DeadlineRegimeError("duplicate graph key %s" % key)
        out[key] = entry

    energy = getattr(resources, "energy_model", None)
    radio = getattr(resources, "radio_model", None)
    return DeadlineRegime(
        regime=regime,
        kappa=kappa,
        alpha=alpha,
        deadline_type=deadline_type,
        cycles_per_bit=cycles_per_bit,
        graphs=out,
        source_manifest_sha256=source_manifest_sha256,
        resources_sha256=resources_sha256(resources),
        energy_model=str(getattr(energy, "model", "") or ""),
        radio_model=str(getattr(radio, "model", "") or ""),
        energy_scope=str(getattr(energy, "energy_scope", "") or ""),
        seed=int(seed),
        generator_commit=generator_commit,
    )


def build_regime_with_witness(
    graphs: Iterable[tuple[int, Any, Any]],
    *,
    regime: str,
    kappa: float,
    alpha: float,
    deadline_type: str,
    resources: ResourceConfig,
    cycles_per_bit: float,
    seed: int = 0,
    class_weights: Mapping[str, float] | None = None,
    source_manifest_sha256: str = "",
    generator_commit: str = "",
    witness_kwargs: Mapping[str, Any] | None = None,
    allow_infeasible: bool = False,
) -> tuple[DeadlineRegime, list[dict[str, Any]]]:
    """Two-stage generation: fastest plan first, then deadlines anchored to it.

    `graphs` yields `(distribution_id, path, task_graph)`. For every graph the
    builder

      1. finds a fastest plan with the REAL scheduler (`witness.find_fastest_plan`),
      2. generates deadlines anchored to that plan's `all_consumers_ready` times,
      3. stamps them and searches for a plan that meets every hard deadline
         (`witness.find_witness`),

    and keeps the graph only when step 3 succeeds -- unless `allow_infeasible`
    (the labelled bucket). Excluded graphs are returned with a reason; nothing is
    silently dropped and nothing is certified by a lower bound.
    """
    from .adapter import to_canonical_dag
    from .witness import find_fastest_plan, find_witness

    if not allow_infeasible and regime == "infeasible_labelled":
        raise DeadlineRegimeError(
            "infeasible_labelled must be built with allow_infeasible=True"
        )
    kwargs = dict(witness_kwargs or {})
    accepted: dict[str, GraphDeadlines] = {}
    excluded: list[dict[str, Any]] = []
    for dist_id, path, task_graph in graphs:
        key = graph_key(dist_id, path)
        digest = file_sha256(path)
        if regime == "none":
            accepted[key] = GraphDeadlines(key, digest, {}, {})
            continue

        dag = to_canonical_dag(task_graph)
        order = [int(t) for t in task_graph.prioritize_sequence]
        fastest = find_fastest_plan(task_graph, resources, **kwargs)
        entry = generate_graph_deadlines(
            dag,
            kappa=kappa,
            alpha=alpha,
            deadline_type=deadline_type,
            resources=resources,
            cycles_per_bit=cycles_per_bit,
            order=order,
            class_weights=class_weights,
            content_sha256=digest,
            graph_key_name=key,
            anchor_ready_s=list(fastest.ready_s),
        )
        stamp_task_graph(task_graph, entry, regime_name=regime)
        witness = find_witness(
            task_graph, resources, seed_actions=list(fastest.actions), **kwargs
        )
        entry = _dc_replace(entry, witness={
            **witness.as_dict(),
            "fastest_plan_makespan_s": float(fastest.makespan_s),
            "fastest_plan_actions": [int(a) for a in fastest.actions],
        })
        if witness.found or allow_infeasible:
            accepted[key] = entry
            if not witness.found:
                excluded.append({"graph": key, "kept": True, "reason": witness.reason})
        else:
            excluded.append({"graph": key, "kept": False, "reason": witness.reason})

    if not accepted:
        raise DeadlineRegimeError(
            "no graph produced a usable sidecar entry (all %d failed the witness "
            "check)" % len(excluded)
        )

    energy = getattr(resources, "energy_model", None)
    radio = getattr(resources, "radio_model", None)
    built = DeadlineRegime(
        regime=regime,
        kappa=kappa,
        alpha=alpha,
        deadline_type=deadline_type,
        cycles_per_bit=cycles_per_bit,
        graphs=accepted,
        source_manifest_sha256=source_manifest_sha256,
        resources_sha256=resources_sha256(resources),
        energy_model=str(getattr(energy, "model", "") or ""),
        radio_model=str(getattr(radio, "model", "") or ""),
        energy_scope=str(getattr(energy, "energy_scope", "") or ""),
        seed=int(seed),
        generator_commit=generator_commit,
    )
    return built, excluded


# --------------------------------------------------------------------------- #
# stamper (loader side)
# --------------------------------------------------------------------------- #
STAMP_ATTR = "_deadline_regime_stamp"


def stamp_task_graph(
    task_graph: Any,
    entry: GraphDeadlines,
    *,
    regime_name: str,
    content_sha256: str | None = None,
    allow_overwrite: bool = False,
) -> int:
    """Write the sidecar deadlines onto a parsed task graph. Returns tasks stamped.

    Fails loudly on: coverage mismatch (missing / extra / duplicate task ids), a
    content hash mismatch, non-finite or non-positive values, and an attempt to
    stamp a DIFFERENT regime over an already stamped object. Idempotent for the
    same regime. Regime `none` (empty task map) writes nothing at all.
    """
    n = int(getattr(task_graph, "task_number", 0))
    tasks = getattr(task_graph, "task_list", None)
    if tasks is None or len(tasks) != n:
        raise DeadlineRegimeError("task_graph has no usable task_list")

    if content_sha256 is not None and entry.content_sha256:
        if content_sha256 != entry.content_sha256:
            raise DeadlineRegimeError(
                "graph hash mismatch for %s: sidecar %s, object %s"
                % (entry.graph_key, entry.content_sha256[:12], str(content_sha256)[:12])
            )

    if not entry.tasks:
        # regime `none`: nothing may change on the object
        if getattr(task_graph, STAMP_ATTR, None) is not None:
            raise DeadlineRegimeError(
                "graph was already stamped with %r but regime 'none' must not touch it"
                % (getattr(task_graph, STAMP_ATTR).get("regime"),)
            )
        return 0

    ids = set(entry.tasks)
    expected = set(range(n))
    if ids != expected:
        missing = sorted(expected - ids)[:5]
        extra = sorted(ids - expected)[:5]
        raise DeadlineRegimeError(
            "%s: task coverage mismatch (missing %s, extra %s)"
            % (entry.graph_key, missing, extra)
        )

    previous = getattr(task_graph, STAMP_ATTR, None)
    if previous is not None and not allow_overwrite:
        same = previous.get("regime") == regime_name and previous.get("values") == {
            tid: td.as_dict() for tid, td in entry.tasks.items()
        }
        if not same:
            raise DeadlineRegimeError(
                "%s: already stamped with regime %r; refusing to overwrite with %r"
                % (entry.graph_key, previous.get("regime"), regime_name)
            )
        return 0  # idempotent

    for tid, td in sorted(entry.tasks.items()):
        task = tasks[tid]
        setattr(task, "deadline_s", float(td.deadline_s))
        setattr(task, "deadline_type", str(td.deadline_type))
        setattr(task, "criticality_class", str(td.criticality_class))
        setattr(task, "tardiness_weight", float(td.tardiness_weight))
    setattr(
        task_graph,
        STAMP_ATTR,
        {
            "regime": regime_name,
            "content_sha256": entry.content_sha256,
            "values": {tid: td.as_dict() for tid, td in entry.tasks.items()},
        },
    )
    return len(entry.tasks)


def stamp_by_key(
    task_graph: Any,
    regime: DeadlineRegime,
    *,
    distribution_id: int,
    path: str | Path,
    allow_overwrite: bool = False,
) -> int:
    """Resolve the sidecar entry by `distribution/file` and verify the file hash."""
    key = graph_key(distribution_id, path)
    entry = regime.graphs.get(key)
    if entry is None:
        raise DeadlineRegimeError("no sidecar entry for %s in regime %r" % (key, regime.regime))
    try:
        digest = file_sha256(path)
    except OSError as exc:
        raise DeadlineRegimeError("cannot hash %s: %s" % (path, exc)) from exc
    return stamp_task_graph(
        task_graph,
        entry,
        regime_name=regime.regime,
        content_sha256=digest,
        allow_overwrite=allow_overwrite,
    )


def require_trainable(regime: DeadlineRegime) -> None:
    """Training entrypoints call this so `infeasible_labelled` can never be used."""
    if not regime.is_trainable:
        raise DeadlineRegimeError(
            "regime %r is evaluation-only (guard/dead-end tests) and must never "
            "be used for training" % regime.regime
        )
