"""Canonical scheduling data model for MARGO Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .validate import require_nonneg_int


class Location(str, Enum):
    UE = "UE"
    MEC = "MEC"
    HELPER = "HELPER"

    def to_action(self) -> int:
        return {Location.UE: 0, Location.MEC: 1, Location.HELPER: 2}[self]

    @classmethod
    def from_action(cls, action: int | str | "Location") -> "Location":
        if isinstance(action, Location):
            return action
        if isinstance(action, str):
            return cls(action)
        mapping = {0: cls.UE, 1: cls.MEC, 2: cls.HELPER}
        if action not in mapping:
            raise ValueError(f"unknown action: {action}")
        return mapping[action]


class ConflictingDuplicateEdgeError(ValueError):
    """Raised when same (src, dst) has distinct edge_output_bytes values."""


DEADLINE_TYPES = ("none", "soft", "firm", "hard")
# Mixed-criticality CLASS is a data-model concept; it is deliberately NOT the
# same thing as the deadline channel. A generator may later map
# HIGH->hard / MEDIUM->firm / LOW->soft, but the two stay separate fields so a
# paper can talk about criticality classes without implying a penalty weight.
CRITICALITY_CLASSES = ("low", "medium", "high")


@dataclass(frozen=True)
class CanonicalTask:
    task_id: int
    compute_workload_bytes: int
    task_output_bytes: int
    external_input_bytes: int = 0
    # Per-task computational intensity. None -> the episode's global
    # `cycles_per_bit` is used (see EnergyModelSpec). Future datasets can carry
    # heterogeneous intensities (e.g. some tasks memory-bound) without a schema
    # change; `compute_cycles()` is the single place that resolves the fallback.
    cycles_per_bit: float | None = None
    # Deadline semantics (②A). `deadline_s` is measured against the time the
    # task's OUTPUT becomes usable by a consumer (F_available), never against
    # the bare compute finish.
    deadline_s: float | None = None
    deadline_type: str = "none"
    # Mixed-criticality class (low | medium | high). Semantic category only.
    criticality_class: str = "medium"
    # Coefficient used by soft-tardiness aggregation. This is NOT "criticality":
    # it is a penalty weight, and it is what a paper must call a weight.
    tardiness_weight: float = 1.0

    def __post_init__(self) -> None:
        require_nonneg_int("task_id", self.task_id)
        require_nonneg_int("compute_workload_bytes", self.compute_workload_bytes)
        require_nonneg_int("task_output_bytes", self.task_output_bytes)
        require_nonneg_int("external_input_bytes", self.external_input_bytes)
        if self.cycles_per_bit is not None and float(self.cycles_per_bit) <= 0.0:
            raise ValueError("cycles_per_bit must be > 0 when given")
        if self.deadline_s is not None and float(self.deadline_s) < 0.0:
            raise ValueError("deadline_s must be non-negative when given")
        if float(self.tardiness_weight) < 0.0:
            raise ValueError("tardiness_weight must be non-negative")
        if self.criticality_class not in CRITICALITY_CLASSES:
            raise ValueError(
                "criticality_class must be one of %s, got %r"
                % (CRITICALITY_CLASSES, self.criticality_class)
            )
        if self.deadline_type not in DEADLINE_TYPES:
            raise ValueError(
                "deadline_type must be one of %s, got %r" % (DEADLINE_TYPES, self.deadline_type)
            )

    def compute_cycles(self, global_cycles_per_bit: float) -> float:
        """Total CPU cycles for this task: C = bytes * 8 * cycles_per_bit."""
        xi = self.cycles_per_bit if self.cycles_per_bit is not None else global_cycles_per_bit
        return float(self.compute_workload_bytes) * 8.0 * float(xi)

    def effective_cycles_per_bit(self, global_cycles_per_bit: float) -> float:
        return float(self.cycles_per_bit if self.cycles_per_bit is not None else global_cycles_per_bit)

    @property
    def has_deadline(self) -> bool:
        return self.deadline_s is not None and self.deadline_type != "none"


@dataclass(frozen=True)
class CanonicalEdge:
    src_task_id: int
    dst_task_id: int
    edge_output_bytes: int

    def __post_init__(self) -> None:
        require_nonneg_int("src_task_id", self.src_task_id)
        require_nonneg_int("dst_task_id", self.dst_task_id)
        require_nonneg_int("edge_output_bytes", self.edge_output_bytes)
        if self.src_task_id == self.dst_task_id:
            raise ValueError(f"self-edge forbidden: {self.src_task_id}->{self.dst_task_id}")


@dataclass
class CanonicalDAG:
    tasks: dict[int, CanonicalTask]
    edges: list[CanonicalEdge]  # unique canonical edges only
    edge_record_count: int
    unique_edge_count: int

    @classmethod
    def from_records(
        cls,
        tasks: Iterable[CanonicalTask],
        raw_edges: Iterable[tuple[int, int, int]],
    ) -> "CanonicalDAG":
        task_map: dict[int, CanonicalTask] = {}
        for t in tasks:
            if t.task_id in task_map:
                raise ValueError(f"duplicate task_id: {t.task_id}")
            task_map[t.task_id] = t
        if not task_map:
            raise ValueError("empty task set")

        records = []
        for s, d, n in raw_edges:
            src = require_nonneg_int("src_task_id", s)
            dst = require_nonneg_int("dst_task_id", d)
            nbytes = require_nonneg_int("edge_output_bytes", n)
            if src == dst:
                raise ValueError(f"self-edge forbidden: {src}->{dst}")
            records.append((src, dst, nbytes))
        by_pair: dict[tuple[int, int], set[int]] = {}
        for src, dst, nbytes in records:
            if src not in task_map or dst not in task_map:
                raise ValueError(f"edge endpoint missing: {src}->{dst}")
            by_pair.setdefault((src, dst), set()).add(nbytes)

        for (src, dst), weights in by_pair.items():
            if len(weights) > 1:
                raise ConflictingDuplicateEdgeError(
                    f"conflicting duplicate edge {src}->{dst}: weights={sorted(weights)}"
                )

        seen: set[tuple[int, int, int]] = set()
        unique: list[CanonicalEdge] = []
        for key in records:
            if key in seen:
                continue
            seen.add(key)
            unique.append(CanonicalEdge(*key))

        unique.sort(key=lambda e: (e.src_task_id, e.dst_task_id, e.edge_output_bytes))
        return cls(
            tasks=task_map,
            edges=unique,
            edge_record_count=len(records),
            unique_edge_count=len(unique),
        )

    def predecessors(self) -> dict[int, list[CanonicalEdge]]:
        preds = {tid: [] for tid in self.tasks}
        for e in self.edges:
            preds[e.dst_task_id].append(e)
        for tid in preds:
            preds[tid].sort(key=lambda e: (e.src_task_id, e.edge_output_bytes))
        return preds

    def successors(self) -> dict[int, list[int]]:
        succs = {tid: [] for tid in self.tasks}
        for e in self.edges:
            if e.dst_task_id not in succs[e.src_task_id]:
                succs[e.src_task_id].append(e.dst_task_id)
        for tid in succs:
            succs[tid].sort()
        return succs

    def sinks(self) -> list[int]:
        succs = self.successors()
        return sorted(tid for tid, outs in succs.items() if not outs)


@dataclass(frozen=True)
class ResourceInterval:
    resource: str
    start: float
    end: float
    task_id: int | None = None
    hop: str | None = None


@dataclass(frozen=True)
class TransferRecord:
    hop: str
    hop_index: int
    bytes: int
    start: float
    end: float
    src_location: Location
    dst_location: Location
    src_task_id: int | None
    dst_task_id: int | None


@dataclass(frozen=True)
class TaskExecutionRecord:
    task_id: int
    location: Location
    start: float
    finish: float
    output_location: Location
    # ②A/②A.1 deadline semantics.
    #   first_available      : earliest arrival at ANY consumer (diagnostic)
    #   all_consumers_ready  : arrival at EVERY required consumer, i.e.
    #                          max_j delivery(i->j); UE return for a sink.
    #                          **This is the primary deadline basis**: a task is
    #                          not "done in time" while one of its consumers
    #                          still cannot read the output.
    first_available: float | None = None
    all_consumers_ready: float | None = None
    deadline_s: float | None = None
    deadline_type: str = "none"
    criticality_class: str = "medium"
    tardiness_weight: float = 1.0
    tardiness_s: float = 0.0

    @property
    def availability_seconds(self) -> float:
        """Primary deadline basis: all required consumers can read the output."""
        if self.all_consumers_ready is not None:
            return float(self.all_consumers_ready)
        if self.first_available is not None:
            return float(self.first_available)
        return float(self.finish)

    @property
    def last_delivery(self) -> float | None:
        """Deprecated alias for `all_consumers_ready` (kept for continuity)."""
        return self.all_consumers_ready

    @property
    def missed(self) -> bool:
        return self.deadline_s is not None and self.tardiness_s > 0.0


@dataclass
class EnergyBreakdown:
    ue_local_cpu_joules: float = 0.0
    ue_mec_uplink_joules: float = 0.0
    ue_mec_downlink_joules: float = 0.0
    ue_v2v_tx_joules: float = 0.0
    ue_v2v_rx_joules: float = 0.0
    helper_compute_joules: float = 0.0
    helper_v2v_tx_joules: float = 0.0
    helper_v2v_rx_joules: float = 0.0
    # Optional accounting only — NOT in total_mobile_joules (ADR-001 / OBJECTIVE §1).
    mec_compute_joules_optional: float = 0.0
    # RSU/MEC transmit energy (physical_v1 only). System-side: counted in
    # total_system_joules, never in total_mobile_joules.
    mec_tx_joules_optional: float = 0.0

    COMPONENT_FIELDS = (
        "ue_local_cpu_joules",
        "ue_mec_uplink_joules",
        "ue_mec_downlink_joules",
        "ue_v2v_tx_joules",
        "ue_v2v_rx_joules",
        "helper_compute_joules",
        "helper_v2v_tx_joules",
        "helper_v2v_rx_joules",
        "mec_compute_joules_optional",
        "mec_tx_joules_optional",
    )

    def add_inplace(self, other: "EnergyBreakdown") -> None:
        for name in self.COMPONENT_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @classmethod
    def sum_many(cls, parts: Iterable["EnergyBreakdown"]) -> "EnergyBreakdown":
        total = cls()
        for part in parts:
            total.add_inplace(part)
        return total

    @property
    def total_ue_joules(self) -> float:
        return (
            self.ue_local_cpu_joules
            + self.ue_mec_uplink_joules
            + self.ue_mec_downlink_joules
            + self.ue_v2v_tx_joules
            + self.ue_v2v_rx_joules
        )

    @property
    def total_helper_joules(self) -> float:
        return (
            self.helper_compute_joules
            + self.helper_v2v_tx_joules
            + self.helper_v2v_rx_joules
        )

    @property
    def total_mobile_joules(self) -> float:
        return self.total_ue_joules + self.total_helper_joules

    @property
    def total_requester_joules(self) -> float:
        """Accounting boundary A: the requesting vehicle's battery = UE only."""
        return self.total_ue_joules

    @property
    def total_system_joules(self) -> float:
        """Accounting boundary C: UE + helper + MEC/RSU compute + MEC TX."""
        return (
            self.total_mobile_joules
            + self.mec_compute_joules_optional
            + self.mec_tx_joules_optional
        )

    @property
    def total_system_joules_optional(self) -> float:
        return self.total_system_joules

    def as_dict(self) -> dict[str, float]:
        return {
            "ue_local_cpu_joules": self.ue_local_cpu_joules,
            "ue_mec_uplink_joules": self.ue_mec_uplink_joules,
            "ue_mec_downlink_joules": self.ue_mec_downlink_joules,
            "ue_v2v_tx_joules": self.ue_v2v_tx_joules,
            "ue_v2v_rx_joules": self.ue_v2v_rx_joules,
            "helper_compute_joules": self.helper_compute_joules,
            "helper_v2v_tx_joules": self.helper_v2v_tx_joules,
            "helper_v2v_rx_joules": self.helper_v2v_rx_joules,
            "mec_compute_joules_optional": self.mec_compute_joules_optional,
            "mec_tx_joules_optional": self.mec_tx_joules_optional,
            "total_ue_joules": self.total_ue_joules,
            "total_helper_joules": self.total_helper_joules,
            "total_mobile_joules": self.total_mobile_joules,
            "total_requester_joules": self.total_requester_joules,
            "total_system_joules": self.total_system_joules,
            "total_system_joules_optional": self.total_system_joules_optional,
        }


@dataclass
class ScheduleResult:
    tasks: dict[int, TaskExecutionRecord]
    transfers: list[TransferRecord]
    resource_intervals: list[ResourceInterval]
    energy: EnergyBreakdown
    makespan_seconds: float
    terminal_return_time: float
    topo_order: list[int] = field(default_factory=list)
    # ②A deadline accounting (all zero/empty when no task carries a deadline).
    soft_tardiness_s: float = 0.0
    soft_tardiness_normalized: float = 0.0
    firm_miss_count: int = 0
    firm_miss_rate: float = 0.0
    hard_miss_count: int = 0
    hard_miss_rate: float = 0.0
    hard_feasible: bool = True
    mean_tardiness_s: float = 0.0
    max_tardiness_s: float = 0.0
    # Which availability notion the deadline was checked against.
    deadline_basis: str = "all_consumers_ready"

    def deadline_metrics(self) -> dict[str, float]:
        return {
            "soft_tardiness_s": self.soft_tardiness_s,
            "soft_tardiness_normalized": self.soft_tardiness_normalized,
            "firm_miss_count": float(self.firm_miss_count),
            "firm_miss_rate": self.firm_miss_rate,
            "hard_miss_count": float(self.hard_miss_count),
            "hard_miss_rate": self.hard_miss_rate,
            "hard_feasible": float(self.hard_feasible),
            "deadline_basis_all_consumers_ready": float(
                self.deadline_basis == "all_consumers_ready"
            ),
            "mean_tardiness_s": self.mean_tardiness_s,
            "max_tardiness_s": self.max_tardiness_s,
        }

    @property
    def total_mobile_joules(self) -> float:
        return self.energy.total_mobile_joules
