"""Canonical scheduling data model for MARGO Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Location(str, Enum):
    UE = "UE"
    MEC = "MEC"
    HELPER = "HELPER"

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


@dataclass(frozen=True)
class CanonicalTask:
    task_id: int
    compute_workload_bytes: int
    task_output_bytes: int
    external_input_bytes: int = 0


@dataclass(frozen=True)
class CanonicalEdge:
    src_task_id: int
    dst_task_id: int
    edge_output_bytes: int


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

        records = [(int(s), int(d), int(n)) for s, d, n in raw_edges]
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
            "total_ue_joules": self.total_ue_joules,
            "total_helper_joules": self.total_helper_joules,
            "total_mobile_joules": self.total_mobile_joules,
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

    @property
    def total_mobile_joules(self) -> float:
        return self.energy.total_mobile_joules
