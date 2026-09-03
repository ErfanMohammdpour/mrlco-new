"""Canonical energy / reporting API (OBJECTIVE_AND_ENERGY.md §§2–5).

Reward telescoping (§6) is a separate Phase 1 commit.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .model import Location, ScheduleResult
from .resources import ResourceConfig

logger = logging.getLogger(__name__)

EPS = 1e-12
OUT_OF_RANGE = "clip_and_log"
LATENCY_WEIGHT = 0.5
ENERGY_WEIGHT = 0.5


@dataclass(frozen=True)
class ReferenceRanges:
    """Episode-local pure-location reference ranges (all_UE / all_MEC / all_HELPER)."""

    L_ue: float
    L_mec: float
    L_helper: float
    E_ue: float
    E_mec: float
    E_helper: float
    source: str = "pure_location_reference_range"
    unit_latency: str = "seconds"
    unit_energy: str = "joules"
    out_of_range: str = OUT_OF_RANGE

    @property
    def L_ref_min(self) -> float:
        return min(self.L_ue, self.L_mec, self.L_helper)

    @property
    def L_ref_max(self) -> float:
        return max(self.L_ue, self.L_mec, self.L_helper)

    @property
    def E_ref_min(self) -> float:
        return min(self.E_ue, self.E_mec, self.E_helper)

    @property
    def E_ref_max(self) -> float:
        return max(self.E_ue, self.E_mec, self.E_helper)

    @property
    def L_scale(self) -> float:
        return max(self.L_ref_max - self.L_ref_min, EPS)

    @property
    def E_scale(self) -> float:
        return max(self.E_ref_max - self.E_ref_min, EPS)


def pure_location_plan(decoder_order: Sequence[int], action: int) -> list[tuple[int, int]]:
    if action not in (0, 1, 2):
        raise ValueError(f"action must be 0/1/2, got {action}")
    return [(int(tid), int(action)) for tid in decoder_order]


def compute_reference_ranges(
    task_graph: Any,
    resources: ResourceConfig,
) -> ReferenceRanges:
    """Schedule three pure-location plans; derive L/E ref min/max."""
    # Lazy import avoids circular import with adapter → energy_api.
    from .adapter import schedule_via_adapter, validate_plan

    order = [int(tid) for tid in task_graph.prioritize_sequence]
    validate_plan(task_graph, pure_location_plan(order, 0))

    metrics: dict[int, tuple[float, float]] = {}
    for action in (0, 1, 2):
        result, _, _ = schedule_via_adapter(
            task_graph, pure_location_plan(order, action), resources
        )
        metrics[action] = (result.makespan_seconds, result.total_mobile_joules)

    return ReferenceRanges(
        L_ue=metrics[0][0],
        L_mec=metrics[1][0],
        L_helper=metrics[2][0],
        E_ue=metrics[0][1],
        E_mec=metrics[1][1],
        E_helper=metrics[2][1],
    )


def normalize(
    value: float,
    vmin: float,
    vmax: float,
    *,
    name: str = "metric",
    out_of_range: str = OUT_OF_RANGE,
) -> float:
    """Map value into [0,1] using reference range; v0.1 out-of-range = clip_and_log."""
    scale = max(vmax - vmin, EPS)
    raw = (value - vmin) / scale
    if out_of_range != OUT_OF_RANGE:
        raise ValueError(f"unsupported out_of_range policy: {out_of_range}")
    if raw < 0.0 or raw > 1.0:
        logger.warning(
            "clip_and_log: %s raw=%s outside [0,1] (value=%s, vmin=%s, vmax=%s)",
            name,
            raw,
            value,
            vmin,
            vmax,
        )
    return min(1.0, max(0.0, raw))


def j_report(makespan_seconds: float, total_mobile_joules: float, refs: ReferenceRanges) -> float:
    """Scientific composite: 0.5 * L_norm + 0.5 * E_norm."""
    l_norm = normalize(
        makespan_seconds, refs.L_ref_min, refs.L_ref_max, name="L", out_of_range=refs.out_of_range
    )
    e_norm = normalize(
        total_mobile_joules,
        refs.E_ref_min,
        refs.E_ref_max,
        name="E",
        out_of_range=refs.out_of_range,
    )
    return LATENCY_WEIGHT * l_norm + ENERGY_WEIGHT * e_norm


def attribute_energy_by_task(
    result: ScheduleResult,
    resources: ResourceConfig,
) -> dict[int, float]:
    """Per-task mobile energy attribution; sum equals total_mobile_joules.

    Compute → executing task. Transfer joules (UE+helper radio for that hop) →
    destination task when present, else source (sink return).
    """
    out: dict[int, float] = defaultdict(float)
    for tid, rec in result.tasks.items():
        dur = rec.finish - rec.start
        if rec.location == Location.UE:
            out[tid] += dur * resources.rho_ue * (resources.f_l**resources.zeta)
        elif rec.location == Location.HELPER:
            out[tid] += dur * resources.rho_helper * (resources.f_v2v**resources.zeta)
        # MEC compute is optional accounting only — not attributed to mobile objective.

    for t in result.transfers:
        owner = t.dst_task_id if t.dst_task_id is not None else t.src_task_id
        if owner is None:
            continue
        dur = t.end - t.start
        if t.hop == "MEC_UL":
            out[owner] += dur * resources.ptx_mec_w
        elif t.hop == "MEC_DL":
            out[owner] += dur * resources.prx_mec_w
        elif t.hop == "V2V":
            out[owner] += dur * (resources.ptx_v2v_w + resources.prx_v2v_w)
    return dict(out)


def transfers_for_task(result: ScheduleResult, task_id: int) -> list:
    """Inbound deps (dst==task) plus sink-return hops (src==task, dst is None)."""
    tid = int(task_id)
    return [
        t
        for t in result.transfers
        if t.dst_task_id == tid or (t.dst_task_id is None and t.src_task_id == tid)
    ]


def split_v2v_times(transfers: Sequence) -> tuple[float, float]:
    """UE→HELPER counted as uplink; HELPER→UE as downlink (UE viewpoint)."""
    up = 0.0
    down = 0.0
    for t in transfers:
        if t.hop != "V2V":
            continue
        dur = t.end - t.start
        if t.src_location == Location.UE:
            up += dur
        elif t.src_location == Location.HELPER:
            down += dur
    return up, down
