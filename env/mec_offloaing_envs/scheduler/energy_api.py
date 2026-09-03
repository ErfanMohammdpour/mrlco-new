"""Canonical energy / reporting API (OBJECTIVE_AND_ENERGY.md §§2–5).

Reward telescoping (§6) lives in `reward.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .model import EnergyBreakdown, Location, ScheduleResult
from .resources import ResourceConfig
from .validate import require_finite

logger = logging.getLogger(__name__)

EPS = 1e-12
OUT_OF_RANGE = "clip_and_log"
# MARGO-SPEC-v0.1 publication freeze (OBJECTIVE_AND_ENERGY.md §4 / frozen_experiment.yaml).
LATENCY_WEIGHT = 0.5
ENERGY_WEIGHT = 0.5
WEIGHT_SUM = 1.0


def _almost_eq(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def frozen_objective_weights(path: Any | None = None) -> tuple[float, float]:
    """Load and validate publication objective weights (must be 0.5/0.5 for v0.1)."""
    from pathlib import Path

    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"
    doc = yaml.safe_load(Path(path).read_text())
    weights = (doc.get("energy") or {}).get("objective_weights") or {}
    lw = require_finite("latency_weight", weights.get("latency_weight", LATENCY_WEIGHT))
    ew = require_finite("energy_weight", weights.get("energy_weight", ENERGY_WEIGHT))
    if lw < 0.0 or ew < 0.0:
        raise ValueError(f"objective weights must be non-negative, got lw={lw}, ew={ew}")
    if not _almost_eq(lw + ew, WEIGHT_SUM):
        raise ValueError(f"objective weights must sum to 1.0, got {lw + ew}")
    if not _almost_eq(lw, LATENCY_WEIGHT) or not _almost_eq(ew, ENERGY_WEIGHT):
        raise ValueError(
            f"MARGO-SPEC-v0.1 freezes latency/energy weights at "
            f"{LATENCY_WEIGHT}/{ENERGY_WEIGHT}, got {lw}/{ew}"
        )
    return float(lw), float(ew)


def require_publication_weights(latency_weight: float, energy_weight: float) -> tuple[float, float]:
    """Fail any override of frozen 0.5/0.5 publication weights."""
    lw = require_finite("latency_weight", latency_weight)
    ew = require_finite("energy_weight", energy_weight)
    if lw < 0.0 or ew < 0.0:
        raise ValueError(f"objective weights must be non-negative, got lw={lw}, ew={ew}")
    if not _almost_eq(lw + ew, WEIGHT_SUM):
        raise ValueError(f"objective weights must sum to 1.0, got {lw + ew}")
    frozen_lw, frozen_ew = frozen_objective_weights()
    if not _almost_eq(lw, frozen_lw) or not _almost_eq(ew, frozen_ew):
        raise ValueError(
            f"publication weights must be {frozen_lw}/{frozen_ew}, got {lw}/{ew}"
        )
    return frozen_lw, frozen_ew


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

    def __post_init__(self) -> None:
        for name in ("L_ue", "L_mec", "L_helper", "E_ue", "E_mec", "E_helper"):
            require_finite(name, getattr(self, name))
        # Derived extrema are always ordered; still assert finite scales usable.
        require_finite("L_ref_min", self.L_ref_min)
        require_finite("L_ref_max", self.L_ref_max)
        require_finite("E_ref_min", self.E_ref_min)
        require_finite("E_ref_max", self.E_ref_max)

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
    value = require_finite(name, value)
    vmin = require_finite(f"{name}_vmin", vmin)
    vmax = require_finite(f"{name}_vmax", vmax)
    if vmax < vmin:
        raise ValueError(f"{name}: vmax ({vmax}) must be >= vmin ({vmin})")
    if out_of_range != OUT_OF_RANGE:
        raise ValueError(f"unsupported out_of_range policy: {out_of_range}")
    scale = max(vmax - vmin, EPS)
    raw = (value - vmin) / scale
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
    """Scientific composite: 0.5 * L_norm + 0.5 * E_norm (clipped)."""
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


def _add_transfer_components(
    bd: EnergyBreakdown,
    hop: str,
    duration: float,
    src_loc: Location,
    resources: ResourceConfig,
) -> None:
    if hop == "MEC_UL":
        bd.ue_mec_uplink_joules += duration * resources.ptx_mec_w
    elif hop == "MEC_DL":
        bd.ue_mec_downlink_joules += duration * resources.prx_mec_w
    elif hop == "V2V":
        if src_loc == Location.HELPER:
            bd.helper_v2v_tx_joules += duration * resources.ptx_v2v_w
            bd.ue_v2v_rx_joules += duration * resources.prx_v2v_w
        else:
            # UE→HELPER, or post-MEC_DL staging at UE toward HELPER
            bd.ue_v2v_tx_joules += duration * resources.ptx_v2v_w
            bd.helper_v2v_rx_joules += duration * resources.prx_v2v_w


def attribute_energy_components_by_task(
    result: ScheduleResult,
    resources: ResourceConfig,
) -> dict[int, EnergyBreakdown]:
    """Per-task energy component breakdown (OBJECTIVE §2).

    Owner rule matches scalar attribution: compute → executor; transfer → dst if
    present else src (sink return). Component-wise sum equals episode breakdown.
    """
    # Every scheduled task gets a breakdown, including zero-mobile (e.g. internal MEC).
    out: dict[int, EnergyBreakdown] = {tid: EnergyBreakdown() for tid in result.tasks}
    for tid, rec in result.tasks.items():
        dur = rec.finish - rec.start
        if rec.location == Location.UE:
            out[tid].ue_local_cpu_joules += dur * resources.rho_ue * (resources.f_l**resources.zeta)
        elif rec.location == Location.HELPER:
            out[tid].helper_compute_joules += (
                dur * resources.rho_helper * (resources.f_v2v**resources.zeta)
            )
        # MEC compute optional remains episode-level only (v0.1 value 0).

    for t in result.transfers:
        owner = t.dst_task_id if t.dst_task_id is not None else t.src_task_id
        if owner is None:
            continue
        _add_transfer_components(out[owner], t.hop, t.end - t.start, t.src_location, resources)
    return out


def attribute_energy_by_task(
    result: ScheduleResult,
    resources: ResourceConfig,
) -> dict[int, float]:
    """Per-task mobile energy scalar; sum equals total_mobile_joules."""
    comps = attribute_energy_components_by_task(result, resources)
    return {tid: bd.total_mobile_joules for tid, bd in comps.items()}


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
