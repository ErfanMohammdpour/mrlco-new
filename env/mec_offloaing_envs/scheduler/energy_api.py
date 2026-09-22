"""Canonical energy / reporting API (OBJECTIVE_AND_ENERGY.md §§2–5).

Reward telescoping (§6) lives in `reward.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .energy_model import hop_energy_fields
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
# Phase 5 Pareto sweep. λ=1.0 is the latency track.
PARETO_LAMBDAS = (1.0, 0.75, 0.5, 0.25, 0.0)
# Reference-range construction.
#   pure_location  : min/max over all_UE / all_MEC / all_HELPER  (MARGO-SPEC-v0.1)
#   candidate_panel: additionally includes greedy_from_mec (and any panel_extra),
#                    i.e. the best/worst plans a policy can realistically emit.
# Measured reason for the panel mode: a mixed plan beats all-MEC on makespan in
# 100% of graphs, so with pure_location ranges the latency term of `j_report`
# clips to 0 and the clipped composite ranks a faster plan WORSE than all-MEC
# (40/40 inversions across four distributions). See spec/energy_reward_audit.py.
REFERENCE_MODE_PURE = "pure_location"
REFERENCE_MODE_PANEL = "candidate_panel"
REFERENCE_MODES = (REFERENCE_MODE_PURE, REFERENCE_MODE_PANEL)


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
    """Episode-local reference ranges.

    `L_ref_min/max` and `E_ref_min/max` come from the pure-location plans
    (MARGO-SPEC-v0.1). When `reference_mode="candidate_panel"` the `*_panel_*`
    fields carry the min/max over the wider candidate panel and are used for
    normalization instead.
    """

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
    reference_mode: str = REFERENCE_MODE_PURE
    L_panel_min: float | None = None
    L_panel_max: float | None = None
    E_panel_min: float | None = None
    E_panel_max: float | None = None

    def __post_init__(self) -> None:
        for name in ("L_ue", "L_mec", "L_helper", "E_ue", "E_mec", "E_helper"):
            require_finite(name, getattr(self, name))
        if self.reference_mode not in REFERENCE_MODES:
            raise ValueError(
                "reference_mode must be one of %s, got %r"
                % (REFERENCE_MODES, self.reference_mode)
            )
        # Derived extrema are always ordered; still assert finite scales usable.
        require_finite("L_ref_min", self.L_ref_min)
        require_finite("L_ref_max", self.L_ref_max)
        require_finite("E_ref_min", self.E_ref_min)
        require_finite("E_ref_max", self.E_ref_max)
        for name in ("L_panel_min", "L_panel_max", "E_panel_min", "E_panel_max"):
            value = getattr(self, name)
            if value is not None:
                require_finite(name, value)
        if self.reference_mode == REFERENCE_MODE_PANEL and (
            self.L_panel_min is None
            or self.L_panel_max is None
            or self.E_panel_min is None
            or self.E_panel_max is None
        ):
            raise ValueError("candidate_panel mode requires the *_panel_* fields")

    # -- bounds actually used for normalization ----------------------------
    @property
    def L_min(self) -> float:
        if self.reference_mode == REFERENCE_MODE_PANEL:
            return float(self.L_panel_min)  # type: ignore[arg-type]
        return self.L_ref_min

    @property
    def L_max(self) -> float:
        if self.reference_mode == REFERENCE_MODE_PANEL:
            return float(self.L_panel_max)  # type: ignore[arg-type]
        return self.L_ref_max

    @property
    def E_min(self) -> float:
        if self.reference_mode == REFERENCE_MODE_PANEL:
            return float(self.E_panel_min)  # type: ignore[arg-type]
        return self.E_ref_min

    @property
    def E_max(self) -> float:
        if self.reference_mode == REFERENCE_MODE_PANEL:
            return float(self.E_panel_max)  # type: ignore[arg-type]
        return self.E_ref_max

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
        return max(self.L_max - self.L_min, EPS)

    @property
    def E_scale(self) -> float:
        return max(self.E_max - self.E_min, EPS)


def pure_location_plan(decoder_order: Sequence[int], action: int) -> list[tuple[int, int]]:
    if action not in (0, 1, 2):
        raise ValueError(f"action must be 0/1/2, got {action}")
    return [(int(tid), int(action)) for tid in decoder_order]


def compute_reference_ranges(
    task_graph: Any,
    resources: ResourceConfig,
    *,
    mode: str = REFERENCE_MODE_PURE,
    panel_extra: Sequence[Sequence[tuple[int, int]]] | None = None,
    panel_max_passes: int = 2,
) -> ReferenceRanges:
    """Schedule the reference plans; derive L/E ref min/max.

    `mode="pure_location"` (default, MARGO-SPEC-v0.1): all_UE / all_MEC / all_HELPER.
    `mode="candidate_panel"`: the three pure plans PLUS `greedy_from_mec` (and any
    caller-supplied `panel_extra` plans). Use this when the reference range must
    bound the plans a policy can actually emit — otherwise mixed plans fall below
    `L_ref_min` and clipped normalization saturates.
    """
    from .adapter import schedule_via_adapter, validate_plan

    if mode not in REFERENCE_MODES:
        raise ValueError("mode must be one of %s, got %r" % (REFERENCE_MODES, mode))

    order = [int(tid) for tid in task_graph.prioritize_sequence]
    validate_plan(task_graph, pure_location_plan(order, 0))

    plans: list[tuple[str, list[tuple[int, int]]]] = [
        ("all_UE", pure_location_plan(order, 0)),
        ("all_MEC", pure_location_plan(order, 1)),
        ("all_HELPER", pure_location_plan(order, 2)),
    ]
    if mode == REFERENCE_MODE_PANEL:
        from .greedy import greedy_from_mec_plan

        greedy_plan_result, _ = greedy_from_mec_plan(
            task_graph, resources, max_passes=int(panel_max_passes)
        )
        plans.append(("greedy_from_mec", list(greedy_plan_result)))
        for idx, extra in enumerate(panel_extra or ()):
            plans.append(("panel_extra_%d" % idx, list(extra)))

    metrics: dict[str, tuple[float, float]] = {}
    for name, plan in plans:
        result, _, _ = schedule_via_adapter(task_graph, plan, resources)
        metrics[name] = (result.makespan_seconds, result.total_mobile_joules)

    latencies = [m[0] for m in metrics.values()]
    energies = [m[1] for m in metrics.values()]
    return ReferenceRanges(
        L_ue=metrics["all_UE"][0],
        L_mec=metrics["all_MEC"][0],
        L_helper=metrics["all_HELPER"][0],
        E_ue=metrics["all_UE"][1],
        E_mec=metrics["all_MEC"][1],
        E_helper=metrics["all_HELPER"][1],
        L_panel_min=min(latencies),
        L_panel_max=max(latencies),
        E_panel_min=min(energies),
        E_panel_max=max(energies),
        source=REFERENCE_MODE_PANEL if mode == REFERENCE_MODE_PANEL else "pure_location_reference_range",
        reference_mode=mode,
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


def lambda_tag(lam: float) -> str:
    return "%.2f" % float(lam)


def j_lambda(
    makespan_seconds: float,
    total_mobile_joules: float,
    refs: ReferenceRanges,
    lam: float,
    *,
    clip: bool = True,
) -> float:
    """J_λ = λ T_norm + (1−λ) E_norm. Search uses clip=False; report uses clip=True.

    Normalization bounds follow `refs.reference_mode`: pure-location plans by
    default, or the wider candidate panel when the reference ranges were built
    with `mode="candidate_panel"`.
    """
    lam = float(lam)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lambda must be in [0,1], got %s" % lam)
    t = require_finite("T", makespan_seconds)
    e = require_finite("E", total_mobile_joules)
    if clip:
        t_n = normalize(t, refs.L_min, refs.L_max, name="L", out_of_range=refs.out_of_range)
        e_n = normalize(e, refs.E_min, refs.E_max, name="E", out_of_range=refs.out_of_range)
    else:
        t_n = (t - refs.L_min) / refs.L_scale
        e_n = (e - refs.E_min) / refs.E_scale
    return lam * t_n + (1.0 - lam) * e_n


def j_report(makespan_seconds: float, total_mobile_joules: float, refs: ReferenceRanges) -> float:
    """Scientific composite: 0.5 * L_norm + 0.5 * E_norm (clipped)."""
    return j_lambda(makespan_seconds, total_mobile_joules, refs, LATENCY_WEIGHT, clip=True)


def _add_transfer_components(
    bd: EnergyBreakdown,
    hop: str,
    duration: float,
    src_loc: Location,
    resources: ResourceConfig,
) -> None:
    """Same hop accounting as the engine (single implementation in energy_model)."""
    for field_name, joules in hop_energy_fields(hop, duration, src_loc, resources).items():
        setattr(bd, field_name, getattr(bd, field_name) + joules)


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
        task = result.graph_tasks.get(tid) if hasattr(result, "graph_tasks") else None
        workload = float(getattr(task, "compute_workload_bytes", 0.0)) if task else None
        if workload is None:
            # ScheduleResult does not carry workloads; fall back to the rate that
            # produced `dur` so C is recovered consistently for either model.
            workload = dur * resources.cpu_rate(rec.location)
        field_name = resources.compute_energy_field(rec.location)
        setattr(
            out[tid],
            field_name,
            getattr(out[tid], field_name)
            + resources.compute_energy_joules(rec.location, workload, dur),
        )

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
