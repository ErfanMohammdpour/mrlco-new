#!/usr/bin/env python3
"""v2 event-based energy ledger — ONE physical implementation, shared with frozen v1.

This module does NOT define a second energy model. It imports the frozen v1 model
(`env/mec_offloaing_envs/scheduler/energy_model.py` for `EnergyModelSpec`/`TierSpec`/
`hop_energy_fields`, and `model.EnergyBreakdown` for the accounting boundaries) and feeds it
the events of a v2 `V2ScheduleResult`.

Accounting boundaries (identical objects and names as v1, never overwritten):

    E_requester = UE CPU + UE radio (UL TX, DL RX when enabled, V2V TX/RX)
    E_mobile    = E_requester + helper CPU + helper radio
    E_system    = E_mobile + MEC compute + MEC TX          <- frozen primary scope

Per-event model (all SI: cycles, Hz, W, J, seconds, bytes):

    C_executed        = executed_bytes * 8 * cycles_per_bit          [cycles]
    P_cpu(tier)       = kappa_tier * f_tier^3                        [W]
    E_cpu             = P_cpu(tier) * cpu_service_seconds            [J]   == integral P dt
    E_tx(hop)         = P_tx(transmitter tier) * active_service_s     [J]

`E_cpu = kappa*C*f^2` (the printed form in the literature) and `P(f)*T` coincide exactly
when the scheduled duration is the physical one, `T = C/f`. The frozen rate table and the
frozen `f` values are NOT consistent (MEC 10 MiB/s vs f=10 GHz with 300 cycles/bit), so v2
uses the **duration form** — the same allocation the scheduler actually executed — and
reports the work form and the ratio as an explicit provenance diagnostic instead of mixing
the two. Active service time, not channel occupancy, is used for radio energy: a
mid-transfer outage pause is not transmission.

Energy that v2 does NOT model is listed in `UNMODELED` and reported as out-of-scope rather
than as a measured zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from env.mec_offloaing_envs.scheduler.energy_api import ReferenceRanges
from env.mec_offloaing_envs.scheduler.energy_model import (
    ENERGY_SCOPES, MODEL_PHYSICAL, EnergyModelSpec, hop_energy_fields,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (
    SCOPE_MOBILE, SCOPE_REQUESTER, SCOPE_SYSTEM, energy_scalar,
)
from env.mec_offloaing_envs.scheduler.model import EnergyBreakdown, Location

from spec.automotive_training.v2.shared_scheduler import HOP_TO_LINK, V2ScheduleResult

#: energy components that are deliberately OUT of the modelled boundary. They are reported
#: as unmodelled, never as a measured zero.
UNMODELED = (
    "receiver/static radio power (include_rx_energy=False in the frozen config)",
    "CPU idle/leakage power between tasks",
    "MEC/RSU static (non-compute) power: server_static_power_w is null",
    "backhaul / core-network transport energy",
    "battery state dynamics, charging efficiency and reserve accounting",
    "warm-standby reservation energy (the reservation costs no joules in this model)",
)

_V2_TO_V1_HOP = {
    "MEC_UL": "MEC_UL",
    "MEC_ULH": "MEC_UL",
    "MEC_DL": "MEC_DL",
    "MEC_DLH": "MEC_DL",
    "V2V": "V2V",
}

_LOC_TO_ENUM = {"UE": Location.UE, "MEC": Location.MEC, "HELPER": Location.HELPER}
_TIER_OF_LOC = {"UE": "ue", "MEC": "mec", "HELPER": "helper"}

#: hoisted once: `hop_energy_fields` reads `resources.energy_model` and, in physical mode,
#: uses nothing else from it.
class _ResourceShim:
    def __init__(self, spec: EnergyModelSpec):
        self.energy_model = spec
        self.energy_scope = spec.energy_scope


class V2EnergyError(RuntimeError):
    """Raised when an event cannot be priced under the frozen physical model."""


@dataclass
class V2EnergyLedger:
    """System/mobile/requester energy for one v2 schedule, plus the provenance."""

    breakdown: EnergyBreakdown
    background: EnergyBreakdown
    foreground: EnergyBreakdown
    primary_scope: str
    spec_sha256: str
    cycles_per_bit: float
    cpu_duration_form_j: float
    cpu_work_form_j: float
    events: int
    per_tier_cpu_seconds: dict = field(default_factory=dict)
    per_hop_service_seconds: dict = field(default_factory=dict)
    warnings: tuple = ()

    # -- boundaries -------------------------------------------------------
    @property
    def requester_joules(self) -> float:
        return energy_scalar(self.breakdown, scope=SCOPE_REQUESTER)

    @property
    def mobile_joules(self) -> float:
        return energy_scalar(self.breakdown, scope=SCOPE_MOBILE)

    @property
    def system_joules(self) -> float:
        return energy_scalar(self.breakdown, scope=SCOPE_SYSTEM)

    @property
    def primary_joules(self) -> float:
        return energy_scalar(self.breakdown, scope=self.primary_scope)

    @property
    def foreground_joules(self) -> float:
        return energy_scalar(self.foreground, scope=self.primary_scope)

    @property
    def background_joules(self) -> float:
        return energy_scalar(self.background, scope=self.primary_scope)

    @property
    def cpu_work_over_duration(self) -> float:
        """R_scheduled/R_physical; the v1/v2 model-consistency diagnostic."""
        if self.cpu_duration_form_j <= 0.0:
            return float("nan")
        return self.cpu_work_form_j / self.cpu_duration_form_j

    def as_dict(self) -> dict:
        return {
            "primary_scope": self.primary_scope,
            "requester_joules": self.requester_joules,
            "mobile_joules": self.mobile_joules,
            "system_joules": self.system_joules,
            "primary_joules": self.primary_joules,
            "foreground_joules": self.foreground_joules,
            "background_joules": self.background_joules,
            "components": self.breakdown.as_dict(),
            "background_components": self.background.as_dict(),
            "energy_model_sha256": self.spec_sha256,
            "cycles_per_bit": self.cycles_per_bit,
            "cpu_duration_form_j": self.cpu_duration_form_j,
            "cpu_work_form_j": self.cpu_work_form_j,
            "cpu_work_over_duration": self.cpu_work_over_duration,
            "per_tier_cpu_seconds": dict(self.per_tier_cpu_seconds),
            "per_hop_service_seconds": dict(self.per_hop_service_seconds),
            "priced_events": int(self.events),
            "unmodeled": list(UNMODELED),
            "warnings": list(self.warnings),
        }


def frozen_energy_spec(path: Path | None = None) -> EnergyModelSpec:
    """The frozen v1 energy configuration (validates the scope declaration)."""
    spec = EnergyModelSpec.from_frozen_yaml(path)
    if not spec.is_physical:
        raise V2EnergyError(
            "the v2 ledger prices events with the physical model; the frozen config says "
            "model=%r" % spec.model)
    return spec


def energy_spec_sha256(spec: EnergyModelSpec) -> str:
    import hashlib
    import json

    payload = json.dumps(spec.as_dict(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _add(breakdown: EnergyBreakdown, fields: Mapping[str, float]) -> None:
    for name, value in fields.items():
        if value:
            if not math.isfinite(float(value)):
                raise V2EnergyError("non-finite energy for %s: %r" % (name, value))
            setattr(breakdown, name, getattr(breakdown, name) + float(value))


def _tier_of(location: str, spec: EnergyModelSpec) -> str:
    tier = _TIER_OF_LOC[location]
    if tier not in spec.tiers:
        raise V2EnergyError("no tier %r in the frozen energy model" % tier)
    return tier


def schedule_energy(result: V2ScheduleResult, *, spec: EnergyModelSpec | None = None,
                    background_dag_ids: Sequence[str] = (),
                    dag_filter: Sequence[str] | None = None) -> V2EnergyLedger:
    """Price every event of a v2 `V2ScheduleResult`.

    `background_dag_ids` splits the ledger into foreground-attributed and background
    energy. Both are charged at the same tiers; only the attribution differs, so the
    system total is the sum of the two and never double counts.
    """
    spec = spec or frozen_energy_spec()
    background_ids = {str(d) for d in background_dag_ids}
    include = None if dag_filter is None else {str(d) for d in dag_filter}
    resources = _ResourceShim(spec)
    total = EnergyBreakdown()
    background = EnergyBreakdown()
    per_tier_seconds: dict = {}
    per_hop_seconds: dict = {}
    warnings: list = []
    events = 0
    duration_form_cpu_j = 0.0
    work_form_cpu_j = 0.0

    # ---- CPU: attempted remote work + final executed work --------------------
    for tm in result.timings.values():
        if include is not None and str(tm.dag_id) not in include:
            continue
        if tm.contact_failure:
            # the failed remote attempt is billed at the tier where it ran and the local
            # restart at the requester tier; nothing is free and nothing is double counted
            attempts = [(str(tm.attempted_location), float(tm.cpu_attempt_seconds),
                         float(tm.cpu_wasted_bytes)),
                        (str(tm.location), float(tm.cpu_s), float(tm.cpu_restart_bytes))]
        else:
            attempts = [(str(tm.location), float(tm.cpu_s), float(tm.cpu_executed_bytes))]
        for location, seconds, bytes_here in attempts:
            if seconds <= 0.0:
                continue
            tier = _tier_of(location, spec)
            price = spec.tiers[tier]
            joule = price.compute_joules_from_duration(seconds)
            field_name = {"ue": "ue_local_cpu_joules",
                          "helper": "helper_compute_joules",
                          "mec": "mec_compute_joules_optional"}[tier]
            _add(total, {field_name: joule})
            if str(tm.dag_id) in background_ids:
                _add(background, {field_name: joule})
            per_tier_seconds[tier] = per_tier_seconds.get(tier, 0.0) + seconds
            duration_form_cpu_j += joule
            work_form_cpu_j += price.compute_joules(bytes_here, spec.cycles_per_bit)
            events += 1
        for value, label in ((tm.cpu_wasted_bytes, "wasted work"),
                             (tm.cpu_restart_bytes, "restart work"),
                             (tm.cpu_attempt_seconds, "attempt seconds")):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise V2EnergyError("non-finite/negative %s in %s/%s: %r"
                                    % (label, tm.dag_id, tm.task_id, value))

    # ---- radio: active service time per booked transfer ----------------------
    for rec in result.radio_ledger:
        if include is not None and str(rec["dag_id"]) not in include:
            continue
        hop = _V2_TO_V1_HOP.get(str(rec["hop"]))
        if hop is None:
            raise V2EnergyError("unknown hop %r in the v2 radio ledger" % (rec["hop"],))
        service = float(rec["service_s"])
        if service < 0.0 or not math.isfinite(service):
            raise V2EnergyError("non-finite radio service time for %r" % (rec,))
        per_hop_seconds[hop] = per_hop_seconds.get(hop, 0.0) + service
        if service <= 0.0:
            continue
        fields = hop_energy_fields(hop, service, _LOC_TO_ENUM[str(rec["src"])], resources)
        if not fields:
            warnings.append("hop %s with src=%s priced to zero joule (check include_rx_energy)"
                            % (hop, rec["src"]))
        _add(total, fields)
        if str(rec["dag_id"]) in background_ids:
            _add(background, fields)
        events += 1

    for name, value in total.as_dict().items():
        if name.startswith("total_"):
            continue
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise V2EnergyError("energy component %s is not a non-negative finite number: %r"
                                % (name, value))

    foreground = EnergyBreakdown()
    for fname in EnergyBreakdown.COMPONENT_FIELDS:
        setattr(foreground, fname,
                getattr(total, fname) - getattr(background, fname))
    # attribution identity: foreground + background == total, component by component
    for fname in EnergyBreakdown.COMPONENT_FIELDS:
        lhs = getattr(foreground, fname) + getattr(background, fname)
        if abs(lhs - getattr(total, fname)) > 1e-9 * max(1.0, abs(getattr(total, fname))):
            raise V2EnergyError("foreground/background attribution does not sum to the total "
                                "for %s" % fname)

    return V2EnergyLedger(
        breakdown=total, background=background, foreground=foreground,
        primary_scope=str(spec.energy_scope), spec_sha256=energy_spec_sha256(spec),
        cycles_per_bit=float(spec.cycles_per_bit),
        cpu_duration_form_j=float(duration_form_cpu_j),
        cpu_work_form_j=float(work_form_cpu_j),
        events=int(events), per_tier_cpu_seconds=per_tier_seconds,
        per_hop_service_seconds=per_hop_seconds, warnings=tuple(warnings))


def reference_ranges_from_plans(plans: Mapping[str, V2ScheduleResult], *,
                                spec: EnergyModelSpec | None = None,
                                scheduler_config_sha256: str) -> ReferenceRanges:
    """All-UE / all-MEC / all-HELPER reference ranges measured ON THE V2 SCHEDULER.

    `E_ue/E_mec/E_helper` are PLAN energies at the frozen primary scope (they are scope
    quantities, not boundaries), exactly as the v1 schema defines them.
    """
    spec = spec or frozen_energy_spec()
    out = {}
    for name in ("ue", "mec", "helper"):
        key = "all_%s" % name
        if key not in plans:
            raise V2EnergyError("missing reference plan %r (have %s)"
                                % (key, sorted(plans)))
        led = schedule_energy(plans[key], spec=spec)
        out["L_%s" % name] = float(plans[key].makespan_s)
        out["E_%s" % name] = float(energy_scalar(led.breakdown, scope=spec.energy_scope))
    if not (0.0 <= min(out.values())) or not all(math.isfinite(v) for v in out.values()):
        raise V2EnergyError("reference ranges must be finite and non-negative: %r" % out)
    return ReferenceRanges(
        L_ue=out["L_ue"], L_mec=out["L_mec"], L_helper=out["L_helper"],
        E_ue=out["E_ue"], E_mec=out["E_mec"], E_helper=out["E_helper"],
        source="v2_all_location_plans_on_the_shared_scheduler",
        energy_scope=str(spec.energy_scope),
        scheduler_config_sha256=str(scheduler_config_sha256))


__all__ = [
    "UNMODELED", "V2EnergyError", "V2EnergyLedger", "energy_spec_sha256",
    "frozen_energy_spec", "reference_ranges_from_plans", "schedule_energy",
    "ENERGY_SCOPES", "MODEL_PHYSICAL", "SCOPE_REQUESTER", "SCOPE_MOBILE", "SCOPE_SYSTEM",
    "HOP_TO_LINK",
]
