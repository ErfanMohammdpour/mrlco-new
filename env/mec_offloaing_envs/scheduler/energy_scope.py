"""Canonical energy-boundary accessor (4.2, part 1: the API only).

One source of truth for the three accounting boundaries, so no consumer picks a
scalar by habit. The definitions are frozen:

    requester = total_requester_joules            (UE / requester boundary)
    mobile    = total_mobile_joules               (requester + helper)
    system    = total_system_joules               (mobile + MEC compute + MEC/RSU TX)

Every caller must name its scope; there is no default and no inference. This
commit ADDS the accessor and does not change any consumer's number: the migration
to explicit scopes happens next, and only then does any scope change semantics.
"""

from __future__ import annotations

import math
from typing import Any

from .energy_model import (  # authoritative definitions live here
    ENERGY_SCOPES,
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
)

__all__ = [
    "ENERGY_SCOPES",
    "SCOPE_REQUESTER",
    "SCOPE_MOBILE",
    "SCOPE_SYSTEM",
    "SCOPE_FIELDS",
    "energy_scalar",
    "configured_energy_scalar",
]

SCOPE_FIELDS = {
    SCOPE_REQUESTER: "total_requester_joules",
    SCOPE_MOBILE: "total_mobile_joules",
    SCOPE_SYSTEM: "total_system_joules",
}


def _breakdown_of(result_or_breakdown: Any) -> Any:
    """Accept a ScheduleResult or an EnergyBreakdown; reject anything else."""
    if hasattr(result_or_breakdown, "energy"):
        return result_or_breakdown.energy
    if any(hasattr(result_or_breakdown, field) for field in SCOPE_FIELDS.values()):
        return result_or_breakdown
    raise TypeError(
        "energy_scalar expects a ScheduleResult or EnergyBreakdown, got %r"
        % type(result_or_breakdown).__name__
    )


def energy_scalar(result_or_breakdown: Any, *, scope: str) -> float:
    """Joules at the requested boundary. `scope` is required and validated."""
    if scope is None or not str(scope):
        raise ValueError(
            "energy scope is required: pass one of %s explicitly" % (list(ENERGY_SCOPES),)
        )
    scope = str(scope)
    if scope not in SCOPE_FIELDS:
        raise ValueError(
            "unknown energy scope %r; allowed: %s" % (scope, list(ENERGY_SCOPES))
        )
    breakdown = _breakdown_of(result_or_breakdown)
    field = SCOPE_FIELDS[scope]
    if not hasattr(breakdown, field):
        raise ValueError(
            "energy breakdown has no %s for scope %r" % (field, scope)
        )
    value = float(getattr(breakdown, field))
    if not math.isfinite(value):
        raise ValueError("energy scalar %s is not finite: %r" % (field, value))
    if value < -1e-9:
        raise ValueError("energy scalar %s is negative: %r" % (field, value))
    return max(0.0, value)


def configured_energy_scalar(result_or_breakdown: Any, resources: Any) -> float:
    """Convenience wrapper: the scope comes from the resolved config, explicitly."""
    scope = str(getattr(resources, "energy_scope", "") or "")
    if not scope:
        scope = str(getattr(getattr(resources, "energy_model", None), "energy_scope", "") or "")
    if not scope:
        raise ValueError(
            "resources carry no energy_scope; refusing to guess a boundary"
        )
    return energy_scalar(result_or_breakdown, scope=scope)
