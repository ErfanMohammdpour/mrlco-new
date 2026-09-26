"""`energy_telemetry_v1`: three-boundary episode scalars from ONE ScheduleResult.

Telemetry is pure post-processing of the schedule the rollout already produced:
`build_energy_telemetry(result, resources)` never schedules anything, so the
reward and the schedule cannot change when telemetry is on. The record carries
the requester/mobile/system boundaries, the primary boundary and its value, and
the scheduler fingerprint it was measured under -- a mismatched result is a loud
failure, never a relabelled number.

Aggregation is episode-weighted: every episode contributes exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .energy_cache import primary_scope_of
from .energy_scope import (
    ENERGY_SCOPES,
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    energy_scalar,
)
from .resources import resolved_config_sha256

TELEMETRY_SCHEMA_VERSION = "energy_telemetry_v1"

# Joules at the three accounting boundaries, in a fixed order.
JOULE_FIELDS = ("requester_joules", "mobile_joules", "system_joules")
TELEMETRY_FIELDS = (
    "requester_joules",
    "mobile_joules",
    "system_joules",
    "primary_scope",
    "primary_joules",
    "scheduler_config_sha256",
    "schema_version",
)

# The production CSV columns (the fingerprint/schema stay in the record only).
CSV_COLUMNS = {
    "requester_joules": "energy/requester_joules",
    "mobile_joules": "energy/mobile_joules",
    "system_joules": "energy/system_joules",
    "primary_joules": "energy/primary_joules",
    "primary_scope": "energy/primary_scope",
}

# `Average energy` is the historical per-task MOBILE metric. It is NOT converted
# to the primary scope: E4.1 keeps it and labels the boundary explicitly so a log
# reader can tell it apart from the scoped telemetry columns.
AVERAGE_ENERGY_LEGACY_KEY = "Average energy,"
AVERAGE_ENERGY_LEGACY_SCOPE = SCOPE_MOBILE


class EnergyTelemetryError(ValueError):
    """Telemetry was missing, malformed, non-finite or from another config."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _finite_number(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise EnergyTelemetryError(
            "%s must be a real number, got %r" % (name, value)
        ) from None
    if not math.isfinite(number):
        raise EnergyTelemetryError("%s is not finite: %r" % (name, value))
    return number


def _finite_nonneg(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise EnergyTelemetryError(
                "%s must be a real number, got %r" % (name, value)
            ) from None
    number = float(value)
    if not math.isfinite(number):
        raise EnergyTelemetryError("%s is not finite: %r" % (name, value))
    if number < 0.0:
        raise EnergyTelemetryError("%s is negative: %r" % (name, value))
    return number


#: compute calendar per physical tier (model.py RESOURCE_NAMES)
TIER_CALENDAR = {"ue": "UE_CPU", "helper": "HELPER_CPU", "mec": "MEC_CPU"}
#: workload-derived compute energy field per tier (model.EnergyBreakdown)
TIER_ENERGY_FIELD = {
    "ue": "ue_local_cpu_joules",
    "helper": "helper_compute_joules",
    "mec": "mec_compute_joules_optional",
}


def duration_consistency(result: Any, resources: Any) -> dict[str, Any]:
    """Workload-derived vs duration-consistent compute energy, per tier.

    `E_workload = kappa*C*f^2` comes from the workload; `E_duration = P(f)*T_busy`
    comes from the duration the scheduler actually used. They agree iff the timing
    model is physical, and otherwise differ by exactly `R_scheduled / R_physical`.
    Nothing here schedules or mutates anything: both numbers are read off the same
    `ScheduleResult` the reward came from.
    """
    spec = getattr(resources, "energy_model", None)
    physical = bool(spec is not None and spec.is_physical)
    intervals = list(getattr(result, "resource_intervals", []) or [])
    breakdown = getattr(result, "energy", None)
    tiers: dict[str, dict[str, float]] = {}
    for tier, calendar in TIER_CALENDAR.items():
        busy = sum(
            float(iv.end) - float(iv.start)
            for iv in intervals if str(getattr(iv, "resource", "")) == calendar
        )
        workload_j = float(getattr(breakdown, TIER_ENERGY_FIELD[tier], 0.0)) if breakdown else 0.0
        if not physical:
            duration_j = 0.0
        else:
            duration_j = spec.tier(tier).compute_joules_from_duration(busy)
        tiers[tier] = {
            "busy_seconds": busy,
            "workload_joules": workload_j,
            "duration_consistent_joules": duration_j,
            "ratio_workload_over_duration": (
                workload_j / duration_j if duration_j > 0.0 else 0.0
            ),
            "implied_power_w": (
                workload_j / busy if busy > 0.0 else 0.0
            ),
        }
    ratios = [
        row["ratio_workload_over_duration"] for row in tiers.values()
        if row["duration_consistent_joules"] > 0.0
    ]
    max_dev = max((abs(r - 1.0) for r in ratios), default=0.0)
    return {
        "model_is_physical": physical,
        "timing_model": str(getattr(resources, "timing_model", "")),
        "ratio_expected_scheduled_over_physical": {
            tier: _rate_ratio(resources, tier) for tier in TIER_CALENDAR
        },
        "tiers": tiers,
        "max_abs_ratio_minus_one": float(max_dev),
        "duration_consistent": bool(physical and max_dev <= 1e-9),
    }


def _rate_ratio(resources: Any, tier: str) -> float:
    """R_scheduled / R_physical for one tier (1.0 when timing is physical).

    `R_scheduled` is the rate the scheduler actually used (frozen table or the
    physical tiers, depending on the timing axis); `R_physical` is the rate implied
    by the ENERGY model's own f and cycles_per_bit. Comparing the two is exactly
    `E_workload / E_duration`.
    """
    from .model import Location

    location = {"ue": Location.UE, "helper": Location.HELPER, "mec": Location.MEC}[tier]
    spec = getattr(resources, "energy_model", None)
    if spec is None or not spec.is_physical:
        return 0.0
    try:
        scheduled = float(resources.cpu_rate(location))
        physical = float(spec.tier(tier).cpu_rate_bytes_per_second(spec.cycles_per_bit))
    except Exception:
        return 0.0
    if physical <= 0.0:
        return 0.0
    return scheduled / physical


def duration_consistency_kvs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flat CSV KVs for one `duration_consistency` record."""
    out = {
        "energy/consistency/model_is_physical": 1.0 if record.get("model_is_physical") else 0.0,
        "energy/consistency/duration_consistent": 1.0 if record.get("duration_consistent") else 0.0,
        "energy/consistency/max_abs_ratio_minus_one": float(
            record.get("max_abs_ratio_minus_one", 0.0) or 0.0
        ),
    }
    for tier, row in (record.get("tiers") or {}).items():
        out["energy/consistency/%s_ratio" % tier] = float(
            row.get("ratio_workload_over_duration", 0.0) or 0.0
        )
        out["energy/consistency/%s_implied_power_w" % tier] = float(
            row.get("implied_power_w", 0.0) or 0.0
        )
    return out


def build_energy_telemetry(result: Any, resources: Any, *,
                           constraint_costs: Any = None,
                           constraint_penalty: float = 0.0) -> dict[str, Any]:
    """Three-boundary telemetry from the rollout's OWN ScheduleResult.

    No schedule/replay happens here: every number is read off `result`. The
    result's scheduler fingerprint must match the resolved resources, otherwise
    the telemetry would carry a valid-looking label for a different config.

    When `constraint_costs` is active its raw/budget/signed/violation values and
    the applied penalty ride along, because with a parallel sampler the trainer
    never sees the worker env's controller state.
    """
    config_sha = str(resolved_config_sha256(resources))
    result_sha = str(getattr(result, "scheduler_config_sha256", "") or "")
    if not result_sha:
        raise EnergyTelemetryError(
            "ScheduleResult carries no scheduler_config_sha256; refusing to label "
            "telemetry from an unverifiable schedule"
        )
    if result_sha != config_sha:
        raise EnergyTelemetryError(
            "scheduler config mismatch: result was scheduled with %s but telemetry "
            "resources are %s" % (result_sha[:12], config_sha[:12])
        )
    primary_scope = primary_scope_of(resources)
    record = {
        "requester_joules": energy_scalar(result, scope=SCOPE_REQUESTER),
        "mobile_joules": energy_scalar(result, scope=SCOPE_MOBILE),
        "system_joules": energy_scalar(result, scope=SCOPE_SYSTEM),
        "primary_scope": primary_scope,
        "primary_joules": energy_scalar(result, scope=primary_scope),
        "scheduler_config_sha256": result_sha,
        "schema_version": TELEMETRY_SCHEMA_VERSION,
    }
    active = bool(getattr(constraint_costs, "active", False))
    if active:
        record["constraint_penalty_applied"] = float(constraint_penalty)
        for name, raw, budget, signed, violation in zip(
            constraint_costs.names,
            constraint_costs.raw,
            constraint_costs.budgets,
            constraint_costs.signed,
            constraint_costs.violations,
        ):
            record["constraint_%s_raw" % name] = float(raw)
            record["constraint_%s_budget" % name] = float(budget)
            record["constraint_%s_signed" % name] = float(signed)
            record["constraint_%s_violation" % name] = float(violation)
    return record


def validate_energy_telemetry(record: Any) -> dict[str, Any]:
    """Normalise and validate one record; raise on anything malformed."""
    if not isinstance(record, Mapping):
        raise EnergyTelemetryError(
            "telemetry record must be a mapping, got %r" % type(record).__name__
        )
    missing = [name for name in TELEMETRY_FIELDS if name not in record]
    if missing:
        raise EnergyTelemetryError("telemetry record is missing fields: %s" % missing)
    schema = str(record["schema_version"])
    if schema != TELEMETRY_SCHEMA_VERSION:
        raise EnergyTelemetryError(
            "telemetry schema %r != %r" % (schema, TELEMETRY_SCHEMA_VERSION)
        )
    scope = str(record["primary_scope"])
    if scope not in ENERGY_SCOPES:
        raise EnergyTelemetryError("primary_scope must be one of %s, got %r" % (list(ENERGY_SCOPES), scope))
    sha = str(record["scheduler_config_sha256"] or "")
    if not _is_sha256(sha):
        raise EnergyTelemetryError("scheduler_config_sha256 is not a 64-hex digest: %r" % (sha,))
    out = {name: _finite_nonneg(name, record[name]) for name in JOULE_FIELDS}
    out["primary_joules"] = _finite_nonneg("primary_joules", record["primary_joules"])
    out["primary_scope"] = scope
    out["scheduler_config_sha256"] = sha
    out["schema_version"] = schema
    for key, value in record.items():
        if isinstance(key, str) and key.startswith("constraint_"):
            out[key] = _finite_number(key, value)
    return out


def aggregate_energy_telemetry(records: Iterable[Any]) -> dict[str, Any]:
    """EPISODE-weighted mean over records: every episode counts exactly once."""
    rows = [validate_energy_telemetry(record) for record in records]
    if not rows:
        raise EnergyTelemetryError("no telemetry episodes to aggregate")
    scopes = {row["primary_scope"] for row in rows}
    if len(scopes) != 1:
        raise EnergyTelemetryError(
            "primary_scope differs across episodes: %s" % sorted(scopes)
        )
    shas = {row["scheduler_config_sha256"] for row in rows}
    if len(shas) != 1:
        raise EnergyTelemetryError(
            "scheduler fingerprint differs across episodes: %s"
            % sorted(value[:12] for value in shas)
        )
    n = float(len(rows))
    aggregate: dict[str, Any] = {
        field: sum(row[field] for row in rows) / n
        for field in (*JOULE_FIELDS, "primary_joules")
    }
    constraint_keys = {k for k in rows[0] if k.startswith("constraint_")}
    for row in rows[1:]:
        if {k for k in row if k.startswith("constraint_")} != constraint_keys:
            raise EnergyTelemetryError(
                "constraint telemetry keys differ across episodes"
            )
    for key in sorted(constraint_keys):
        aggregate[key] = sum(row[key] for row in rows) / n
    aggregate["primary_scope"] = rows[0]["primary_scope"]
    aggregate["scheduler_config_sha256"] = rows[0]["scheduler_config_sha256"]
    aggregate["schema_version"] = TELEMETRY_SCHEMA_VERSION
    aggregate["n_episodes"] = n
    return aggregate


def telemetry_csv_kvs(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """The five production CSV columns for an aggregated record."""
    checked = validate_energy_telemetry(aggregate)
    return {
        CSV_COLUMNS[field]: checked[field] for field in
        ("requester_joules", "mobile_joules", "system_joules", "primary_joules", "primary_scope")
    }


def legacy_average_energy_kvs(avg_energy: Any) -> dict[str, Any]:
    """The legacy `Average energy` value plus its explicit MOBILE-scope label."""
    return {
        AVERAGE_ENERGY_LEGACY_KEY: avg_energy,
        "energy/average_energy_scope": AVERAGE_ENERGY_LEGACY_SCOPE,
    }


def collect_energy_telemetry(samples_data_batch: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten the per-meta-task `energy_telemetry` lists from samples_data."""
    rows: list[dict[str, Any]] = []
    for samples in samples_data_batch or ():
        telemetry = None
        if hasattr(samples, "get"):
            telemetry = samples.get("energy_telemetry")
        if telemetry is None:
            continue
        for record in telemetry:
            if record is not None:
                rows.append(record)
    return rows
