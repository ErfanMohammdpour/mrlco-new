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


class EnergyTelemetryError(ValueError):
    """Telemetry was missing, malformed, non-finite or from another config."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


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


def build_energy_telemetry(result: Any, resources: Any) -> dict[str, Any]:
    """Three-boundary telemetry from the rollout's OWN ScheduleResult.

    No schedule/replay happens here: every number is read off `result`. The
    result's scheduler fingerprint must match the resolved resources, otherwise
    the telemetry would carry a valid-looking label for a different config.
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
    return {
        "requester_joules": energy_scalar(result, scope=SCOPE_REQUESTER),
        "mobile_joules": energy_scalar(result, scope=SCOPE_MOBILE),
        "system_joules": energy_scalar(result, scope=SCOPE_SYSTEM),
        "primary_scope": primary_scope,
        "primary_joules": energy_scalar(result, scope=primary_scope),
        "scheduler_config_sha256": result_sha,
        "schema_version": TELEMETRY_SCHEMA_VERSION,
    }


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
