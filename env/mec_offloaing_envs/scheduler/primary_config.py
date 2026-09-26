"""The ONE resolved scheduler config for primary train/validation/workers.

Axes are passed EXPLICITLY: the coarse `model="physical_v1"` also switches the
rates (its historical meaning), which is not what the energy-constraint
experiment wants. Here timing stays on the frozen rate table while energy
accounting is physical and scoped to the system boundary.

Import this module rather than rebuilding a config anywhere else: a second
construction site is how a model gets dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .resources import ResourceConfig

PRIMARY_SCHEDULER_AXES: dict[str, str] = {
    "timing_model": "legacy_frozen_rates",
    "radio_timing_model": "legacy_frozen_rates",
    "energy_model": "physical_v1",
    "radio_model": "physical_v1",
    "energy_scope": "system",
}

DEFAULT_FROZEN_YAML = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"


#: Co-physical preset: the scheduled duration and the joule accounting come from
#: the SAME tier model, so E = P(f)*T_busy holds exactly for the reported energy.
PHYSICAL_SCHEDULER_AXES: dict[str, str] = {
    "timing_model": "physical_rates",
    "radio_timing_model": "physical_rates",
    "energy_model": "physical_v1",
    "radio_model": "physical_v1",
    "energy_scope": "system",
}


class EnergyPhysicsMismatch(RuntimeError):
    """Energy arithmetic and scheduled time come from different machine models."""


def energy_timing_consistency(config: Any) -> dict[str, Any]:
    """Describe whether `config`'s energy and timing share one physics.

    `mixed_physics` is True when joule accounting is physical while the scheduled
    durations come from the frozen rate table: the reported energy is then NOT
    `P(f) x T_scheduled`. Pure inspection - nothing is mutated.
    """
    energy_physical = bool(getattr(config, "physical", False))
    timing_physical = bool(getattr(config, "timing_is_physical", False))
    radio_physical = bool(getattr(config, "radio_timing_is_physical", False))
    mixed = energy_physical and not timing_physical
    return {
        "energy_model": str(getattr(getattr(config, "energy_model", None), "model", "")),
        "timing_model": str(getattr(config, "timing_model", "")),
        "radio_timing_model": str(getattr(config, "radio_timing_model", "")),
        "energy_accounting_physical": energy_physical,
        "timing_physical": timing_physical,
        "radio_timing_physical": radio_physical,
        "mixed_physics": mixed,
        "label": "mixed_physics_legacy_timing_physical_energy" if mixed
        else ("co_physical" if energy_physical and timing_physical else "legacy"),
    }


def require_energy_timing_consistency(config: Any) -> dict[str, Any]:
    """Raise unless energy arithmetic and timing share one physics."""
    facts = energy_timing_consistency(config)
    if facts["mixed_physics"]:
        raise EnergyPhysicsMismatch(
            "energy is %s but timing is %s: reported joules are not P(f)*T_scheduled"
            % (facts["energy_model"], facts["timing_model"])
        )
    return facts


def resolved_physical_scheduler_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResourceConfig:
    """The co-physical counterpart of `resolved_primary_scheduler_config`."""
    merged = dict(overrides or {})
    unknown = set(merged) - set(PRIMARY_SCHEDULER_AXES)
    if unknown:
        raise ValueError("unknown scheduler axis override(s): %s" % sorted(unknown))
    for key, value in PHYSICAL_SCHEDULER_AXES.items():
        merged.setdefault(key, value)
    return resolved_primary_scheduler_config(path=path, overrides=merged)


def resolved_primary_scheduler_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResourceConfig:
    axes = dict(PRIMARY_SCHEDULER_AXES)
    axes.update(dict(overrides or {}))
    unknown = set(axes) - set(PRIMARY_SCHEDULER_AXES)
    if unknown:
        raise ValueError("unknown scheduler axis override(s): %s" % sorted(unknown))
    return ResourceConfig.from_frozen_yaml(
        Path(path) if path else DEFAULT_FROZEN_YAML,
        energy_model=axes["energy_model"],
        radio_model=axes["radio_model"],
        timing_model=axes["timing_model"],
        radio_timing_model=axes["radio_timing_model"],
        energy_scope=axes["energy_scope"],
    )


def evaluator_scheduler_config(
    scheduler_config: ResourceConfig | None = None,
    *,
    standalone: bool = False,
    standalone_axes: Mapping[str, Any] | None = None,
) -> ResourceConfig:
    """The config a held-out/meta evaluator must use.

    Primary evaluation consumes the SAME resolved config as training (equal
    fingerprint), so held-out numbers are computed under the training contract.
    A standalone audit may ask for an independent config, but then every axis must
    be stated explicitly: no ambiguous default, and the result must not be
    presented as a primary number.
    """
    if scheduler_config is not None:
        if standalone:
            raise ValueError("pass either scheduler_config or standalone_axes, not both")
        return scheduler_config
    if not standalone:
        return resolved_primary_scheduler_config()
    axes = dict(standalone_axes or {})
    missing = set(PRIMARY_SCHEDULER_AXES) - set(axes)
    if missing:
        raise ValueError(
            "standalone evaluator config must state every axis explicitly; "
            "missing %s" % sorted(missing)
        )
    return resolved_primary_scheduler_config(overrides=axes)
