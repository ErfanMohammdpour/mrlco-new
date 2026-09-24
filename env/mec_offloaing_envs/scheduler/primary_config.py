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
