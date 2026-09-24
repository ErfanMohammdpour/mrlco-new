"""Frozen resource rates and power coefficients for the production scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .energy_model import MODEL_LEGACY, MODEL_PHYSICAL, EnergyModelSpec
from .radio import RADIO_LEGACY, RADIO_PHYSICAL, RadioModelSpec
from .model import Location
from .validate import require_nonneg_float, require_positive_rate


# --------------------------------------------------------------------------- #
# Independent axes. Timing (rates -> durations -> schedule) and energy accounting
# must be switchable separately: turning on physical ENERGY must never silently
# change CPU/radio rates, makespan, start/finish times, calendar intervals or
# action feasibility. `model="legacy"|"physical_v1"` remains the coarse switch for
# old callers; the explicit kwargs below win and are recorded in provenance.
# --------------------------------------------------------------------------- #
TIMING_LEGACY = "legacy_frozen_rates"
TIMING_PHYSICAL = "physical_rates"
TIMING_MODELS = (TIMING_LEGACY, TIMING_PHYSICAL)
ENERGY_SCOPES = ("requester", "mobile", "system")


@dataclass(frozen=True)


class ResourceConfig:
    ue_cpu_bytes_per_second: float
    mec_cpu_bytes_per_second: float
    helper_cpu_bytes_per_second: float
    mec_uplink_bytes_per_second: float
    mec_downlink_bytes_per_second: float
    v2v_bytes_per_second: float
    rho_ue: float
    f_l: float
    zeta: float
    ptx_mec_w: float
    prx_mec_w: float
    ptx_v2v_w: float
    prx_v2v_w: float
    rho_helper: float
    f_v2v: float
    # None -> legacy normalized model (MARGO-SPEC-v0.1) is authoritative.
    # An EnergyModelSpec with model='physical_v1' switches rate derivation,
    # compute energy and radio accounting to SI physics.
    energy_model: Any | None = None
    # None -> legacy Mbps table. A physical RadioModelSpec derives the hop
    # rate from bandwidth_hz * spectral_efficiency (or SINR).
    radio_model: Any | None = None
    # --- independent timing axis (rates). Default: the frozen byte-rate table,
    # i.e. the schedule does not move when the energy model changes.
    timing_model: str = TIMING_LEGACY
    radio_timing_model: str = TIMING_LEGACY
    # tier source for timing_model="physical_rates" only; required in that case so
    # nothing is inferred from the energy model behind the caller's back
    timing_tiers: Any | None = None
    # requested accounting scope (provenance + validation; consumers use the
    # canonical accessor, never a hidden default)
    energy_scope: str = ""
    # hash of the yaml/document this config was resolved from, when applicable
    source_config_sha256: str = ""

    def __post_init__(self) -> None:
        require_positive_rate("ue_cpu_bytes_per_second", self.ue_cpu_bytes_per_second)
        require_positive_rate("mec_cpu_bytes_per_second", self.mec_cpu_bytes_per_second)
        require_positive_rate("helper_cpu_bytes_per_second", self.helper_cpu_bytes_per_second)
        require_positive_rate("mec_uplink_bytes_per_second", self.mec_uplink_bytes_per_second)
        require_positive_rate("mec_downlink_bytes_per_second", self.mec_downlink_bytes_per_second)
        require_positive_rate("v2v_bytes_per_second", self.v2v_bytes_per_second)
        require_nonneg_float("rho_ue", self.rho_ue)
        require_nonneg_float("f_l", self.f_l)
        require_nonneg_float("zeta", self.zeta)
        require_nonneg_float("ptx_mec_w", self.ptx_mec_w)
        require_nonneg_float("prx_mec_w", self.prx_mec_w)
        require_nonneg_float("ptx_v2v_w", self.ptx_v2v_w)
        require_nonneg_float("prx_v2v_w", self.prx_v2v_w)
        require_nonneg_float("rho_helper", self.rho_helper)
        require_nonneg_float("f_v2v", self.f_v2v)
        if self.timing_model not in TIMING_MODELS:
            raise ValueError(
                "timing_model must be one of %s, got %r"
                % (", ".join(TIMING_MODELS), self.timing_model)
            )
        if self.radio_timing_model not in TIMING_MODELS:
            raise ValueError(
                "radio_timing_model must be one of %s, got %r"
                % (", ".join(TIMING_MODELS), self.radio_timing_model)
            )
        if self.timing_model == TIMING_PHYSICAL and self.timing_tiers is None:
            raise ValueError(
                "timing_model=%r requires timing_tiers; refusing to infer the "
                "rate source from the energy model" % TIMING_PHYSICAL
            )
        if self.radio_timing_model == TIMING_PHYSICAL and self.radio_model is None:
            raise ValueError(
                "radio_timing_model=%r requires a radio_model" % TIMING_PHYSICAL
            )
        if self.energy_scope and self.energy_scope not in ENERGY_SCOPES:
            raise ValueError(
                "energy_scope must be one of %s, got %r"
                % (", ".join(ENERGY_SCOPES), self.energy_scope)
            )

    def cpu_rate(self, loc: Location) -> float:
        if self.timing_is_physical:
            return self.cpu_rate_bytes_per_second(loc)
        return {
            Location.UE: self.ue_cpu_bytes_per_second,
            Location.MEC: self.mec_cpu_bytes_per_second,
            Location.HELPER: self.helper_cpu_bytes_per_second,
        }[loc]

    def hop_rate(self, hop: str) -> float:
        """Effective throughput [bytes/s], from the RADIO TIMING axis.

        Radio energy accounting (`radio_model`) is a separate axis: enabling
        physical hop energy must not change hop durations.
        """
        if self.radio_timing_model == TIMING_PHYSICAL:
            return self.radio_model.rate_for_hop(hop)
        return {
            "MEC_UL": self.mec_uplink_bytes_per_second,
            "MEC_DL": self.mec_downlink_bytes_per_second,
            "V2V": self.v2v_bytes_per_second,
        }[hop]

    # -- axis predicates --------------------------------------------------
    @property
    def physical(self) -> bool:
        """True when ENERGY ACCOUNTING is physical. Never used for rates."""
        return self.energy_model is not None and self.energy_model.is_physical

    @property
    def timing_is_physical(self) -> bool:
        return self.timing_model == TIMING_PHYSICAL

    @property
    def radio_timing_is_physical(self) -> bool:
        return self.radio_timing_model == TIMING_PHYSICAL

    def cpu_rate_bytes_per_second(
        self, loc: "Location", cycles_per_bit: float | None = None
    ) -> float:
        """Scheduling rate. physical_v1 derives it from f and cycles_per_bit so
        that C/f == duration exactly; legacy keeps the byte-rate table.

        `cycles_per_bit` carries a per-task override when the task provides one;
        None means "use the episode-global cycles_per_bit".
        """
        if self.timing_is_physical:
            from .energy_model import tier_for_location

            source = self.timing_tiers
            xi = (
                float(source.cycles_per_bit)
                if cycles_per_bit is None
                else float(cycles_per_bit)
            )
            return source.tier(tier_for_location(loc)).cpu_rate_bytes_per_second(xi)
        return self.cpu_rate(loc)

    def cpu_rate_for_task(self, loc: "Location", task: Any) -> float:
        """Per-task rate (physical_v1 honours task.cycles_per_bit)."""
        xi = None
        if (
            self.timing_is_physical
            and task is not None
            and getattr(task, "cycles_per_bit", None) is not None
        ):
            xi = float(task.cycles_per_bit)
        return self.cpu_rate_bytes_per_second(loc, xi)

    def compute_energy_joules(
        self,
        loc: "Location",
        workload_bytes: float,
        duration: float,
        cycles_per_bit: float | None = None,
    ) -> float:
        """Compute energy for one task on one tier."""
        if self.physical:
            from .energy_model import tier_for_location

            xi = (
                float(self.energy_model.cycles_per_bit)
                if cycles_per_bit is None
                else float(cycles_per_bit)
            )
            return self.energy_model.tier(tier_for_location(loc)).compute_joules(
                workload_bytes, xi
            )
        if loc == Location.UE:
            return duration * self.rho_ue * (self.f_l ** self.zeta)
        if loc == Location.HELPER:
            return duration * self.rho_helper * (self.f_v2v ** self.zeta)
        return 0.0  # MEC compute is out of scope in the legacy model

    def compute_energy_field(self, loc: "Location") -> str:
        from .energy_model import compute_energy_field, tier_for_location

        return compute_energy_field(tier_for_location(loc))

    @classmethod
    def from_frozen_yaml(
        cls,
        path: Path | None = None,
        *,
        model: str | None = "legacy",
        energy_model: str | None = None,
        radio_model: str | None = None,
        timing_model: str | None = None,
        radio_timing_model: str | None = None,
        energy_scope: str | None = None,
    ) -> "ResourceConfig":
        """Build the frozen resource config.

        `model` selects the energy model explicitly:
          "legacy" (default) -> MARGO-SPEC-v0.1 numbers, byte-rate table, MEC
                                compute = 0. Existing callers keep their exact
                                previous behaviour unless they opt in.
          "physical_v1"      -> per-tier kappa/f physics, SI units; tier and
                                radio parameters are read from the yaml.
          None               -> use whatever the yaml block says (primary config)

        `energy_model` / `radio_model` override the two switches INDEPENDENTLY, so
        a radio-only audit (legacy compute physics, physical radio) is possible
        without conflating the two changes. When omitted they follow `model`.

        TIMING is a separate axis. `model="physical_v1"` keeps its historical
        meaning (physical accounting AND physical rates) for existing callers,
        while an explicit `timing_model=TIMING_LEGACY` gives physical accounting on
        the frozen rate table -- the mode the energy-constraint experiment needs.
        `model=None` honours the yaml for ACCOUNTING only and keeps timing legacy
        unless `timing_model` asks otherwise: nothing is inferred silently.
        """
        import yaml

        if path is None:
            path = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"
        import hashlib

        text = path.read_text()
        doc = yaml.safe_load(text)
        rates = doc["resource_rates"]
        power = doc["power"]
        resolved_energy = _select_energy_model(
            doc, model if energy_model is None else energy_model
        )
        resolved_radio = _select_radio_model(
            doc, model if radio_model is None else radio_model
        )
        coarse_physical = model == MODEL_PHYSICAL
        resolved_timing = (
            timing_model
            if timing_model is not None
            else (TIMING_PHYSICAL if coarse_physical else TIMING_LEGACY)
        )
        resolved_radio_timing = (
            radio_timing_model
            if radio_timing_model is not None
            else (TIMING_PHYSICAL if coarse_physical else TIMING_LEGACY)
        )
        declared_scope = (
            energy_scope
            if energy_scope is not None
            else str(getattr(resolved_energy, "energy_scope", "") or "")
        )
        return cls(
            ue_cpu_bytes_per_second=float(rates["ue_cpu_bytes_per_second"]),
            mec_cpu_bytes_per_second=float(rates["mec_cpu_bytes_per_second"]),
            helper_cpu_bytes_per_second=float(rates["helper_cpu_bytes_per_second"]),
            mec_uplink_bytes_per_second=float(rates["mec_uplink_bytes_per_second"]),
            mec_downlink_bytes_per_second=float(rates["mec_downlink_bytes_per_second"]),
            v2v_bytes_per_second=float(rates["v2v_bytes_per_second"]),
            rho_ue=float(power["rho_ue"]),
            f_l=float(power["f_l"]),
            zeta=float(power["zeta"]),
            ptx_mec_w=float(power["ptx_mec_w"]),
            prx_mec_w=float(power["prx_mec_w"]),
            ptx_v2v_w=float(power["ptx_v2v_w"]),
            prx_v2v_w=float(power["prx_v2v_w"]),
            rho_helper=float(power["rho_helper"]),
            f_v2v=float(power["f_v2v"]),
            energy_model=resolved_energy,
            radio_model=resolved_radio,
            timing_model=resolved_timing,
            radio_timing_model=resolved_radio_timing,
            timing_tiers=resolved_energy if resolved_timing == TIMING_PHYSICAL else None,
            energy_scope=declared_scope,
            source_config_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


def _select_energy_model(doc: dict, model: str | None) -> EnergyModelSpec:
    """Resolve the energy model from the yaml block plus an explicit override."""
    if model == MODEL_LEGACY:
        return EnergyModelSpec(model=MODEL_LEGACY)
    spec = EnergyModelSpec.from_dict(doc.get("energy_model"))
    if model is None or model == spec.model:
        return spec
    if model == MODEL_PHYSICAL:
        # force physical_v1 while still reading tiers/radio from the yaml
        return EnergyModelSpec.from_dict({**doc.get("energy_model", {}), "model": MODEL_PHYSICAL})
    raise ValueError("model must be legacy, physical_v1, or None; got %r" % (model,))


def _select_radio_model(doc: dict, model: str | None) -> RadioModelSpec:
    """Same explicit-selection rule as the energy model."""
    if model == MODEL_LEGACY:
        return RadioModelSpec(model=RADIO_LEGACY)
    spec = RadioModelSpec.from_dict(doc.get("radio_model"))
    if model is None or model == spec.model:
        return spec
    if model == MODEL_PHYSICAL:
        return RadioModelSpec.from_dict({**doc.get("radio_model", {}), "model": RADIO_PHYSICAL})
    raise ValueError("model must be legacy, physical_v1, or None; got %r" % (model,))
