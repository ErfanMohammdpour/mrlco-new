"""Frozen resource rates and power coefficients for the production scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .energy_model import MODEL_LEGACY, MODEL_PHYSICAL, EnergyModelSpec
from .radio import RADIO_LEGACY, RADIO_PHYSICAL, RadioModelSpec
from .model import Location
from .validate import require_nonneg_float, require_positive_rate


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

    def cpu_rate(self, loc: Location) -> float:
        if self.energy_model is not None and self.energy_model.is_physical:
            return self.cpu_rate_bytes_per_second(loc)
        return {
            Location.UE: self.ue_cpu_bytes_per_second,
            Location.MEC: self.mec_cpu_bytes_per_second,
            Location.HELPER: self.helper_cpu_bytes_per_second,
        }[loc]

    def hop_rate(self, hop: str) -> float:
        """Effective throughput [bytes/s]. physical radio uses B*eta (or SINR)."""
        if self.radio_model is not None and self.radio_model.is_physical:
            return self.radio_model.rate_for_hop(hop)
        return {
            "MEC_UL": self.mec_uplink_bytes_per_second,
            "MEC_DL": self.mec_downlink_bytes_per_second,
            "V2V": self.v2v_bytes_per_second,
        }[hop]

    # -- physical_v1 aware helpers (legacy path untouched) ----------------
    @property
    def physical(self) -> bool:
        return self.energy_model is not None and self.energy_model.is_physical

    def cpu_rate_bytes_per_second(
        self, loc: "Location", cycles_per_bit: float | None = None
    ) -> float:
        """Scheduling rate. physical_v1 derives it from f and cycles_per_bit so
        that C/f == duration exactly; legacy keeps the byte-rate table.

        `cycles_per_bit` carries a per-task override when the task provides one;
        None means "use the episode-global cycles_per_bit".
        """
        if self.physical:
            from .energy_model import tier_for_location

            xi = (
                float(self.energy_model.cycles_per_bit)
                if cycles_per_bit is None
                else float(cycles_per_bit)
            )
            return self.energy_model.tier(tier_for_location(loc)).cpu_rate_bytes_per_second(xi)
        return self.cpu_rate(loc)

    def cpu_rate_for_task(self, loc: "Location", task: Any) -> float:
        """Per-task rate (physical_v1 honours task.cycles_per_bit)."""
        xi = None
        if self.physical and task is not None and getattr(task, "cycles_per_bit", None) is not None:
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
        """
        import yaml

        if path is None:
            path = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"
        doc = yaml.safe_load(path.read_text())
        rates = doc["resource_rates"]
        power = doc["power"]
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
            energy_model=_select_energy_model(
                doc, model if energy_model is None else energy_model
            ),
            radio_model=_select_radio_model(
                doc, model if radio_model is None else radio_model
            ),
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
