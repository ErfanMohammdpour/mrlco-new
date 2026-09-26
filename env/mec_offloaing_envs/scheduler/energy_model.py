"""Physical energy model (`physical_v1`) vs the legacy normalized model.

Contract (approved 2026-09-21):

    C_i = D_i * cycles_per_bit                      # cycles (task property)
    T_cpu(i, x) = C_i / f_x                         # s
    E_cpu(i, x) = kappa_x * C_i * f_x^2             # J     (== (kappa_x f_x^3) * T)

Three accounting boundaries are kept SIMULTANEOUSLY, never overwritten:

    total_requester_joules = E_UE
    total_mobile_joules    = E_UE + E_helper
    total_system_joules    = E_UE + E_helper + E_MEC/RSU      <- primary scope

Model selection:
    energy_model="legacy"       reproduces MARGO-SPEC-v0.1 bit-for-bit
                                (E_local = t*rho_ue*f_l^zeta, MEC compute 0,
                                 rho_helper=0.7, rx powers included)
    energy_model="physical_v1"  kappa/C/f physics per tier, SI units, TX-only
                                radio by default (no literature-pinned p_rx)

Units are SI everywhere: cycles, cycles/s, bytes, bytes/s (== 8*bit/s),
seconds, watts, joules. `cycles_per_bit` is a TASK property; `f_hz` and
`kappa` are TIER properties. Never mix cycles-per-bit and cycles-per-byte
without converting: 1 cycle/bit == 8 cycles/byte.

kappa_MEC == 0 is forbidden in physical mode (that is exactly the asymmetry
this model exists to remove); it is only reachable through `legacy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .validate import require_finite, require_nonneg_float, require_positive_rate

MODEL_LEGACY = "legacy"
MODEL_PHYSICAL = "physical_v1"
ENERGY_MODELS = (MODEL_LEGACY, MODEL_PHYSICAL)

SCOPE_REQUESTER = "requester"
SCOPE_MOBILE = "mobile"
SCOPE_SYSTEM = "system"
ENERGY_SCOPES = (SCOPE_REQUESTER, SCOPE_MOBILE, SCOPE_SYSTEM)

TIER_UE = "ue"
TIER_HELPER = "helper"
TIER_MEC = "mec"
TIERS = (TIER_UE, TIER_HELPER, TIER_MEC)

DEFAULT_CYCLES_PER_BIT = 300.0  # Zhao2018 Table I (sweep 300/500/1000)

# --- radio field names on EnergyBreakdown (single source of truth) ---------
F_UE_MEC_UL = "ue_mec_uplink_joules"
F_UE_MEC_DL = "ue_mec_downlink_joules"
F_UE_V2V_TX = "ue_v2v_tx_joules"
F_UE_V2V_RX = "ue_v2v_rx_joules"
F_HELPER_CPU = "helper_compute_joules"
F_HELPER_V2V_TX = "helper_v2v_tx_joules"
F_HELPER_V2V_RX = "helper_v2v_rx_joules"
F_UE_CPU = "ue_local_cpu_joules"
F_MEC_CPU = "mec_compute_joules_optional"
F_MEC_TX = "mec_tx_joules_optional"


@dataclass(frozen=True)
class TierSpec:
    """CPU frequency and switched-capacitance coefficient for one tier."""

    f_hz: float
    kappa: float
    source: str = "unspecified"

    def __post_init__(self) -> None:
        require_positive_rate("f_hz", self.f_hz)
        require_nonneg_float("kappa", self.kappa)

    @property
    def implied_dynamic_power_w(self) -> float:
        """P = kappa * f^3 (dynamic CPU power at full frequency)."""
        return float(self.kappa) * (float(self.f_hz) ** 3)

    def cpu_rate_bytes_per_second(self, cycles_per_bit: float) -> float:
        """Rate consistent with C/f == duration:  rate = f / (8 * cycles_per_bit)."""
        require_positive_rate("cycles_per_bit", float(cycles_per_bit))
        return float(self.f_hz) / (8.0 * float(cycles_per_bit))

    def compute_joules(self, workload_bytes: float, cycles_per_bit: float) -> float:
        """E = kappa * C * f^2 with C = bytes * 8 * cycles_per_bit."""
        cycles = float(workload_bytes) * 8.0 * float(cycles_per_bit)
        return float(self.kappa) * cycles * (float(self.f_hz) ** 2)

    def compute_seconds(self, workload_bytes: float, cycles_per_bit: float) -> float:
        cycles = float(workload_bytes) * 8.0 * float(cycles_per_bit)
        return cycles / float(self.f_hz)

    def compute_joules_from_duration(self, seconds: float) -> float:
        """``E = P(f) * T`` - the duration-consistent form of ``E = kappa*C*f^2``.

        ``compute_joules`` derives the energy from the WORKLOAD (``C/f`` gives the
        physical duration), so it agrees with ``P(f)*T`` only when the scheduled
        duration is also physical. Under the frozen rate table the two disagree by
        exactly ``R_scheduled / R_physical``; this accessor makes that measurable.
        """
        return self.implied_dynamic_power_w * require_nonneg_float("seconds", float(seconds))

    def as_dict(self) -> dict[str, Any]:
        return {
            "f_hz": self.f_hz,
            "kappa": self.kappa,
            "implied_dynamic_power_w": self.implied_dynamic_power_w,
            "source": self.source,
        }


@dataclass(frozen=True)
class EnergyModelSpec:
    """Model selector + physical parameters + accounting boundary."""

    model: str = MODEL_LEGACY
    energy_scope: str = SCOPE_MOBILE
    cycles_per_bit: float = DEFAULT_CYCLES_PER_BIT
    include_rx_energy: bool = False
    tiers: dict[str, TierSpec] = field(default_factory=dict)
    ue_tx_w: float | None = None
    mec_tx_w: float | None = None
    helper_tx_w: float | None = None
    ue_rx_w: float | None = None
    helper_rx_w: float | None = None
    # Sensitivity only; not used by the primary dynamic model.
    server_static_power_w: float | None = None
    server_utilization_model: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.model not in ENERGY_MODELS:
            raise ValueError("model must be one of %s, got %r" % (ENERGY_MODELS, self.model))
        if self.energy_scope not in ENERGY_SCOPES:
            raise ValueError(
                "energy_scope must be one of %s, got %r" % (ENERGY_SCOPES, self.energy_scope)
            )
        if self.model == MODEL_PHYSICAL:
            require_positive_rate("cycles_per_bit", float(self.cycles_per_bit))
            if not self.tiers:
                raise ValueError("physical_v1 requires per-tier specs")
            missing = [t for t in TIERS if t not in self.tiers]
            if missing:
                raise ValueError("physical_v1 missing tiers: %s" % missing)
            for name in TIERS:
                tier = self.tiers[name]
                if tier.kappa <= 0.0:
                    raise ValueError(
                        "physical_v1 requires kappa > 0 for tier %r (kappa=%s). "
                        "A zero MEC compute coefficient is exactly the legacy "
                        "asymmetry this model removes." % (name, tier.kappa)
                    )
            for name in ("ue_tx_w", "mec_tx_w", "helper_tx_w"):
                if getattr(self, name) is None:
                    raise ValueError("physical_v1 requires %s" % name)
                require_nonneg_float(name, float(getattr(self, name)))
            if self.include_rx_energy:
                for name in ("ue_rx_w", "helper_rx_w"):
                    if getattr(self, name) is None:
                        raise ValueError(
                            "include_rx_energy=True requires %s (no literature-pinned "
                            "receive power exists; it must be an explicit assumption)" % name
                        )
                    require_nonneg_float(name, float(getattr(self, name)))
        if self.server_static_power_w is not None:
            require_nonneg_float("server_static_power_w", float(self.server_static_power_w))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: dict[str, Any] | None) -> "EnergyModelSpec":
        if not doc:
            return cls()
        tiers_doc = doc.get("tiers") or {}
        tiers = {
            name: TierSpec(
                f_hz=require_finite("%s.f_hz" % name, float(spec["f_hz"])),
                kappa=require_finite("%s.kappa" % name, float(spec["kappa"])),
                source=str(spec.get("source", "unspecified")),
            )
            for name, spec in tiers_doc.items()
        }
        radio = doc.get("radio") or {}
        return cls(
            model=str(doc.get("model", MODEL_LEGACY)),
            energy_scope=str(doc.get("energy_scope", SCOPE_MOBILE)),
            cycles_per_bit=float(doc.get("cycles_per_bit", DEFAULT_CYCLES_PER_BIT)),
            include_rx_energy=bool(doc.get("include_rx_energy", False)),
            tiers=tiers,
            ue_tx_w=_opt(radio.get("ue_tx_w")),
            mec_tx_w=_opt(radio.get("mec_tx_w")),
            helper_tx_w=_opt(radio.get("helper_tx_w")),
            ue_rx_w=_opt(radio.get("ue_rx_w")),
            helper_rx_w=_opt(radio.get("helper_rx_w")),
            server_static_power_w=_opt(doc.get("server_static_power_w")),
            server_utilization_model=bool(doc.get("server_utilization_model", False)),
            provenance=str(doc.get("provenance", "")),
        )

    @classmethod
    def from_frozen_yaml(cls, path: Path | None = None) -> "EnergyModelSpec":
        import yaml

        if path is None:
            path = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"
        doc = yaml.safe_load(Path(path).read_text()) or {}
        spec = cls.from_dict(doc.get("energy_model"))
        validate_frozen_energy_scope(doc, spec)
        return spec

    # -- queries -----------------------------------------------------------
    @property
    def is_physical(self) -> bool:
        return self.model == MODEL_PHYSICAL

    def tier(self, name: str) -> TierSpec:
        return self.tiers[name]

    def cpu_rate_bytes_per_second(self, name: str) -> float:
        return self.tiers[name].cpu_rate_bytes_per_second(self.cycles_per_bit)

    def dynamic_power_w(self, name: str) -> float:
        return self.tiers[name].implied_dynamic_power_w

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "energy_scope": self.energy_scope,
            "cycles_per_bit": self.cycles_per_bit,
            "include_rx_energy": self.include_rx_energy,
            "tiers": {name: spec.as_dict() for name, spec in self.tiers.items()},
            "radio_w": {
                "ue_tx_w": self.ue_tx_w,
                "mec_tx_w": self.mec_tx_w,
                "helper_tx_w": self.helper_tx_w,
                "ue_rx_w": self.ue_rx_w,
                "helper_rx_w": self.helper_rx_w,
            },
            "server_static_power_w": self.server_static_power_w,
            "server_utilization_model": self.server_utilization_model,
            "provenance": self.provenance,
        }


def _opt(value: Any) -> float | None:
    if value is None:
        return None
    return require_finite("value", float(value))


# ---------------------------------------------------------------------------
# Radio hop energy — one implementation shared by engine.py and energy_api.py
# ---------------------------------------------------------------------------
def hop_energy_fields(
    hop: str,
    duration: float,
    src_loc: Any,
    resources: Any,
) -> dict[str, float]:
    """Return {EnergyBreakdown field -> joules} for one communication hop.

    legacy: UE pays UL tx + DL rx; V2V tx/rx split between UE and helper.
    physical_v1: the TRANSMITTER pays its own TX (system-side energy lives in
    `mec_tx_joules_optional`), and RX is added only when include_rx_energy=True.
    """
    from .model import Location

    spec = getattr(resources, "energy_model", None)
    physical = spec is not None and spec.is_physical
    duration = float(duration)
    out: dict[str, float] = {}

    def add(key: str, joules: float) -> None:
        if joules:
            out[key] = out.get(key, 0.0) + joules

    if not physical:
        # --- legacy (MARGO-SPEC-v0.1) -------------------------------------
        if hop == "MEC_UL":
            add(F_UE_MEC_UL, duration * resources.ptx_mec_w)
        elif hop == "MEC_DL":
            add(F_UE_MEC_DL, duration * resources.prx_mec_w)
        elif hop == "V2V":
            if src_loc == Location.HELPER:
                add(F_HELPER_V2V_TX, duration * resources.ptx_v2v_w)
                add(F_UE_V2V_RX, duration * resources.prx_v2v_w)
            else:
                add(F_UE_V2V_TX, duration * resources.ptx_v2v_w)
                add(F_HELPER_V2V_RX, duration * resources.prx_v2v_w)
        return out

    # --- physical_v1 ------------------------------------------------------
    rx = bool(spec.include_rx_energy)
    if hop == "MEC_UL":  # UE -> MEC, UE transmits
        add(F_UE_MEC_UL, duration * float(spec.ue_tx_w))
        if rx and spec.mec_tx_w is not None:
            pass  # MEC receive power is not modelled; receivers are UE/helper only
    elif hop == "MEC_DL":  # MEC -> UE, the RSU pays its own TX
        add(F_MEC_TX, duration * float(spec.mec_tx_w))
        if rx and spec.ue_rx_w is not None:
            add(F_UE_MEC_DL, duration * float(spec.ue_rx_w))
    elif hop == "V2V":
        if src_loc == Location.HELPER:  # helper -> UE
            add(F_HELPER_V2V_TX, duration * float(spec.helper_tx_w))
            if rx and spec.ue_rx_w is not None:
                add(F_UE_V2V_RX, duration * float(spec.ue_rx_w))
        else:  # UE -> helper (or UE -> helper after MEC_DL staging)
            add(F_UE_V2V_TX, duration * float(spec.ue_tx_w))
            if rx and spec.helper_rx_w is not None:
                add(F_HELPER_V2V_RX, duration * float(spec.helper_rx_w))
    return out


def compute_energy_field(tier: str) -> str:
    return {
        TIER_UE: F_UE_CPU,
        TIER_HELPER: F_HELPER_CPU,
        TIER_MEC: F_MEC_CPU,
    }[tier]


def tier_for_location(loc: Any) -> str:
    from .model import Location

    return {
        Location.UE: TIER_UE,
        Location.HELPER: TIER_HELPER,
        Location.MEC: TIER_MEC,
    }[loc]


def validate_frozen_energy_scope(doc: dict, spec: "EnergyModelSpec") -> str:
    """`energy.accounting_primary_scope` is authoritative and must agree with
    `energy_model.energy_scope`. Two conflicting scope definitions in one config
    file is exactly the class of bug that silently mis-scores a reward later, so
    this is a hard error, not a warning.
    """
    declared = (doc.get("energy") or {}).get("accounting_primary_scope")
    if declared is None:
        return spec.energy_scope
    declared = str(declared)
    if declared not in ENERGY_SCOPES:
        raise ValueError(
            "energy.accounting_primary_scope must be one of %s, got %r"
            % (ENERGY_SCOPES, declared)
        )
    if declared != spec.energy_scope:
        raise ValueError(
            "energy accounting scope conflict: energy.accounting_primary_scope=%r "
            "but energy_model.energy_scope=%r. Exactly one definition is allowed; "
            "make them agree." % (declared, spec.energy_scope)
        )
    return declared


def objective_weights_status(doc: dict) -> str:
    """Return "legacy_only" or "active" for energy.objective_weights.

    In the constrained formulation, energy is a constraint channel with a dual
    multiplier, not a weighted objective term; the legacy 0.5/0.5 weights remain
    only so MARGO-SPEC-v0.1 runs reproduce.
    """
    weights = (doc.get("energy") or {}).get("objective_weights") or {}
    return str(weights.get("status", "active"))
