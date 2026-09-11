"""STATUS: baseline/ablation only (ADR-007)

CAVIA episode cost with energy on/off. Physics already emits joules; this selects the scalar.

`use_energy=False` (phase-1 default): cost = makespan_seconds. Joules still logged.
`use_energy=True`: cost = latency_weight * L_norm + energy_weight * E_norm
using frozen pure-location ranges. Publication weights are 0.5/0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

from env.mec_offloaing_envs.scheduler.energy_api import (
    ENERGY_WEIGHT,
    LATENCY_WEIGHT,
    j_report,
    normalize,
)

CAVIA_Z_DIM = 32
CAVIA_INNER_STEPS = 20
CAVIA_INNER_LR = 5.0e-4
CAVIA_STRONG_INNER_STEPS = 50
CAVIA_STRONG_INNER_LR = 1.0e-2
CAVIA_IDENTITY_T_VAL_LO = 575.0
CAVIA_IDENTITY_T_VAL_HI = 590.0
CAVIA_BCCONT_IDENTITY_T_VAL_LO = 568.0
CAVIA_BCCONT_IDENTITY_T_VAL_HI = 582.0
CAVIA_HELP_SEC = 15.0
CAVIA_WEAK_SEC = 5.0
CAVIA_HURT_RATIO = 1.15
CAVIA_HURT_T = 650.0
CAVIA_LOCAL_HURT = 0.35
CAVIA_LOCAL_SOFT = 0.28


@dataclass(frozen=True)
class CaviaObjective:
    use_energy: bool = False
    latency_weight: float = LATENCY_WEIGHT
    energy_weight: float = ENERGY_WEIGHT

    def __post_init__(self):
        lw = float(self.latency_weight)
        ew = float(self.energy_weight)
        if lw < 0.0 or ew < 0.0:
            raise ValueError("weights must be non-negative, got lw=%s ew=%s" % (lw, ew))
        if bool(self.use_energy):
            if abs(lw + ew - 1.0) > 1e-12:
                raise ValueError("energy-on weights must sum to 1.0, got %s" % (lw + ew))
        object.__setattr__(self, "latency_weight", lw)
        object.__setattr__(self, "energy_weight", ew)
        object.__setattr__(self, "use_energy", bool(self.use_energy))

    @property
    def method_suffix(self):
        return "energy" if self.use_energy else "frozen"

    def cost(self, makespan_seconds, total_mobile_joules, refs=None):
        """Return (cost, T, E). Always returns raw T and E for logs."""
        t = float(makespan_seconds)
        e = float(total_mobile_joules)
        if not self.use_energy:
            return t, t, e
        if refs is None:
            raise ValueError("energy-on cost needs ReferenceRanges")
        pub = (
            abs(self.latency_weight - LATENCY_WEIGHT) <= 1e-12
            and abs(self.energy_weight - ENERGY_WEIGHT) <= 1e-12
        )
        if pub:
            return float(j_report(t, e, refs)), t, e
        l_norm = normalize(t, refs.L_ref_min, refs.L_ref_max, name="L", out_of_range=refs.out_of_range)
        e_norm = normalize(e, refs.E_ref_min, refs.E_ref_max, name="E", out_of_range=refs.out_of_range)
        return float(self.latency_weight * l_norm + self.energy_weight * e_norm), t, e


def centered_advantages(costs):
    import numpy as np

    c = np.asarray(costs, dtype=np.float64).reshape(-1)
    if c.size == 0:
        raise ValueError("advantages need at least one cost")
    a = c - float(np.mean(c))
    return a.astype(np.float32)


def classify_cavia_verdict(t_k0, t_k20, local_k20, identity_t=None, identity_lo=None, identity_hi=None):
    t0 = float(t_k0)
    t20 = float(t_k20)
    loc = float(local_k20)
    lo = CAVIA_IDENTITY_T_VAL_LO if identity_lo is None else float(identity_lo)
    hi = CAVIA_IDENTITY_T_VAL_HI if identity_hi is None else float(identity_hi)
    if identity_t is not None:
        ident = float(identity_t)
        if ident < lo or ident > hi:
            return "identity_fail"
    if loc >= CAVIA_LOCAL_HURT or t20 >= t0 * CAVIA_HURT_RATIO or t20 >= CAVIA_HURT_T:
        return "cavia_hurts"
    delta = t0 - t20
    if delta >= CAVIA_HELP_SEC and loc < CAVIA_LOCAL_SOFT:
        return "cavia_helps"
    if delta >= CAVIA_WEAK_SEC:
        return "cavia_weak"
    return "cavia_no_gain"


def is_cavia_var_name(name):
    n = str(name)
    return "cavia_z" in n or "cavia_film" in n
