#!/usr/bin/env python3
"""v2 link model: ESTIMATED vs REALIZED link state (seeded, reproducible).

Contract
--------
* The scheduler always executes with the **realized** rate at the time a transfer is
  reserved (`realized(link, t)`).
* The policy may only observe the **estimated** rate/confidence available at plan time
  (`estimate_at(link, t)`), optionally plus a predicted contact time.
* All randomness comes from one seeded `numpy.random.RandomState`; the same seed gives
  bit-identical rate series (tested).
* The regime numbers live in `link_regimes.yaml` and are labelled
  `explicit_assumption_synthetic` - they are sensitivity knobs, not literature values.

Outage semantics: while a link is out, the realized multiplier is 0.0 (no transfer can be
booked at a positive rate). The scheduler raises if asked to transfer at a zero realized
rate, so an outage forces either waiting or a different placement on the next plan - the
open-loop plan cannot react, which is exactly the robustness question v2 must measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

REGIME_YAML = Path(__file__).resolve().parent / "link_regimes.yaml"
LINK_UL, LINK_DL, LINK_V2V = "mec_ul", "mec_dl", "v2v"


class LinkModelError(RuntimeError):
    """Raised on invalid regime configuration or a zero-rate transfer attempt."""


def load_regimes(path: Path | None = None) -> dict:
    doc = yaml.safe_load(Path(path or REGIME_YAML).read_text())
    regimes = doc.get("link_regimes") or {}
    for name, spec in regimes.items():
        if spec.get("evidence_class") != "explicit_assumption_synthetic":
            raise LinkModelError(
                "regime %r must declare evidence_class=explicit_assumption_synthetic "
                "until it is trace/literature grounded" % name)
    return regimes


@dataclass
class LinkRegime:
    name: str
    sigma: float
    persistence: float
    outage_prob_per_step: float
    outage_steps: int
    estimation_sigma: float
    estimation_bias: float
    dt_s: float
    mean_multiplier: float = 1.0
    evidence_class: str = "explicit_assumption_synthetic"

    @classmethod
    def from_spec(cls, name: str, spec: Mapping) -> "LinkRegime":
        return cls(name=name, sigma=float(spec["sigma"]), persistence=float(spec["persistence"]),
                   outage_prob_per_step=float(spec["outage_prob_per_step"]),
                   outage_steps=int(spec["outage_steps"]),
                   estimation_sigma=float(spec["estimation_sigma"]),
                   estimation_bias=float(spec["estimation_bias"]),
                   dt_s=float(spec["dt_s"]),
                   mean_multiplier=float(spec.get("mean_multiplier", 1.0)),
                   evidence_class=str(spec.get("evidence_class", "explicit_assumption_synthetic")))


@dataclass
class LinkProcess:
    """Piecewise-constant realized/estimated multiplier series per link."""

    regime: LinkRegime
    seed: int
    horizon_s: float = 60.0
    links: tuple = (LINK_UL, LINK_DL, LINK_V2V)
    _realized: dict = field(default_factory=dict, repr=False)
    _estimated: dict = field(default_factory=dict, repr=False)
    _outage: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.RandomState(int(self.seed))
        n = max(2, int(math.ceil(self.horizon_s / self.regime.dt_s)) + 1)
        # variance-only reshaping: exp(AR(1)) has a positive drift, so centre the log
        # process and rescale by the declared mean multiplier (degraded regime is 0.7x)
        if self.regime.sigma > 0 and self.regime.persistence < 1.0:
            log_var = self.regime.sigma ** 2 / (1.0 - self.regime.persistence ** 2)
        else:
            log_var = 0.0
        for link in self.links:
            realized = np.ones(n, dtype=np.float64)
            state = 0.0
            outage_left = 0
            outage_flags = np.zeros(n, dtype=bool)
            for i in range(n):
                if outage_left > 0:
                    realized[i] = 0.0
                    outage_flags[i] = True
                    outage_left -= 1
                    state = 0.0
                    continue
                state = self.regime.persistence * state + rng.normal(0.0, self.regime.sigma)
                multiplier = (float(self.regime.mean_multiplier) * math.exp(state - 0.5 * log_var)
                              if self.regime.sigma > 0 else float(self.regime.mean_multiplier))
                if self.regime.outage_prob_per_step > 0 and rng.rand() < self.regime.outage_prob_per_step:
                    outage_left = max(0, self.regime.outage_steps - 1)
                    realized[i] = 0.0
                    outage_flags[i] = True
                else:
                    realized[i] = multiplier
            if self.regime.estimation_sigma > 0:
                noise = rng.normal(0.0, self.regime.estimation_sigma, size=n)
                estimated = realized * np.exp(noise) * self.regime.estimation_bias
            else:
                estimated = realized.copy()
            # the estimate cannot see an outage before it starts, and it never sees a zero
            estimated = np.maximum(estimated, 0.05)
            self._realized[link] = realized
            self._estimated[link] = estimated
            self._outage[link] = outage_flags

    # -- accessors ---------------------------------------------------------
    def _idx(self, t: float) -> int:
        return min(len(self._realized[self.links[0]]) - 1,
                   max(0, int(math.floor(max(0.0, float(t)) / self.regime.dt_s))))

    def realized(self, link: str, t: float = 0.0) -> float:
        return float(self._realized[link][self._idx(t)])

    def estimate_at(self, link: str, t: float = 0.0) -> float:
        return float(self._estimated[link][self._idx(t)])

    def past_outage_fraction(self, link: str, t_now: float = 0.0) -> float:
        """Outage fraction over the OBSERVED window [0, t_now] (decision-time information).

        Using the whole horizon would be future leakage: an admission decision taken at t=0
        must not know about an outage that happens later. At t_now = 0 the window is the
        single CURRENT step, i.e. the observable present link state (not a regime prior and
        not a statement about the future). Steps strictly after `t_now` never enter the mean.
        """
        idx = self._idx(t_now)
        window = self._outage[link][: idx + 1]
        if window.size == 0:
            return float(self.regime.outage_prob_per_step)
        return float(np.mean(window))

    def confidence(self, link: str, t: float = 0.0) -> float:
        """Estimate-model confidence: a function of the ESTIMATION model, not of truth.

        The audited defect was `1 - |estimate - realized| / ...`, which reads hidden realized
        truth at decision time. Confidence is now derived from the declared estimation noise
        of the regime (a property of the estimator), so it is observable at plan time.
        """
        sigma = float(self.regime.estimation_sigma)
        return float(max(0.0, min(1.0, 1.0 / (1.0 + sigma))))

    def estimate_age_s(self, link: str, t: float = 0.0) -> float:
        """Age of the OBSERVABLE information the estimate rests on, in seconds.

        Definition (declared): the time since the last OBSERVED outage ended on this link,
        within the observed window [0, t]. With no outage observed yet it is 0.0, documented
        as "the estimator has seen a clean link so far" rather than as a missing value. This
        is genuinely decision-time information: it never reads a step after `t`.
        """
        idx = self._idx(t)
        window = self._outage[link][: idx + 1]
        observed = np.nonzero(window)[0]
        if observed.size == 0:
            return 0.0
        last_out_step = int(observed[-1])
        last_out_end_s = (last_out_step + 1) * float(self.regime.dt_s)
        return float(max(0.0, float(t) - last_out_end_s))

    def observed_outage_fraction(self, link: str, t: float = 0.0) -> float:
        """Alias of `past_outage_fraction` with a name that says what it measures."""
        return self.past_outage_fraction(link, t)

    def confidence_vs_truth(self, link: str, t: float = 0.0) -> float:
        """Diagnostic ONLY (post-hoc analysis). Never feed this into an observation,
        admission rule or policy-accessible score."""
        idx = self._idx(t)
        r = float(self._realized[link][idx])
        e = float(self._estimated[link][idx])
        scale = max(r, e, 1e-9)
        return float(max(0.0, 1.0 - abs(e - r) / scale))

    def outage_at(self, link: str, t: float = 0.0) -> bool:
        return bool(self._outage[link][self._idx(t)])

    def realized_rate(self, base_rate: float, link: str, t: float = 0.0) -> float:
        rate = float(base_rate) * self.realized(link, t)
        if rate <= 0.0:
            raise LinkModelError(
                "realized %s rate is 0 at t=%.3fs (outage): the open-loop plan cannot "
                "proceed; the scheduler must wait or re-plan" % (link, t))
        return rate

    def estimated_rate(self, base_rate: float, link: str, t: float = 0.0) -> float:
        return float(base_rate) * self.estimate_at(link, t)

    def summary(self, base_rates: Mapping[str, float]) -> dict:
        out = {"regime": self.regime.name, "seed": int(self.seed),
               "evidence_class": self.regime.evidence_class, "links": {}}
        for link, base in base_rates.items():
            r = self._realized[link]
            e = self._estimated[link]
            out["links"][link] = {
                "base_bytes_per_s": float(base),
                "realized_mean_multiplier": float(np.mean(r)),
                "realized_std_multiplier": float(np.std(r)),
                "realized_min_multiplier": float(np.min(r)),
                "estimated_mean_multiplier": float(np.mean(e)),
                "outage_steps": int(np.sum(self._outage[link])),
                "outage_fraction": float(np.mean(self._outage[link])),
                "mean_confidence_estimate_model": float(self.confidence(link)),
                "mean_confidence_vs_truth_diagnostic_only": float(np.mean(np.clip(
                    1.0 - np.abs(e - r) / np.maximum(np.maximum(r, e), 1e-9), 0.0, 1.0))),
            }
        return out


def make_process(regime_name: str, seed: int, *, horizon_s: float = 60.0,
                 regimes: dict | None = None) -> LinkProcess:
    table = regimes if regimes is not None else load_regimes()
    if regime_name not in table:
        raise LinkModelError("unknown regime %r (have %s)" % (regime_name, sorted(table)))
    return LinkProcess(LinkRegime.from_spec(regime_name, table[regime_name]), int(seed),
                       horizon_s=horizon_s)
