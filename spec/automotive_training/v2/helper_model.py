#!/usr/bin/env python3
"""v2 helper model: a helper is a vehicle with its own CPU, occupancy and contact window.

EVIDENCE CLASS: explicit_assumption_synthetic. The mobility/contact numbers below are
sensitivity knobs, not measured traces. Contact is modelled as a window
`[contact_start, contact_end)`; the planner sees a PREDICTED window while execution is
bounded by the REALIZED one (the same estimated-vs-realized split as the link model).

Helper semantics implemented here:
* `busy_until_s` - the helper's own workload keeps its CPU busy until then (queue penalty);
* admissibility - a helper placement is only admissible if transfer + compute fit inside
  the PREDICTED window with a declared safety margin;
* execution - if the task would finish after the REALIZED contact end, the placement
  fails. Minimal fallback (documented, not claimed as reliable): the remaining work is
  restarted locally, so the episode stays finite and the failure is logged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


class HelperModelError(RuntimeError):
    """Raised on invalid helper configuration."""


@dataclass
class HelperState:
    helper_id: int
    cpu_bytes_per_s: float
    owner: int = -1
    busy_until_s: float = 0.0
    contact_start_s: float = 0.0
    contact_end_s: float = math.inf           # realized window
    predicted_contact_end_s: float | None = None  # what the planner sees
    recovery_fraction: float = 0.0            # restart cost as a fraction of the work

    def __post_init__(self) -> None:
        if self.cpu_bytes_per_s <= 0.0 or not math.isfinite(self.cpu_bytes_per_s):
            raise HelperModelError("helper %s has a non-positive cpu rate" % self.helper_id)
        if self.predicted_contact_end_s is None:
            self.predicted_contact_end_s = self.contact_end_s

    @property
    def available_from_s(self) -> float:
        return max(self.busy_until_s, self.contact_start_s)


def sample_contact(rng: np.random.RandomState, *, mean_s: float = 2.0,
                   cv: float = 0.5, min_s: float = 0.2) -> float:
    """Explicit-assumption contact duration (lognormal, mean `mean_s`, CV `cv`)."""
    if mean_s <= 0.0:
        raise HelperModelError("mean contact must be positive")
    if cv <= 0.0:
        return float(mean_s)
    sigma = math.sqrt(math.log(1.0 + cv ** 2))
    mu = math.log(mean_s) - 0.5 * sigma ** 2
    return float(max(min_s, rng.lognormal(mu, sigma)))


def predicted_contact(realized_end_s: float, *, bias: float = 1.15) -> float:
    """The planner's optimistic prediction (explicit assumption: bias > 1)."""
    if not math.isfinite(realized_end_s):
        return realized_end_s
    return float(realized_end_s * float(bias))


def required_helper_time_s(*, payload_in_bytes: float, compute_bytes: float,
                           v2v_bytes_per_s: float, helper_bytes_per_s: float,
                           hops: int = 2) -> float:
    """Transfer in (+ hops) plus compute, i.e. what must fit in the contact window."""
    if v2v_bytes_per_s <= 0.0 or helper_bytes_per_s <= 0.0:
        raise HelperModelError("non-positive rate in the helper admissibility check")
    transfer = hops * float(payload_in_bytes) / float(v2v_bytes_per_s)
    compute = float(compute_bytes) / float(helper_bytes_per_s)
    return transfer + compute


def admissible(helper: HelperState, *, now_s: float, payload_in_bytes: float,
               compute_bytes: float, v2v_bytes_per_s: float, margin: float = 0.9) -> bool:
    """Can the placement finish inside the PREDICTED contact window (with margin)?"""
    start = max(float(now_s), helper.available_from_s)
    predicted_end = helper.predicted_contact_end_s
    if predicted_end is None or not math.isfinite(predicted_end):
        return True
    window = (predicted_end - start) * float(margin)
    need = required_helper_time_s(payload_in_bytes=payload_in_bytes,
                                  compute_bytes=compute_bytes,
                                  v2v_bytes_per_s=v2v_bytes_per_s,
                                  helper_bytes_per_s=helper.cpu_bytes_per_s)
    return window >= need


def make_helpers(specs: Mapping[int, Mapping], *, seed: int, contact_mean_s: float = 2.0,
                 contact_cv: float = 0.5, prediction_bias: float = 1.15) -> dict:
    """Build realized helper states (seedable) from a {helper_id: {cpu, busy, owner}} map."""
    rng = np.random.RandomState(int(seed))
    out = {}
    for hid, spec in sorted(specs.items()):
        duration = sample_contact(rng, mean_s=contact_mean_s, cv=contact_cv)
        out[int(hid)] = HelperState(
            helper_id=int(hid), cpu_bytes_per_s=float(spec["cpu_bytes_per_s"]),
            owner=int(spec.get("owner", -1)), busy_until_s=float(spec.get("busy_until_s", 0.0)),
            contact_start_s=float(spec.get("contact_start_s", 0.0)),
            contact_end_s=duration,
            predicted_contact_end_s=predicted_contact(duration, bias=prediction_bias),
            recovery_fraction=float(spec.get("recovery_fraction", 0.0)))
    return out
