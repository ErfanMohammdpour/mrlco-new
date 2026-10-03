#!/usr/bin/env python3
"""v2 criticality-aware remote-execution reliability (and the minimal fallback hook).

Distinctions kept explicit (never conflated):
* `deadline_s`      - WHEN a task must finish (urgency; unchanged from v1 semantics).
* `epsilon_class`   - how much REMOTE-EXECUTION FAILURE probability the criticality class
                      accepts. Anchored on dataset-ledgered 3GPP reliability requirements
                      for HIGH/MEDIUM, explicit assumption for LOW.
* `warm_standby`    - whether a local standby reservation is required for remote tokens.

The success-probability model is an explicit assumption built from observable quantities:
estimated/realized link confidence, realized outage fraction, and (for helpers) the
contact margin. It is NOT a measured failure curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

CLASSES_YAML = Path(__file__).resolve().parent / "reliability_classes.yaml"
ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class ReliabilityError(RuntimeError):
    """Raised on invalid reliability configuration or use."""


@dataclass(frozen=True)
class ReliabilityClass:
    name: str
    epsilon: float
    evidence_class: str
    requirement_id: str | None
    reliability_percent: float | None
    source: str
    evidence_sha256: str | None
    warm_standby: bool
    fallback_semantics: str

    @property
    def required_success(self) -> float:
        return 1.0 - float(self.epsilon)


def load_classes(path: Path | None = None) -> dict:
    doc = yaml.safe_load(Path(path or CLASSES_YAML).read_text())
    raw = doc.get("reliability_classes") or {}
    out = {}
    for name, spec in raw.items():
        if name not in ORDER:
            raise ReliabilityError("unknown criticality class %r" % name)
        out[name] = ReliabilityClass(
            name=name, epsilon=float(spec["epsilon"]),
            evidence_class=str(spec["evidence_class"]),
            requirement_id=spec.get("requirement_id"),
            reliability_percent=(None if spec.get("reliability_percent") is None
                                 else float(spec["reliability_percent"])),
            source=str(spec.get("source", "")),
            evidence_sha256=spec.get("evidence_sha256"),
            warm_standby=bool(spec.get("warm_standby", False)),
            fallback_semantics=str(spec.get("fallback_semantics", "none")))
    if set(out) != set(ORDER):
        raise ReliabilityError("classes must define exactly %s" % sorted(ORDER))
    if not (out["HIGH"].epsilon < out["MEDIUM"].epsilon < out["LOW"].epsilon):
        raise ReliabilityError("epsilon ordering must satisfy HIGH < MEDIUM < LOW")
    for name, cls in out.items():
        if cls.evidence_class == "dataset_standard_derived" and not cls.evidence_sha256:
            raise ReliabilityError("standard-derived class %r must carry an evidence sha" % name)
    return out


def remote_success_probability(*, link_confidence: float, outage_fraction: float,
                               contact_margin: float = 1.0,
                               location: str = "MEC") -> float:
    """Explicit success model for one remote placement.

    p = clip(link_confidence * (1 - outage_fraction) * contact_factor)
    where contact_factor is 1.0 for MEC and the clamped contact margin for HELPER.
    """
    if not (0.0 <= link_confidence <= 1.0):
        raise ReliabilityError("link_confidence must be in [0,1]")
    if not (0.0 <= outage_fraction <= 1.0):
        raise ReliabilityError("outage_fraction must be in [0,1]")
    factor = 1.0 if location != "HELPER" else max(0.0, min(1.0, float(contact_margin)))
    p = float(link_confidence) * (1.0 - float(outage_fraction)) * factor
    return float(max(0.0, min(1.0, p)))


def remote_admissible(criticality: str, p_success: float, classes: Mapping | None = None) -> bool:
    table = classes if classes is not None else load_classes()
    if criticality not in table:
        raise ReliabilityError("unknown criticality %r" % criticality)
    return float(p_success) >= table[criticality].required_success


def make_gate(classes: Mapping | None = None):
    """Return (location, criticality, evidence) -> (admissible, p_success) callable."""
    table = classes if classes is not None else load_classes()

    def gate(location: str, criticality: str, evidence: Mapping):
        p = remote_success_probability(
            link_confidence=float(evidence.get("link_confidence", 1.0)),
            outage_fraction=float(evidence.get("outage_fraction", 0.0)),
            contact_margin=float(evidence.get("contact_margin", 1.0)),
            location=location)
        return remote_admissible(criticality, p, table), p

    return gate


def standby_required(criticality: str, classes: Mapping | None = None) -> bool:
    table = classes if classes is not None else load_classes()
    if criticality not in table:
        raise ReliabilityError("unknown criticality %r" % criticality)
    return bool(table[criticality].warm_standby)
