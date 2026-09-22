"""④ RADIO_MODEL_V1: explicit radio physics — bandwidth, spectral efficiency
(or SINR), and the effective rate the scheduler actually consumes.

Before this module the scheduler used a single implicit number per hop
(`resource_rates.mec_uplink_mbps` → bytes/s), which hides a 1 bit/s/Hz
assumption inside "Mbps".  Bandwidth is a frequency span, not a throughput:

    R = B * eta                          (eta model, bit/s/Hz)
    R = B * log2(1 + SINR_linear)        (SINR model)

Conversions used everywhere below:
    B [Hz] * eta [bit/s/Hz] = R [bit/s] ;  bytes/s = R / 8

Model selection:
    radio_model="legacy"       -> frozen Mbps table, byte-for-byte as before
    radio_model="physical_v1"  -> B / eta (or SINR) per link, SI units

Every link carries provenance and an `assumption` flag so a paper cannot
silently present an assumed V2V spectral efficiency as a measured one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .validate import require_finite, require_positive_rate

RADIO_LEGACY = "legacy"
RADIO_PHYSICAL = "physical_v1"
RADIO_MODELS = (RADIO_LEGACY, RADIO_PHYSICAL)

RATE_MODEL_ETA = "eta"
RATE_MODEL_SINR = "sinr"
RATE_MODELS = (RATE_MODEL_ETA, RATE_MODEL_SINR)

LINK_V2I_UL = "v2i_ul"
LINK_V2I_DL = "v2i_dl"
LINK_V2V = "v2v"
LINK_NAMES = (LINK_V2I_UL, LINK_V2I_DL, LINK_V2V)

# hop name (routes.py) -> link name
HOP_TO_LINK = {"MEC_UL": LINK_V2I_UL, "MEC_DL": LINK_V2I_DL, "V2V": LINK_V2V}


@dataclass(frozen=True)
class RadioLinkSpec:
    """One wireless link: bandwidth + efficiency model."""

    bandwidth_hz: float
    rate_model: str = RATE_MODEL_ETA
    spectral_efficiency: float | None = None      # bit/s/Hz, required for eta
    sinr_db: float | None = None                  # required for the SINR model
    assumption: bool = False                      # True = not literature-pinned
    source: str = "unspecified"

    def __post_init__(self) -> None:
        require_positive_rate("bandwidth_hz", self.bandwidth_hz)
        if self.rate_model not in RATE_MODELS:
            raise ValueError(
                "rate_model must be one of %s, got %r" % (RATE_MODELS, self.rate_model)
            )
        if self.rate_model == RATE_MODEL_ETA:
            if self.spectral_efficiency is None:
                raise ValueError("eta model requires spectral_efficiency (bit/s/Hz)")
            if float(self.spectral_efficiency) <= 0.0:
                raise ValueError("spectral_efficiency must be > 0")
        else:
            if self.sinr_db is None:
                raise ValueError("sinr model requires sinr_db")
            require_finite("sinr_db", float(self.sinr_db))

    @property
    def effective_rate_bps(self) -> float:
        """R [bit/s]. No hidden 1 bit/s/Hz: eta (or SINR) is always explicit."""
        if self.rate_model == RATE_MODEL_ETA:
            return float(self.bandwidth_hz) * float(self.spectral_efficiency)
        sinr_linear = 10.0 ** (float(self.sinr_db) / 10.0)
        return float(self.bandwidth_hz) * math.log2(1.0 + sinr_linear)

    @property
    def effective_rate_bytes_per_second(self) -> float:
        return self.effective_rate_bps / 8.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bandwidth_hz": self.bandwidth_hz,
            "rate_model": self.rate_model,
            "spectral_efficiency": self.spectral_efficiency,
            "sinr_db": self.sinr_db,
            "effective_rate_bps": self.effective_rate_bps,
            "effective_rate_bytes_per_second": self.effective_rate_bytes_per_second,
            "assumption": self.assumption,
            "source": self.source,
        }


@dataclass(frozen=True)
class RadioModelSpec:
    model: str = RADIO_LEGACY
    links: dict[str, RadioLinkSpec] = field(default_factory=dict)
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.model not in RADIO_MODELS:
            raise ValueError(
                "model must be one of %s, got %r" % (RADIO_MODELS, self.model)
            )
        if self.model == RADIO_PHYSICAL:
            missing = [name for name in LINK_NAMES if name not in self.links]
            if missing:
                raise ValueError("physical_v1 radio missing links: %s" % missing)

    @classmethod
    def from_dict(cls, doc: dict[str, Any] | None) -> "RadioModelSpec":
        if not doc:
            return cls()
        links_doc = doc.get("links") or {}
        links = {
            str(name): RadioLinkSpec(
                bandwidth_hz=require_finite(
                    "%s.bandwidth_hz" % name, float(spec["bandwidth_hz"])
                ),
                rate_model=str(spec.get("rate_model", RATE_MODEL_ETA)),
                spectral_efficiency=(
                    None
                    if spec.get("spectral_efficiency") is None
                    else float(spec["spectral_efficiency"])
                ),
                sinr_db=None if spec.get("sinr_db") is None else float(spec["sinr_db"]),
                assumption=bool(spec.get("assumption", False)),
                source=str(spec.get("source", "unspecified")),
            )
            for name, spec in links_doc.items()
        }
        return cls(
            model=str(doc.get("model", RADIO_LEGACY)),
            links=links,
            provenance=str(doc.get("provenance", "")),
        )

    @classmethod
    def from_frozen_yaml(cls, path: Path | None = None) -> "RadioModelSpec":
        import yaml

        if path is None:
            path = Path(__file__).resolve().parents[3] / "spec" / "frozen_experiment.yaml"
        doc = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(doc.get("radio_model"))

    @property
    def is_physical(self) -> bool:
        return self.model == RADIO_PHYSICAL

    def link(self, name: str) -> RadioLinkSpec:
        try:
            return self.links[name]
        except KeyError as exc:
            raise KeyError("unknown radio link %r" % (name,)) from exc

    def rate_bytes_per_second(self, link_name: str) -> float:
        return self.link(link_name).effective_rate_bytes_per_second

    def rate_for_hop(self, hop: str) -> float:
        return self.rate_bytes_per_second(HOP_TO_LINK[hop])

    def assumptions(self) -> list[str]:
        return sorted(name for name, spec in self.links.items() if spec.assumption)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "links": {name: spec.as_dict() for name, spec in self.links.items()},
            "assumed_links": self.assumptions(),
            "provenance": self.provenance,
        }


def effective_rate_bps(bandwidth_hz: float, *, eta=None, sinr_db=None) -> float:
    """Standalone helper mirroring RadioLinkSpec.effective_rate_bps."""
    if eta is not None:
        if float(eta) <= 0.0:
            raise ValueError("eta must be > 0")
        return float(bandwidth_hz) * float(eta)
    if sinr_db is None:
        raise ValueError("provide eta or sinr_db")
    require_positive_rate("bandwidth_hz", float(bandwidth_hz))
    return float(bandwidth_hz) * math.log2(1.0 + 10.0 ** (float(sinr_db) / 10.0))


def transfer_time_seconds(nbytes: int, rate_bytes_per_second: float) -> float:
    """t = bytes / (R/8); rejects a non-positive rate instead of dividing by 0."""
    require_positive_rate("rate_bytes_per_second", float(rate_bytes_per_second))
    return float(nbytes) / float(rate_bytes_per_second)
