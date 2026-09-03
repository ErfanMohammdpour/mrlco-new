"""Frozen resource rates and power coefficients for the production scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import Location


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

    def cpu_rate(self, loc: Location) -> float:
        return {
            Location.UE: self.ue_cpu_bytes_per_second,
            Location.MEC: self.mec_cpu_bytes_per_second,
            Location.HELPER: self.helper_cpu_bytes_per_second,
        }[loc]

    def hop_rate(self, hop: str) -> float:
        return {
            "MEC_UL": self.mec_uplink_bytes_per_second,
            "MEC_DL": self.mec_downlink_bytes_per_second,
            "V2V": self.v2v_bytes_per_second,
        }[hop]

    @classmethod
    def from_frozen_yaml(cls, path: Path | None = None) -> "ResourceConfig":
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
        )
