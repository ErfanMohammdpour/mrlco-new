"""Frozen resource rates and power coefficients for the production scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
