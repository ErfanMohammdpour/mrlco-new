#!/usr/bin/env python3
"""Per-graph CO-PHYSICAL scheduler configuration for the automotive primary path.

The frozen dataset carries one resource profile per graph:
    f_UE, f_HELPER, f_MEC, R_MEC_UL, R_MEC_DL, R_V2V
and the training scheduler must consume exactly those numbers. The primary
automotive configuration therefore uses the co-physical axes required by the
integration contract:

    timing_model       = physical_rates     (T_i^x = 8*xi*W_i / f_x)
    radio_timing_model = physical_rates     (T_tx,e = 8*B_e / R_link)
    energy_model       = physical_v1
    radio_model        = physical_v1
    energy_scope       = system

Co-physical AND graph-specific at the same time: the CPU tier frequencies come
from the graph, and each radio link is instantiated as an EXPLICIT physical link
whose bandwidth is derived from the graph's frozen achievable capacity and the
frozen spectral efficiency (R = B*eta, so B = R/eta). No rate is invented and no
3GPP service-required rate is used as capacity.

The historical mixed configuration (legacy timing + physical energy) stays
available under an explicit name for reproduction/control only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_YAML = REPO_ROOT / "spec" / "frozen_experiment.yaml"
DATASET_WORKLOAD_YAML = REPO_ROOT / "spec" / "automotive_mc_v1" / "workload_model_v2.yaml"

#: The eight resource keys every frozen graph must carry.
REQUIRED_RESOURCE_KEYS = (
    "f_ue_hz",
    "f_helper_hz",
    "f_mec_hz",
    "r_mec_ul_bps",
    "r_mec_dl_bps",
    "r_v2v_bps",
)

CO_PHYSICAL_AXES = {
    "timing_model": "physical_rates",
    "radio_timing_model": "physical_rates",
    "energy_model": "physical_v1",
    "radio_model": "physical_v1",
    "energy_scope": "system",
}
LEGACY_MIXED_AXES = {
    "timing_model": "legacy_frozen_rates",
    "radio_timing_model": "legacy_frozen_rates",
    "energy_model": "physical_v1",
    "radio_model": "physical_v1",
    "energy_scope": "system",
}


class AutomotiveResourceError(ValueError):
    """Refuses to build a config from malformed or missing frozen resources."""


def dataset_reference_parameters() -> dict:
    """f_ref / xi as frozen by the dataset workload model."""
    doc = yaml.safe_load(DATASET_WORKLOAD_YAML.read_text())
    ref = doc["reference_compute_model"]
    return {
        "f_ref_hz": float(ref["f_ref_hz"]),
        "xi_cycles_per_bit": float(ref["xi_cycles_per_bit"]),
        "reference_tier": str(ref["reference_tier"]),
        "formula": str(ref["formula"]),
    }


def frozen_radio_eta(path: Path | None = None) -> dict:
    """The frozen spectral efficiency per radio link (bit/s/Hz), with its flags."""
    doc = yaml.safe_load(Path(path or FROZEN_YAML).read_text())
    out = {}
    for name, spec in (doc.get("radio_model", {}).get("links") or {}).items():
        if spec.get("rate_model") != "eta" or spec.get("spectral_efficiency") is None:
            raise AutomotiveResourceError(
                "the automotive primary needs the eta form for link %r; SINR links "
                "would need an explicit SINR per graph" % name
            )
        out[name] = {
            "spectral_efficiency": float(spec["spectral_efficiency"]),
            "assumption": bool(spec.get("assumption", False)),
            "source": str(spec.get("source", "")),
        }
    for required in ("v2i_ul", "v2i_dl", "v2v"):
        if required not in out:
            raise AutomotiveResourceError("frozen radio model lacks link %r" % required)
    return out


def frozen_energy_parameters(path: Path | None = None) -> dict:
    """Per-tier kappa and the TX powers used by the physical energy model."""
    doc = yaml.safe_load(Path(path or FROZEN_YAML).read_text())
    em = doc.get("energy_model") or {}
    tiers = {
        name: float(spec["kappa"]) for name, spec in (em.get("tiers") or {}).items()
    }
    for required in ("ue", "helper", "mec"):
        if required not in tiers:
            raise AutomotiveResourceError("frozen energy model lacks tier %r" % required)
    radio = em.get("radio") or {}
    return {
        "kappa": tiers,
        "ue_tx_w": float(radio["ue_tx_w"]),
        "mec_tx_w": float(radio["mec_tx_w"]),
        "helper_tx_w": float(radio["helper_tx_w"]),
        "include_rx_energy": bool(em.get("include_rx_energy", False)),
        "cycles_per_bit": float(em.get("cycles_per_bit", 300.0)),
        "provenance": str(em.get("provenance", "")),
    }


def _check_resources(resource: Mapping[str, Any]) -> dict:
    out = {}
    for key in REQUIRED_RESOURCE_KEYS:
        if key not in resource:
            raise AutomotiveResourceError(f"graph resource is missing {key!r}")
        value = float(resource[key])
        if not (value > 0.0) or value != value or value in (float("inf"),):
            raise AutomotiveResourceError(f"graph resource {key}={value!r} is not a positive finite rate")
        out[key] = value
    return out


def _energy_spec(resource: Mapping[str, Any], xi: float, frozen_path: Path | None):
    from env.mec_offloaing_envs.scheduler.energy_model import (
        EnergyModelSpec,
        TierSpec,
        MODEL_PHYSICAL,
    )

    energy = frozen_energy_parameters(frozen_path)
    tiers = {
        "ue": TierSpec(f_hz=resource["f_ue_hz"], kappa=energy["kappa"]["ue"],
                       source="graph f_ue_hz + frozen kappa_ue"),
        "helper": TierSpec(f_hz=resource["f_helper_hz"], kappa=energy["kappa"]["helper"],
                           source="graph f_helper_hz + frozen kappa_helper"),
        "mec": TierSpec(f_hz=resource["f_mec_hz"], kappa=energy["kappa"]["mec"],
                        source="graph f_mec_hz + frozen kappa_mec"),
    }
    return EnergyModelSpec(
        model=MODEL_PHYSICAL,
        energy_scope="system",
        cycles_per_bit=float(xi),
        include_rx_energy=energy["include_rx_energy"],
        tiers=tiers,
        ue_tx_w=energy["ue_tx_w"],
        mec_tx_w=energy["mec_tx_w"],
        helper_tx_w=energy["helper_tx_w"],
        provenance="frozen_experiment.yaml energy_model + per-graph tier frequencies",
    )


def _radio_spec(resource: Mapping[str, Any], frozen_path: Path | None):
    from env.mec_offloaing_envs.scheduler.radio import (
        LINK_V2I_DL,
        LINK_V2I_UL,
        LINK_V2V,
        RADIO_PHYSICAL,
        RadioLinkSpec,
        RadioModelSpec,
    )

    eta = frozen_radio_eta(frozen_path)
    mapping = {
        LINK_V2I_UL: ("r_mec_ul_bps", "v2i_ul"),
        LINK_V2I_DL: ("r_mec_dl_bps", "v2i_dl"),
        LINK_V2V: ("r_v2v_bps", "v2v"),
    }
    links = {}
    for link_name, (key, frozen_name) in mapping.items():
        rate_bps = resource[key]
        spec = eta[frozen_name]
        links[link_name] = RadioLinkSpec(
            bandwidth_hz=rate_bps / spec["spectral_efficiency"],
            rate_model="eta",
            spectral_efficiency=spec["spectral_efficiency"],
            assumption=spec["assumption"],
            source=(
                "bandwidth inverted from the frozen achievable capacity %s=%.6g bps "
                "under the frozen spectral efficiency %.6g bit/s/Hz" % (key, rate_bps,
                                                                       spec["spectral_efficiency"])
            ),
        )
    return RadioModelSpec(
        model=RADIO_PHYSICAL,
        links=links,
        provenance="per-graph achievable capacity from the frozen automotive resource profile",
    )


def _base_fields(resource: Mapping[str, Any], xi: float) -> dict:
    return {
        "ue_cpu_bytes_per_second": resource["f_ue_hz"] / (8.0 * xi),
        "mec_cpu_bytes_per_second": resource["f_mec_hz"] / (8.0 * xi),
        "helper_cpu_bytes_per_second": resource["f_helper_hz"] / (8.0 * xi),
        "mec_uplink_bytes_per_second": resource["r_mec_ul_bps"] / 8.0,
        "mec_downlink_bytes_per_second": resource["r_mec_dl_bps"] / 8.0,
        "v2v_bytes_per_second": resource["r_v2v_bps"] / 8.0,
    }


def co_physical_config_for_graph(
    resource: Mapping[str, Any],
    *,
    frozen_path: Path | None = None,
    source_sha256: str = "",
    xi: float | None = None,
):
    """Co-physical, graph-specific primary scheduler config."""
    from env.mec_offloaing_envs.scheduler.resources import (
        TIMING_PHYSICAL,
        ResourceConfig,
    )

    res = _check_resources(resource)
    ref = dataset_reference_parameters()
    xi = float(ref["xi_cycles_per_bit"] if xi is None else xi)
    energy = _energy_spec(res, xi, frozen_path)
    radio = _radio_spec(res, frozen_path)
    return ResourceConfig(
        **_base_fields(res, xi),
        rho_ue=1.0,
        f_l=1.0,
        zeta=2.0,
        ptx_mec_w=0.1,
        prx_mec_w=0.05,
        ptx_v2v_w=0.06,
        prx_v2v_w=0.03,
        rho_helper=0.7,
        f_v2v=1.0,
        energy_model=energy,
        radio_model=radio,
        timing_model=TIMING_PHYSICAL,
        timing_tiers=energy,
        radio_timing_model=TIMING_PHYSICAL,
        radio_timing_spec=radio,
        energy_scope="system",
        source_config_sha256=str(source_sha256 or ""),
    )


def legacy_mixed_config_for_graph(
    resource: Mapping[str, Any],
    *,
    frozen_path: Path | None = None,
    source_sha256: str = "",
    xi: float | None = None,
):
    """Reproduction/control only: physical energy, legacy timing and radio table.

    This is the historical MARGO primary configuration. It is kept so the earlier
    mixed-physics numbers stay reproducible; the automotive primary never uses it.
    """
    from env.mec_offloaing_envs.scheduler.resources import (
        TIMING_LEGACY,
        ResourceConfig,
    )

    res = _check_resources(resource)
    ref = dataset_reference_parameters()
    xi = float(ref["xi_cycles_per_bit"] if xi is None else xi)
    energy = _energy_spec(res, xi, frozen_path)
    return ResourceConfig(
        **_base_fields(res, xi),
        rho_ue=1.0,
        f_l=1.0,
        zeta=2.0,
        ptx_mec_w=0.1,
        prx_mec_w=0.05,
        ptx_v2v_w=0.06,
        prx_v2v_w=0.03,
        rho_helper=0.7,
        f_v2v=1.0,
        energy_model=energy,
        radio_model=None,
        timing_model=TIMING_LEGACY,
        radio_timing_model=TIMING_LEGACY,
        energy_scope="system",
        source_config_sha256=str(source_sha256 or ""),
    )


def config_fingerprint(config: Any) -> str:
    from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

    return resolved_config_sha256(config)


def axes_of(config: Any) -> dict:
    return {
        "timing_model": config.timing_model,
        "radio_timing_model": config.radio_timing_model,
        "energy_model": str(getattr(getattr(config, "energy_model", None), "model", "")),
        "radio_model": str(getattr(getattr(config, "radio_model", None), "model", "")),
        "energy_scope": str(config.energy_scope),
    }


def version_axes(config: Any) -> dict:
    """Alias of `axes_of` used by the parity test (kept for readability)."""
    return axes_of(config)


def axes_fingerprint(frozen_path: Path | None = None) -> str:
    """Graph-INDEPENDENT fingerprint of the co-physical scheduling contract.

    The legacy energy-telemetry aggregator requires ONE scheduler sha per run, while
    the automotive primary legitimately schedules each graph with its own rates. This
    value pins the axes + the frozen eta/kappa/xi model that is common to every graph;
    the per-graph config sha is recorded separately (`graph_scheduler_config_sha256`).
    """
    import hashlib
    import json as _json

    material = {
        "axes": CO_PHYSICAL_AXES,
        "radio_eta": frozen_radio_eta(frozen_path),
        "energy": {k: v for k, v in frozen_energy_parameters(frozen_path).items()
                   if k != "provenance"},
        "reference": dataset_reference_parameters(),
        "rate_rule": "R_graph / eta with B = R/eta; per-graph rates from resource_profiles.yaml",
    }
    return hashlib.sha256(_json.dumps(material, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
