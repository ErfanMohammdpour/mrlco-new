"""Resource profile grid + ResourceConfig builders. Phase 4. paper_result=false."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from env.mec_offloaing_envs.scheduler.resources import ResourceConfig

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parent
PROFILES_YAML = SPEC / "resource_profiles.yaml"
FROZEN_YAML = SPEC / "frozen_experiment.yaml"

UE_CPU = 1048576.0
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def mbps_to_bytes_per_second(mbps: float) -> float:
    return float(mbps) * MBPS_TO_BPS


def load_profiles_doc(path: Path = PROFILES_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def profile_id_from_parts(ul_mbps: float, v2v_mbps: float, ratio: int) -> str:
    ul = int(ul_mbps)
    v2v = int(v2v_mbps)
    ratio = int(ratio)
    if ul == 7 and v2v == 5 and ratio == 10:
        return "frozen_7_5_10"
    return "p_%d_%d_%d" % (ul, v2v, ratio)


def parse_profile_id(pid: str) -> tuple[float, float, int]:
    pid = str(pid)
    if pid == "frozen_7_5_10":
        return 7.0, 5.0, 10
    if not pid.startswith("p_"):
        raise ValueError("bad profile_id %r" % pid)
    parts = pid.split("_")
    if len(parts) != 4:
        raise ValueError("bad profile_id %r" % pid)
    return float(parts[1]), float(parts[2]), int(parts[3])


def all_grid_profile_ids(doc: dict | None = None) -> list[str]:
    doc = doc or load_profiles_doc()
    out = []
    for ul in doc["mec_ul_dl_mbps"]:
        for v2v in doc["v2v_mbps"]:
            for ratio in doc["mec_ue_cpu_ratio"]:
                out.append(profile_id_from_parts(ul, v2v, ratio))
    return out


def role_profile_ids(role: str, doc: dict | None = None) -> list[str]:
    doc = doc or load_profiles_doc()
    role = str(role)
    if role == "meta_train":
        return list(doc["meta_train_profile_ids"])
    if role == "validation_heldout":
        return list(doc["validation_heldout_profile_ids"])
    if role == "meta_test_heldout":
        return list(doc["meta_test_heldout_profile_ids"])
    if role == "frozen":
        return [str(doc["frozen_profile_id"])]
    raise ValueError("role %r" % role)


def resource_config_for_profile(
    profile_id: str,
    power_from_frozen: bool = True,
) -> ResourceConfig:
    """Build ResourceConfig for profile_id. Power coeffs from frozen yaml."""
    ul, v2v, ratio = parse_profile_id(profile_id)
    ue = float(UE_CPU)
    mec_cpu = float(ratio) * ue
    ul_bps = mbps_to_bytes_per_second(ul)
    v2v_bps = mbps_to_bytes_per_second(v2v)
    # Power from frozen
    from env.mec_offloaing_envs.scheduler.resources import ResourceConfig as RC

    frozen = RC.from_frozen_yaml()
    return ResourceConfig(
        ue_cpu_bytes_per_second=ue,
        mec_cpu_bytes_per_second=mec_cpu,
        helper_cpu_bytes_per_second=ue,
        mec_uplink_bytes_per_second=ul_bps,
        mec_downlink_bytes_per_second=ul_bps,
        v2v_bytes_per_second=v2v_bps,
        rho_ue=frozen.rho_ue,
        f_l=frozen.f_l,
        zeta=frozen.zeta,
        ptx_mec_w=frozen.ptx_mec_w,
        prx_mec_w=frozen.prx_mec_w,
        ptx_v2v_w=frozen.ptx_v2v_w,
        prx_v2v_w=frozen.prx_v2v_w,
        rho_helper=frozen.rho_helper,
        f_v2v=frozen.f_v2v,
    )


def resources_cluster_for_profile(profile_id: str, energy_config: dict | None = None):
    """OffloadingEnvironment Resources cluster (Mi-Mbps + CPU B/s)."""
    from env.mec_offloaing_envs.offloading_env import Resources

    ul, v2v, ratio = parse_profile_id(profile_id)
    ue = float(UE_CPU)
    if energy_config is None:
        energy_config = {
            "use_energy": True,
            "reward_mode": "latency_over_all_mec",
            "energy_weight": 0.5,
            "latency_weight": 0.5,
            "rho": 1.0,
            "f_l": 1.0,
            "zeta": 2.0,
            "ptx": 0.1,
            "prx": 0.05,
            "ptx_v2v": 0.06,
            "prx_v2v": 0.03,
            "rho_v2v": 0.7,
            "f_v2v": 1.0,
            "normalize_energy": True,
        }
    return Resources(
        mec_process_capable=float(ratio) * ue,
        mobile_process_capable=ue,
        bandwidth_up=float(ul),
        bandwidth_dl=float(ul),
        v2v_process_capable=ue,
        v2v_bandwidth=float(v2v),
        use_energy=True,
        energy_config=energy_config,
    )


def resource_feature_vector(profile_id: str) -> list[float]:
    """Raw [log UL_bps, log V2V_bps, log MEC_CPU, log UE_CPU] before z-score."""
    import math

    cfg = resource_config_for_profile(profile_id)
    return [
        math.log(cfg.mec_uplink_bytes_per_second),
        math.log(cfg.v2v_bytes_per_second),
        math.log(cfg.mec_cpu_bytes_per_second),
        math.log(cfg.ue_cpu_bytes_per_second),
    ]


def assert_frozen_matches_yaml():
    """Frozen profile ResourceConfig equals frozen_experiment.yaml field-by-field."""
    cfg = resource_config_for_profile("frozen_7_5_10")
    frozen = ResourceConfig.from_frozen_yaml()
    fields = (
        "ue_cpu_bytes_per_second",
        "mec_cpu_bytes_per_second",
        "helper_cpu_bytes_per_second",
        "mec_uplink_bytes_per_second",
        "mec_downlink_bytes_per_second",
        "v2v_bytes_per_second",
    )
    for f in fields:
        a = float(getattr(cfg, f))
        b = float(getattr(frozen, f))
        if abs(a - b) > 1e-6:
            raise AssertionError("frozen mismatch %s: profile=%s yaml=%s" % (f, a, b))
