"""Phase 4 EAS on held-out resource profiles. paper_result=false. No meta-test."""

from __future__ import annotations

from spec.phase4_campaign import DIAG_EAS_PROFILES_METHOD_ID, diag_bc_profiles_run_dir
from spec.resource_profiles import role_profile_ids

GATE_DT_S = 15.0  # spec: ΔT ≥ 15s vs zero-shot on held-out profiles


def bc_profiles_ckpt(seed):
    return diag_bc_profiles_run_dir(seed) / "ckpt" / "bc_core.ckpt"


def heldout_profile_ids():
    ids = list(role_profile_ids("validation_heldout"))
    for pid in ids:
        if pid in set(role_profile_ids("meta_test_heldout")):
            raise ValueError("held-out val profile leaked into meta-test: %s" % pid)
    return ids


def make_val_env_for_profile(profile_id):
    """Validation graphs only, encoded under this profile's resources + obs v2."""
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from spec.resource_profiles import resources_cluster_for_profile
    from spec.split_loader import assert_held_out_prefixes, validation_graph_prefixes

    val_paths = validation_graph_prefixes()
    assert_held_out_prefixes(val_paths, "validation")
    env = OffloadingEnvironment(
        resource_cluster=resources_cluster_for_profile(profile_id),
        batch_size=100,
        graph_number=100,
        graph_file_paths=val_paths,
        time_major=False,
    )
    if int(env.input_dim) != 54:
        raise RuntimeError("eas_profiles need obs v2 packed 54, got %s" % env.input_dim)
    return env


assert DIAG_EAS_PROFILES_METHOD_ID == "margo_v0.3_eas_profiles"
