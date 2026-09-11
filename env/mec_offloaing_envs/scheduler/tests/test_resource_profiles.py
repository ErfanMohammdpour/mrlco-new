#!/usr/bin/env python3
"""Phase 4 resource profile axis tests. No GPU."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.resource_profiles import (  # noqa: E402
    PROFILES_YAML,
    all_grid_profile_ids,
    assert_frozen_matches_yaml,
    mbps_to_bytes_per_second,
    resource_config_for_profile,
    resource_feature_vector,
    role_profile_ids,
    sha256_file,
)
from spec.split_loader import count_meta_tasks, iter_meta_tasks  # noqa: E402


class TestResourceProfiles(unittest.TestCase):
    def test_grid_and_roles(self):
        self.assertEqual(len(all_grid_profile_ids()), 45)
        self.assertEqual(len(role_profile_ids("meta_train")), 13)
        self.assertEqual(len(role_profile_ids("validation_heldout")), 2)
        self.assertEqual(len(role_profile_ids("meta_test_heldout")), 8)
        self.assertIn("frozen_7_5_10", role_profile_ids("meta_train"))
        train = set(role_profile_ids("meta_train"))
        val = set(role_profile_ids("validation_heldout"))
        test = set(role_profile_ids("meta_test_heldout"))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_mbps_convention(self):
        self.assertAlmostEqual(mbps_to_bytes_per_second(7), 917504.0)
        self.assertAlmostEqual(mbps_to_bytes_per_second(5), 655360.0)
        self.assertAlmostEqual(mbps_to_bytes_per_second(3), 3 * 1048576 / 8)

    def test_frozen_roundtrip(self):
        assert_frozen_matches_yaml()
        cfg = resource_config_for_profile("frozen_7_5_10")
        self.assertAlmostEqual(cfg.mec_uplink_bytes_per_second, 917504.0)
        self.assertAlmostEqual(cfg.v2v_bytes_per_second, 655360.0)
        self.assertAlmostEqual(cfg.mec_cpu_bytes_per_second, 10485760.0)

    def test_feature_vector_changes_with_profile(self):
        a = resource_feature_vector("p_3_3_5")
        b = resource_feature_vector("p_11_7_20")
        self.assertEqual(len(a), 4)
        self.assertNotEqual(a, b)
        self.assertEqual(a, resource_feature_vector("p_3_3_5"))

    def test_iter_meta_tasks_counts(self):
        self.assertEqual(count_meta_tasks("meta_train"), 195)
        self.assertEqual(count_meta_tasks("validation_heldout"), 10)
        self.assertEqual(count_meta_tasks("validation_frozen"), 5)
        self.assertEqual(count_meta_tasks("validation"), 15)
        self.assertEqual(count_meta_tasks("meta_test"), 40)
        train = set(role_profile_ids("meta_train"))
        for _d, pid in iter_meta_tasks("meta_test"):
            self.assertNotIn(pid, train)

    def test_v1_stats_hash_untouched(self):
        path = ROOT / "spec" / "encoder_feature_stats.json"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            h,
            "94e598759e4b02544bf216220a48225521abd6977419880c4a818631bb9c5e83",
        )

    def test_obs_v2_resource_channels_constant_across_nodes(self):
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
        from env.mec_offloaing_envs.scheduler import encoder_obs as eo
        from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag
        from env.mec_offloaing_envs.scheduler.encoder_obs import (
            raw_node_features,
            set_obs_version,
        )
        from spec.hamming2_probe import _gv_path
        from spec.split_loader import meta_train_distribution_ids

        dist = meta_train_distribution_ids()[0]
        gv = _gv_path(dist, 0)
        if not Path(gv).is_file():
            self.skipTest("no graph file")
        prev = eo.OBS_VERSION
        try:
            set_obs_version("v2")
            self.assertEqual(eo.FEATURE_DIM, 15)
            self.assertEqual(eo.PACKED_DIM, 54)
            tg = OffloadingTaskGraph(str(gv))
            dag = to_canonical_dag(tg)
            order = sorted(dag.tasks)
            rv = resource_feature_vector("p_3_3_5")
            rows = raw_node_features(dag, order, resource_vec=rv)
            self.assertEqual(rows.shape[1], 15)
            self.assertTrue(np.allclose(rows[:, 11:], np.asarray(rv).reshape(1, 4)))
            rv2 = resource_feature_vector("p_11_7_20")
            rows2 = raw_node_features(dag, order, resource_vec=rv2)
            self.assertFalse(np.allclose(rows[:, 11:], rows2[:, 11:]))
        finally:
            set_obs_version(prev)

    def test_yaml_sha_stable(self):
        h = sha256_file(PROFILES_YAML)
        self.assertEqual(len(h), 64)
        self.assertEqual(h, hashlib.sha256(PROFILES_YAML.read_bytes()).hexdigest())

    def test_continuity_one_sided_no_regression(self):
        from spec.bc_profiles import continuity_no_regression

        better = continuity_no_regression(493.8)
        self.assertTrue(better["continuity_pass"])
        self.assertLess(better["continuity_delta_vs_phase1"], 0)
        same = continuity_no_regression(577.02)
        self.assertTrue(same["continuity_pass"])
        worse = continuity_no_regression(590.0)
        self.assertFalse(worse["continuity_pass"])

    def test_eas_profiles_heldout_not_metatest(self):
        from spec.eas_profiles import GATE_DT_S, heldout_profile_ids

        ids = heldout_profile_ids()
        self.assertEqual(ids, ["p_5_5_10", "p_9_5_10"])
        self.assertEqual(GATE_DT_S, 15.0)
        self.assertFalse(set(ids) & set(role_profile_ids("meta_test_heldout")))

    def test_ctx_profiles_mask_and_gate(self):
        from spec.ctx_profiles import GATE_DT_S, TRAIN_QUERY_GIDX, TRAIN_SUPPORT_GIDX, heldout_profile_ids, mask_task

        self.assertEqual(len(TRAIN_SUPPORT_GIDX), 20)
        self.assertEqual(len(TRAIN_QUERY_GIDX), 80)
        self.assertFalse(set(TRAIN_SUPPORT_GIDX) & set(TRAIN_QUERY_GIDX))
        self.assertEqual(GATE_DT_S, 10.0)
        self.assertEqual(heldout_profile_ids(), ["p_5_5_10", "p_9_5_10"])
        dist = np.array([1, 1, 1, 3], dtype=np.int32)
        prof = np.array(["p_3_3_5", "p_3_3_5", "p_7_7_5", "p_3_3_5"])
        gidx = np.array([0, 21, 0, 5], dtype=np.int32)
        m = mask_task(dist, prof, gidx, "p_3_3_5", 1, TRAIN_SUPPORT_GIDX)
        self.assertEqual(m.tolist(), [True, False, False, False])

    def test_bok_profiles_eval_not_metatest(self):
        from spec.bok_profiles import EVAL_PROFILE_IDS, FROZEN_PROFILE_ID, eval_profile_ids

        ids = eval_profile_ids()
        self.assertEqual(ids, ["frozen_7_5_10", "p_5_5_10", "p_9_5_10"])
        self.assertEqual(FROZEN_PROFILE_ID, "frozen_7_5_10")
        self.assertEqual(tuple(ids), EVAL_PROFILE_IDS)
        self.assertFalse(set(ids) & set(role_profile_ids("meta_test_heldout")))
        self.assertIn(FROZEN_PROFILE_ID, role_profile_ids("meta_train"))


if __name__ == "__main__":
    unittest.main()
