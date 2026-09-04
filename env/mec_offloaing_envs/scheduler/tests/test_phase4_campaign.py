#!/usr/bin/env python3
"""Phase 4 campaign-contract tests. Numpy/stdlib only. No GPU. No training."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.phase4_campaign import (
    GPU_ENV,
    GPU_FLAG,
    K_REPORT,
    META_TEST_IDS,
    META_TRAIN_IDS,
    OUTER_ITERS,
    PARENT_SHA,
    SEEDS,
    VALIDATION_IDS,
    GpuPermissionError,
    campaign_jobs,
    eval_json_path,
    provenance_template,
    require_gpu_permission,
    seed_run_dir,
    write_plan,
)
from spec.split_loader import (
    meta_test_distribution_ids,
    meta_train_distribution_ids,
    validation_distribution_ids,
)


class TestCampaignFreeze(unittest.TestCase):
    def test_seeds_and_budget(self):
        self.assertEqual(SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(OUTER_ITERS, 3500)
        self.assertEqual(K_REPORT, (0, 3))
        self.assertEqual(PARENT_SHA, "0c776924b49da6c66c511c12a8cde70be732e25d")

    def test_split_matches_loader(self):
        self.assertEqual(list(META_TRAIN_IDS), meta_train_distribution_ids())
        self.assertEqual(list(VALIDATION_IDS), validation_distribution_ids())
        self.assertEqual(list(META_TEST_IDS), meta_test_distribution_ids())

    def test_jobs_cover_all_test_dists(self):
        jobs = campaign_jobs()
        trains = [j for j in jobs if j["kind"] == "train"]
        evals = [j for j in jobs if j["kind"] == "meta_test_eval"]
        self.assertEqual(len(trains), 5)
        self.assertEqual(len(evals), 5 * 5 * 2)
        for seed in SEEDS:
            for dist_id in META_TEST_IDS:
                for k in K_REPORT:
                    match = [
                        j
                        for j in evals
                        if j["seed"] == seed and j["distribution_id"] == dist_id and j["k_steps"] == k
                    ]
                    self.assertEqual(len(match), 1)
                    self.assertTrue(match[0]["artifact"].endswith("dist_%d_k%d.json" % (dist_id, k)))

    def test_unknown_seed_rejected(self):
        with self.assertRaises(ValueError):
            seed_run_dir(99)

    def test_gpu_blocked_without_flag_or_env(self):
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(False, environ={})
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(True, environ={})
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(True, environ={GPU_ENV: "0"})
        require_gpu_permission(True, environ={GPU_ENV: "1"})

    def test_execute_cli_blocked(self):
        from spec.phase4_campaign import main

        with self.assertRaises(GpuPermissionError):
            main(["--execute-train", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--gpu-smoke", "--seed", "0"])
        with mock.patch.dict(os.environ, {GPU_ENV: "1"}, clear=False):
            with self.assertRaises(GpuPermissionError):
                main(["--execute-train", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--gpu-smoke", "--seed", "0"])

    def test_provenance_protocol_fields(self):
        row = provenance_template(0)
        self.assertEqual(row["outer_update_method"], "mrlco_first_order_mean_pseudogradient")
        self.assertEqual(row["hyperparameter_provenance.policy"], "fixed_literature_derived_defaults")
        self.assertEqual(row["k_steps"], 3)
        self.assertFalse(row["gpu_requested"])
        self.assertFalse(row["gpu_finished"])
        self.assertFalse(row["paper_result"])
        self.assertFalse(row["optimization_claim"])
        self.assertEqual(row["units_time"], "seconds")
        self.assertEqual(row["units_energy"], "joules")

    def test_plan_write(self):
        plan = write_plan()
        self.assertEqual(plan["gpu"]["require_cli_flag"], GPU_FLAG)
        self.assertTrue(plan["gpu"]["require_chat_approval"])
        self.assertEqual(plan["status"], "IN PROGRESS")

    def test_train_driver_not_imported_by_campaign(self):
        source = (ROOT / "spec" / "phase4_campaign.py").read_text()
        self.assertNotIn("import tensorflow", source)
        self.assertNotIn("from meta_trainer import", source)
        self.assertIn("require_gpu_permission", source)

    def test_eval_path_layout(self):
        run_dir = seed_run_dir(0)
        path = eval_json_path(run_dir, 12, 3)
        self.assertEqual(path.name, "dist_12_k3.json")
        self.assertIn("meta_test", str(path))

    def test_smoke_dir_is_not_primary(self):
        from spec.phase4_campaign import smoke_run_dir

        primary = seed_run_dir(0)
        smoke = smoke_run_dir(0)
        self.assertIn("margo_v0.1_primary", str(primary))
        self.assertIn("gpu_smoke", str(smoke))
        self.assertNotEqual(primary, smoke)

    def test_status_not_closed(self):
        text = (ROOT / "spec" / "PHASE4_STATUS.md").read_text()
        self.assertRegex(text, r"(?m)^Status:\s*IN PROGRESS\b")
        self.assertNotRegex(text, r"^Status:\s*CLOSED\b")


if __name__ == "__main__":
    unittest.main()
