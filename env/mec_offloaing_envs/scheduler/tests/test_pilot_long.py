#!/usr/bin/env python3
"""P2: long latency-only diagnostic runner (pure parts + dry run)."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})


from spec.pilot_long import (  # noqa: E402
    CONTRACT,
    INTERIM_ITERS,
    assert_fresh_run_dir,
    checkpoint_plan,
    classification,
    interim_summary,
    main,
    manifest,
    sha256_file,
    watchdog_flags,
)


class TestPlanAndDirs(unittest.TestCase):
    def test_checkpoint_plan_is_the_frozen_set_plus_final(self):
        self.assertEqual(
            checkpoint_plan(999), (0, 50, 100, 200, 300, 500, 750, 999)
        )

    def test_interim_iters(self):
        self.assertEqual(tuple(INTERIM_ITERS), (200, 500))

    def test_fresh_run_dir_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "off" / "seed_0"
            assert_fresh_run_dir(path)  # missing is fine
            path.mkdir(parents=True)
            assert_fresh_run_dir(path)  # empty is fine
            (path / "x").write_text("1")
            with self.assertRaises(RuntimeError):
                assert_fresh_run_dir(path)

    def test_contract_is_latency_only_diagnostic(self):
        self.assertEqual(CONTRACT["reward_mode"], "latency_only")
        self.assertEqual(CONTRACT["objective_mode"], "log_only")
        self.assertEqual(CONTRACT["constraints"], "off")
        self.assertEqual(CONTRACT["deadlines"], "none")
        self.assertEqual(CONTRACT["mask_mode"], "off")
        self.assertEqual(CONTRACT["obs_version"], "v3")
        self.assertEqual(CONTRACT["entropy_coefficient"], 0.0)
        self.assertIs(CONTRACT["paper_result"], False)


class TestClassification(unittest.TestCase):
    def _row(self, itr, k3, gap):
        return {"itr": itr, "k0": k3 + 50.0, "k3": k3, "gap_to_all_mec": gap,
                "gap_to_greedy": gap + 1.0}

    def test_pass_needs_two_improving_points(self):
        rows = [self._row(-1, 900.0, 270.0), self._row(0, 880.0, 250.0),
                self._row(50, 860.0, 230.0)]
        out = classification(rows)
        self.assertEqual(out["verdict"], "PASS_LATENCY_LEARNING")
        self.assertEqual(out["improved_validation_points"], 2)
        self.assertEqual(out["best_itr"], 50)

    def test_flat_is_no_improvement(self):
        rows = [self._row(-1, 900.0, 270.0), self._row(0, 905.0, 275.0),
                self._row(50, 910.0, 280.0)]
        self.assertEqual(classification(rows)["verdict"], "NO_CONSISTENT_IMPROVEMENT")

    def test_insufficient_or_nonfinite(self):
        self.assertEqual(classification([])["verdict"], "INSUFFICIENT_OR_NONFINITE")
        bad = [self._row(-1, 900.0, 270.0), {"itr": 0, "k3": float("nan"),
                                             "gap_to_all_mec": 1.0, "k0": 1.0,
                                             "gap_to_greedy": 2.0}]
        self.assertEqual(classification(bad)["verdict"], "INSUFFICIENT_OR_NONFINITE")


class TestWatchdog(unittest.TestCase):
    def test_value_limit_and_non_finite_stop(self):
        flags = watchdog_flags({"critic/value_abs_max": [1.0, 1e3], "policy/approx_kl": [0.1]})
        self.assertTrue(flags["stop"]["value_abs_max_over_limit"])
        self.assertTrue(flags["stop_any"])
        flags = watchdog_flags({"policy/approx_kl": [float("inf")]})
        self.assertTrue(flags["stop"]["non_finite"])

    def test_low_share_and_entropy_warn_only(self):
        flags = watchdog_flags({
            "action_fraction/local": [0.005] * 5,
            "action_fraction/v2v": [0.2] * 5,
            "policy/entropy_valid": [0.005] * 5,
        })
        self.assertTrue(flags["warn"]["local_below_floor"])
        self.assertTrue(flags["warn"]["entropy_near_zero"])
        self.assertFalse(flags["stop_any"])


class TestInterimAndManifest(unittest.TestCase):
    def test_interim_summary_reports_gap_reduction(self):
        init = {"itr": -1, "k3": 900.0, "gap_to_all_mec": 270.0}
        rows = [{"itr": 200, "k0": 880.0, "k3": 800.0, "gap_to_all_mec": 170.0,
                 "gap_to_greedy": 180.0}]
        out = interim_summary(rows, init, 200)
        self.assertTrue(out["better_than_true_init"])
        self.assertAlmostEqual(out["gap_reduction_to_all_mec"], 100.0)
        self.assertTrue(out["k3_better_than_k0"])

    def test_manifest_lists_files_with_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a")
            (root / "sub").mkdir()
            (root / "sub" / "b.bin").write_bytes(b"bb")
            rows = manifest(root)
            self.assertEqual([r["relative_path"] for r in rows], ["a.txt", "sub/b.bin"])
            self.assertEqual(rows[0]["sha256"], sha256_file(root / "a.txt"))
            self.assertEqual(rows[1]["bytes"], 2)

    def test_dry_run_does_not_touch_the_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            code = main(["--root", str(root), "--seed", "0", "--itr", "1000", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(root.exists())

    def test_kish_target_uses_the_long_runner_and_a_separate_run_dir(self):
        script = (ROOT / "spec" / "kish_gpu.sh").read_text()
        self.assertIn("pilot-a-long-1000)", script)
        self.assertIn("python -m spec.pilot_long", script)
        self.assertIn("--root runs/long_latency_v1", script)
        block = script.split("pilot-a-long-1000)")[1].split(";;")[0]
        self.assertNotIn("--root runs/mask_sanity_v3", block)
        self.assertIn("--root runs/long_latency_v1", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
