#!/usr/bin/env python3
"""Tests for the dedicated ⑥b trainer entrypoint (spec/mask_sanity.py).

The training itself needs TF/GPU, but the parts that decide whether a 500-iteration
run is even allowed -- run-dir isolation, the cheap preflight and the post-run CSV
validation -- are pure Python and are tested here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.mask_sanity import (  # noqa: E402
    ALLOWED_ITERS,
    METRIC_KEYS,
    TELEMETRY_KEYS,
    VALUE_ABS_MAX_LIMIT,
    ZERO_RATE_KEYS,
    run_dir,
    validate_progress_csv,
)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in columns))
    path.write_text("\n".join(lines) + "\n")


def good_row(**overrides):
    row = {key: 0.0 for key in METRIC_KEYS}
    row["policy/entropy_valid"] = 1.05
    row["critic/value_abs_max"] = 0.25
    # E4.1/E4.2 scoped telemetry columns the smoke validator now requires
    row.update({
        "energy/requester_joules": 1.0,
        "energy/mobile_joules": 2.0,
        "energy/system_joules": 3.0,
        "energy/primary_joules": 3.0,
        "energy/primary_scope": "system",
    })
    row.update(overrides)
    return row


class TestRunDirLayout(unittest.TestCase):
    def test_mode_and_seed_are_separated(self):
        root = Path("/tmp/runs")
        self.assertEqual(run_dir("off", 0, root), root / "off" / "seed_0")
        self.assertEqual(run_dir("static", 3, root), root / "static" / "seed_3")

    def test_off_and_static_cannot_collide(self):
        root = Path("/tmp/runs")
        self.assertNotEqual(run_dir("off", 0, root), run_dir("static", 0, root))

    def test_allowed_iterations_are_exactly_one_and_five_hundred(self):
        self.assertEqual(tuple(ALLOWED_ITERS), (1, 500))


class TestProgressCsvValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mask_sanity_"))
        self.csv = self.tmp / "logs" / "progress.csv"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_csv_passes(self):
        write_csv(self.csv, list(METRIC_KEYS) + list(TELEMETRY_KEYS), [good_row(), good_row()])
        out = validate_progress_csv(self.tmp, 1)
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["rows"], 2)
        self.assertAlmostEqual(out["value_abs_max"], 0.25)

    def test_missing_metric_column_is_fatal(self):
        columns = [k for k in METRIC_KEYS if k != "mask/forced_rate"]
        write_csv(self.csv, columns, [{k: 0.0 for k in columns}])
        out = validate_progress_csv(self.tmp, 1)
        self.assertTrue(any(f["check"] == "metric_column_present" for f in out["failures"]))

    def test_nonzero_control_rate_is_fatal(self):
        write_csv(self.csv, list(METRIC_KEYS), [good_row(**{"mask/active_rate": 0.25})])
        out = validate_progress_csv(self.tmp, 1)
        self.assertTrue(any(f["check"] == "control_rate_zero" for f in out["failures"]))
        self.assertEqual(len(ZERO_RATE_KEYS), 5)

    def test_non_finite_metric_is_fatal(self):
        write_csv(self.csv, list(METRIC_KEYS), [good_row(**{"policy/entropy_valid": "nan"})])
        out = validate_progress_csv(self.tmp, 1)
        self.assertTrue(any(f["check"] == "metric_value_finite" for f in out["failures"]))

    def test_value_abs_max_limit(self):
        write_csv(
            self.csv,
            list(METRIC_KEYS),
            [good_row(**{"critic/value_abs_max": VALUE_ABS_MAX_LIMIT})],
        )
        out = validate_progress_csv(self.tmp, 1)
        self.assertTrue(any(f["check"] == "value_abs_max_below_limit" for f in out["failures"]))

    def test_missing_file_is_fatal(self):
        out = validate_progress_csv(self.tmp, 1)
        self.assertTrue(any(f["check"] == "progress_csv_exists" for f in out["failures"]))

    def test_row_count_is_checked(self):
        write_csv(self.csv, list(METRIC_KEYS), [good_row()])
        out = validate_progress_csv(self.tmp, 3)
        self.assertTrue(any(f["check"] == "row_count" for f in out["failures"]))


class TestPreflightCli(unittest.TestCase):
    """Runs the real CLI in a subprocess so its env changes stay contained."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mask_sanity_cli_"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *extra):
        argv = [
            sys.executable,
            "-m",
            "spec.mask_sanity",
            "--runs-root",
            str(self.tmp),
            "--skip-git-check",
            *extra,
        ]
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        env.pop("MARGO_OBS_VERSION", None)
        env.pop("MARGO_MASK_MODE", None)
        env.pop("MARGO_CONSTRAINTS", None)
        return subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, env=env)

    def _json(self, proc):
        idx = proc.stdout.find("{")
        self.assertGreaterEqual(idx, 0, proc.stdout + proc.stderr)
        return json.loads(proc.stdout[idx:])

    def test_preflight_only_passes_and_records_provenance(self):
        proc = self._run("--mode", "static", "--itr", "1", "--preflight-only")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = self._json(proc)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["obs_version"], "v3")
        self.assertEqual(report["mask_mode"], "static")
        self.assertEqual(report["constraints"], "off")
        self.assertEqual(report["run_dir"], str(self.tmp / "static" / "seed_0"))
        self.assertTrue(report["git_sha"])
        cfg = self.tmp / "static" / "seed_0" / "config.resolved.json"
        self.assertTrue(cfg.is_file())
        self.assertEqual(json.loads(cfg.read_text())["mask_mode"], "static")

    def test_preflight_off_mode_has_no_placeholder_requirement(self):
        proc = self._run("--mode", "off", "--itr", "1", "--preflight-only")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self._json(proc)["mask_mode"], "off")

    def test_unsupported_iteration_is_rejected(self):
        proc = self._run("--mode", "static", "--itr", "5", "--preflight-only")
        self.assertNotEqual(proc.returncode, 0)

    def test_unsupported_mode_is_rejected(self):
        proc = self._run("--mode", "runtime", "--itr", "1", "--preflight-only")
        self.assertNotEqual(proc.returncode, 0)


class TestLauncherTargets(unittest.TestCase):
    def setUp(self):
        self.script = (ROOT / "spec" / "kish_gpu.sh").read_text()

    def test_dedicated_targets_exist(self):
        for target in (
            "mask-train-smoke-off",
            "mask-train-smoke-static",
            "mask-sanity-500-off",
            "mask-sanity-500-static",
        ):
            self.assertIn("%s)" % target, self.script, target)

    def test_dedicated_targets_use_the_sanity_entrypoint(self):
        self.assertIn("--mode off --itr 1", self.script)
        self.assertIn("--mode static --itr 1", self.script)
        self.assertIn("--mode off --itr 500", self.script)
        self.assertIn("--mode static --itr 500", self.script)
        # four runs, one cheap preflight target, the E4.2 energy smoke and the
        # Part A constraint smoke
        self.assertEqual(self.script.count("python -m spec.mask_sanity"), 7)
        self.assertIn("--preflight-only", self.script)

    def test_constraint_smoke_uses_the_expected_scenario(self):
        self.assertIn("energy-constraint-smoke)", self.script)
        self.assertIn("--constraints-scenario total_small", self.script)

    def test_energy_smoke_uses_the_new_reward_and_objective_contract(self):
        self.assertIn("energy-train-smoke)", self.script)
        self.assertIn("--reward-mode latency_only", self.script)
        self.assertIn("--objective-mode log_only", self.script)
        self.assertIn("energy-tests)", self.script)
        # the image has no pytest; the stdlib runner must be used
        self.assertIn("python -m unittest discover", self.script)
        self.assertNotIn("python -m pytest", self.script)

    def test_legacy_par500_is_untouched(self):
        # the old v0.1 diagnostic must not be reused for the ⑥b runs
        self.assertIn("par500)", self.script)
        self.assertIn("--diagnostic-500", self.script)


class TestPrimarySchedulerConfig(unittest.TestCase):
    def test_smoke_stack_uses_the_resolved_system_config(self):
        from spec.mask_sanity import primary_scheduler_config

        cfg = primary_scheduler_config()
        self.assertEqual(cfg.energy_scope, "system")
        self.assertTrue(str(getattr(cfg.energy_model, "model", "")))

    def test_smoke_entry_passes_the_config_strictly(self):
        import inspect
        import spec.mask_sanity as mod

        source = inspect.getsource(mod._train_masked)
        self.assertIn("scheduler_config=primary_scheduler_config()", source)
        self.assertIn("strict_scheduler_config=True", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
