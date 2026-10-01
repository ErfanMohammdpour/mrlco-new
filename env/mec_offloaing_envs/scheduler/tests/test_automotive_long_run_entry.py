#!/usr/bin/env python3
"""Regression test for the LONG-run entry point.

The one-iteration smoke exercises `run_automotive_gpu_smoke`; the long run goes
through `run_automotive_primary_seed`, which is a different code path. A missing
argument there once produced `NameError: name 'meta_batch_size' is not defined` only
after the campaign had been launched, so this test pins the call contract WITHOUT
starting any training: `_train` is replaced by a recorder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # the driver imports TensorFlow
    import tensorflow as _tf  # noqa: F401

    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False


@unittest.skipUnless(HAS_TF, "long-run entry test requires TensorFlow (run in the TF1.15 image)")
class TestAutomotiveLongRunEntry(unittest.TestCase):
    def test_long_run_calls_train_with_the_automotive_dataset(self):
        from spec import phase4_train_driver as drv

        recorded = {}

        def fake_train(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = dict(kwargs)
            return None

        original_train = drv._train
        original_write = drv._write_payload
        original_gpu = drv.require_gpu_permission
        drv._train = fake_train
        drv._write_payload = lambda run_dir, payload: None
        drv.require_gpu_permission = lambda allow_gpu: None
        try:
            run_dir = drv.run_automotive_primary_seed(0, True, n_itr=1)
        finally:
            drv._train = original_train
            drv._write_payload = original_write
            drv.require_gpu_permission = original_gpu

        self.assertTrue(str(run_dir).endswith("primary/seed_0"))
        kwargs = recorded["kwargs"]
        self.assertEqual(kwargs.get("dataset"), "automotive_mc_v1")
        self.assertEqual(int(kwargs.get("meta_batch_size", 0)), 10)
        self.assertEqual(int(kwargs.get("support_trajectories", 0)), 20)
        self.assertEqual(int(recorded["args"][1]), 1, "n_itr must reach _train")

    def test_smoke_entry_uses_the_smoke_dataset_too(self):
        from spec import phase4_train_driver as drv

        recorded = {}
        drv._train = lambda *a, **k: recorded.update({"kwargs": k})
        original_write = drv._write_payload
        original_gpu = drv.require_gpu_permission
        drv._write_payload = lambda run_dir, payload: None
        drv.require_gpu_permission = lambda allow_gpu: None
        try:
            drv.run_automotive_gpu_smoke(0, True, n_itr=1)
        finally:
            drv._write_payload = original_write
            drv.require_gpu_permission = original_gpu
        self.assertEqual(recorded["kwargs"].get("dataset"), "automotive_mc_v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
