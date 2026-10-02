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


class TestEntryArgumentParsing(unittest.TestCase):
    """The CLI must parse (no TensorFlow needed): a duplicate option string once made
    every campaign seed exit 1 before any iteration ran."""

    def test_help_exits_zero_and_options_are_unique(self):
        import subprocess

        out = subprocess.run([sys.executable, "spec/automotive_gpu_smoke.py", "--help"],
                             cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        for flag in ("--seed", "--iters", "--run-kind", "--long", "--i-allow-gpu", "--gpu"):
            self.assertIn(flag, out.stdout)
        # a duplicated option makes argparse itself fail (exit 2 + "conflicting option
        # string"), so a clean exit 0 plus a present option is the regression guard
        self.assertNotIn("conflicting option", out.stderr + out.stdout)
        self.assertIn("--iters", out.stdout)

    def test_long_without_gpu_permission_is_refused(self):
        import subprocess

        out = subprocess.run([sys.executable, "spec/automotive_gpu_smoke.py", "--long",
                              "--seed", "0"], cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("--long requires --i-allow-gpu", out.stderr + out.stdout)


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
        self.assertEqual(kwargs.get("run_kind"), "primary")

    def test_pilot_run_kind_and_iterations_reach_train(self):
        from spec import phase4_train_driver as drv

        recorded = {}
        original_train = drv._train
        original_write = drv._write_payload
        original_gpu = drv.require_gpu_permission
        drv._train = lambda *a, **k: recorded.update({"args": a, "kwargs": k})
        drv._write_payload = lambda run_dir, payload: None
        drv.require_gpu_permission = lambda allow_gpu: None
        try:
            run_dir = drv.run_automotive_primary_seed(0, True, n_itr=500,
                                                      run_kind="primary_500")
        finally:
            drv._train = original_train
            drv._write_payload = original_write
            drv.require_gpu_permission = original_gpu
        self.assertTrue(str(run_dir).endswith("primary_500/seed_0"))
        self.assertEqual(int(recorded["args"][1]), 500)
        self.assertEqual(recorded["kwargs"].get("run_kind"), "primary_500")

    def test_train_forwards_only_supported_kwargs_to_the_stack_builder(self):
        """STATIC check: every kwarg the `_train` call site forwards must exist on
        build_frozen_primary_stack.

        A missing `run_kind` parameter once aborted the launch with
        TypeError: build_frozen_primary_stack() got an unexpected keyword argument.
        Parsed from the AST instead of executed, so the test cannot be fooled by a
        stubbed callee.
        """
        import ast
        import inspect

        import meta_trainer as mt
        from spec import phase4_train_driver as drv

        # read the module FILE: a monkeypatched drv._train attribute must not change
        # what this static check inspects
        source = Path(inspect.getsourcefile(drv)).read_text()
        tree = ast.parse(source)
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "build_frozen_primary_stack":
                    forwarded |= {kw.arg for kw in node.keywords if kw.arg}
        self.assertIn("run_kind", forwarded)
        self.assertIn("dataset", forwarded)
        params = set(inspect.signature(mt.build_frozen_primary_stack).parameters)
        self.assertEqual(sorted(forwarded - params), [],
                         "unsupported kwargs at the _train call site")

    def test_validation_can_run_more_than_once(self):
        """REGRESSION: the adaptation PPO must be built once, not per validation.

        Building it per call raised "Variable ppo_update_validation_policy/... already
        exists" at the SECOND validation (iteration 50), which killed all five seeds of
        the first pilot campaign after 50 iterations. This test builds the real stack
        and evaluates k=0/k=3 three times in one session.
        """
        import tempfile

        import tensorflow as tf

        from spec.automotive_training.automotive_primary import (
            build_automotive_primary_stack,
        )

        with tempfile.TemporaryDirectory() as tmp:
            trainer, algo = build_automotive_primary_stack(
                seed=0, n_itr=1, ckpt_dir=tmp)
            evaluator = trainer.held_out_evaluator
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                algo.sync_task_policies_from_core()
                first = evaluator.evaluate_all(k_steps=3)
                second = evaluator.evaluate_all(k_steps=3)
                third = evaluator.evaluate_all(k_steps=0)
            self.assertGreater(first["rollouts"], 0)
            self.assertGreater(second["rollouts"], 0)
            self.assertGreater(third["rollouts"], 0)
            self.assertEqual(getattr(evaluator, "adaptation_ppo_constructions", 0), 1,
                             "the adaptation PPO must be constructed exactly once")

    def test_smoke_entry_uses_the_smoke_dataset_too(self):
        from spec import phase4_train_driver as drv

        recorded = {}
        original_train = drv._train
        original_write = drv._write_payload
        original_gpu = drv.require_gpu_permission
        drv._train = lambda *a, **k: recorded.update({"kwargs": k})
        drv._write_payload = lambda run_dir, payload: None
        drv.require_gpu_permission = lambda allow_gpu: None
        try:
            drv.run_automotive_gpu_smoke(0, True, n_itr=1)
        finally:
            drv._train = original_train
            drv._write_payload = original_write
            drv.require_gpu_permission = original_gpu
        self.assertEqual(recorded["kwargs"].get("dataset"), "automotive_mc_v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
