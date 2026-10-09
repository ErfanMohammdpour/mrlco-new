#!/usr/bin/env python3
"""F34: TF-gated test of the OBSERVED kernel-placement profiler.

`run_tf.py` only ever asserted the REQUESTED device. With `allow_soft_placement=True` an op that
was placed on `/gpu:0` can execute on the CPU, so "GPU" was claimed without being measured. The
`KernelProfiler` traces ONE `sess.run` with `RunOptions(FULL_TRACE)` + `RunMetadata` and records
the device each kernel ACTUALLY ran on (`step_stats.dev_stats`).

Runs in the TF1.15 container (skipped without TensorFlow); the image has no pytest, so this is a
plain `unittest` module:

    python -m unittest env.mec_offloaing_envs.scheduler.tests.test_v2_f34_kernel_profile_tf
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import tensorflow as tf

    HAS_TF = bool(getattr(tf, "__version__", "")) and hasattr(tf, "compat") and \
        hasattr(getattr(tf, "compat", None), "v1") and \
        hasattr(getattr(tf.compat.v1, "RunOptions", None), "FULL_TRACE")
except Exception:
    HAS_TF = False


def _profile_module():
    from spec.automotive_training.v2 import run_tf

    return run_tf


@unittest.skipUnless(HAS_TF, "requires TensorFlow (run in the TF1.15 image)")
class TestKernelProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tfv1 = tf.compat.v1
        tfv1.disable_eager_execution()
        tfv1.reset_default_graph()
        tfv1.set_random_seed(0)

    def _small_graph(self):
        """A graph large enough to exceed the profiler's op-count threshold, plus a STATEFUL
        counter that increments exactly once per `sess.run` of the fetch.

        A stateful counter (rather than a random op) makes "the profiler must not execute the
        graph an extra time" a crisp, deterministic assertion: if the profiler secretly ran the
        fetches for its trace, the counter would advance twice for one logical call.
        """
        tfv1 = tf.compat.v1
        a = tfv1.placeholder(tf.float32, shape=[64, 64], name="profile_a")
        b = tfv1.placeholder(tf.float32, shape=[64, 64], name="profile_b")
        prod = tf.matmul(a, b, name="profile_matmul")
        counter = tfv1.Variable(0.0, name="profile_counter")
        bumped = tfv1.assign_add(counter, 1.0, name="profile_bump")
        summed = tfv1.reduce_sum(prod) + bumped
        return [a, b], [summed, prod, counter]

    def test_profiler_records_observed_device_per_op(self):
        tfv1 = tf.compat.v1
        run_tf = _profile_module()
        phs, fetches = self._small_graph()
        rng = np.random.RandomState(0)
        feed = {phs[0]: rng.randn(64, 64).astype(np.float32),
                phs[1]: rng.randn(64, 64).astype(np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            profiler = run_tf.KernelProfiler(path, requested_device="cpu", min_ops=3)
            with tfv1.Session() as sess:
                sess.run(tfv1.global_variables_initializer())
                profiler.start(sess)
                sess.run(fetches, feed_dict=feed)
                profiler.stop(sess)
                doc = profiler.write(graph=tfv1.get_default_graph())
            on_disk = json.loads(path.read_text())
        self.assertTrue(on_disk.get("ok"), "the profiler must capture a real run: %s"
                        % (on_disk.get("reason"),))
        self.assertEqual(on_disk["schema"], "v2_kernel_profile_v1")
        observed = on_disk["observed"]
        self.assertGreaterEqual(int(observed["n_ops"]), 3)
        self.assertEqual(int(observed["n_ops"]),
                         int(observed["gpu_ops"]) + int(observed["cpu_ops"])
                         + int(observed["other_ops"]))
        # every recorded op carries the device the kernel RAN on, and (when the graph is passed)
        # the device the placer CHOSE, so a fallback is visible
        self.assertTrue(on_disk["op_device"], "no per-op device entries were recorded")
        names = {entry["op"] for entry in on_disk["op_device"]}
        self.assertIn("profile_matmul", names)
        for entry in on_disk["op_device"]:
            self.assertTrue(entry["device"], "an op entry has no device")
            self.assertIn("placed_device", entry)
        self.assertEqual(doc["observed"]["n_ops"], observed["n_ops"])

    def test_profiling_does_not_add_a_run_or_change_the_fetched_values(self):
        """Profiling must not execute the graph an extra time and must return the true outputs.

        The fixture's stateful counter advances exactly once per `sess.run` of the fetches, so an
        implementation that ran the graph for its trace would show `profiled[2] == 3.0` (two
        increments for one call) instead of `2.0`. The deterministic fetch (`prod`) must be
        bit-identical across all three calls, and the counter-dependent fetch must be exactly the
        graph's value at the advanced counter.
        """
        tfv1 = tf.compat.v1
        run_tf = _profile_module()
        phs, fetches = self._small_graph()
        rng = np.random.RandomState(1)
        feed = {phs[0]: rng.randn(64, 64).astype(np.float32),
                phs[1]: rng.randn(64, 64).astype(np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            profiler = run_tf.KernelProfiler(Path(tmp) / "p.json", min_ops=3)
            with tfv1.Session() as sess:
                sess.run(tfv1.global_variables_initializer())
                plain1 = sess.run(fetches, feed_dict=feed)
                profiler.start(sess)
                profiled = sess.run(fetches, feed_dict=feed)
                profiler.stop(sess)
                plain2 = sess.run(fetches, feed_dict=feed)
        # the deterministic fetch is unaffected by profiling
        np.testing.assert_array_equal(profiled[1], plain1[1])
        np.testing.assert_array_equal(profiled[1], plain2[1])
        # exactly ONE increment per logical run: no extra traced execution
        self.assertEqual(float(plain1[2]), 1.0)
        self.assertEqual(float(profiled[2]), 2.0,
                         "the profiler executed the graph more than once for one call")
        self.assertEqual(float(plain2[2]), 3.0)
        # the counter-dependent fetch is the graph's own value, not a substituted one
        self.assertAlmostEqual(float(profiled[0]) - float(plain1[0]), 1.0, places=3)
        self.assertAlmostEqual(float(plain2[0]) - float(profiled[0]), 1.0, places=3)

    def test_missing_capture_is_reported_not_faked(self):
        run_tf = _profile_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "never.json"
            profiler = run_tf.KernelProfiler(path, min_ops=10 ** 9)
            doc = profiler.write(graph=None)
            self.assertFalse(doc["ok"])
            self.assertIn("reason", doc)
            self.assertEqual(json.loads(path.read_text())["ok"], False)

    def test_visible_device_info_is_measured(self):
        run_tf = _profile_module()
        info = run_tf._visible_device_info()
        self.assertIn("cuda_visible_devices", info)
        self.assertIn("physical_devices", info)
        self.assertIn("allow_soft_placement", info)


# --- NON-TF artifact logic, exercised against TF-shaped stub messages ---------------------------
# These run EVERYWHERE (no TensorFlow needed): they cover the parsing, the GPU/CPU summary, the
# "keep the placed device so a fallback is visible" rule and the profiler's run-wrapper contract.
class _Memory:
    def __init__(self, total_bytes=0):
        self.total_bytes = int(total_bytes)


class _Node:
    def __init__(self, name, start=0, end=10, memory=0):
        self.node_name = name
        self.all_start_micros = int(start)
        self.op_start_rel_micros = 0
        self.op_end_rel_micros = int(end)
        self.all_end_rel_micros = int(end)
        self.memory = [_Memory(memory)]


class _Dev:
    def __init__(self, device, nodes):
        self.device = device
        self.node_stats = list(nodes)


class _StepStats:
    def __init__(self, devs):
        self.dev_stats = list(devs)


class _RunMetadata:
    def __init__(self, devs):
        self.step_stats = _StepStats(devs)


class _Op:
    def __init__(self, name, device):
        self.name = name
        self.device = device


class _Graph:
    def __init__(self, ops):
        self._ops = list(ops)

    def get_operations(self):
        return list(self._ops)


class TestKernelProfileParsing(unittest.TestCase):
    """`_profiled_device_record` is PURE PARSING: it runs everywhere, with no TensorFlow."""

    def _record(self, devs):
        from spec.automotive_training.v2 import run_tf

        return run_tf._profiled_device_record(_RunMetadata(devs))

    def test_device_counts_split_gpu_and_cpu(self):
        record = self._record([
            _Dev("/job:localhost/replica:0/task:0/device:CPU:0",
                 [_Node("a"), _Node("b")]),
            _Dev("/job:localhost/replica:0/task:0/device:GPU:0", [_Node("c")]),
        ])
        self.assertEqual(record["n_ops"], 3)
        self.assertEqual(record["cpu_ops"], 2)
        self.assertEqual(record["gpu_ops"], 1)
        self.assertEqual(record["other_ops"], 0)
        self.assertEqual(record["device_counts"],
                         {"/job:localhost/replica:0/task:0/device:CPU:0": 2,
                          "/job:localhost/replica:0/task:0/device:GPU:0": 1})
        self.assertEqual({e["op"] for e in record["ops"]}, {"a", "b", "c"})

    def test_placed_device_is_kept_so_a_fallback_is_visible(self):
        from spec.automotive_training.v2 import run_tf

        metadata = _RunMetadata([
            _Dev("/job:localhost/replica:0/task:0/device:CPU:0", [_Node("matmul")])])
        graph = _Graph([_Op("matmul", "/device:GPU:0")])
        record = run_tf._profiled_device_record(metadata, graph=graph)
        self.assertEqual(record["ops"][0]["placed_device"], "/device:GPU:0")
        self.assertEqual(record["ops"][0]["device"],
                         "/job:localhost/replica:0/task:0/device:CPU:0")

    def test_no_step_stats_is_reported(self):
        from spec.automotive_training.v2 import run_tf

        record = run_tf._profiled_device_record(object())
        self.assertEqual(record["n_ops"], 0)
        self.assertIn("note", record)
        self.assertEqual(record["ops"], [])


class TestKernelProfileArtifactLogic(unittest.TestCase):
    """`KernelProfiler.start`/`write` touch TF, so this class is TF-gated."""

    def setUp(self):
        if not HAS_TF:
            self.skipTest("requires TensorFlow (run in the TF1.15 image)")

    def test_profiler_run_returns_the_untraced_value_and_writes_an_artifact(self):
        """`KernelProfiler.start` wraps `sess.run`; the traced call must return exactly the
        untraced value and the artifact must carry the observed op devices."""
        import types
        from spec.automotive_training.v2 import run_tf

        devs = [_Dev("/device:CPU:0", [_Node("x%d" % i) for i in range(12)])]
        metadata = _RunMetadata(devs)
        original_kwargs = run_tf._profile_run_kwargs
        run_tf._profile_run_kwargs = lambda: ({"options": object(), "run_metadata": metadata},
                                              metadata)
        try:
            captured = {}

            class _Sess:
                def run(self, fetches, feed_dict=None, **kwargs):
                    captured["kwargs"] = sorted(kwargs)
                    return ["value"]

            sess = _Sess()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "kernel.json"
                profiler = run_tf.KernelProfiler(path, requested_device="cpu", min_ops=5)
                profiler.start(sess)
                out = sess.run(["fetch"], feed_dict={})
                profiler.stop(sess)
                doc = profiler.write(graph=None)
                on_disk = json.loads(path.read_text())
            self.assertEqual(out, ["value"])
            self.assertEqual(captured["kwargs"], ["options", "run_metadata"])
            self.assertTrue(doc["ok"])
            self.assertEqual(doc["observed"]["cpu_ops"], 12)
            self.assertEqual(doc["observed"]["n_ops"], 12)
            self.assertIn("visible", doc)
            self.assertEqual(on_disk["schema"], "v2_kernel_profile_v1")
            # the wrapper is removed again
            self.assertFalse(hasattr(sess, "_orig_run"))
            self.assertEqual(sess.run(["f"]), ["value"])
        finally:
            run_tf._profile_run_kwargs = original_kwargs

    def test_low_op_count_run_is_not_captured(self):
        import types
        from spec.automotive_training.v2 import run_tf

        metadata = _RunMetadata([_Dev("/device:CPU:0", [_Node("x")])])
        original_kwargs = run_tf._profile_run_kwargs
        run_tf._profile_run_kwargs = lambda: ({"options": object(), "run_metadata": metadata},
                                              metadata)
        try:
            class _Sess:
                def run(self, fetches, feed_dict=None, **kwargs):
                    return "ok"

            sess = _Sess()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "kernel.json"
                profiler = run_tf.KernelProfiler(path, requested_device="gpu", min_ops=10)
                profiler.start(sess)
                sess.run(["tiny"])
                profiler.stop(sess)
                doc = profiler.write(graph=None)
            self.assertFalse(doc["ok"])
            self.assertIn("threshold", doc["reason"])
        finally:
            run_tf._profile_run_kwargs = original_kwargs

    def test_gpu_request_with_zero_gpu_ops_is_flagged(self):
        from spec.automotive_training.v2 import run_tf

        metadata = _RunMetadata([_Dev("/device:CPU:0", [_Node("x%d" % i)
                                                        for i in range(12)])])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel.json"
            profiler = run_tf.KernelProfiler(path, requested_device="gpu", min_ops=5)
            profiler.captured = metadata
            profiler.captured_run_number = 1
            doc = profiler.write(graph=None)
        self.assertTrue(doc["ok"])
        self.assertIn("NO op executed on a GPU device", doc["warning"])
        self.assertEqual(doc["gpu_share"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
