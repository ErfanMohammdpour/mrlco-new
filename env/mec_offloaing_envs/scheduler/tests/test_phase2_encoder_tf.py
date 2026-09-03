#!/usr/bin/env python3
"""Phase 2 TensorFlow encoder smoke.

Requires TensorFlow 1.x with tf.contrib (the training stack). Missing TF is a
failure, not a skip: Phase 2 must not close on packing tests alone.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
    PACKED_DIM,
    FeatureStats,
    encode_canonical_dag,
    encode_task_graph,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask  # noqa: E402


def _require_tf():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Phase 2 encoder smoke requires TensorFlow 1.15; import failed"
        ) from exc
    return tf


def _tasks(n: int) -> list[CanonicalTask]:
    out = []
    for tid in range(n):
        out.append(
            CanonicalTask(
                task_id=tid,
                compute_workload_bytes=10,
                task_output_bytes=4,
                external_input_bytes=8 if tid == 0 else 0,
            )
        )
    return out


def _matching_pair():
    """20-node graphs, identical node features, different edges."""
    tasks = _tasks(20)
    graph_a = CanonicalDAG.from_records(tasks, [(0, 1, 4), (2, 3, 4)])
    graph_b = CanonicalDAG.from_records(tasks, [(0, 3, 4), (2, 1, 4)])
    order = list(range(20))
    stats = FeatureStats.identity()
    return (
        encode_canonical_dag(graph_a, order, stats=stats, enforce_task_count=True),
        encode_canonical_dag(graph_b, order, stats=stats, enforce_task_count=True),
    )


def _pred_pair():
    """Sink 19 has different predecessor; node-19 features match; fw of 19 empty."""
    tasks = _tasks(20)
    graph_a = CanonicalDAG.from_records(tasks, [(0, 19, 4)])
    graph_b = CanonicalDAG.from_records(tasks, [(1, 19, 4)])
    order = list(range(20))
    stats = FeatureStats.identity()
    return (
        encode_canonical_dag(graph_a, order, stats=stats, enforce_task_count=True),
        encode_canonical_dag(graph_b, order, stats=stats, enforce_task_count=True),
    )


class TestEncoderTFSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf = _require_tf()
        cls.tf = tf
        tfv1 = tf.compat.v1 if hasattr(tf, "compat") else tf
        cls.tfv1 = tfv1
        if hasattr(tfv1, "disable_eager_execution"):
            tfv1.disable_eager_execution()
        tfv1.reset_default_graph()
        if hasattr(tf, "set_random_seed"):
            tf.set_random_seed(0)
        elif hasattr(tfv1, "set_random_seed"):
            tfv1.set_random_seed(0)
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        cls.policy = Seq2SeqPolicy(
            obs_dim=PACKED_DIM,
            encoder_units=32,
            decoder_units=32,
            vocab_size=3,
            name="phase2_smoke",
        )
        cls.sess = tfv1.Session()
        cls.sess.run(tfv1.global_variables_initializer())

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "sess", None) is not None:
            cls.sess.close()

    def _encode(self, packed: np.ndarray) -> np.ndarray:
        out = self.sess.run(
            self.policy.network.encoder_outputs,
            feed_dict={self.policy.obs: packed[None, ...]},
        )
        return np.asarray(out)

    def test_real_gv_forward_finite_shapes(self):
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

        gv = ROOT / "env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.0.gv"
        tg = OffloadingTaskGraph(str(gv))
        packed = np.asarray(
            encode_task_graph(tg, list(range(tg.task_number))),
            dtype=np.float32,
        )
        self.assertEqual(packed.shape, (20, PACKED_DIM))
        outputs = self._encode(packed)
        state = self.sess.run(
            self.policy.network.encoder_state,
            feed_dict={self.policy.obs: packed[None, ...]},
        )
        self.assertEqual(outputs.shape[0], 1)
        self.assertEqual(outputs.shape[1], 20)
        self.assertTrue(np.all(np.isfinite(outputs)))
        h = state[-1].h if isinstance(state, (tuple, list)) else state.h
        self.assertTrue(np.all(np.isfinite(h)))

    def test_dropout_off_is_deterministic(self):
        packed_a, _ = _matching_pair()
        y1 = self._encode(packed_a)
        y2 = self._encode(packed_a)
        np.testing.assert_array_equal(y1, y2)

    def test_topology_changes_embeddings(self):
        packed_a, packed_b = _matching_pair()
        ya = self._encode(packed_a)
        yb = self._encode(packed_b)
        self.assertFalse(np.allclose(ya, yb))

    def test_predecessor_wiring_changes_sink_embedding(self):
        packed_a, packed_b = _pred_pair()
        ya = self._encode(packed_a)[0, 19]
        yb = self._encode(packed_b)[0, 19]
        self.assertFalse(np.allclose(ya, yb))

    def test_task_id_relabel_preserves_aligned_embeddings(self):
        stats = FeatureStats.identity()
        tasks_a = [
            CanonicalTask(i, 10, 4, 8 if i == 0 else 0) for i in range(20)
        ]
        tasks_b = [
            CanonicalTask(i + 100, 10, 4, 8 if i == 0 else 0) for i in range(20)
        ]
        edges_a = [(0, 1, 4), (2, 3, 4)]
        edges_b = [(100, 101, 4), (102, 103, 4)]
        dag_a = CanonicalDAG.from_records(tasks_a, edges_a)
        dag_b = CanonicalDAG.from_records(tasks_b, edges_b)
        order_a = list(range(20))
        order_b = [i + 100 for i in range(20)]
        packed_a = encode_canonical_dag(dag_a, order_a, stats=stats, enforce_task_count=True)
        packed_b = encode_canonical_dag(dag_b, order_b, stats=stats, enforce_task_count=True)
        ya = self._encode(packed_a)
        yb = self._encode(packed_b)
        np.testing.assert_allclose(ya, yb, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
