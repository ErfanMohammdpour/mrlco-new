#!/usr/bin/env python3
"""Phase 1 encoder ablation tests. Numpy always. TF skipped if missing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
    MAX_TASKS,
    FeatureStats,
    encode_canonical_dag,
    reachability_mask,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask  # noqa: E402
from spec.encoder_fixture import (  # noqa: E402
    OUTPUT_PATH,
    PACKED_PATH,
    SEED,
    chain_dag,
    fork_join_dag,
)

HIDDEN = 128


def _tasks(*ids, workload=10, output=4, external=0):
    out = []
    for tid in ids:
        out.append(
            CanonicalTask(
                task_id=tid,
                compute_workload_bytes=workload,
                task_output_bytes=output,
                external_input_bytes=external if tid == ids[0] else 0,
            )
        )
    return out


def _tf_or_skip():
    try:
        import tensorflow as tf
    except ImportError:
        raise unittest.SkipTest("TensorFlow not installed")
    if not hasattr(tf, "contrib"):
        raise unittest.SkipTest("need tf.contrib (TensorFlow 1.15)")
    return tf


class TestReachabilityMask(unittest.TestCase):
    def test_four_node_chain(self):
        tasks = _tasks(0, 1, 2, 3, external=3)
        dag = CanonicalDAG.from_records(tasks, [(0, 1, 7), (1, 2, 7), (2, 3, 7)])
        got = reachability_mask(dag, [0, 1, 2, 3])
        want = np.ones((4, 4), dtype=np.float32)
        np.testing.assert_array_equal(got, want)

    def test_fork_join(self):
        tasks = _tasks(0, 1, 2, 3, external=3)
        dag = CanonicalDAG.from_records(
            tasks, [(0, 1, 5), (0, 2, 6), (1, 3, 7), (2, 3, 8)]
        )
        got = reachability_mask(dag, [0, 1, 2, 3])
        want = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(got, want)


class TestMeanaggFixtureRegression(unittest.TestCase):
    def test_meanagg_matches_prechange_fixture(self):
        tf = _tf_or_skip()
        if not PACKED_PATH.is_file() or not OUTPUT_PATH.is_file():
            raise unittest.SkipTest("meanagg fixture missing; run spec/dump_meanagg_fixture.py")
        packed = np.load(str(PACKED_PATH))
        want = np.load(str(OUTPUT_PATH))
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(SEED)
        np.random.seed(SEED)
        from policies.graph2seq_encoder import Graph2SeqEncoderAdapter

        ph = tf.compat.v1.placeholder(tf.float32, [None, None, packed.shape[-1]])
        adapter = Graph2SeqEncoderAdapter(
            input_dim=packed.shape[-1],
            hidden_dim=HIDDEN,
            num_layers=2,
            encoder_type="meanagg",
            readout_type="triple",
        )
        outputs, _ = adapter.encode(ph)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            got = sess.run(outputs, feed_dict={ph: packed})
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def _run_encoder(tf, packed, encoder_type, reach=None):
    from policies.graph2seq_encoder import Graph2SeqEncoderAdapter

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(0)
    np.random.seed(0)
    ph = tf.compat.v1.placeholder(tf.float32, [None, None, packed.shape[-1]])
    reach_ph = None
    if encoder_type == "dagformer":
        reach_ph = tf.compat.v1.placeholder(tf.float32, [None, None, None])
    adapter = Graph2SeqEncoderAdapter(
        input_dim=packed.shape[-1],
        hidden_dim=HIDDEN,
        num_layers=2,
        encoder_type=encoder_type,
        readout_type="triple",
        reachability_mask=reach_ph,
    )
    outputs, _ = adapter.encode(ph)
    fd = {ph: packed}
    if reach_ph is not None:
        fd[reach_ph] = reach
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        return sess.run(outputs, feed_dict=fd)


class TestEncoderPermutationAndTopology(unittest.TestCase):
    def _stats(self):
        return FeatureStats.identity()

    def test_permutation_all_encoders(self):
        tf = _tf_or_skip()
        from env.mec_offloaing_envs.scheduler.encoder_obs import pack_observation, unpack_observation

        order = list(range(MAX_TASKS))
        dag = chain_dag()
        packed0 = encode_canonical_dag(dag, order, stats=self._stats())
        perm = np.arange(MAX_TASKS)
        perm[0], perm[5] = perm[5], perm[0]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(MAX_TASKS)
        feat, fw, bw, mask = unpack_observation(packed0)

        def remap(adj):
            out = np.asarray(adj, dtype=np.int32).copy()
            valid = out >= 0
            out[valid] = inv[out[valid]]
            return out

        packed_p = pack_observation(feat[perm], remap(fw[perm]), remap(bw[perm]), mask[perm])
        reach = reachability_mask(dag, order)
        reach_p = reach[np.ix_(perm, perm)]
        packed = packed0[None, ...]
        packed_p = packed_p[None, ...]
        reach = reach[None, ...]
        reach_p = reach_p[None, ...]
        for enc in ("meanagg", "gatv2", "dagformer"):
            out = _run_encoder(tf, packed, enc, reach)
            out_p = _run_encoder(tf, packed_p, enc, reach_p)
            np.testing.assert_allclose(out_p[0], out[0][perm], atol=1e-5, err_msg=enc)

    def test_topology_changes_embeddings(self):
        tf = _tf_or_skip()
        order = list(range(MAX_TASKS))
        stats = self._stats()
        a = encode_canonical_dag(chain_dag(), order, stats=stats)[None, ...]
        b = encode_canonical_dag(fork_join_dag(), order, stats=stats)[None, ...]
        ra = reachability_mask(chain_dag(), order)[None, ...]
        rb = reachability_mask(fork_join_dag(), order)[None, ...]
        for enc in ("meanagg", "gatv2", "dagformer"):
            oa = _run_encoder(tf, a, enc, ra)
            ob = _run_encoder(tf, b, enc, rb)
            self.assertFalse(np.allclose(oa, ob, atol=1e-5), msg=enc)


if __name__ == "__main__":
    unittest.main()
