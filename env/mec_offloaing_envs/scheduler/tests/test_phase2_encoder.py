#!/usr/bin/env python3
"""Phase 2 encoder tests: DAG adj, remap, degrees, stats isolation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    MAX_NEIGH,
    MAX_TASKS,
    PACKED_DIM,
    PAD_INDEX,
    EncoderGraphError,
    FeatureStats,
    encode_canonical_dag,
    fit_feature_stats,
    global_neighbor_indices,
    load_feature_stats,
    neighbor_index_tables,
    packed_edge_set,
    raw_node_features,
    sha256_canonical_text,
    spec_source_hashes,
    unpack_observation,
)
from env.mec_offloaing_envs.scheduler.model import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
)


def _tasks(*ids: int, workload: int = 10, output: int = 4, external: int = 0) -> list[CanonicalTask]:
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


def _chain(ids: list[int], edge_bytes: int = 7) -> CanonicalDAG:
    tasks = _tasks(*ids, external=3)
    edges = [(ids[i], ids[i + 1], edge_bytes) for i in range(len(ids) - 1)]
    return CanonicalDAG.from_records(tasks, edges)


class TestEncoderPacking(unittest.TestCase):
    def test_chain_is_not_clique(self):
        dag = _chain([0, 1, 2, 3])
        order = [0, 1, 2, 3]
        packed = encode_canonical_dag(dag, order, stats=FeatureStats.identity())
        _, fw, bw, mask = unpack_observation(packed)
        self.assertEqual(packed.shape, (4, PACKED_DIM))
        np.testing.assert_array_equal(mask, np.ones(4))
        self.assertEqual(int((fw[0] >= 0).sum()), 1)
        self.assertEqual(int(fw[0, 0]), 1)
        self.assertEqual(int((fw[3] >= 0).sum()), 0)
        self.assertEqual(int((bw[3] >= 0).sum()), 1)
        self.assertEqual(int(bw[3, 0]), 2)
        self.assertNotEqual(int((fw >= 0).sum()), 4 * 4)

    def test_remap_task_id_to_decoder_index(self):
        dag = _chain([0, 1, 2])
        order = [2, 0, 1]
        fw, bw, fw_len, bw_len = neighbor_index_tables(dag, order)
        # DAG edges 0->1, 1->2. decoder: 2@0, 0@1, 1@2
        self.assertEqual(int(fw_len[1]), 1)
        self.assertEqual(int(fw[1, 0]), 2)  # task 0 -> task 1 at decoder 2
        self.assertEqual(int(fw_len[2]), 1)
        self.assertEqual(int(fw[2, 0]), 0)  # task 1 -> task 2 at decoder 0
        self.assertEqual(int(bw_len[0]), 1)
        self.assertEqual(int(bw[0, 0]), 2)

    def test_no_six_neighbor_truncation(self):
        ids = list(range(9))
        tasks = _tasks(*ids, external=1)
        edges = [(i, 8, 11) for i in range(8)]
        dag = CanonicalDAG.from_records(tasks, edges)
        packed = encode_canonical_dag(dag, ids, stats=FeatureStats.identity())
        _, fw, bw, _ = unpack_observation(packed)
        self.assertEqual(int((bw[8] >= 0).sum()), 8)
        self.assertGreater(int((bw[8] >= 0).sum()), 6)
        self.assertEqual(sorted(int(x) for x in bw[8] if x >= 0), list(range(8)))
        self.assertTrue(np.all(fw[:8, 0] == 8))

    def test_uses_edge_output_bytes(self):
        tasks = _tasks(0, 1, external=5)
        light = CanonicalDAG.from_records(tasks, [(0, 1, 3)])
        heavy = CanonicalDAG.from_records(tasks, [(0, 1, 99)])
        idx = FEATURE_NAMES.index("incoming_edge_bytes")
        a = raw_node_features(light, [0, 1])
        b = raw_node_features(heavy, [0, 1])
        self.assertEqual(a[1, idx], 3)
        self.assertEqual(b[1, idx], 99)
        self.assertNotEqual(a[1, idx], b[1, idx])

    def test_no_legacy_scheduler_cost_slots(self):
        forbidden = ("T_loc", "T_up", "T_mec", "local_process_cost", "up_link_cost")
        for name in forbidden:
            self.assertNotIn(name, FEATURE_NAMES)
        self.assertEqual(FEATURE_DIM, len(FEATURE_NAMES))
        self.assertNotIn("pred_id", "".join(FEATURE_NAMES))

    def test_padding_mask_and_pad_index(self):
        dag = _chain([0, 1])
        packed = encode_canonical_dag(dag, [0, 1], stats=FeatureStats.identity())
        _, fw, bw, mask = unpack_observation(packed)
        self.assertTrue(np.all(mask == 1.0))
        self.assertEqual(int(fw[1, 0]), PAD_INDEX)
        self.assertEqual(int(bw[0, 0]), PAD_INDEX)
        self.assertTrue(np.all(fw[1, 1:] == PAD_INDEX))

    def test_global_neighbor_indices_dummy_pad(self):
        dag = _chain([0, 1, 2])
        packed = encode_canonical_dag(dag, [0, 1, 2], stats=FeatureStats.identity())
        _, fw, _, _ = unpack_observation(packed)
        global_adj, lengths = global_neighbor_indices(fw[None, ...])
        self.assertEqual(global_adj.shape, (3, MAX_NEIGH))
        self.assertEqual(int(lengths[0]), 1)
        self.assertEqual(int(global_adj[0, 0]), 1)
        dummy = 3
        self.assertTrue(np.all(global_adj[0, 1:] == dummy))
        self.assertEqual(int(lengths[2]), 0)
        self.assertTrue(np.all(global_adj[2] == dummy))

    def test_duplicate_edges_collapse_to_one_neighbor(self):
        tasks = _tasks(0, 1, external=2)
        dag = CanonicalDAG.from_records(tasks, [(0, 1, 5), (0, 1, 5), (0, 1, 5)])
        self.assertEqual(dag.unique_edge_count, 1)
        fw, bw, fw_len, bw_len = neighbor_index_tables(dag, [0, 1])
        self.assertEqual(int(fw_len[0]), 1)
        self.assertEqual(int(bw_len[1]), 1)
        self.assertEqual(int(fw[0, 0]), 1)

    def test_noncontiguous_task_ids(self):
        dag = _chain([0, 2, 5])
        order = [5, 0, 2]
        packed = encode_canonical_dag(dag, order, stats=FeatureStats.identity())
        edges = packed_edge_set(packed, order)
        self.assertEqual(edges, {(0, 2), (2, 5)})

    def test_degree_19_star_is_full_capacity(self):
        self.assertEqual(MAX_NEIGH, MAX_TASKS - 1)
        ids = list(range(MAX_TASKS))
        tasks = _tasks(*ids, external=1)
        edges = [(i, MAX_TASKS - 1, 11) for i in range(MAX_NEIGH)]
        dag = CanonicalDAG.from_records(tasks, edges)
        packed = encode_canonical_dag(dag, ids, stats=FeatureStats.identity())
        _, fw, bw, _ = unpack_observation(packed)
        self.assertEqual(int((bw[MAX_TASKS - 1] >= 0).sum()), MAX_NEIGH)
        self.assertEqual(
            sorted(int(x) for x in bw[MAX_TASKS - 1] if x >= 0),
            list(range(MAX_NEIGH)),
        )
        self.assertTrue(np.all(fw[:MAX_NEIGH, 0] == MAX_TASKS - 1))

    def test_degree_above_max_neigh_raises(self):
        n = MAX_NEIGH + 2
        ids = list(range(n))
        tasks = _tasks(*ids, external=1)
        edges = [(i, n - 1, 1) for i in range(n - 1)]
        dag = CanonicalDAG.from_records(tasks, edges)
        with self.assertRaises(EncoderGraphError):
            encode_canonical_dag(dag, ids, stats=FeatureStats.identity())


class TestTopologyAndPermutation(unittest.TestCase):
    def test_topology_sensitivity_same_nodes_different_edges(self):
        # Same per-node raw features (including degree) ; only wiring differs.
        tasks = _tasks(0, 1, 2, 3, external=8)
        graph_a = CanonicalDAG.from_records(tasks, [(0, 1, 4), (2, 3, 4)])
        graph_b = CanonicalDAG.from_records(tasks, [(0, 3, 4), (2, 1, 4)])
        order = [0, 1, 2, 3]
        stats = FeatureStats.identity()
        np.testing.assert_array_equal(
            raw_node_features(graph_a, order), raw_node_features(graph_b, order)
        )
        packed_a = encode_canonical_dag(graph_a, order, stats=stats)
        packed_b = encode_canonical_dag(graph_b, order, stats=stats)
        _, fw_a, _, _ = unpack_observation(packed_a)
        _, fw_b, _, _ = unpack_observation(packed_b)
        self.assertFalse(np.array_equal(fw_a, fw_b))
        self.assertNotEqual(
            packed_edge_set(packed_a, order), packed_edge_set(packed_b, order)
        )

    def test_permutation_consistency(self):
        tasks = _tasks(0, 1, 2, 3, external=9)
        dag = CanonicalDAG.from_records(
            tasks, [(0, 1, 2), (0, 2, 3), (1, 3, 4), (2, 3, 5)]
        )
        order_a = [0, 1, 2, 3]
        order_b = [3, 0, 2, 1]
        stats = FeatureStats.identity()
        packed_a = encode_canonical_dag(dag, order_a, stats=stats)
        packed_b = encode_canonical_dag(dag, order_b, stats=stats)
        self.assertEqual(
            packed_edge_set(packed_a, order_a), packed_edge_set(packed_b, order_b)
        )
        expected = {(0, 1), (0, 2), (1, 3), (2, 3)}
        self.assertEqual(packed_edge_set(packed_a, order_a), expected)


class TestMetaTrainStats(unittest.TestCase):
    def test_standardize_uses_injected_stats(self):
        dag = _chain([0, 1])
        mean = np.zeros(FEATURE_DIM)
        std = np.ones(FEATURE_DIM)
        idx = FEATURE_NAMES.index("incoming_edge_bytes")
        mean[idx] = 7.0
        std[idx] = 2.0
        stats = FeatureStats(
            feature_names=FEATURE_NAMES,
            mean=mean,
            std=std,
            n_graphs=1,
            n_nodes=2,
            dataset_manifest_sha256="a" * 64,
            split_policy_sha256="b" * 64,
        )
        packed = encode_canonical_dag(dag, [0, 1], stats=stats)
        feats, _, _, _ = unpack_observation(packed)
        self.assertAlmostEqual(float(feats[1, idx]), (7.0 - 7.0) / 2.0)

    def test_fit_from_rows_only(self):
        rows = [np.arange(FEATURE_DIM, dtype=np.float64).reshape(1, -1)]
        stats = fit_feature_stats(rows, n_graphs=1, max_indegree_unique=1, max_outdegree_unique=1)
        self.assertEqual(stats.role, "meta_train")
        self.assertEqual(stats.n_graphs, 1)

    def test_frozen_stats_are_meta_train_only(self):
        path = ROOT / "spec" / "encoder_feature_stats.json"
        self.assertTrue(path.is_file(), "encoder_feature_stats.json missing; run fit_encoder_stats.py")
        stats = load_feature_stats(path)
        self.assertEqual(stats.role, "meta_train")
        data = json.loads(path.read_text())
        self.assertEqual(data["role"], "meta_train")
        n_train = 0
        with (ROOT / "spec" / "dataset_manifest.jsonl").open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("role") == "meta_train":
                    n_train += 1
        self.assertEqual(stats.n_graphs, n_train)
        self.assertGreater(stats.n_graphs, 0)
        self.assertEqual(MAX_NEIGH, MAX_TASKS - 1)
        self.assertEqual(data["max_tasks"], MAX_TASKS)
        self.assertEqual(data["max_neigh"], MAX_NEIGH)
        self.assertEqual(tuple(data["feature_names"]), FEATURE_NAMES)
        manifest_hash, split_hash = spec_source_hashes(ROOT / "spec")
        self.assertEqual(data["dataset_manifest_sha256"], manifest_hash)
        self.assertEqual(data["split_policy_sha256"], split_hash)
        self.assertLessEqual(stats.max_indegree_unique, MAX_NEIGH)
        self.assertLessEqual(stats.max_outdegree_unique, MAX_NEIGH)
        self.assertEqual(data["packed_dim"], PACKED_DIM)
        self.assertEqual(data["hash_normalization"], "canonical_lf")
        self.assertEqual(
            data["split_policy_sha256"],
            "1ae009a3ad30b0780e58c1b3e85d47406b538b80de8a5ff4ae013fd4e71957ba",
        )

    def test_hash_is_lf_canonical(self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            lf = Path(tmp) / "lf.json"
            crlf = Path(tmp) / "crlf.json"
            lf.write_bytes(b'{"k": 1}\n')
            crlf.write_bytes(b'{"k": 1}\r\n')
            self.assertEqual(sha256_canonical_text(lf), sha256_canonical_text(crlf))
            self.assertNotEqual(
                hashlib.sha256(lf.read_bytes()).hexdigest(),
                hashlib.sha256(crlf.read_bytes()).hexdigest(),
            )

    def test_nan_mean_rejected(self):
        mean = np.zeros(FEATURE_DIM)
        mean[0] = np.nan
        with self.assertRaises(EncoderGraphError):
            FeatureStats(
                feature_names=FEATURE_NAMES,
                mean=mean,
                std=np.ones(FEATURE_DIM),
                n_graphs=1,
                n_nodes=2,
                dataset_manifest_sha256="a" * 64,
                split_policy_sha256="b" * 64,
            )

    def test_neighbor_index_out_of_range_rejected(self):
        dag = _chain([0, 1, 2])
        features = np.zeros((3, FEATURE_DIM), dtype=np.float32)
        fw = np.full((3, MAX_NEIGH), PAD_INDEX, dtype=np.int32)
        bw = np.full((3, MAX_NEIGH), PAD_INDEX, dtype=np.int32)
        fw[0, 0] = 99
        from env.mec_offloaing_envs.scheduler.encoder_obs import pack_observation

        with self.assertRaises(EncoderGraphError):
            pack_observation(features, fw, bw)

    def test_spec_task_count_enforced_on_production_path(self):
        ids = list(range(MAX_TASKS + 1))
        tasks = _tasks(*ids, external=1)
        dag = CanonicalDAG.from_records(tasks, [(0, 1, 1)])
        with self.assertRaises(EncoderGraphError):
            encode_canonical_dag(
                dag, ids, stats=FeatureStats.identity(), enforce_task_count=True
            )
        packed = encode_canonical_dag(
            dag, ids, stats=FeatureStats.identity(), enforce_task_count=False
        )
        self.assertEqual(packed.shape[0], MAX_TASKS + 1)


class TestTaskGraphEncodePath(unittest.TestCase):
    def test_ranking_and_cost_uses_canonical_pack(self):
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

        gv = ROOT / "env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.0.gv"
        tg = OffloadingTaskGraph(str(gv))
        order = list(range(tg.task_number))
        packed = np.asarray(
            tg.encode_point_sequence_with_ranking_and_cost(order, resource_cluster=None)
        )
        self.assertEqual(packed.shape, (tg.task_number, PACKED_DIM))
        edges = packed_edge_set(packed, order)
        expected = {(int(e[0]), int(e[4])) for e in tg.edge_set}
        self.assertEqual(edges, expected)
        reversed_order = list(reversed(order))
        packed_rev = np.asarray(
            tg.encode_point_sequence_with_ranking_and_cost(reversed_order, resource_cluster=None)
        )
        self.assertEqual(packed_edge_set(packed_rev, reversed_order), expected)


if __name__ == "__main__":
    unittest.main()
