#!/usr/bin/env python3
"""Executed R/S CRN evaluation: pairing, independence, nested aggregation, no best-of-S."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2 import eval_loop  # noqa: E402
from spec.automotive_training.v2.crn import gumbel_argmax, gumbel_noise, seed_pair  # noqa: E402
from spec.automotive_training.v2.eval_loop import (  # noqa: E402
    AGG_KEYS, DETERMINISTIC_CANDIDATES, ENV_SEED_DOMAIN, STOCHASTIC_CANDIDATE, EvalLoopError,
    EvalProtocol, evaluate_candidate, mc_identity, physics_fingerprint,
    reference_policy_logits, replicate_mc, replicate_world, run_evaluation,
)
from spec.automotive_training.v2.adapters import plan_map_from_actions, pure_plan  # noqa: E402
from spec.automotive_training.v2.energy import schedule_energy  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import V2ConstraintError  # noqa: E402

from env.mec_offloaing_envs.scheduler.constraints import ConstraintMetrics  # noqa: E402


def _graph(index=0):
    return load_dataset().validation_query()[index]


PROTO = EvalProtocol(r_select=2, s_select=3, tokens=20)


class TestAxesAndPairing(unittest.TestCase):
    def test_both_axes_are_populated_and_indexed(self):
        out = evaluate_candidate(_graph(), "all_mec", PROTO)
        self.assertEqual(out["r_select"], 2)
        self.assertEqual(out["s_select"], 3)
        self.assertEqual(len(out["cells"]), 6)
        pairs = {(c["r"], c["s"]) for c in out["cells"]}
        self.assertEqual(pairs, {(r, s) for r in range(2) for s in range(3)})
        self.assertTrue(out["deterministic"])

    def test_the_stochastic_candidate_actually_samples(self):
        out = evaluate_candidate(_graph(), STOCHASTIC_CANDIDATE, PROTO)
        self.assertFalse(out["deterministic"])
        latencies = [c["episode_latency_s"] for c in out["cells"]]
        self.assertGreater(len(set(np.round(latencies, 9))), 1,
                           "S must produce more than one realized sample")
        # the same (r, s) must reproduce the same cell exactly
        again = evaluate_candidate(_graph(), STOCHASTIC_CANDIDATE, PROTO)
        self.assertEqual([c["episode_latency_s"] for c in again["cells"]], latencies)

    def test_crn_noise_is_shared_across_candidates(self):
        graph = _graph()
        graph_id = str(graph.graph_id)
        logits = np.zeros((20, 3))
        a = gumbel_argmax(logits, gumbel_noise(PROTO.protocol_id, 1, 2, graph_id, 20, 3))
        b = gumbel_argmax(logits, gumbel_noise(PROTO.protocol_id, 1, 2, graph_id, 20, 3))
        c = gumbel_argmax(logits, gumbel_noise(PROTO.protocol_id, 1, 3, graph_id, 20, 3))
        np.testing.assert_array_equal(a, b, "the same (r, s) key must give the same draw")
        self.assertFalse(np.array_equal(a, c), "a different s must be an independent draw")

    def test_candidates_share_one_world_per_replicate(self):
        graph = _graph()
        w0, _ = replicate_world(graph, PROTO, 0)
        w1, _ = replicate_world(graph, PROTO, 0)
        self.assertEqual(w0.fingerprint_sha256(), w1.fingerprint_sha256(),
                         "the environmental realization must be reproducible per r")
        w2, _ = replicate_world(graph, PROTO, 1)
        # F04/F27(1): the R axis must move the PHYSICS, proven with a fingerprint that EXCLUDES
        # `world_id`. The old assertion compared `fingerprint_sha256()`, which contains the
        # world label, so it passed even when every r produced identical demand and latency.
        self.assertNotEqual(physics_fingerprint(w0), physics_fingerprint(w2),
                            "a different r must be a different PHYSICAL realization")
        self.assertNotEqual([t.compute_bytes for t in w0.foreground.tasks],
                            [t.compute_bytes for t in w2.foreground.tasks],
                            "the demand MC realization must differ across r")
        self.assertNotEqual(mc_identity(replicate_mc(graph, PROTO, 0)),
                            mc_identity(replicate_mc(graph, PROTO, 1)))
        # a candidate change never regenerates the world
        a = w0.with_foreground_plan_map(plan_map_from_actions(graph, [1] * 20))
        b = w0.with_foreground_plan_map(plan_map_from_actions(graph, [0] * 20))
        self.assertEqual(a.structure_fingerprint_sha256(), b.structure_fingerprint_sha256(),
                         "changing the candidate must NOT regenerate the world")
        self.assertNotEqual(a.fingerprint_sha256(), b.fingerprint_sha256(),
                            "the two scored configurations are genuinely different")

    def test_env_and_policy_seed_namespaces_cannot_collide(self):
        """F03: the env seed material must not be the s=0 policy-draw material."""
        graph = _graph()
        graph_id = str(graph.graph_id)
        for r in range(3):
            env_pair = seed_pair(ENV_SEED_DOMAIN, r, 0, graph_id)
            policy_pair = seed_pair(PROTO.protocol_id, r, 0, graph_id)
            self.assertNotEqual(env_pair, policy_pair,
                                "the environmental seed and the s=0 policy draw are two reads "
                                "of one namespace")
        # and the draw itself is an independent stream
        logits = np.zeros((20, 3))
        env_noise = gumbel_noise(ENV_SEED_DOMAIN, 0, 0, graph_id, 20, 3)
        policy_noise = gumbel_noise(PROTO.protocol_id, 0, 0, graph_id, 20, 3)
        self.assertFalse(np.array_equal(env_noise, policy_noise))
        self.assertTrue(np.array_equal(
            gumbel_argmax(logits, env_noise), gumbel_argmax(logits, env_noise)))

    def test_r_axis_moves_latency_not_only_the_world_label(self):
        out = evaluate_candidate(_graph(), "all_mec", PROTO)
        lat = [c["episode_latency_s"] for c in out["cells"] if c["s"] == 0]
        self.assertGreater(len(set(np.round(lat, 12))), 1,
                           "deterministic all-MEC latency must differ across r once the R "
                           "axis draws demand")


class TestAggregation(unittest.TestCase):
    def test_aggregation_is_nested_and_never_best_of_s(self):
        out = evaluate_candidate(_graph(), STOCHASTIC_CANDIDATE, PROTO)
        values = [c["episode_latency_s"] for c in out["cells"]]
        by_r = {}
        for cell in out["cells"]:
            by_r.setdefault(cell["r"], []).append(cell["episode_latency_s"])
        expected = float(np.mean([np.mean(v) for v in by_r.values()]))
        got = out["nested"]["episode_latency_s"]["mean"]
        self.assertAlmostEqual(got, expected, places=9)
        self.assertLess(got, max(values), "best-of-S would report the minimum, not the mean")
        self.assertEqual(out["nested"]["episode_latency_s"]["replicates"], 2)

    def test_every_candidate_reports_finite_metrics_on_the_same_objective(self):
        for name in list(DETERMINISTIC_CANDIDATES) + [STOCHASTIC_CANDIDATE]:
            out = evaluate_candidate(_graph(), name, PROTO)
            for key in AGG_KEYS:
                value = out["nested"][key]["mean"]
                self.assertTrue(np.isfinite(value), "%s/%s" % (name, key))
            for cell in out["cells"]:
                self.assertGreaterEqual(cell["world_makespan_s"] + 1e-12,
                                        cell["episode_latency_s"])
                self.assertGreater(cell["system_joules"], 0.0)
                self.assertLessEqual(cell["requester_joules"], cell["mobile_joules"] + 1e-9)

    def test_paired_deltas_are_reported_against_the_reference(self):
        out = run_evaluation([_graph()], protocol=PROTO,
                             candidates=("all_mec", "all_ue", STOCHASTIC_CANDIDATE),
                             reference_candidate="all_mec")
        graph_row = out["graphs"][0]
        self.assertIn("all_ue", graph_row["paired_deltas"])
        delta = graph_row["paired_deltas"]["all_ue"]["episode_latency_s"]
        self.assertIn("mean", delta)
        self.assertGreater(delta["mean"], 0.0,
                           "all-UE must be slower than all-MEC in this fixture")
        self.assertIn("stderr", delta)

    def test_replay_preserves_a_graph_own_metrics(self):
        graph = _graph()
        a = evaluate_candidate(graph, "greedy_cd", PROTO)
        b = evaluate_candidate(graph, "greedy_cd", PROTO)
        self.assertEqual(a["replicate_means"], b["replicate_means"])


class TestCellIdentitiesAndDecodingBlocks(unittest.TestCase):
    def test_every_cell_carries_its_world_identity(self):
        """F27(3): a per-cell row must identify the world it was scored on."""
        out = evaluate_candidate(_graph(), "all_ue", PROTO)
        world, _ = replicate_world(_graph(), PROTO, 0)
        for cell in out["cells"]:
            for key in ("world_fingerprint_sha256", "mc_id", "config_sha",
                        "dataset_graph_id", "world_id", "graph_id"):
                self.assertIn(key, cell, key)
            self.assertTrue(np.isfinite(cell["episode_latency_s"]))
        # the r=0 cell must name the r=0 world
        cell0 = next(c for c in out["cells"] if c["r"] == 0)
        self.assertEqual(cell0["world_fingerprint_sha256"], world.fingerprint_sha256())
        self.assertEqual(cell0["dataset_graph_id"], 0)
        identities = {c["mc_id"] for c in out["cells"]}
        self.assertEqual(len(identities), PROTO.r_select,
                         "each replicate must carry its own demand identity")

    def test_deterministic_and_sampled_blocks_are_separate(self):
        """F27(4)/(5): never merge decodings, never report best-of-S."""
        out = run_evaluation([_graph()], protocol=PROTO,
                             candidates=("all_mec", STOCHASTIC_CANDIDATE),
                             reference_candidate="all_mec")
        self.assertIn("deterministic_decoding", out)
        self.assertIn("sampled_decoding", out)
        self.assertIn("all_mec", out["deterministic_decoding"]["candidates"])
        self.assertNotIn(STOCHASTIC_CANDIDATE, out["deterministic_decoding"]["candidates"])
        self.assertIn(STOCHASTIC_CANDIDATE, out["sampled_decoding"]["candidates"])
        self.assertNotIn("all_mec", out["sampled_decoding"]["candidates"])
        self.assertFalse(out["deterministic_decoding"]["best_of_s_used"])
        self.assertFalse(out["sampled_decoding"]["best_of_s_used"])
        self.assertIn("NEVER", out["sampled_decoding"]["selection_rule"])
        # the reported value is the nested MEAN, never the minimum over s
        cells = out["graphs"][0]["candidates"][STOCHASTIC_CANDIDATE]["cells"]
        lat = [c["episode_latency_s"] for c in cells]
        reported = out["panel_mean_over_graphs"][STOCHASTIC_CANDIDATE]["episode_latency_s"]
        self.assertAlmostEqual(reported, float(np.mean(lat)), places=9)
        self.assertGreater(reported, min(lat) - 1e-12)

    def test_run_evaluation_records_the_r_axis_evidence(self):
        out = run_evaluation([_graph()], protocol=PROTO, candidates=("all_mec",),
                             reference_candidate="all_mec")
        row = out["graphs"][0]
        self.assertEqual(len(row["realization_axis"]), PROTO.r_select)
        self.assertTrue(row["r_axis_moves_physics"])


class TestF28FailLoudly(unittest.TestCase):
    def test_score_rejects_a_non_finite_required_metric(self):
        """F28: `_score` must run the fail-loudly gate on the evaluator metric path."""
        graph = _graph()
        world, _ = replicate_world(graph, PROTO, 0)
        bad = ConstraintMetrics(
            ue_energy_j=1.0, helper_energy_j=1.0, total_energy_j=float("nan"),
            helper_compute_j=1.0, v2v_airtime_s=1.0, v2v_task_fraction=0.1,
            makespan_s=1.0, n_tasks=3, n_helper_tasks=1, n_mec_tasks=1)
        with mock.patch.object(eval_loop, "v2_metrics", return_value=bad):
            with self.assertRaises(V2ConstraintError):
                eval_loop._score(world, pure_plan(graph, 1))

    def test_score_reports_unmodelled_components(self):
        graph = _graph()
        world, _ = replicate_world(graph, PROTO, 0)
        scored = eval_loop._score(world, pure_plan(graph, 1))
        self.assertEqual(scored["unmodelled"]["status"], "UNMODELLED")
        self.assertTrue(any("receiver" in c for c in scored["unmodelled"]["components"]))
        self.assertTrue(any("backhaul" in c for c in scored["unmodelled"]["components"]))
        self.assertGreater(scored["n_judged_tasks"], 0)


class TestProtocolGuards(unittest.TestCase):
    def test_protocol_id_is_frozen(self):
        with self.assertRaises(EvalLoopError):
            EvalProtocol(protocol_id="something_else")

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(EvalLoopError):
            evaluate_candidate(_graph(), "not_a_candidate", PROTO)

    def test_logits_are_decision_time_only(self):
        graph = _graph()
        world, _ = replicate_world(graph, PROTO, 0)
        logits = reference_policy_logits(world, graph, PROTO)
        self.assertEqual(logits.shape, (20, 3))
        # rebuilding with a DIFFERENT realized link process must not change the logits,
        # because they are computed from nominal (plan-time) rates
        from spec.automotive_training.v2.link_model import make_process

        other = make_process("degraded", 999)
        result = world.with_foreground_plan_map(
            plan_map_from_actions(graph, [0] * 20)).schedule(link_process=other)
        self.assertGreater(result.makespan_s, 0.0)
        again = reference_policy_logits(world, graph, PROTO)
        np.testing.assert_allclose(logits, again, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
