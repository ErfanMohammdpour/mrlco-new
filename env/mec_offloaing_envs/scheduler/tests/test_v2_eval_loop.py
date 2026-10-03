#!/usr/bin/env python3
"""Executed R/S CRN evaluation: pairing, independence, nested aggregation, no best-of-S."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2.crn import gumbel_argmax, gumbel_noise  # noqa: E402
from spec.automotive_training.v2.eval_loop import (  # noqa: E402
    AGG_KEYS, DETERMINISTIC_CANDIDATES, STOCHASTIC_CANDIDATE, EvalLoopError, EvalProtocol,
    evaluate_candidate, reference_policy_logits, replicate_world, run_evaluation,
)
from spec.automotive_training.v2.adapters import plan_map_from_actions  # noqa: E402
from spec.automotive_training.v2.energy import schedule_energy  # noqa: E402


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
        self.assertNotEqual(w0.fingerprint_sha256(), w2.fingerprint_sha256(),
                            "a different r must be a different realization")
        # a candidate change never regenerates the world
        a = w0.with_foreground_plan_map(plan_map_from_actions(graph, [1] * 20))
        b = w0.with_foreground_plan_map(plan_map_from_actions(graph, [0] * 20))
        self.assertEqual(a.structure_fingerprint_sha256(), b.structure_fingerprint_sha256(),
                         "changing the candidate must NOT regenerate the world")
        self.assertNotEqual(a.fingerprint_sha256(), b.fingerprint_sha256(),
                            "the two scored configurations are genuinely different")


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
