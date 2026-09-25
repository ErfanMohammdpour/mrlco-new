#!/usr/bin/env python3
"""P1: pure parts of the read-only checkpoint evaluator."""

from __future__ import annotations

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


from spec.evaluate_checkpoint import sha256_file, summarize  # noqa: E402


def _side(policy, mec, greedy):
    return {
        "query_mean_latency": policy,
        "query_all_mec_latency": mec,
        "query_greedy_latency": greedy,
        "query_action_fraction/mec": 0.4,
        "query_entropy_valid": 0.9,
        "per_distribution": [{"ignored": 1}],
        "validation_per_graph_plans": ["heavy"],
        "validation_plan_identities": ["heavy"],
    }


class TestSummarize(unittest.TestCase):
    def test_gaps_and_k3_flag(self):
        out = summarize(_side(982.8035, 630.2798, 626.6017),
                        _side(893.3548, 630.2798, 626.6017))
        self.assertAlmostEqual(out["gaps"]["k3_gap_to_all_mec"], 893.3548 - 630.2798, places=6)
        self.assertAlmostEqual(out["gaps"]["k3_gap_to_greedy"], 893.3548 - 626.6017, places=6)
        self.assertAlmostEqual(out["gaps"]["k0_gap_to_all_mec"], 982.8035 - 630.2798, places=6)
        self.assertTrue(out["k3_better_than_k0"])
        self.assertAlmostEqual(out["k3"]["query_mean_latency"], 893.3548, places=6)

    def test_heavy_payload_is_not_copied(self):
        out = summarize(_side(1.0, 1.0, 1.0), _side(1.0, 1.0, 1.0))
        for side in ("k0", "k3"):
            self.assertNotIn("per_distribution", out[side])
            self.assertNotIn("validation_per_graph_plans", out[side])

    def test_missing_baseline_does_not_crash(self):
        out = summarize({"query_mean_latency": 1.0}, {"query_mean_latency": 0.5})
        self.assertEqual(out["gaps"], {})
        self.assertTrue(out["k3_better_than_k0"])


class TestSha(unittest.TestCase):
    def test_sha256_file_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b"abc")
            self.assertEqual(sha256_file(p), sha256_file(p))
            self.assertEqual(len(sha256_file(p)), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
