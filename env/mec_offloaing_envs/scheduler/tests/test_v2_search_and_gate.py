#!/usr/bin/env python3
"""v2 stage-6 tests: stronger-search accounting/labels and gate verdict logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.geometry_gate import REGIMES  # noqa: E402
from spec.automotive_training.v2.stronger_search import stronger_search  # noqa: E402


class TestStrongerSearch(unittest.TestCase):
    def _objective(self, plan):
        # cheap for all-MEC, cheaper still if token 2 is UE and token 3 is HELPER
        cost = 0.10 * sum(1 for a in plan.values() if a != 1)
        if plan.get(2) == 0:
            cost -= 0.05
        if plan.get(3) == 2:
            cost -= 0.05
        return cost

    def test_labels_and_budget(self):
        result = stronger_search(list(range(6)), self._objective, starts=3, seed=0, budget=200)
        self.assertEqual(result.label, "candidate-search lower bound")
        self.assertLessEqual(result.evaluations, 200)
        self.assertGreater(result.evaluations, 0)
        self.assertIn(2, result.plan)
        self.assertTrue(result.convergence)

    def test_reproducible(self):
        a = stronger_search(list(range(6)), self._objective, starts=3, seed=5, budget=300)
        b = stronger_search(list(range(6)), self._objective, starts=3, seed=5, budget=300)
        self.assertEqual(a.plan, b.plan)
        self.assertEqual(a.objective, b.objective)
        self.assertEqual(a.evaluations, b.evaluations)

    def test_starts_from_all_mec_and_never_worse(self):
        result = stronger_search(list(range(5)), self._objective, starts=1, seed=0, budget=100)
        self.assertLessEqual(result.objective, self._objective({i: 1 for i in range(5)}) + 1e-12)


class TestGateDefinition(unittest.TestCase):
    def test_regimes_cover_the_required_axes(self):
        self.assertEqual(len(REGIMES), 12)
        for key in ("A_mec_idle_stable_v2i", "D_mec_heavy", "E_helper_idle_stable_v2v",
                    "G_helper_short_contact", "H_poor_v2v", "I_network_high_variance",
                    "J_high_unreliable_remote", "K_high_reliable_idle_mec",
                    "L_low_good_remote"):
            self.assertIn(key, REGIMES)
        self.assertEqual(REGIMES["D_mec_heavy"]["load"], 8)
        self.assertEqual(REGIMES["J_high_unreliable_remote"]["criticality"], "HIGH")
        self.assertEqual(REGIMES["L_low_good_remote"]["criticality"], "LOW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
