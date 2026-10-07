#!/usr/bin/env python3
"""F13: the v2 reward must implement the declared discount-compatible shaping contract.

`spec/objective_contract.py` declares, for `latency_only`:

    J_t = L_t / L_scale ;  r_t = J_{t-1} - gamma * J_t
    optimised criterion R = sum_t gamma^(t-1) r_t = J_0 - gamma^N * J_N

That form is path-independent, so ranking by R agrees with ranking by the terminal plan
objective `-J_N`. The audited implementation emitted the plain delta `J_{t-1} - J_t` and let the
processor discount it on top, which does NOT telescope: two plans with identical terminal
latency but different intermediate prefixes got different returns (7.97 vs 7.93).

These tests pin the contract and would have caught the mismatch.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.objective_contract import (  # noqa: E402
    SHAPING_DISCOUNT, discounted_return, telescoped_from_potentials,
)


class DiscountCompatibleShapingTests(unittest.TestCase):
    def _shaped(self, potentials, gamma=SHAPING_DISCOUNT):
        """r_t = J_{t-1} - gamma*J_t from the prefix potentials [J_0, J_1, ..., J_N]."""
        return [potentials[t - 1] - gamma * potentials[t]
                for t in range(1, len(potentials))]

    def test_discounted_return_equals_terminal_affine_map(self):
        gamma = SHAPING_DISCOUNT
        for potentials in ([10.0, 5.0, 2.0], [10.0, 9.0, 2.0], [10.0, 1.0, 2.0, 0.5]):
            r = self._shaped(potentials, gamma)
            got = discounted_return(r, gamma)
            n = len(potentials) - 1
            want = potentials[0] - (gamma ** n) * potentials[-1]
            self.assertAlmostEqual(got, want, places=9,
                                   msg="discounted return is not J_0 - gamma^N J_N")

    def test_equal_terminal_objective_ranks_equal(self):
        """Same J_0 and J_N, different intermediate prefixes -> same discounted return."""
        gamma = SHAPING_DISCOUNT
        a = discounted_return(self._shaped([10.0, 5.0, 2.0], gamma), gamma)
        b = discounted_return(self._shaped([10.0, 9.0, 2.0], gamma), gamma)
        self.assertAlmostEqual(a, b, places=9,
                               msg="ranking depends on the intermediate prefix")

    def test_legacy_delta_form_is_path_dependent(self):
        """The pre-fix form does NOT telescope -- proof the defect was real."""
        gamma = SHAPING_DISCOUNT
        def delta(pot):
            return [pot[t - 1] - pot[t] for t in range(1, len(pot))]
        a = discounted_return(delta([10.0, 5.0, 2.0]), gamma)
        b = discounted_return(delta([10.0, 9.0, 2.0]), gamma)
        self.assertNotAlmostEqual(a, b, places=4,
                                  msg="the deprecated gamma=1 delta form must be path-dependent")

    def test_terminal_weight_makes_penalty_exact(self):
        """A terminal charge divided by gamma^(N-1) contributes exactly -penalty to R."""
        gamma = SHAPING_DISCOUNT
        n = 20
        w = gamma ** (n - 1)
        base = self._shaped([1.0] * (n + 1), gamma)
        penalty = 0.37
        charged = list(base)
        charged[-1] -= penalty / w
        d_base = discounted_return(base, gamma)
        d_charged = discounted_return(charged, gamma)
        self.assertAlmostEqual(d_base - d_charged, penalty, places=9,
                               msg="the constraint cost is not exactly -penalty in the return")

    def test_telescoped_from_potentials_helper_agrees(self):
        gamma = SHAPING_DISCOUNT
        pot = [3.0, 2.0, 1.0]
        self.assertAlmostEqual(telescoped_from_potentials(pot, gamma),
                               discounted_return(self._shaped(pot, gamma), gamma), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class EnvShapingEmitsContractTests(unittest.TestCase):
    """The REAL `V2AutomotiveEnv._telescoping` must emit `J_{t-1} - gamma*J_t`."""

    def _env(self):
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import AutomotiveResourceCluster
        from spec.automotive_training.v2.env import V2AutomotiveEnv
        graphs = list(load_dataset().validation_query())[:2]
        env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="meta_train",
                              slots_per_task=2, base_seed=0, single_dist=True,
                              shaping_discount=SHAPING_DISCOUNT)
        env.set_task({"dist_index": 0, "graph_indices": np.arange(2, dtype=np.int32)})
        env.reset()
        return env

    def test_env_rewards_match_the_contract_form(self):
        env = self._env()
        slot = 0
        n = len(env.base.orders[env.base._graph_index(slot)])
        gamma = SHAPING_DISCOUNT
        actions = [1] * n                       # all-MEC prefix plan
        rewards = env._telescoping(slot, actions)
        # reconstruct the prefix potentials the env used
        pot = []
        prev = env._schedule_slot(slot, [0] * n, validate=False).makespan_s
        graph = env.base.graph_objects[env.base._graph_index(slot)]
        ls = max(float(graph.D_G_s), 1e-12)
        pot.append(prev / ls)
        for k in range(n):
            prefix = [int(a) for a in actions[:k + 1]] + [0] * (n - k - 1)
            cur = env._schedule_slot(slot, prefix, validate=False).makespan_s
            pot.append(cur / ls)
        want = [pot[t - 1] - gamma * pot[t] for t in range(1, len(pot))]
        np.testing.assert_allclose(np.asarray(rewards), np.asarray(want), rtol=0, atol=1e-12,
                                   err_msg="env reward is not J_{t-1} - gamma*J_t")
        # and the discounted return is the affine terminal map
        got = discounted_return(np.asarray(rewards), gamma)
        self.assertAlmostEqual(got, pot[0] - (gamma ** n) * pot[-1], places=9)
