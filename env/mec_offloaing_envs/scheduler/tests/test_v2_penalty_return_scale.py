#!/usr/bin/env python3
"""F13: the discounted return must be an exact affine function of the APPROVED cost.

Approved terminal cost: constrained latency `J_N + P` (`spec/objective_contract.py`: the
optimised criterion must rank checkpoints the same as the terminal plan objective).

The earlier `gamma_(N-1)` form made the penalty's discounted contribution exactly `P`, giving
`R = J_0 - gamma^N*J_N - P`, which is the DIFFERENT weighted objective `J_N + P/gamma^N`: the
penalty carried `1/gamma^N` (~1.2226 at gamma=0.99, N=20) times the weight of `J_N`.

Contract `gamma_N_v2` sets the terminal charge so the discounted contribution is `gamma^N * P`:
    R = J_0 - gamma^N * (J_N + P)
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.constraints import ConstraintCosts  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    PENALTY_RETURN_SCALE, V2ConstraintManager,
)
from spec.objective_contract import SHAPING_DISCOUNT, discounted_return  # noqa: E402

N = 20


class _Stub:
    def __init__(self, penalty):
        self.controller = type("C", (), {"penalty": lambda _s, _c: float(penalty)})()
        self.applied_penalty = 0.0


def _costs(signed=1.0):
    return ConstraintCosts(names=("total_energy",), raw=(signed,), budgets=(1.0,), scales=(1.0,),
                           signed=(signed,), violations=(max(0.0, signed),))


def _shaped_rewards(potentials, gamma=SHAPING_DISCOUNT):
    return [potentials[t - 1] - gamma * potentials[t] for t in range(1, len(potentials))]


class PenaltyReturnScaleTests(unittest.TestCase):
    def _return(self, J0, JN, P, gamma=SHAPING_DISCOUNT, n=N):
        pot = [J0] + [0.0] * (n - 1) + [JN]
        r = _shaped_rewards(pot, gamma)
        self.assertEqual(len(r), n)
        stub = _Stub(P)
        V2ConstraintManager.apply_penalty(stub, r, _costs(1.0),
                                          terminal_weight=gamma ** (n - 1),
                                          terminal_scale=gamma ** n)
        return discounted_return(r, gamma)

    def test_contract_id_is_versioned(self):
        self.assertEqual(PENALTY_RETURN_SCALE, "gamma_N_v2")

    def test_return_is_exactly_j0_minus_gammaN_jn_plus_p(self):
        gamma = SHAPING_DISCOUNT
        R = self._return(1.0, 0.5, 0.37, gamma)
        want = 1.0 - (gamma ** N) * (0.5 + 0.37)
        self.assertAlmostEqual(R / want, 1.0, places=12)

    def test_ranking_matches_the_approved_cost(self):
        gamma = SHAPING_DISCOUNT
        rows = [(0.4, 0.1), (0.5, 0.37), (0.2, 0.9), (0.9, 0.0)]
        by_return = sorted(rows, key=lambda ab: self._return(1.0, ab[0], ab[1], gamma))
        by_cost = sorted(rows, key=lambda ab: -(ab[0] + ab[1]))
        self.assertEqual([r[0] for r in by_return], [r[0] for r in by_cost],
                         "ranking by R must equal ranking by -(J_N + P)")

    def test_zero_penalty_reduces_to_the_pure_latency_form(self):
        gamma = SHAPING_DISCOUNT
        R = self._return(1.0, 0.5, 0.0, gamma)
        self.assertAlmostEqual(R / (1.0 - (gamma ** N) * 0.5), 1.0, places=12)

    def test_path_independence_with_a_penalty(self):
        gamma = SHAPING_DISCOUNT
        vals = []
        for mid in (0.05, 0.4, 0.8):
            pot = [1.0] + [mid] * (N - 1) + [0.5]
            r = _shaped_rewards(pot, gamma)
            stub = _Stub(0.37)
            V2ConstraintManager.apply_penalty(stub, r, _costs(1.0),
                                              terminal_weight=gamma ** (N - 1),
                                              terminal_scale=gamma ** N)
            vals.append(discounted_return(r, gamma))
        for v in vals[1:]:
            self.assertAlmostEqual(v, vals[0], places=12,
                                   msg="return depends on the intermediate path")

    def test_old_scale_is_measurably_different(self):
        """The superseded form carried 1/gamma^N times the weight of J_N."""
        gamma = SHAPING_DISCOUNT
        r_old = _shaped_rewards([1.0] + [0.0] * (N - 1) + [0.5])
        stub = _Stub(0.37)
        V2ConstraintManager.apply_penalty(stub, r_old, _costs(1.0),
                                          terminal_weight=gamma ** (N - 1),
                                          terminal_scale=1.0)     # penalty contributes P
        R_old = discounted_return(r_old, gamma)
        # R_old = J_0 - gamma^N*J_N - P  (penalty contributes exactly P, hence 1/gamma^N times
        # the weight of J_N relative to the approved form)
        self.assertAlmostEqual(R_old, 1.0 - (gamma ** N) * 0.5 - 0.37, places=12)
        # and the approved form is what this module now produces
        R_new = self._return(1.0, 0.5, 0.37, gamma)
        self.assertAlmostEqual(R_new, 1.0 - (gamma ** N) * (0.5 + 0.37), places=12)
        self.assertNotAlmostEqual(R_old, R_new, places=6,
                                  msg="the superseded and approved scales must differ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
