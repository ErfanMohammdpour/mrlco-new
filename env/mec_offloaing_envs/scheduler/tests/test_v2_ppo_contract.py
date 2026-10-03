#!/usr/bin/env python3
"""PPO / MRLCO audit that is EXECUTABLE without TensorFlow, plus source-level guards.

The frozen PPO and the MRLCO outer update run in a TensorFlow 1.15 graph, so their execution
is NOT RUN here. What CAN be verified on CPU is:

1. the numerical CONTRACT the graph implements — a numpy reference of the categorical
   log-likelihood and of `likelihood_ratio_sym`, proving the ratio is `p_new(a)/p_old(a)` at
   the STORED action and equals 1 for unchanged logits (and differs if actions were resampled,
   which is the defect this guards);
2. the RESET CONTRACT — asserted directly against the source text, so a future edit that
   removes the per-task optimizer reset, the per-iteration core sync, or swaps the first-order
   pseudogradient for a claimed second-order update fails this test.

Both are labelled: `executed` for (1), `reviewed (static)` for (2). Neither is a substitute
for running the TF graph.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MRLCO = ROOT / "meta_algos" / "MRLCO.py"
TRAINER = ROOT / "meta_trainer.py"
CATEGORICAL = ROOT / "policies" / "distributions" / "categorical_pd.py"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_likelihood(logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """numpy reference of `categorical_pd.log_likelihood_sym` (log p(a) under `logits`)."""
    return np.log(np.clip(_softmax(logits), 1e-30, None))[
        np.arange(actions.shape[0]), actions]


def likelihood_ratio(old_logits: np.ndarray, new_logits: np.ndarray,
                     actions: np.ndarray) -> np.ndarray:
    """numpy reference of `categorical_pd.likelihood_ratio_sym` on STORED actions."""
    return np.exp(log_likelihood(new_logits, actions) - log_likelihood(old_logits, actions))


class TestLikelihoodRatioContract(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.actions = rng.randint(0, 3, size=64)
        self.old = rng.randn(64, 3).astype(np.float64)
        self.scale = rng.uniform(0.5, 2.0, size=(64, 1))

    def test_unchanged_logits_give_a_ratio_of_exactly_one(self):
        ratio = likelihood_ratio(self.old, self.old.copy(), self.actions)
        np.testing.assert_allclose(ratio, 1.0, rtol=0, atol=1e-12)

    def test_ratio_equals_new_over_old_probability_at_the_stored_action(self):
        new = self.old * self.scale
        ratio = likelihood_ratio(self.old, new, self.actions)
        p_old = _softmax(self.old)[np.arange(64), self.actions]
        p_new = _softmax(new)[np.arange(64), self.actions]
        np.testing.assert_allclose(ratio, p_new / p_old, rtol=1e-12, atol=1e-15)

    def test_resampling_actions_would_change_the_ratio(self):
        """The audited defect this guards: computing the ratio at RE-SAMPLED actions."""
        new = self.old * self.scale
        stored = likelihood_ratio(self.old, new, self.actions)
        resampled = np.argmax(new, axis=1)          # a greedy re-decode, NOT the stored action
        wrong = likelihood_ratio(self.old, new, resampled)
        self.assertGreater(float(np.max(np.abs(stored - wrong))), 1e-6,
                           "the fixture must distinguish stored from resampled actions")

    def test_ratio_is_per_token_not_per_sequence_collapsed(self):
        old = np.zeros((5, 3))
        new = np.zeros((5, 3))
        new[2] = [0.0, 5.0, 0.0]                     # only token 2 changed
        actions = np.zeros(5, dtype=int)
        ratio = likelihood_ratio(old, new, actions)
        np.testing.assert_allclose(ratio[:2], 1.0, atol=1e-12)
        np.testing.assert_allclose(ratio[3:], 1.0, atol=1e-12)
        self.assertLess(ratio[2], 1.0)               # the stored action became less likely


class TestSourceLevelResetContract(unittest.TestCase):
    """Static guards: `reviewed`, not executed. Each asserts a property the TF run would show."""

    def test_ratio_is_built_from_the_stored_actions_and_old_logits(self):
        source = MRLCO.read_text()
        self.assertIn(
            "likelihood_ratio_sym(\n                    self.actions[i], "
            "self.old_logits[i], self.new_logits[i]",
            source,
            "the ratio must be computed from the STORED actions, the stored old logits and "
            "the current new logits")
        self.assertIn("self.actions.append(self.policy.meta_policies[i].decoder_targets)",
                      source,
                      "`actions` must be the decoder targets that were actually executed, "
                      "not a fresh sample")
        self.assertIn("tf.exp(logli_new - logli_old)",
                      CATEGORICAL.read_text(),
                      "the ratio must be exp(log p_new - log p_old) at the same action")

    def test_each_task_resets_its_inner_optimizer_before_adapting(self):
        source = MRLCO.read_text()
        body = source.split("def UpdatePPOTargetPerTask")[1]
        self.assertIn("self.reset_inner_optimizer(task_id)", body.split("def ")[0],
                      "the per-task inner optimizer slot state must be reset before the "
                      "task's inner PPO, otherwise Adam moments leak across tasks")
        self.assertIn("tf.compat.v1.variables_initializer(slot_vars)", source)

    def test_the_trainer_syncs_from_the_core_at_the_start_of_every_iteration(self):
        source = TRAINER.read_text()
        loop = source.split("for itr in range(self.start_itr, self.n_itr):")[1]
        head = loop[:1200]
        self.assertIn("sync_task_policies_from_core()", head,
                      "task policies must be re-synced from the core before the meta-batch is "
                      "sampled, so no adaptation survives into the next iteration")
        self.assertIn("UpdateMetaPolicy()", loop)

    def test_the_outer_update_is_a_declared_first_order_pseudogradient(self):
        source = MRLCO.read_text()
        self.assertIn("mean_pseudogradient(theta0, adapted, self.inner_lr, "
                      "self.num_inner_grad_steps)", source)
        for claim in ("second_order", "second-order", "hessian"):
            self.assertIsNone(re.search(claim, source),
                              "the outer update must not be described as second-order MAML")
        # the outer step is built from OPTIMIZER slots over a pseudogradient placeholder, not
        # from a differentiable gradient-of-gradient graph (which is what would make it MAML)
        self.assertNotIn("tf.gradients", source,
                         "a second-order outer update would differentiate the inner gradients; "
                         "the frozen algorithm does not and must not be described as doing so")
        self.assertIn("self.outer_optimizer.apply_gradients(outer_grads_and_var)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
