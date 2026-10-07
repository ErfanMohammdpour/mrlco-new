#!/usr/bin/env python3
"""F21: the NumPy categorical log-likelihood must be a stable, correct log p(x).

The audited implementation returned `[-inf, -inf]` for zero logits and actions [0, 1] instead of
`-log(3)`: it normalised with the Python builtin `sum` (wrong axis when batched), took
`log(one_hot * p)` (log(0) for the unselected classes), and built `final_shape` from
`list.append` (which returns None). These tests require TensorFlow because the module imports it
for the symbolic path; they are TF-gated like the other TF contract tests.

    python -m pytest \
      env/mec_offloaing_envs/scheduler/tests/test_categorical_log_likelihood.py -q
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:                                                    # TF is a hard dependency of the module
    import tensorflow as tf                             # noqa: F401
    HAVE_TF = True
except Exception:                                       # pragma: no cover
    HAVE_TF = False

if HAVE_TF:
    from policies.distributions.categorical_pd import CategoricalPd  # noqa: E402


@unittest.skipUnless(HAVE_TF, "TensorFlow is required (categorical_pd imports tf)")
class NumPyCategoricalLogLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.pd = CategoricalPd(self.dim)

    def test_zero_logits_are_uniform(self):
        out = self.pd.log_likelihood(np.array([0, 1]), np.zeros((2, self.dim)))
        np.testing.assert_allclose(np.asarray(out, dtype=float),
                                   np.full(2, -np.log(self.dim)), rtol=0, atol=1e-9)

    def test_scalar_input(self):
        out = self.pd.log_likelihood(0, np.zeros(self.dim))
        np.testing.assert_allclose(np.asarray(out, dtype=float), -np.log(self.dim),
                                   rtol=0, atol=1e-9)

    def test_batched_logits_normalise_per_row(self):
        logits = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = np.asarray(self.pd.log_likelihood(np.array([0, 0]), logits), dtype=float)
        # row 0: uniform -> -log 3 ; row 1: action 0 has the highest logit
        self.assertAlmostEqual(out[0], -np.log(3), places=9)
        lse = np.log(np.sum(np.exp([1.0, 0.0, 0.0])))
        self.assertAlmostEqual(out[1], 1.0 - lse, places=9)

    def test_time_indexed_input(self):
        rng = np.random.RandomState(0)
        logits = rng.randn(4, 5, self.dim)
        xs = rng.randint(0, self.dim, size=(4, 5))
        out = np.asarray(self.pd.log_likelihood(xs, logits), dtype=float)
        self.assertEqual(out.shape, (4, 5))
        # closed form reference
        ref = np.take_along_axis(
            logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True)),
            xs[..., None], axis=-1)[..., 0]
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-9)

    def test_large_logits_do_not_overflow(self):
        logits = np.array([[1000.0, 0.0, -1000.0]])
        out = float(np.asarray(self.pd.log_likelihood(np.array([0]), logits), dtype=float)[0])
        self.assertTrue(np.isfinite(out))
        self.assertAlmostEqual(out, 0.0, places=6)

    def test_matches_tensorflow_symbolic_path(self):
        import tensorflow as tf
        rng = np.random.RandomState(1)
        logits = rng.randn(6, self.dim).astype(np.float32)
        xs = rng.randint(0, self.dim, size=(6,)).astype(np.int32)
        with tf.Session() as sess:
            sym = sess.run(self.pd.log_likelihood_sym(tf.constant(xs), tf.constant(logits)))
        num = np.asarray(self.pd.log_likelihood(xs, logits), dtype=float)
        np.testing.assert_allclose(num, np.asarray(sym, dtype=float), rtol=1e-5, atol=1e-5)

    def test_bad_action_index_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pd.log_likelihood(np.array([5]), np.zeros((1, self.dim)))

    def test_wrong_action_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pd.log_likelihood(np.array([0]), np.zeros((2, 4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
