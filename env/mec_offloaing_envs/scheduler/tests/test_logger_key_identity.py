#!/usr/bin/env python3
"""`HumanOutputFormat.writekvs` must not lose keys that share a truncated prefix.

Field defect (MARGO v2, 500-iteration run): `writekvs` keyed its print dict on the
TRUNCATED key name, and since it iterates `sorted(kvs.items())` the lexicographically LAST
key in each 20-character-prefix group silently overwrote the others. Consequences in the
completed run:

* `constraint/total_energy_signed` was replaced by `constraint/total_energy_violation`;
* `constraint/mean_signed_total_energy` (+0.0522, the quantity that drives dual ascent) was
  replaced by `constraint/mean_signed_ue_energy` (-0.4945);

so a finished run appeared to contradict itself — a strongly negative "mean signed" cost
beside `lambda_total_energy` ascending 0 -> 1.3055. The training dual was correct; only the
printed table was wrong. These tests pin the fix and would have caught it.

The module is loaded from source because `utils/__init__.py` imports TensorFlow, which is not
available in the non-TF suite.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load_logger():
    sys.modules.setdefault("joblib", types.ModuleType("joblib"))
    spec = importlib.util.spec_from_file_location("margo_logger", str(ROOT / "utils" / "logger.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TruncationCollisionTests(unittest.TestCase):

    def _render(self, kvs):
        module = _load_logger()
        buffer = io.StringIO()
        module.HumanOutputFormat(buffer).writekvs(kvs)
        return buffer.getvalue()

    def test_keys_sharing_a_truncated_prefix_all_survive(self):
        """The exact MARGO v2 collision: 5 real keys collapse to 3 printed labels before."""
        out = self._render({
            "constraint/mean_signed_total_energy": 0.0522,
            "constraint/mean_signed_ue_energy": -0.4945,
            "constraint/total_energy_signed": 0.0973,
            "constraint/total_energy_violation": 0.0974,
            "constraint/total_violation": 0.0975,
        })
        for key in ("constraint/mean_signed_total_energy", "constraint/mean_signed_ue_energy",
                    "constraint/total_energy_signed", "constraint/total_energy_violation",
                    "constraint/total_violation"):
            self.assertIn(key, out, "key %r was lost from the printed table" % key)
        # and each key keeps ITS OWN value
        self.assertIn("0.0522", out)
        self.assertIn("-0.494", out)
        self.assertIn("0.0973", out)
        self.assertIn("0.0974", out)
        self.assertIn("0.0975", out)

    def test_long_keys_do_not_overwrite_each_other(self):
        out = self._render({
            "constraint/lambda_total_energy": 1.31,
            "constraint/lambda_ue_energy": 0.0,
        })
        self.assertIn("constraint/lambda_total_energy", out)
        self.assertIn("constraint/lambda_ue_energy", out)
        self.assertIn("1.31", out)

    def test_values_are_still_compact(self):
        """Only the KEY identity changed; long VALUES are still truncated for display."""
        out = self._render({"some/long/value/key_here": "x" * 200})
        self.assertIn("some/long/value/key_here", out)
        self.assertNotIn("x" * 100, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
