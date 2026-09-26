#!/usr/bin/env python3
"""Doc-consistency guards: the architecture docs must not silently contradict code.

Two claims were verified stale during the v0.3 audit and are pinned here:
  * obs dims: the docs described obs v1 only (FEATURE_DIM=11/PACKED_DIM=50) while
    the pilot/eval path runs obs v3 (FEATURE_DIM=31/PACKED_DIM=70);
  * the method version: `MARGO-METHOD-v0.2-cavia` is superseded by ADR-007, which
    removes CAVIA-on-z from the method.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})

ADR = "decisions/ADR-007-adaptation-engine.md"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


class TestObsDims(unittest.TestCase):
    def test_encoder_defines_three_versions_with_v3_as_the_pilot_contract(self):
        source = (ROOT / "env/mec_offloaing_envs/scheduler/encoder_obs.py").read_text()
        self.assertIn("v1 (default): FEATURE_DIM=11, PACKED_DIM=50", source)
        self.assertIn("v2: FEATURE_DIM=15, PACKED_DIM=54", source)
        self.assertIn("v3: FEATURE_DIM=31, PACKED_DIM=70", source)
        self.assertIn("PACKED_DIM = FEATURE_DIM + 2 * MAX_NEIGH + 1", source)

    def test_architecture_doc_does_not_claim_50_is_the_only_dim(self):
        doc = _read("spec/CURRENT_ARCHITECTURE.md")
        self.assertIn("PACKED_DIM  = 11 + 2*19 + 1 = 50", doc)
        # the v1 numbers must be labelled, and v3 must be stated for the pilot path
        self.assertIn("v3 (قرارداد فعلی", doc)
        self.assertIn("PACKED_DIM=70", doc)


class TestMethodVersion(unittest.TestCase):
    def test_adr_007_supersedes_the_cavia_method(self):
        adr = _read(f"spec/{ADR}")
        self.assertIn("Supersedes: `MARGO-METHOD-v0.2-cavia`", adr)
        self.assertIn("removed from the method", adr)

    def test_docs_carrying_the_dead_method_version_point_at_adr_007(self):
        for relative in (
            "spec/FINAL_README.md",
            "spec/FINAL_IMPLEMENTATION_PLAN.md",
            "spec/FINAL_DATAFLOW.md",
            "spec/CURRENT_ARCHITECTURE.md",
        ):
            with self.subTest(doc=relative):
                self.assertIn(ADR, _read(relative))


if __name__ == "__main__":
    unittest.main(verbosity=2)
