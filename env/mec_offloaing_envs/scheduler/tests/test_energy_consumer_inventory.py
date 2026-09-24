#!/usr/bin/env python3
"""E2.1: machine inventory of direct boundary reads.

After the consumer migration, no production module may read a boundary property
(`total_{ue,helper,mobile,requester,system}_joules`) directly: values come from
`energy_scope.energy_scalar(..., scope=...)`. The three surviving sites are the
energy IMPLEMENTATION and two independent SERIALIZATION/oracle paths, listed
here so a future regression fails loudly instead of drifting.
"""

from __future__ import annotations

import re
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

# attribute access only: a bare string "total_mobile_joules" is documentation
BOUNDARY_READ = re.compile(
    r"\.total_(?:ue|helper|mobile|requester|system)_joules(?:_optional)?\b"
)

ALLOWLIST = {
    "env/mec_offloaing_envs/scheduler/model.py": (
        "boundary definitions and EnergyBreakdown.as_dict serialization"
    ),
    "spec/toy_oracles/oracle_checker.py": (
        "independent legacy oracle serialization/comparison"
    ),
    "spec/energy_pareto_audit.py": (
        "pareto audit serialization of all three boundaries"
    ),
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "tests"}


def _scan():
    hits: dict[str, list[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        parts = set(path.relative_to(ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if BOUNDARY_READ.search(line):
                hits.setdefault(rel, []).append("%d: %s" % (lineno, line.strip()))
    return hits


class TestBoundaryReadInventory(unittest.TestCase):
    def test_only_allowlisted_direct_reads_remain(self):
        hits = _scan()
        offenders = {rel: lines for rel, lines in hits.items() if rel not in ALLOWLIST}
        self.assertEqual(
            offenders,
            {},
            "direct boundary reads must go through energy_scalar(scope=...): %r"
            % offenders,
        )

    def test_allowlist_is_not_stale(self):
        hits = _scan()
        for rel, reason in ALLOWLIST.items():
            self.assertIn(rel, hits, "allowlisted file no longer reads a boundary (%s)" % reason)

    def test_no_migrated_production_module_reads_a_boundary(self):
        hits = _scan()
        for rel in (
            "env/mec_offloaing_envs/scheduler/reward.py",
            "env/mec_offloaing_envs/scheduler/constraints.py",
            "env/mec_offloaing_envs/scheduler/objective.py",
            "env/mec_offloaing_envs/offloading_env.py",
            "meta_evaluator.py",
        ):
            self.assertNotIn(rel, hits, "%s still reads a boundary directly" % rel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
