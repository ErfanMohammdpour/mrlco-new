#!/usr/bin/env python3
"""F05: the HIDDEN MC realization must not reach decision-time observations/admission.

Audited at HEAD 9e80a16 (BROKEN):

* `V2AutomotiveEnv.reference_ranges_for_slot` built the three pure-location reference worlds
  with the slot's realized MC (`base._slot_mc[slot]`). `adapters.dag_spec_from_graph` turns
  that MC into the survivor set + `effective_equiv`, so `refs`, the six `est_wait_*` /
  `est_energy_*` observation columns and the total-energy budget anchor moved with the hidden
  realization. Executed proof: swapping ONLY `base._slot_mc[0]` changed the packed
  observation in exactly those 6 columns.
* `contact_slack` was computed from the MC-filtered `world.foreground`, so the slack column was
  realization-dependent too (1.0 vs 0.937546 with a short contact window, masked at the
  default 2.0 s mean).

The requirement is a constraint on OBSERVATIONS / REFERENCES / ADMISSION, never on execution:
the plan the policy finally submits is still scheduled against the realized MC (the world is
built from `_slot_mc` at `reset()` and `_schedule_slot` still executes it).
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.context_fields import V2_CONTEXT_FIELDS  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402

SLOTS = 2
TOKENS = 20

#: the six decision-time columns the audit found realization-dependent
REFERENCE_COLUMNS = (
    "est_wait_mec_s", "est_wait_ue_s", "est_wait_helper_s",
    "est_energy_ue_j", "est_energy_mec_j", "est_energy_helper_j",
)


def _env(seed=2024, **kw):
    graphs = list(load_dataset().validation_query()[:SLOTS])
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=SLOTS, base_seed=seed, single_dist=True,
                          constraints_enabled=True, reliability=True,
                          budget_fractions={"total_energy_mec": 0.5, "ue_energy": 1.0},
                          scheduler_config_sha256="d" * 64, **kw)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(SLOTS, dtype=np.int32)})
    return env


def _different_realization(mc, factor):
    """A DIFFERENT realization of the SAME graph: same survivors, scaled effective work.

    `adapters.dag_spec_from_graph` reads exactly `executed_task_ids` (survivors) and
    `execution[*]["effective_equiv"]` (compute bytes), so scaling the effective work is a
    real, second realization of the same graph.
    """
    alt = copy.deepcopy(mc)
    for record in alt["execution"].values():
        record["effective_equiv"] = float(record["effective_equiv"]) * float(factor)
    return alt


def _realized_bytes(world):
    foreground = world.foreground
    return (max([float(t.external_input_bytes) for t in foreground.tasks] or [0.0]),
            float(sum(t.compute_bytes for t in foreground.tasks)),
            max([float(t.output_bytes) for t in foreground.tasks] or [0.0]))


class TestHiddenRealizationIsNotObservable(unittest.TestCase):
    def _snapshot(self, env, slot=0):
        """Every DECISION-TIME quantity a policy or an admission rule can read."""
        env.last_v2_context = env._contexts()
        ctx = np.asarray(env.v2_context(), dtype=np.float32)
        obs = np.asarray(env._packed_observation(env.last_v2_context), dtype=np.float32)
        refs = env.reference_ranges_for_slot(slot)
        n_tokens = len(env.base.orders[env.base._graph_index(slot)])
        env._schedule_slot(slot, [1] * n_tokens, validate=False)
        return {"ctx": ctx, "obs": obs, "refs": refs,
                "evidence": dict(env.last_admission_evidence.get(slot, {}))}

    def _clear_reference_caches(self, env):
        """Exactly the caches `reset()` clears, so only `_slot_mc` differs."""
        env._reference_ranges = {}
        env._reference_results = {}
        env._reference_ledgers = {}

    def test_swapping_only_the_hidden_realization_leaves_decision_time_bit_identical(self):
        env = _env(seed=2024)
        env.reset()
        slot = 0
        mc_before = env.base._slot_mc[slot]
        alt = _different_realization(mc_before, 3.0)
        # the swap IS a different realization of the same graph...
        self.assertEqual([int(t) for t in mc_before["executed_task_ids"]],
                         [int(t) for t in alt["executed_task_ids"]])
        self.assertTrue(any(float(mc_before["execution"][t]["effective_equiv"])
                            != float(alt["execution"][t]["effective_equiv"])
                            for t in alt["execution"]))
        before = self._snapshot(env, slot)
        world_fingerprint = env.world_for_slot(slot).fingerprint_sha256()
        # ... and ONLY the hidden realization is swapped: the world is NOT rebuilt
        env.base._slot_mc[slot] = alt
        self._clear_reference_caches(env)
        after = self._snapshot(env, slot)

        self.assertEqual(env.world_for_slot(slot).fingerprint_sha256(), world_fingerprint,
                         "the swap must not change the world the decision is taken in")
        # -- v2_context() and the packed policy observation, BIT-identical ----------
        np.testing.assert_array_equal(before["ctx"], after["ctx"])
        np.testing.assert_array_equal(before["obs"], after["obs"])
        self.assertEqual(before["ctx"].tobytes(), after["ctx"].tobytes())
        self.assertEqual(before["obs"].tobytes(), after["obs"].tobytes())
        # -- the six reference-plan columns specifically ---------------------------
        columns = [list(V2_CONTEXT_FIELDS).index(name) for name in REFERENCE_COLUMNS]
        np.testing.assert_array_equal(before["ctx"][slot][columns],
                                      after["ctx"][slot][columns])
        self.assertTrue(np.all(before["ctx"][slot][columns] != 0.0),
                        "the reference columns must be measured, not zero")
        # -- the admission/reference payload ---------------------------------------
        self.assertEqual(before["refs"], after["refs"])
        self.assertEqual(before["evidence"], after["evidence"],
                         "the admission evidence changed with the hidden realization")

    def test_contact_slack_is_taken_from_the_canonical_graph_not_the_realization(self):
        # short contact window: the audited leak is NOT masked (at the 2.0 s default the
        # slack saturates at 1.0 and the leak is invisible in this column)
        env = _env(seed=31, helper_contact_mean_s=0.05, helper_contact_cv=0.0)
        env.reset()
        slot = 0
        column = list(V2_CONTEXT_FIELDS).index("contact_slack")
        mc_before = env.base._slot_mc[slot]
        env.last_v2_context = env._contexts()
        before = float(np.asarray(env.v2_context(), dtype=np.float32)[slot][column])
        bytes_before = _realized_bytes(env.world_for_slot(slot))
        # a genuinely different realization, with the world REBUILT from it (this is the
        # path that used to feed `world.foreground` into `contact_slack`)
        env.base._slot_mc[slot] = _different_realization(mc_before, 20.0)
        env._worlds[slot] = env._build_slot_world(slot)
        self._clear_reference_caches(env)
        env.last_v2_context = env._contexts()
        after = float(np.asarray(env.v2_context(), dtype=np.float32)[slot][column])
        bytes_after = _realized_bytes(env.world_for_slot(slot))
        self.assertNotEqual(bytes_before, bytes_after,
                            "the two realizations must differ in the MC-filtered foreground")
        self.assertEqual(before, after,
                         "contact_slack moved with the hidden realization (it must be a "
                         "property of the canonical graph)")

    def test_admission_budgets_and_scales_do_not_move_with_the_realization(self):
        env = _env(seed=77)
        env.reset()
        plan = np.ones((SLOTS, TOKENS), dtype=int)
        _o, _r, _d, info = env.step(plan)
        before = [dict(record["constraints"]) for record in info[2]]
        refs_before = env.reference_ranges_for_slot(0)
        env.base._slot_mc[0] = _different_realization(env.base._slot_mc[0], 3.0)
        self._clear_reference_caches(env)
        _o, _r, _d, info = env.step(plan)
        after = [dict(record["constraints"]) for record in info[2]]
        for a, b in zip(before, after):
            self.assertEqual(a["names"], b["names"])
            for name in a["names"]:
                for suffix in ("_budget", "_scale"):
                    self.assertEqual(a[name + suffix], b[name + suffix],
                                     "%s%s moved with the hidden realization"
                                     % (name, suffix))
        self.assertEqual(env.reference_ranges_for_slot(0), refs_before)

    def test_execution_still_uses_the_realized_mc(self):
        """The fix constrains observations, not execution (explicit non-regression)."""
        env = _env(seed=5)
        env.reset()
        slot = 0
        plan = [1] * len(env.base.orders[env.base._graph_index(slot)])
        baseline = env._schedule_slot(slot, plan, validate=False)
        env.base._slot_mc[slot] = _different_realization(env.base._slot_mc[slot], 3.0)
        env._worlds[slot] = env._build_slot_world(slot)
        moved = env._schedule_slot(slot, plan, validate=False)
        self.assertNotEqual(float(baseline.makespan_s), float(moved.makespan_s),
                            "execution must still be scheduled against the realized MC")


    def test_energy_budget_ratio_mirrors_the_constraint_anchor(self):
        """The observed budget ratio must use the SAME anchor knob as the constraint.

        The production config anchors the total-energy budget on the pure all-MEC plan
        (`total_energy_frac_of_all_mec`). The pre-fix observation code read only the
        `..._of_all_ue` knob, so an active MEC-anchored constraint was observed as a budget
        ratio of exactly 0.0.
        """
        env = _env(seed=3)
        env.reset()
        column = list(V2_CONTEXT_FIELDS).index("energy_budget_ratio")
        ratio = float(np.asarray(env.v2_context(), dtype=np.float32)[0][column])
        self.assertGreater(ratio, 0.0)
        refs = env.reference_ranges_for_slot(0)
        energies = {name: float(ledger.system_joules)
                    for name, ledger in env._reference_ledgers[0].items()}
        expected = (0.5 * float(refs.E_mec)) / max(max(energies.values()), 1e-12)
        self.assertAlmostEqual(ratio, expected, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
