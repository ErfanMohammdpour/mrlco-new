#!/usr/bin/env python3
"""F29 regression: the checkpoint SELECTION RULE is wired, labelled, and truthfully reported.

Covers:
  * the frozen lexicographic rule order (hard feasibility, MC violations, HIGH tardiness,
    MEDIUM tardiness, latency) and the "a violating fixture is never selected over a feasible
    one" invariant;
  * `select_checkpoint` being CALLED on the v2 validation/selection path (it was the only rule
    that labels a no-feasible-checkpoint case and no run path ever called it);
  * the checkpoint sidecar recording the ACTUAL applied reward mode, constraint names and
    lambda vector, and WHICH selection criterion chose the checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_primary import (  # noqa: E402
    CHECKPOINT_RULE, CHECKPOINT_RULE_ID, lexicographic_key,
)
from spec.automotive_training.automotive_trainer import AutomotiveTrainerMixin  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    V2ConstraintError, select_checkpoint,
)


def _metrics(**over):
    base = {
        "graph_hard_violation_rate": 0.0,
        "mc_policy_violation_count": 0.0,
        "mean_high_tardiness_s": 0.0,
        "mean_medium_tardiness_s": 0.0,
        "query_mean_latency_seconds": 1.0,
    }
    base.update(over)
    return base


class TestLexicographicRule(unittest.TestCase):
    def test_rule_is_the_frozen_six_step_order(self):
        self.assertEqual(len(CHECKPOINT_RULE), 6)
        self.assertEqual(CHECKPOINT_RULE_ID, "automotive_lexicographic_v1")

    def test_step_order_hard_mc_high_medium_latency(self):
        reference = _metrics()
        # 1. hard feasibility dominates everything, even a much better latency
        hard_bad = _metrics(graph_hard_violation_rate=0.1, query_mean_latency_seconds=1e-6)
        self.assertLess(lexicographic_key(reference), lexicographic_key(hard_bad))
        # 2. MC policy violations
        mc_bad = _metrics(mc_policy_violation_count=1.0, query_mean_latency_seconds=1e-6)
        self.assertLess(lexicographic_key(reference), lexicographic_key(mc_bad))
        # 3. HIGH tardiness
        high_bad = _metrics(mean_high_tardiness_s=0.5, query_mean_latency_seconds=1e-6)
        self.assertLess(lexicographic_key(reference), lexicographic_key(high_bad))
        # 4. MEDIUM tardiness
        med_bad = _metrics(mean_medium_tardiness_s=0.5, query_mean_latency_seconds=1e-6)
        self.assertLess(lexicographic_key(reference), lexicographic_key(med_bad))
        # 5. latency last
        slow = _metrics(query_mean_latency_seconds=2.0)
        self.assertLess(lexicographic_key(reference), lexicographic_key(slow))
        # and HIGH tardiness dominates MEDIUM tardiness regardless of latency
        low_high_high_medium = _metrics(mean_medium_tardiness_s=5.0,
                                        query_mean_latency_seconds=10.0)
        high_high = _metrics(mean_high_tardiness_s=0.1, query_mean_latency_seconds=0.001)
        self.assertLess(lexicographic_key(low_high_high_medium),
                        lexicographic_key(high_high),
                        "step 3 (HIGH tardiness) is compared before step 4 (MEDIUM)")

    def test_a_violating_fixture_is_never_selected_over_a_feasible_one(self):
        feasible = _metrics(query_mean_latency_seconds=10.0)
        violating = _metrics(graph_hard_violation_rate=0.2, query_mean_latency_seconds=0.001)
        self.assertLess(lexicographic_key(feasible), lexicographic_key(violating),
                        "a hard-violating checkpoint can never win on latency alone")
        records = [
            {"objective": 0.001, "feasible": False, "total_violation": 0.2, "itr": 1},
            {"objective": 10.0, "feasible": True, "total_violation": 0.0, "itr": 2},
        ]
        best = select_checkpoint(records)
        self.assertTrue(best["feasible"])
        self.assertEqual(best["selected_index"], 1)

    def test_no_feasible_checkpoint_is_labelled_not_silently_kept(self):
        records = [{"objective": 0.1, "feasible": False, "total_violation": 5.0, "itr": 3},
                   {"objective": 0.9, "feasible": False, "total_violation": 1.0, "itr": 4}]
        best = select_checkpoint(records)
        self.assertFalse(best["feasible"])
        self.assertIn("selection_note", best)
        self.assertIn("infeasible", best["selection_note"])
        with self.assertRaises(V2ConstraintError):
            select_checkpoint([{"objective": 0.1, "feasible": True}])


class _CorePolicy(object):
    def __init__(self):
        self.saved = []

    def save_variables(self, save_path=None):
        self.saved.append(save_path)


class _Policy(object):
    def __init__(self):
        self.core_policy = _CorePolicy()


class _Algo(object):
    def sync_task_policies_from_core(self):
        return None


class _Manager(object):
    names = ("C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS")

    def lambdas_by_name(self):
        return {"C_GRAPH_HARD_DEADLINE": 0.25, "C_HI_TASK_TARDINESS": 0.75}


class _Evaluator(object):
    sync_count = 1
    last_sync_max_abs_diff = 0.0
    core_unchanged_after_adaptation = True
    adaptation_ppo_constructions = 1

    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate_all(self, k_steps, sess=None):
        out = dict(self.metrics)
        out["replicates"] = 5
        out["applied_r_select"] = 5
        out["applied_s_select"] = 5
        return out


class _FakeTrainer(AutomotiveTrainerMixin):
    def __init__(self, metrics):
        self.held_out_evaluator = _Evaluator(metrics)
        self.algo = _Algo()
        self.policy = _Policy()
        self.best_selection_key = None
        self.best_val_objective = None
        self.auto_protocol_id = "automotive_eval_protocol_v1"
        self.auto_protocol_sha = "sha"
        self.auto_decoding = "deterministic"
        self.auto_last_validation = {}
        self.auto_lambda_broadcast_targets = 2
        self.auto_penalty_steps = 1
        self.auto_v2_constraint_manager = _Manager()
        self.auto_reward_mode = "latency_only"
        self.auto_energy_constraint = "configured:total_energy"
        self.auto_selection_criterion = "automotive_lexicographic_v1"
        self.auto_oracle_selection_criterion = "best_of_panel_latency"
        self.auto_constraint_budget_fractions = {"total_energy_mec": 0.5}
        self.auto_checkpoint_records = []
        self.auto_correctness = {}
        self.dir = tempfile.mkdtemp()

    def _ckpt_path(self, name):
        return os.path.join(self.dir, name)


class TestSelectionIsWiredAndTruthfullyReported(unittest.TestCase):
    def _run(self, metrics):
        from spec.automotive_training.automotive_trainer import AutomotiveTrainerMixin

        fake_logger = types.SimpleNamespace(logkv=lambda *a, **k: None,
                                            logkv_mean=lambda *a, **k: None)
        fake_utils = types.ModuleType("utils")
        fake_utils.logger = fake_logger
        trainer = _FakeTrainer(metrics)
        with mock.patch.dict(sys.modules, {"utils": fake_utils}):
            k0, k3 = AutomotiveTrainerMixin._run_validation(trainer, 7)
        sidecar = json.loads(Path(trainer._ckpt_path(
            "meta_model_best_val.metric.json")).read_text())
        return trainer, k0, k3, sidecar

    def test_select_checkpoint_is_called_and_the_criterion_is_declared(self):
        trainer, _k0, _k3, sidecar = self._run(_metrics())
        self.assertTrue(hasattr(trainer, "auto_checkpoint_selection"))
        self.assertTrue(trainer.auto_checkpoint_selection["feasible"])
        self.assertEqual(len(trainer.auto_checkpoint_records), 1)
        self.assertEqual(sidecar["selection_criterion"], "automotive_lexicographic_v1")
        self.assertIn("objective_contract_v1", sidecar["selection_contract_alternatives"])
        self.assertEqual(sidecar["oracle_selection_criterion"], "best_of_panel_latency")
        self.assertEqual(sidecar["rule_steps"], list(CHECKPOINT_RULE))
        self.assertTrue(sidecar["checkpoint_feasible"])
        self.assertTrue(sidecar["candidate_feasible"])
        self.assertEqual(sidecar["applied_r_select"], 5)
        self.assertEqual(sidecar["applied_s_select"], 5)

    def test_sidecar_records_the_actual_contract_not_a_hardcoded_label(self):
        trainer, _k0, _k3, sidecar = self._run(_metrics())
        self.assertEqual(sidecar["reward_mode"], "latency_only")
        self.assertEqual(sidecar["energy_constraint"], "configured:total_energy")
        self.assertEqual(sidecar["constraints"]["names"],
                         ["C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS"])
        self.assertEqual(sidecar["constraints"]["lambda_vector"],
                         {"C_GRAPH_HARD_DEADLINE": 0.25, "C_HI_TASK_TARDINESS": 0.75})
        self.assertEqual(sidecar["constraints"]["budget_fractions"],
                         {"total_energy_mec": 0.5})
        self.assertEqual(sidecar["constraints"]["applied_lambda_count"], 2)

    def test_infeasible_run_is_labelled_in_the_sidecar(self):
        trainer, _k0, _k3, sidecar = self._run(
            _metrics(graph_hard_violation_rate=0.25, mc_policy_violation_count=2.0))
        self.assertFalse(trainer.auto_checkpoint_selection["feasible"])
        self.assertIn("selection_note", trainer.auto_checkpoint_selection)
        self.assertFalse(sidecar["checkpoint_feasible"])
        self.assertFalse(sidecar["candidate_feasible"])
        self.assertIn("infeasible", sidecar["selection_note"])

    def test_a_violating_next_candidate_never_replaces_a_feasible_checkpoint(self):
        from spec.automotive_training.automotive_trainer import AutomotiveTrainerMixin

        fake_logger = types.SimpleNamespace(logkv=lambda *a, **k: None,
                                            logkv_mean=lambda *a, **k: None)
        fake_utils = types.ModuleType("utils")
        fake_utils.logger = fake_logger
        feasible = _FakeTrainer(_metrics(query_mean_latency_seconds=10.0))
        with mock.patch.dict(sys.modules, {"utils": fake_utils}):
            AutomotiveTrainerMixin._run_validation(feasible, 1)
            first_key = tuple(feasible.best_selection_key)
            # a later iteration reports a great latency but violates the hard deadline
            feasible.held_out_evaluator = _Evaluator(
                _metrics(graph_hard_violation_rate=0.5, query_mean_latency_seconds=0.001))
            AutomotiveTrainerMixin._run_validation(feasible, 2)
        self.assertEqual(tuple(feasible.best_selection_key), first_key,
                         "the violating candidate must not replace the feasible checkpoint")
        self.assertEqual(len(feasible.auto_checkpoint_records), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
