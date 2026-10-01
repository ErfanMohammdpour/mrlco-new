#!/usr/bin/env python3
"""Tests for the MARGO-AUTOMOTIVE-MC-v1 constraint channels (numpy/stdlib only).

Focus: truthfulness and observable dual behaviour.

  * the three exact channel names and their semantics (only D_G is hard)
  * energy is `not_configured` and cannot be enabled without provenance
  * violating vs non-violating fixtures: lambda up vs lambda EXACTLY unchanged
  * a batch with mean violation v moves lambda by exactly dual_lr * v
  * one projected ascent step, clip [0, 1e6], no NaN/Inf, empty batches are no-ops
  * the latency objective and the constraint penalty stay in distinct keys
  * evaluate_constraints against a ScheduleResult-like object, hand-computed
  * the split guard: validation never moves training state
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training import automotive_constraints as ac  # noqa: E402

YAML_PATH = ROOT / "spec" / "automotive_training" / "constraints_automotive_v1.yaml"


# --------------------------------------------------------------------------- #
# Fixtures that mimic the scheduler result shape (no TF, no scheduler import)
# --------------------------------------------------------------------------- #
class _FakeTaskRecord:
    def __init__(self, task_id, all_consumers_ready, criticality_class=None,
                 deadline_s=None):
        self.task_id = task_id
        self.all_consumers_ready = all_consumers_ready
        self.criticality_class = criticality_class
        self.deadline_s = deadline_s


class _FakeScheduleResult:
    def __init__(self, tasks, makespan_seconds=None):
        self.tasks = {t.task_id: t for t in tasks}
        self.makespan_seconds = makespan_seconds


class _FakeGraphTask:
    def __init__(self, task_id, deadline_s, criticality=None):
        self.task_id = task_id
        self.deadline_s = deadline_s
        if criticality is not None:
            self.criticality = criticality


class _FakeGraph:
    def __init__(self, D_G_s, tasks):
        self.D_G_s = D_G_s
        self.tasks = tasks


def hand_computed_fixture():
    """D_G=10, makespan=12; HIGH tardiness 0.5, MEDIUM tardiness 2.0."""
    graph = _FakeGraph(10.0, [
        _FakeGraphTask(0, 1.0),   # criticality comes from the result record
        _FakeGraphTask(1, 2.0),
        _FakeGraphTask(2, 1.0),
        _FakeGraphTask(3, 1.0),   # LOW: never counted
    ])
    result = _FakeScheduleResult([
        _FakeTaskRecord(0, all_consumers_ready=1.5, criticality_class="HIGH"),
        _FakeTaskRecord(1, all_consumers_ready=2.0, criticality_class="HIGH"),
        _FakeTaskRecord(2, all_consumers_ready=3.0, criticality_class="MEDIUM"),
        _FakeTaskRecord(3, all_consumers_ready=5.0, criticality_class="LOW"),
    ], makespan_seconds=12.0)
    return graph, result


def make_controller(**kwargs):
    return ac.AutomotiveDualController(ac.default_constraint_specs(), **kwargs)


# --------------------------------------------------------------------------- #
# 1. names / specs
# --------------------------------------------------------------------------- #
class TestNamesAndSpecs(unittest.TestCase):
    def test_exact_constraint_names(self):
        self.assertEqual(ac.CONSTRAINT_NAMES, (
            "C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS", "C_MED_TASK_TARDINESS"))
        self.assertEqual(set(ac.default_constraint_specs()), set(ac.CONSTRAINT_NAMES))

    def test_only_the_graph_deadline_is_hard(self):
        specs = ac.default_constraint_specs()
        self.assertEqual(specs[ac.C_GRAPH_HARD_DEADLINE].kind, "hard")
        self.assertEqual(specs[ac.C_HI_TASK_TARDINESS].kind, "firm")
        self.assertEqual(specs[ac.C_MED_TASK_TARDINESS].kind, "firm")
        # HIGH subdeadlines are never renamed to hard, and carry no fake budget.
        for name in (ac.C_HI_TASK_TARDINESS, ac.C_MED_TASK_TARDINESS):
            self.assertEqual(specs[name].budget, 0.0)
            self.assertEqual(specs[name].unit, "s")
            self.assertTrue(specs[name].enabled)
            self.assertEqual(specs[name].source, "frozen_dataset_contract:d_i")
            self.assertEqual(specs[name].status, "active")
        hard = specs[ac.C_GRAPH_HARD_DEADLINE]
        self.assertIsNone(hard.budget)  # per-graph D_G_s, not a constant
        self.assertEqual(hard.source, "frozen_dataset_contract:D_G_s")
        self.assertEqual(hard.unit, "s")

    def test_spec_dataclass_field_order_and_validation(self):
        spec = ac.ConstraintSpec("C_X", 1.5, "firm", "s", True, "somewhere")
        self.assertEqual(spec.name, "C_X")
        self.assertEqual(spec.budget, 1.5)
        self.assertEqual(spec.kind, "firm")
        self.assertEqual(spec.unit, "s")
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.source, "somewhere")
        with self.assertRaises(ValueError):
            ac.ConstraintSpec("C_X", 1.0, "soft", "s")
        with self.assertRaises(ValueError):
            ac.ConstraintSpec("C_X", -1.0, "firm", "s")


# --------------------------------------------------------------------------- #
# 2. energy truthfulness
# --------------------------------------------------------------------------- #
class TestEnergyTruthfulness(unittest.TestCase):
    def test_energy_is_not_configured_by_default(self):
        self.assertEqual(ac.energy_constraint, "not_configured")
        self.assertEqual(ac.ENERGY_NOT_CONFIGURED, "not_configured")
        spec = ac.energy_spec()
        self.assertFalse(spec.enabled)
        self.assertIsNone(spec.budget)          # no fake budget, ever
        self.assertEqual(spec.status, "not_configured")
        self.assertNotIn(ac.ENERGY_CONSTRAINT_NAME, ac.default_constraint_specs())

    def test_enabling_energy_without_provenance_raises(self):
        for budget, provenance in ((10.0, None), (10.0, ""), (10.0, "   "),
                                   (0.0, None), (0.0, "   "), (None, "measured:x")):
            with self.assertRaises(ac.EnergyConstraintError):
                ac.energy_spec(budget, provenance)

    def test_no_budget_and_no_provenance_is_still_not_configured(self):
        spec = ac.energy_spec(None, "   ")
        self.assertFalse(spec.enabled)
        self.assertIsNone(spec.budget)
        self.assertEqual(spec.status, "not_configured")

    def test_energy_budget_itself_must_be_a_finite_number(self):
        for budget in (float("nan"), float("inf"), "10", True):
            with self.assertRaises(ValueError):
                ac.energy_spec(budget, "measured:somewhere")
        with self.assertRaises(ac.EnergyConstraintError):
            ac.energy_spec(-1.0, "measured:somewhere")

    def test_energy_with_provenance_is_explicitly_tagged(self):
        spec = ac.energy_spec(12.5, "measured:2024-09-01/energy_reward_audit.py")
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.budget, 12.5)
        self.assertEqual(spec.unit, "j")
        self.assertTrue(spec.source.startswith("provenance:"))
        self.assertEqual(spec.status, "active")

    def test_controller_never_fabricates_an_energy_lambda(self):
        specs = dict(ac.default_constraint_specs())
        specs[ac.ENERGY_CONSTRAINT_NAME] = ac.energy_spec()
        ctrl = ac.AutomotiveDualController(specs, dual_lr=0.1)
        self.assertNotIn(ac.ENERGY_CONSTRAINT_NAME, ctrl.lambdas)
        out = ctrl.observe({ac.ENERGY_CONSTRAINT_NAME: 99.0})
        self.assertEqual(out["recorded"], [])
        step = ctrl.dual_step()
        self.assertIn(ac.ENERGY_CONSTRAINT_NAME, step["skipped"])
        self.assertEqual(ctrl.as_dict()["constraints"][ac.ENERGY_CONSTRAINT_NAME]["status"],
                         "not_configured")
        self.assertIsNone(
            ctrl.as_dict()["constraints"][ac.ENERGY_CONSTRAINT_NAME]["lambda"])

    def test_controller_refuses_an_enabled_energy_spec_without_provenance(self):
        bad = ac.ConstraintSpec(ac.ENERGY_CONSTRAINT_NAME, 3.0, "firm", "j", True, "who_knows")
        with self.assertRaises(ac.EnergyConstraintError):
            ac.AutomotiveDualController({ac.ENERGY_CONSTRAINT_NAME: bad}, dual_lr=0.1)

    def test_module_performs_no_file_io_at_import(self):
        """The module must not read spec/constraints.yaml (or any file) on import."""
        import builtins
        import importlib
        from unittest import mock

        source = (ROOT / "spec" / "automotive_training" / "automotive_constraints.py").read_text()
        self.assertNotIn("import yaml", source)
        self.assertNotIn("read_text", source)
        self.assertNotIn("open(", source)
        with mock.patch.object(builtins, "open",
                               side_effect=AssertionError("module performed file IO")), \
                mock.patch.object(Path, "read_text",
                                  side_effect=AssertionError("module performed file IO")):
            importlib.reload(ac)
        self.assertEqual(ac.CONSTRAINT_NAMES, (
            "C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS", "C_MED_TASK_TARDINESS"))
        self.assertEqual(ac.energy_constraint, "not_configured")


# --------------------------------------------------------------------------- #
# 3. dual controller: real, projected ascent
# --------------------------------------------------------------------------- #
class TestDualController(unittest.TestCase):
    def test_violating_fixture_increases_lambda(self):
        ctrl = make_controller(dual_lr=0.05)
        ctrl.observe({ac.C_HI_TASK_TARDINESS: 4.0})
        before = ctrl.lambdas[ac.C_HI_TASK_TARDINESS]
        step = ctrl.dual_step()
        after = ctrl.lambdas[ac.C_HI_TASK_TARDINESS]
        self.assertGreater(after, before)
        self.assertIn(ac.C_HI_TASK_TARDINESS, step["updated"])
        self.assertAlmostEqual(after - before, 0.05 * 4.0, delta=1e-12)
        # satisfied channels did not move at all
        self.assertEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], 0.0)
        self.assertEqual(ctrl.lambdas[ac.C_GRAPH_HARD_DEADLINE], 0.0)

    def test_non_violating_fixture_leaves_lambda_exactly_unchanged(self):
        ctrl = make_controller(dual_lr=0.5)
        ctrl.observe({ac.C_GRAPH_HARD_DEADLINE: 0.0,
                      ac.C_HI_TASK_TARDINESS: 0.0,
                      ac.C_MED_TASK_TARDINESS: 0.0})
        before = dict(ctrl.lambdas)
        step = ctrl.dual_step()
        self.assertEqual(ctrl.lambdas, before)          # exact equality, no -0.0
        self.assertEqual(step["updated"], [])
        for value in ctrl.lambdas.values():
            self.assertEqual(value, 0.0)
            self.assertFalse(math.copysign(1.0, value) < 0.0)

    def test_batch_mean_moves_lambda_by_exactly_dual_lr_times_v(self):
        v = 3.25
        lr = 0.02
        ctrl = make_controller(dual_lr=lr)
        ctrl.observe({ac.C_MED_TASK_TARDINESS: v})
        ctrl.observe({ac.C_MED_TASK_TARDINESS: v})
        ctrl.dual_step()
        expected = lr * float(np.mean([v, v]))
        self.assertAlmostEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], expected, delta=1e-12)
        self.assertEqual(ctrl.as_dict()["constraints"][ac.C_MED_TASK_TARDINESS]
                         ["last_mean_violation"], v)

    def test_one_step_per_call_and_reset_batch(self):
        ctrl = make_controller(dual_lr=0.1)
        ctrl.observe({ac.C_HI_TASK_TARDINESS: 1.0})
        ctrl.dual_step()
        ctrl.dual_step()  # same batch: the mean is unchanged, so it moves once more
        self.assertAlmostEqual(ctrl.lambdas[ac.C_HI_TASK_TARDINESS], 0.2, delta=1e-12)
        ctrl.reset_batch()
        before = dict(ctrl.lambdas)
        step = ctrl.dual_step()
        self.assertEqual(ctrl.lambdas, before)
        self.assertIsNone(step["means"][ac.C_HI_TASK_TARDINESS])

    def test_projection_keeps_lambda_within_clip(self):
        ctrl = make_controller(dual_lr=0.5, clip=(0.0, 1.0))
        ctrl.observe({ac.C_HI_TASK_TARDINESS: 100.0})
        ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[ac.C_HI_TASK_TARDINESS], 1.0)  # clipped high
        # lower projection: a negative multiplier is pulled back to the floor
        ctrl.lambdas[ac.C_MED_TASK_TARDINESS] = -5.0
        ctrl.observe({ac.C_MED_TASK_TARDINESS: 0.1})
        ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], 0.0)
        for value in ctrl.lambdas.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_empty_batch_is_a_no_op_without_nan(self):
        ctrl = make_controller(dual_lr=0.3)
        step = ctrl.dual_step()
        self.assertEqual(ctrl.lambdas, {n: 0.0 for n in ac.CONSTRAINT_NAMES})
        for value in step["means"].values():
            self.assertIsNone(value)
        self.assertEqual(step["updated"], [])
        breakdown = ac.objective_breakdown(1.0, {}, ctrl)
        self.assertTrue(all(math.isfinite(v) for v in breakdown["constraint_lambdas"].values()))
        self.assertEqual(breakdown["constraint_penalty"], 0.0)
        self.assertEqual(breakdown["penalized_objective"], 1.0)

    def test_non_finite_and_bad_input_raises(self):
        ctrl = make_controller(dual_lr=0.1)
        for bad in (float("nan"), float("inf"), float("-inf"), "1.0", None):
            with self.assertRaises(ValueError):
                ctrl.observe({ac.C_HI_TASK_TARDINESS: bad})
        with self.assertRaises(ValueError):
            ctrl.observe([1.0, 2.0])
        with self.assertRaises(ValueError):
            ac.AutomotiveDualController(ac.default_constraint_specs(), dual_lr=float("nan"))
        with self.assertRaises(ValueError):
            ac.AutomotiveDualController(ac.default_constraint_specs(), dual_lr=-0.1)
        with self.assertRaises(ValueError):
            ac.AutomotiveDualController(ac.default_constraint_specs(), dual_lr=0.1,
                                        clip=(1.0, 0.0))

    def test_penalty_is_additive_over_enabled_channels(self):
        ctrl = make_controller(dual_lr=0.1)
        ctrl.lambdas[ac.C_HI_TASK_TARDINESS] = 2.0
        ctrl.lambdas[ac.C_MED_TASK_TARDINESS] = 0.5
        penalty = ctrl.penalty({ac.C_HI_TASK_TARDINESS: 3.0,
                                ac.C_MED_TASK_TARDINESS: 4.0})
        self.assertAlmostEqual(penalty, 2.0 * 3.0 + 0.5 * 4.0, delta=1e-12)
        self.assertEqual(ctrl.penalty({}), 0.0)
        self.assertAlmostEqual(ctrl.penalty({ac.C_HI_TASK_TARDINESS: 1.0}), 2.0, delta=1e-12)
        with self.assertRaises(ValueError):
            ctrl.penalty({ac.C_HI_TASK_TARDINESS: float("nan")})

    def test_as_dict_reports_lambda_mean_updates_status(self):
        specs = dict(ac.default_constraint_specs())
        specs[ac.ENERGY_CONSTRAINT_NAME] = ac.energy_spec()
        ctrl = ac.AutomotiveDualController(specs, dual_lr=0.25)
        ctrl.observe({ac.C_GRAPH_HARD_DEADLINE: 2.0})
        ctrl.dual_step()
        doc = ctrl.as_dict()
        self.assertEqual(doc["dual_lr"], 0.25)
        self.assertEqual(doc["clip"], [0.0, 1e6])
        row = doc["constraints"][ac.C_GRAPH_HARD_DEADLINE]
        self.assertAlmostEqual(row["lambda"], 0.5, delta=1e-12)
        self.assertEqual(row["last_mean_violation"], 2.0)
        self.assertEqual(row["updates"], 1)
        self.assertEqual(row["status"], "active")
        self.assertEqual(doc["constraints"][ac.ENERGY_CONSTRAINT_NAME]["status"],
                         "not_configured")
        self.assertEqual(doc["constraints"][ac.C_HI_TASK_TARDINESS]["status"], "active")
        self.assertEqual(doc["constraints"][ac.C_HI_TASK_TARDINESS]["updates"], 0)
        self.assertIsNone(doc["constraints"][ac.C_HI_TASK_TARDINESS]["last_mean_violation"])

    def test_disabled_channel_is_status_disabled_and_has_no_lambda(self):
        spec = ac.ConstraintSpec(ac.C_HI_TASK_TARDINESS, 0.0, "firm", "s", False, "turned_off")
        ctrl = ac.AutomotiveDualController({ac.C_HI_TASK_TARDINESS: spec}, dual_lr=0.1)
        self.assertEqual(ctrl.lambdas, {})
        ctrl.observe({ac.C_HI_TASK_TARDINESS: 5.0})
        ctrl.dual_step()
        self.assertEqual(ctrl.as_dict()["constraints"][ac.C_HI_TASK_TARDINESS]["status"],
                         "disabled")


# --------------------------------------------------------------------------- #
# 4. split guard
# --------------------------------------------------------------------------- #
class TestSplitGuard(unittest.TestCase):
    def test_validation_split_does_not_move_lambdas(self):
        ctrl = make_controller(dual_lr=0.5)
        out = ctrl.observe({ac.C_HI_TASK_TARDINESS: 10.0}, split="validation")
        self.assertFalse(out["accepted"])
        step = ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[ac.C_HI_TASK_TARDINESS], 0.0)
        self.assertIsNone(step["means"][ac.C_HI_TASK_TARDINESS])
        doc = ctrl.as_dict()
        self.assertEqual(doc["rejected_splits"], {"validation": 1})
        self.assertEqual(ctrl.rejected_splits, {"validation": 1})

    def test_meta_test_split_does_not_move_lambdas(self):
        ctrl = make_controller(dual_lr=0.5)
        ctrl.observe({ac.C_MED_TASK_TARDINESS: 7.0}, split="meta_test")
        ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], 0.0)
        self.assertEqual(ctrl.rejected_splits, {"meta_test": 1})

    def test_training_splits_are_accepted(self):
        for split in ("meta_train", "train"):
            ctrl = make_controller(dual_lr=0.5)
            out = ctrl.observe({ac.C_MED_TASK_TARDINESS: 2.0}, split=split)
            self.assertTrue(out["accepted"])
            ctrl.dual_step()
            self.assertAlmostEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], 1.0, delta=1e-12)
            self.assertEqual(ctrl.rejected_splits, {})

    def test_non_finite_input_is_refused_even_on_a_rejected_split(self):
        ctrl = make_controller(dual_lr=0.5)
        with self.assertRaises(ValueError):
            ctrl.observe({ac.C_HI_TASK_TARDINESS: float("nan")}, split="validation")
        self.assertEqual(ctrl.lambdas[ac.C_HI_TASK_TARDINESS], 0.0)


# --------------------------------------------------------------------------- #
# 5. objective breakdown keeps the keys separate
# --------------------------------------------------------------------------- #
class TestObjectiveBreakdown(unittest.TestCase):
    def test_penalty_is_never_folded_into_the_latency_value(self):
        ctrl = make_controller(dual_lr=0.1)
        ctrl.lambdas[ac.C_HI_TASK_TARDINESS] = 1.5
        ctrl.lambdas[ac.C_MED_TASK_TARDINESS] = 0.25
        violations = {ac.C_HI_TASK_TARDINESS: 2.0, ac.C_MED_TASK_TARDINESS: 4.0,
                      ac.C_GRAPH_HARD_DEADLINE: 0.0}
        base = 0.75
        out = ac.objective_breakdown(base, violations, ctrl)
        penalty = 1.5 * 2.0 + 0.25 * 4.0
        self.assertEqual(out["latency_only_objective"], base)      # untouched
        self.assertAlmostEqual(out["constraint_penalty"], penalty, delta=1e-12)
        self.assertAlmostEqual(out["penalized_objective"], base + penalty, delta=1e-12)
        self.assertNotEqual(out["latency_only_objective"], out["penalized_objective"])
        self.assertEqual(out["constraint_violations"][ac.C_HI_TASK_TARDINESS], 2.0)
        self.assertEqual(out["constraint_lambdas"][ac.C_HI_TASK_TARDINESS], 1.5)
        self.assertEqual(set(out), {
            "latency_only_objective", "constraint_violations", "constraint_lambdas",
            "constraint_penalty", "penalized_objective"})

    def test_zero_penalty_keeps_the_two_values_equal_but_separate_keys(self):
        out = ac.objective_breakdown(0.5, {ac.C_HI_TASK_TARDINESS: 0.0},
                                     lambdas={ac.C_HI_TASK_TARDINESS: 3.0})
        self.assertEqual(out["constraint_penalty"], 0.0)
        self.assertEqual(out["penalized_objective"], 0.5)
        self.assertIn("constraint_penalty", out)
        self.assertNotEqual(id(out["latency_only_objective"]),
                            id(out["constraint_penalty"]))

    def test_breakdown_rejects_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            ac.objective_breakdown(float("nan"), {})
        ctrl = make_controller(dual_lr=0.1)
        with self.assertRaises(ValueError):
            ac.objective_breakdown(1.0, {ac.C_HI_TASK_TARDINESS: float("inf")}, ctrl)


# --------------------------------------------------------------------------- #
# 6. evaluate_constraints
# --------------------------------------------------------------------------- #
class TestEvaluateConstraints(unittest.TestCase):
    def test_schedule_result_like_object_hand_computed(self):
        graph, result = hand_computed_fixture()
        out = ac.evaluate_constraints(graph, result)
        self.assertEqual(set(out), set(ac.CONSTRAINT_NAMES))
        for name in ac.CONSTRAINT_NAMES:
            self.assertEqual(set(out[name]),
                             {"violation", "value", "budget", "n_violating_tasks"})
        hard = out[ac.C_GRAPH_HARD_DEADLINE]
        self.assertAlmostEqual(hard["violation"], 2.0, delta=1e-12)
        self.assertAlmostEqual(hard["value"], 12.0, delta=1e-12)
        self.assertAlmostEqual(hard["budget"], 10.0, delta=1e-12)
        self.assertEqual(hard["n_violating_tasks"], 1)
        hi = out[ac.C_HI_TASK_TARDINESS]
        self.assertAlmostEqual(hi["violation"], 0.5, delta=1e-12)
        self.assertAlmostEqual(hi["value"], 0.5, delta=1e-12)
        self.assertEqual(hi["budget"], 0.0)
        self.assertEqual(hi["n_violating_tasks"], 1)
        med = out[ac.C_MED_TASK_TARDINESS]
        self.assertAlmostEqual(med["violation"], 2.0, delta=1e-12)
        self.assertEqual(med["n_violating_tasks"], 1)

    def test_satisfied_schedule_has_zero_violations(self):
        graph = _FakeGraph(20.0, [
            _FakeGraphTask(0, 5.0, criticality="HIGH"),
            _FakeGraphTask(1, 5.0, criticality="MEDIUM"),
        ])
        result = _FakeScheduleResult([
            _FakeTaskRecord(0, 1.0, "high", 5.0),
            _FakeTaskRecord(1, 2.0, "medium", 5.0),
        ], makespan_seconds=3.0)
        out = ac.evaluate_constraints(graph, result)
        for name in ac.CONSTRAINT_NAMES:
            self.assertEqual(out[name]["violation"], 0.0)
            self.assertEqual(out[name]["n_violating_tasks"], 0)
        self.assertEqual(out[ac.C_GRAPH_HARD_DEADLINE]["budget"], 20.0)

    def test_accepts_a_plain_availability_mapping(self):
        graph = _FakeGraph(10.0, [_FakeGraphTask(0, 1.0, criticality="HIGH")])
        # makespan is derived from the availability mapping when not given
        out = ac.evaluate_constraints(graph, {0: 3.0})
        self.assertAlmostEqual(out[ac.C_GRAPH_HARD_DEADLINE]["value"], 3.0, delta=1e-12)
        self.assertAlmostEqual(out[ac.C_HI_TASK_TARDINESS]["violation"], 2.0, delta=1e-12)
        self.assertEqual(out[ac.C_MED_TASK_TARDINESS]["violation"], 0.0)

    def test_graph_side_criticality_takes_precedence_over_result_records(self):
        graph = _FakeGraph(10.0, [_FakeGraphTask(0, 1.0, criticality="MEDIUM")])
        result = _FakeScheduleResult([_FakeTaskRecord(0, 4.0, "high", 1.0)],
                                     makespan_seconds=4.0)
        out = ac.evaluate_constraints(graph, result)
        self.assertEqual(out[ac.C_HI_TASK_TARDINESS]["violation"], 0.0)
        self.assertAlmostEqual(out[ac.C_MED_TASK_TARDINESS]["violation"], 3.0, delta=1e-12)

    def test_refuses_missing_availability_or_deadline(self):
        graph = _FakeGraph(10.0, [_FakeGraphTask(0, 1.0, criticality="HIGH")])
        with self.assertRaises(ValueError):
            ac.evaluate_constraints(graph, {1: 3.0})           # unknown task id
        no_deadline = _FakeGraph(10.0, [_FakeGraphTask(0, 1.0)])
        with self.assertRaises(ValueError):
            ac.evaluate_constraints(no_deadline, {0: 3.0})     # no d_i anywhere
        with self.assertRaises(ValueError):
            ac.evaluate_constraints(object(), {0: 3.0})              # no .tasks
        with self.assertRaises(ValueError):
            ac.evaluate_constraints({"tasks": []}, {0: 3.0})         # no frozen D_G_s
        with self.assertRaises(ValueError):
            ac.evaluate_constraints(_FakeGraph(10.0, [
                _FakeGraphTask(0, 1.0, criticality="HIGH")]), {0: float("nan")})

    def test_accepts_the_real_scheduler_schedule_result(self):
        """Integration check: the frozen TaskExecutionRecord shape is accepted."""
        from env.mec_offloaing_envs.scheduler.model import (
            EnergyBreakdown, ScheduleResult, TaskExecutionRecord)

        def rec(task_id, ready, cls, deadline):
            return TaskExecutionRecord(
                task_id=task_id, location="ue", start=0.0, finish=max(0.0, ready - 0.5),
                output_location="ue", all_consumers_ready=ready,
                criticality_class=cls, deadline_s=deadline)

        result = ScheduleResult(
            tasks={0: rec(0, 1.5, "high", 1.0), 1: rec(1, 3.0, "medium", 1.0),
                   2: rec(2, 1.0, "low", 1.0)},
            transfers=[], resource_intervals=[], energy=EnergyBreakdown(),
            makespan_seconds=12.0, terminal_return_time=3.0)
        graph, _ = hand_computed_fixture()
        graph.tasks = [_FakeGraphTask(0, 1.0), _FakeGraphTask(1, 1.0),
                       _FakeGraphTask(2, 1.0)]
        out = ac.evaluate_constraints(graph, result)
        self.assertAlmostEqual(out[ac.C_GRAPH_HARD_DEADLINE]["violation"], 2.0, delta=1e-12)
        self.assertAlmostEqual(out[ac.C_HI_TASK_TARDINESS]["violation"], 0.5, delta=1e-12)
        self.assertAlmostEqual(out[ac.C_MED_TASK_TARDINESS]["violation"], 2.0, delta=1e-12)

    def test_violation_feeds_the_dual_channel_arithmetic(self):
        graph, result = hand_computed_fixture()
        violations = {name: row["violation"]
                      for name, row in ac.evaluate_constraints(graph, result).items()}
        ctrl = make_controller(dual_lr=0.1)
        ctrl.observe(violations, split="meta_train")
        ctrl.dual_step()
        self.assertAlmostEqual(ctrl.lambdas[ac.C_GRAPH_HARD_DEADLINE], 0.2, delta=1e-12)
        self.assertAlmostEqual(ctrl.lambdas[ac.C_HI_TASK_TARDINESS], 0.05, delta=1e-12)
        self.assertAlmostEqual(ctrl.lambdas[ac.C_MED_TASK_TARDINESS], 0.2, delta=1e-12)


# --------------------------------------------------------------------------- #
# 7. the YAML definition file
# --------------------------------------------------------------------------- #
class TestYamlDefinition(unittest.TestCase):
    def setUp(self):
        self.doc = yaml.safe_load(YAML_PATH.read_text())

    def test_yaml_schema_and_channels(self):
        self.assertEqual(self.doc["schema_version"], "automotive-constraints-v1")
        self.assertIsInstance(self.doc["dual_lr"], float)
        self.assertGreater(self.doc["dual_lr"], 0.0)
        rows = {row["name"]: row for row in self.doc["constraints"]}
        self.assertEqual(set(rows), set(ac.CONSTRAINT_NAMES))
        for name in ac.CONSTRAINT_NAMES:
            self.assertTrue(rows[name]["enabled"])
            self.assertEqual(rows[name]["unit"], "s")
            self.assertEqual(rows[name]["source"], "frozen_dataset_contract")
        self.assertEqual(rows[ac.C_GRAPH_HARD_DEADLINE]["kind"], "hard")
        self.assertEqual(rows[ac.C_GRAPH_HARD_DEADLINE]["budget"], "from_graph:D_G_s")
        for name in (ac.C_HI_TASK_TARDINESS, ac.C_MED_TASK_TARDINESS):
            self.assertEqual(rows[name]["kind"], "firm")
            self.assertEqual(rows[name]["budget"], 0.0)

    def test_yaml_energy_is_not_configured_with_a_reason(self):
        self.assertEqual(self.doc["energy_constraint"], "not_configured")
        self.assertTrue(str(self.doc["energy_constraint_reason"]).strip())
        self.assertIn("proposal_not_frozen", self.doc["energy_constraint_reason"])

    def test_yaml_forbidden_list(self):
        self.assertEqual(self.doc["forbidden"], [
            "fake_energy_budget", "renaming_HIGH_subdeadlines_as_hard",
            "penalty_inside_latency"])

    def test_yaml_parses_into_truthful_specs(self):
        specs = ac.specs_from_mapping(self.doc)
        self.assertEqual(set(specs), set(ac.CONSTRAINT_NAMES))
        self.assertIsNone(specs[ac.C_GRAPH_HARD_DEADLINE].budget)
        self.assertEqual(specs[ac.C_HI_TASK_TARDINESS].budget, 0.0)
        self.assertNotIn(ac.ENERGY_CONSTRAINT_NAME, specs)
        ctrl = ac.AutomotiveDualController(specs, dual_lr=float(self.doc["dual_lr"]))
        self.assertEqual(set(ctrl.lambdas), set(ac.CONSTRAINT_NAMES))

    def test_yaml_energy_cannot_be_enabled_without_provenance(self):
        doc = {"constraints": [], "energy_constraint": {"enabled": True, "budget": 5.0}}
        with self.assertRaises(ac.EnergyConstraintError):
            ac.specs_from_mapping(doc)


if __name__ == "__main__":
    unittest.main()
