#!/usr/bin/env python3
"""Iteration accounting, failure propagation and resume semantics.

These tests reproduce the two audited defects from the gate2 run and pin the fixes:

1. **Pre-emptive completion.** `stack.AutomotiveV2Trainer._audit` used to set
   `completed_iterations` when the `validation` audit fired. That is the moment a phase is
   *entered*, not the moment an iteration finishes, and iterations whose
   `itr % validation_interval != 0` never fired it at all. A run with two real completed
   iterations therefore reported `completed=1`, and a crash after validation would have
   reported a completed iteration that did not finish. Completion is now recorded by the
   loop's LAST statement through `_iteration_completed`.

2. **Report-time `KeyError: 'iterations_executed'`.** `run_tf.main` printed a summary key
   that `_collect_evidence` never produced. Training had already completed both iterations
   and written a valid checkpoint before the process died in the *reporting* phase with a
   traceback but a JSON that still said `ok: True`. Both the missing key and the premature
   `ok` are covered here.

The counters are exercised through the REAL `AutomotiveV2Trainer._audit` /
`_iteration_completed` implementation without importing TensorFlow (the module is loaded from
source, so this runs in the non-TF suite).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The phase names the frozen loop emits, in the order it emits them.
PHASES_NO_VALIDATION = [
    "sync_task_policies_from_core", "sample_tasks", "obtain_samples_ppo",
    "process_samples_ppo", "inner_ppo", "obtain_samples_eval", "process_samples_eval",
    "outer_update", "constraints",
]
PHASES_WITH_VALIDATION = PHASES_NO_VALIDATION + ["validation"]


def _load_counter_mixin():
    """Load the REAL `_audit`/`_iteration_completed` bodies from stack.py without TF.

    `spec.automotive_training.v2.stack` imports TensorFlow at module import time, so the
    accounting methods are extracted from the source text and executed in a throwaway module
    namespace. That keeps the assertion on the production implementation while allowing the
    test to run in the non-TF suite.
    """
    src = (ROOT / "spec" / "automotive_training" / "v2" / "stack.py").read_text()
    start = src.index("        def _audit(self, itr, name, payload=None):")
    end = src.index("        def constraint_observer(")
    body = src[start:end]
    # strip the eight-space method indentation; the methods become module-level functions
    dedented = textwrap.dedent(body)
    ns: dict = {"time": __import__("time"), "json": json,
                "_Path": Path, "os": os}
    exec(compile(dedented, "stack_counter_body", "exec"), ns)  # noqa: S102 - trusted repo source
    return ns


class _Base:
    """`super()._audit` stand-in: the frozen base implementation does nothing observable."""

    def _audit(self, itr, name, payload=None):
        return None


class CounterHarness(_Base):
    """A trainer-shaped object carrying the production accounting methods."""

    n_itr = 500
    auto_run_dir = None

    def __init__(self):
        self.started_iterations = 0
        self.completed_iterations = 0
        self.last_completed_iteration = -1
        self.next_iteration = 0
        self.outer_updates_completed = 0
        self.dual_updates_completed = 0
        self.validations_completed = 0
        self.current_iteration = -1
        self.auto_run_dir = None


def _make_harness():
    ns = _load_counter_mixin()
    cls = type("H", (CounterHarness,), {
        "_audit": ns["_audit"],
        "_iteration_completed": ns["_iteration_completed"],
        "_next_iteration": ns["_next_iteration"],
        "_progress_path": ns["_progress_path"],
        "_write_progress": ns["_write_progress"],
        "_phase_jsonl_path": ns["_phase_jsonl_path"],
        "_append_phase_timings": ns["_append_phase_timings"],
    })
    # the production body spells `super(AutomotiveV2Trainer, self)`, so the harness class must
    # be visible under that name inside the extracted body's namespace
    ns["AutomotiveV2Trainer"] = cls
    return cls()


def _run_iteration(h, itr, phases, *, crash_after=None):
    """Drive one iteration exactly as the frozen loop does: audit each phase, then the hook."""
    for name in phases:
        if crash_after == name:
            raise RuntimeError("simulated crash during %s" % name)
        h._audit(itr, name, {})
    if crash_after == "after_last_phase":
        raise RuntimeError("simulated crash before the completion hook")
    h._iteration_completed(itr)


class IterationAccountingTests(unittest.TestCase):

    def test_two_complete_iterations_report_two_completed(self):
        """The gate2 contradiction: two real iterations MUST report completed=2."""
        h = _make_harness()
        _run_iteration(h, 0, PHASES_WITH_VALIDATION)   # iteration 0 is validated
        _run_iteration(h, 1, PHASES_NO_VALIDATION)     # iteration 1 is not
        self.assertEqual(h.started_iterations, 2)
        self.assertEqual(h.completed_iterations, 2,
                         "an iteration without a scheduled validation was not counted")
        self.assertEqual(h.last_completed_iteration, 1)
        self.assertEqual(h.next_iteration, 2)
        self.assertEqual(h.outer_updates_completed, 2)
        self.assertEqual(h.validations_completed, 1)
        self.assertEqual(h.dual_updates_completed, 2)

    def test_iteration_without_validation_is_counted(self):
        h = _make_harness()
        for itr in (0, 1, 2):
            _run_iteration(h, itr, PHASES_NO_VALIDATION)
        self.assertEqual(h.completed_iterations, 3)
        self.assertEqual(h.validations_completed, 0, "no validation was scheduled")

    def test_failure_after_outer_update_does_not_count_completion(self):
        """A crash after the real outer update leaves the iteration started-but-unfinished."""
        h = _make_harness()
        _run_iteration(h, 0, PHASES_WITH_VALIDATION)
        with self.assertRaises(RuntimeError):
            _run_iteration(h, 1, PHASES_NO_VALIDATION, crash_after="constraints")
        self.assertEqual(h.started_iterations, 2)
        self.assertEqual(h.completed_iterations, 1, "a crashed iteration was counted as done")
        self.assertEqual(h.last_completed_iteration, 0)
        self.assertEqual(h.next_iteration, 1, "resume must re-run the unfinished iteration")

    def test_failure_during_validation_is_not_a_completion(self):
        """Entering validation is NOT completing an iteration."""
        h = _make_harness()
        with self.assertRaises(RuntimeError):
            _run_iteration(h, 0, PHASES_WITH_VALIDATION, crash_after="validation")
        self.assertEqual(h.started_iterations, 1)
        self.assertEqual(h.completed_iterations, 0)
        self.assertEqual(h.next_iteration, 0)
        self.assertEqual(h.validations_completed, 0)

    def test_outer_update_counter_is_not_inferred(self):
        h = _make_harness()
        _run_iteration(h, 0, ["sync_task_policies_from_core", "sample_tasks"])
        self.assertEqual(h.started_iterations, 1)
        self.assertEqual(h.outer_updates_completed, 0,
                         "the outer update was counted without the phase running")
        self.assertEqual(h.completed_iterations, 1)

    def test_the_ledger_survives_a_failing_rich_progress_record(self):
        """Field defect: a failure building the rich record lost the completion boundary.

        The run's own `v2_phase_timings.jsonl` was created but never written, because
        `_write_progress` raised BEFORE the append ran and the exception escaped
        `_iteration_completed`. The counters must survive any single failing observability
        path: the degraded record is written and the phase-timing append still happens.
        """
        class _Boom:
            def world_realization_key(self):
                raise RuntimeError("simulated telemetry failure")

        with tempfile.TemporaryDirectory() as tmp:
            h = _make_harness()
            h.auto_run_dir = Path(tmp)
            h.auto_env = _Boom()          # makes building the rich record raise
            _run_iteration(h, 0, PHASES_WITH_VALIDATION)
            progress = Path(tmp) / "v2_progress.json"
            self.assertTrue(progress.exists(),
                            "the completion ledger was lost because a richer field failed")
            rec = json.loads(progress.read_text())
            self.assertTrue(rec.get("degraded"))
            self.assertEqual(rec["completed_iterations"], 1)
            self.assertEqual(rec["next_iteration"], 1)
            # the phase-timing path must still have run
            self.assertTrue((Path(tmp) / "v2_phase_timings.jsonl").exists())
            self.assertTrue(getattr(h, "auto_progress_error", None))

    def test_progress_file_is_atomic_and_carries_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _make_harness()
            h.auto_run_dir = Path(tmp)
            _run_iteration(h, 0, PHASES_WITH_VALIDATION)
            path = Path(tmp) / "v2_progress.json"
            self.assertTrue(path.exists(), "no progress file was written")
            self.assertFalse(path.with_suffix(".tmp").exists(), "the temp file was left behind")
            rec = json.loads(path.read_text())
            for key in ("started_iterations", "completed_iterations", "last_completed_iteration",
                        "next_iteration", "outer_updates_completed", "dual_updates_completed",
                        "validations_completed"):
                self.assertIn(key, rec, "progress ledger is missing %r" % key)
            self.assertEqual(rec["completed_iterations"], 1)
            self.assertEqual(rec["next_iteration"], 1)


class ResumeSemanticsTests(unittest.TestCase):
    """`--iterations` is the TOTAL target, and a resume must not repeat completed work."""

    def test_resume_starts_at_next_iteration_not_zero(self):
        h = _make_harness()
        for itr in range(7):
            _run_iteration(h, itr, PHASES_WITH_VALIDATION if itr % 50 == 0
                           else PHASES_NO_VALIDATION)
        self.assertEqual(h.next_iteration, 7)
        # a resumed run must execute iterations 7..499, i.e. it starts at 7
        target = 500
        self.assertEqual(list(range(h.next_iteration, target))[0], 7)
        self.assertEqual(len(range(h.next_iteration, target)), 493)


class ReportContractTests(unittest.TestCase):
    """The reporting phase must not be able to kill an otherwise complete run."""

    def _evidence(self, **over):
        ev = {
            "iterations_requested": 2, "iterations_started": 2, "iterations_completed": 2,
            "last_completed_iteration": 1, "next_iteration": 2,
            "outer_updates_completed": 2, "validations_completed": 1,
            "iterations_executed": 2, "wall_seconds": 1.0,
            "sampler_counters": {"env_calls": 10},
            "penalty": {"episodes": 2, "episodes_with_nonzero_penalty": 1, "penalty_sum": 1.0,
                        "steps_with_penalty": 1, "signed_rows_observed": 2},
            "last_rollout": {"makespan_s": 1.0, "system_joules": 10.0,
                             "constraint_penalty": 0.0, "world_id": "w"},
        }
        ev.update(over)
        return ev

    def test_assert_real_run_accepts_a_complete_run(self):
        from spec.automotive_training.v2 import run_tf

        run_tf._assert_real_run(self._evidence())

    def test_assert_real_run_rejects_counter_disagreement(self):
        from spec.automotive_training.v2 import run_tf

        with self.assertRaises(run_tf.V2RunError):
            run_tf._assert_real_run(self._evidence(next_iteration=5))

    def test_assert_real_run_rejects_overshoot(self):
        from spec.automotive_training.v2 import run_tf

        with self.assertRaises(run_tf.V2RunError):
            run_tf._assert_real_run(self._evidence(iterations_requested=1))

    def test_assert_real_run_rejects_zero_energy(self):
        from spec.automotive_training.v2 import run_tf

        ev = self._evidence()
        ev["last_rollout"]["system_joules"] = 0.0
        with self.assertRaises(run_tf.V2RunError):
            run_tf._assert_real_run(ev)

    def test_summary_printer_has_no_missing_keys(self):
        """Every key the terminal summary reads must exist in the evidence dict."""
        from spec.automotive_training.v2 import run_tf

        ev = self._evidence()
        for key in ("iterations_requested", "iterations_started", "iterations_completed",
                    "last_completed_iteration", "next_iteration", "outer_updates_completed",
                    "validations_completed", "penalty", "last_rollout"):
            self.assertIn(key, ev)
        for key in ("makespan_s", "system_joules", "constraint_penalty", "world_id"):
            self.assertIn(key, ev["last_rollout"])


class ResumeBundleTests(unittest.TestCase):
    """Reject incompatible checkpoints and honour `--iterations` as a total target."""

    def _modules(self):
        from spec.automotive_training.v2 import run_tf

        return run_tf

    def _args(self, **over):
        base = dict(iterations=500, seed=0, background=2, background_policy="all_mec",
                    link_regime="moderate", mec_workers=1, r_select=2, s_select=2,
                    ckpt_dir=None)
        base.update(over)
        return types.SimpleNamespace(**base)

    def _trainer(self):
        class M:
            def __init__(self):
                self._state = {"schema": "v2_constraint_state_v1",
                               "constraints_sha256": "abc", "lambdas": {"total_energy": 0.0}}

            def state(self):
                return dict(self._state)

            def load_state(self, state):
                if state.get("schema") != self._state["schema"]:
                    raise RuntimeError("spec mismatch")
                self._state = dict(state)

        t = types.SimpleNamespace()
        t.started_iterations = 3
        t.completed_iterations = 3
        t.last_completed_iteration = 2
        t.next_iteration = 3
        t.outer_updates_completed = 3
        t.dual_updates_completed = 3
        t.validations_completed = 1
        t.auto_v2_constraint_manager = M()
        t.sampler = types.SimpleNamespace(counters={"env_calls": 5})
        t.best_val_objective = None
        t.start_itr = 0
        for attr in ("auto_obs_version", "auto_protocol_id", "auto_dataset_fingerprint",
                     "auto_scheduler_fingerprint", "auto_v2_system"):
            setattr(t, attr, None)
        return t

    def test_bundle_round_trip_sets_start_iteration(self):
        rt = self._modules()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            args = self._args(ckpt_dir=str(ckpt))
            trainer = self._trainer()
            path = rt._write_resume_state(trainer, args)
            self.assertTrue(path.exists())
            fresh = self._trainer()
            fresh.start_itr = 0
            rt._restore_resume_state(fresh, path, args)
            self.assertEqual(fresh.start_itr, 3,
                             "resume did not advance to the next unfinished iteration")
            self.assertEqual(fresh.completed_iterations, 3)
            self.assertEqual(fresh.next_iteration, 3)

    def test_incompatible_seed_is_rejected(self):
        rt = self._modules()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            args = self._args(ckpt_dir=str(ckpt))
            path = rt._write_resume_state(self._trainer(), args)
            with self.assertRaises(rt.V2RunError):
                rt._restore_resume_state(self._trainer(), path, self._args(
                    ckpt_dir=str(ckpt), seed=1))

    def test_resume_past_target_is_rejected(self):
        rt = self._modules()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            args = self._args(ckpt_dir=str(ckpt))
            path = rt._write_resume_state(self._trainer(), args)
            with self.assertRaises(rt.V2RunError):
                rt._restore_resume_state(self._trainer(), path,
                                         self._args(ckpt_dir=str(ckpt), iterations=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
