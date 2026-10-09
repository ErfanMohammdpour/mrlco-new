#!/usr/bin/env python3
"""F19: the resume CONTRACT is compared, not just the seed.

`run_tf.py` wrote `obs_version`, `dataset_fingerprint` and `scheduler_fingerprint` into the v2
resume bundle but `_restore_resume_state` only compared seed/background/link/background_policy/
mec_workers, so a checkpoint produced under a DIFFERENT observation schema, dataset, scheduler or
sampling contract resumed silently as if it were the same run. The comparison is now a pure
function (`run_tf._resume_mismatches`) and this module exercises it with synthetic bundles -- no
TensorFlow, no session, no checkpoint on disk.

Pure-Python (`python3 -m pytest ... -q -p no:nameko`); also runnable as
`python -m unittest env.mec_offloaing_envs.scheduler.tests.test_v2_f19_resume_contract`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2 import run_tf  # noqa: E402


class _Sampler:
    def __init__(self, cursor=0, contract="crn_token_keyed_v2"):
        self._draw_cursor = cursor
        self.sampling_contract_version = contract
        self.draw_branch = contract
        self.counters = {"support_calls": 0, "query_calls": 0}


class _Wrapper:
    """Forwarding wrapper: a WRITE here must not be mistaken for the sampler's state."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __getattr__(self, item):
        return getattr(self.sampler, item)


class _Trainer:
    def __init__(self, cursor=17, contract="crn_token_keyed_v2"):
        self.sampler = _Sampler(cursor=cursor, contract=contract)
        self.auto_sampler = _Wrapper(self.sampler)
        self.auto_obs_version = "automotive_v2_obs_v1"
        self.auto_dataset_fingerprint = "dataset-sha"
        self.auto_scheduler_fingerprint = "scheduler-sha"
        self.auto_protocol_id = "automotive_crn_gumbel_v1"
        self.auto_sampling_contract = contract
        self.auto_v2_system = {"link_regime": "moderate", "mec_workers": 1}
        self.auto_v2_constraints = {"enabled": True, "names": ["total_energy_mec"],
                                    "spec_sha256": "spec-sha",
                                    "objectives": {"primary": "constrained_latency"}}
        self.auto_v2_constraint_manager = None
        self.started_iterations = 4
        self.completed_iterations = 4
        self.last_completed_iteration = 3
        self.next_iteration = 4
        self.outer_updates_completed = 4
        self.dual_updates_completed = 4
        self.validations_completed = 0
        self.best_val_objective = None


def _args(**over):
    args = argparse.Namespace(seed=0, background=2, background_policy="all_mec",
                              link_regime="moderate", mec_workers=1, iterations=8,
                              r_select=2, s_select=2, validation_interval=50,
                              sampling_contract=run_tf._sampling_contract_default())
    for key, value in over.items():
        setattr(args, key, value)
    return args


def _bundle(trainer, args, **over):
    raw = {
        "schema": run_tf.RESUME_SCHEMA,
        "seed": int(args.seed),
        "background_dags": int(args.background),
        "background_policy": str(args.background_policy),
        "link_regime": str(args.link_regime),
        "mec_workers": int(args.mec_workers),
        "obs_version": trainer.auto_obs_version,
        "dataset_fingerprint": trainer.auto_dataset_fingerprint,
        "scheduler_fingerprint": trainer.auto_scheduler_fingerprint,
        "sampling_contract": run_tf._sampling_contract_of(trainer, args),
        "validation_interval": int(args.validation_interval),
        "objective_contract": run_tf._objective_contract_of(trainer),
        "constraint_contract": run_tf._constraint_contract_of(trainer),
        "iteration_target": 4,
    }
    raw.update(over)
    return raw


class TestResumeContractComparison(unittest.TestCase):
    def test_matching_contract_has_no_mismatch(self):
        trainer = _Trainer()
        args = _args()
        self.assertEqual(run_tf._resume_mismatches(_bundle(trainer, args), args, trainer), [])

    def test_obs_version_change_is_rejected_by_name(self):
        trainer = _Trainer()
        args = _args()
        raw = _bundle(trainer, args, obs_version="automotive_mc_obs_v1")
        mismatches = run_tf._resume_mismatches(raw, args, trainer)
        self.assertTrue(mismatches, "a different observation schema must be refused")
        self.assertTrue(any(m.startswith("obs_version:") for m in mismatches), mismatches)
        self.assertIn("automotive_mc_obs_v1", mismatches[0])
        self.assertIn("automotive_v2_obs_v1", mismatches[0])

    def test_dataset_and_scheduler_fingerprints_are_compared(self):
        trainer = _Trainer()
        args = _args()
        raw = _bundle(trainer, args, dataset_fingerprint="other-dataset",
                      scheduler_fingerprint="other-scheduler")
        mismatches = run_tf._resume_mismatches(raw, args, trainer)
        self.assertTrue(any(m.startswith("dataset_fingerprint:") for m in mismatches), mismatches)
        self.assertTrue(any(m.startswith("scheduler_fingerprint:") for m in mismatches),
                        mismatches)

    def test_sampling_contract_change_is_rejected(self):
        """A resume that asks for a DIFFERENT sampling contract than the checkpoint ran under is
        a different run: the draws would not be reproducible from the stored cursor."""
        # the checkpoint ran under the v2 contract; this process REQUESTS the legacy one
        checkpoint_trainer = _Trainer(contract="crn_token_keyed_v2")
        raw = _bundle(checkpoint_trainer, _args(sampling_contract="crn_token_keyed_v2"))
        current_trainer = _Trainer(contract="legacy_stateful_categorical")
        args = _args(sampling_contract="legacy_stateful_categorical")
        mismatches = run_tf._resume_mismatches(raw, args, current_trainer)
        self.assertTrue(any(m.startswith("sampling_contract:") for m in mismatches), mismatches)
        self.assertIn("legacy_stateful_categorical", mismatches[0])
        # ... and the reverse direction is refused too
        raw2 = _bundle(_Trainer(contract="legacy_stateful_categorical"),
                       _args(sampling_contract="legacy_stateful_categorical"))
        self.assertTrue(run_tf._resume_mismatches(
            raw2, _args(), _Trainer(contract="crn_token_keyed_v2")))

    def test_the_resume_contract_is_selectable_on_the_command_line(self):
        """`--sampling-contract` makes the contract an explicit, comparable request whose default
        is the v2 token-keyed contract the stack already runs."""
        self.assertEqual(run_tf._sampling_contract_default(), "crn_token_keyed_v2")
        self.assertIn("--sampling-contract", run_tf._build_arg_parser().format_help())

    def test_manifest_sampling_contract_is_surfaced(self):
        """The generation manifest records `sampling_contract`; a manifest that disagrees with
        its own bundle is a mismatch of its own, never silently resolved."""
        trainer = _Trainer()
        args = _args()
        raw = _bundle(trainer, args)
        manifest = {"sampling_contract": "some_other_contract"}
        mismatches = run_tf._resume_mismatches(raw, args, trainer, manifest)
        self.assertTrue(any("manifest=" in m for m in mismatches), mismatches)

    def test_validation_interval_and_objective_contract_are_compared(self):
        trainer = _Trainer()
        args = _args(validation_interval=10)
        raw = _bundle(trainer, args, validation_interval=50)
        mismatches = run_tf._resume_mismatches(raw, args, trainer)
        self.assertTrue(any(m.startswith("validation_interval:") for m in mismatches), mismatches)
        raw = _bundle(trainer, args, objective_contract="deadbeef")
        mismatches = run_tf._resume_mismatches(raw, args, trainer)
        self.assertTrue(any(m.startswith("objective_contract:") for m in mismatches), mismatches)
        raw = _bundle(trainer, args, constraint_contract="deadbeef")
        mismatches = run_tf._resume_mismatches(raw, args, trainer)
        self.assertTrue(any(m.startswith("constraint_contract:") for m in mismatches), mismatches)

    def test_extra_manifest_keys_do_not_break_an_old_bundle(self):
        """A bundle written before a key existed must still resume: absent keys compare clean."""
        trainer = _Trainer()
        args = _args()
        raw = _bundle(trainer, args)
        for key in ("obs_version", "dataset_fingerprint", "scheduler_fingerprint",
                    "sampling_contract", "validation_interval", "objective_contract",
                    "constraint_contract"):
            raw.pop(key, None)
        self.assertEqual(run_tf._resume_mismatches(raw, args, trainer), [])


class TestRestoreRejections(unittest.TestCase):
    def _write(self, doc):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "v2_resume_state.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
        return path

    def test_legacy_v1_bundle_is_refused_with_an_explanation(self):
        trainer = _Trainer()
        args = _args()
        path = self._write({"schema": run_tf.RESUME_SCHEMA_V1, "seed": 0})
        with self.assertRaises(run_tf.V2RunError) as ctx:
            run_tf._restore_resume_state(trainer, path, args)
        message = str(ctx.exception)
        self.assertIn(run_tf.RESUME_SCHEMA_V1, message)
        self.assertIn("--warm-start-weights-from", message)
        self.assertIn("world sequence", message)

    def test_unknown_schema_is_refused(self):
        trainer = _Trainer()
        args = _args()
        path = self._write({"schema": "v2_resume_state_v99"})
        with self.assertRaises(run_tf.V2RunError):
            run_tf._restore_resume_state(trainer, path, args)

    def test_incompatible_schema_is_refused_with_key_and_values(self):
        trainer = _Trainer()
        args = _args()
        raw = _bundle(trainer, args, obs_version="automotive_mc_obs_v1")
        path = self._write(raw)
        with self.assertRaises(run_tf.V2RunError) as ctx:
            run_tf._restore_resume_state(trainer, path, args)
        message = str(ctx.exception)
        self.assertIn("obs_version", message)
        self.assertIn("automotive_mc_obs_v1", message)
        self.assertIn("automotive_v2_obs_v1", message)

    def test_resume_restores_the_ledger_and_the_draw_cursor_owner(self):
        """The ledger and the cursor are applied, the cursor on the SAMPLER (not the wrapper)."""
        trainer = _Trainer(cursor=17)
        args = _args()
        raw = _bundle(trainer, args)
        raw.update({"draw_cursor": 42, "draw_cursor_path": "auto_sampler.sampler._draw_cursor",
                    "last_completed_iteration": 6, "completed_iterations": 7, "next_iteration": 7,
                    "started_iterations": 7, "outer_updates_completed": 7,
                    "dual_updates_completed": 7, "validations_completed": 1,
                    "iteration_target": 7})
        path = self._write(raw)
        state = run_tf._restore_resume_state(trainer, path, args)
        self.assertEqual(trainer.next_iteration, 7)
        self.assertEqual(trainer.start_itr, 7)
        self.assertEqual(trainer.completed_iterations, 7)
        self.assertEqual(trainer.validations_completed, 1)
        self.assertEqual(trainer.sampler._draw_cursor, 42)
        self.assertNotIn("_draw_cursor", vars(trainer.auto_sampler),
                         "the cursor must not be written to the forwarding wrapper")
        # 7 completed of a stored target of 7 resumed at 8: the target was increased, which is
        # PERMITTED (the total target is what the caller asks for)
        self.assertTrue(state["target_increased"])
        self.assertEqual(state["stored_iteration_target"], 7)

    def test_total_target_increase_is_permitted(self):
        trainer = _Trainer()
        args = _args(iterations=12)
        raw = _bundle(trainer, args, iteration_target=4, next_iteration=4,
                      completed_iterations=4, last_completed_iteration=3)
        path = self._write(raw)
        state = run_tf._restore_resume_state(trainer, path, args)
        self.assertTrue(state["target_increased"])
        self.assertEqual(state["stored_iteration_target"], 4)
        self.assertEqual(trainer.start_itr, 4)

    def test_target_below_completed_work_is_refused(self):
        trainer = _Trainer()
        args = _args(iterations=3)
        raw = _bundle(trainer, args, next_iteration=5, completed_iterations=5)
        path = self._write(raw)
        with self.assertRaises(run_tf.V2RunError) as ctx:
            run_tf._restore_resume_state(trainer, path, args)
        self.assertIn("past the requested total target", str(ctx.exception))

    def test_restore_writes_the_cursor_to_a_deeper_wrapper_owner(self):
        """`trainer.auto_sampler.sampler` is the v2 layout; the old two-level probe missed it."""
        trainer = _Trainer(cursor=5)
        trainer.sampler = _Wrapper(trainer.sampler)          # three levels now
        args = _args()
        path = self._write(_bundle(trainer, args, draw_cursor=99,
                                   draw_cursor_path="auto_sampler.sampler.sampler._draw_cursor"))
        run_tf._restore_resume_state(trainer, path, args)
        self.assertEqual(trainer.sampler.sampler._draw_cursor, 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
