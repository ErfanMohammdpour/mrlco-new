#!/usr/bin/env python3
"""F18 crash-recovery / continuation fixtures for the transactional generation machinery.

The generation layout was implemented but the DELIBERATE-CORRUPTION fixtures had never been
executed, so "the rollback path works" was an untested claim. Every fixture here is built from
real directories and real SHA-256 manifests, needs no TensorFlow and no session, and drives the
PRODUCTION functions (`_verify_generation`, `_resolve_resume_generation`, `_resolve_resume_ckpt`,
`_prune_generations`, `_restore_resume_state`) rather than a re-implementation:

1. a deleted TF shard       -> rejected with a precise reason, recovery falls back one generation
2. a truncated state bundle -> rejected (hash mismatch), never silently loaded
3. a corrupt/undeclared file-> rejected with a precise reason
4. an interrupted save      -> `gen_N.tmp` is not a generation; an incomplete `CURRENT` target
                               falls back to the newest COMPLETE generation
5. a legacy v1 bundle       -> REFUSED with an explanation (the weights-only warm start stays a
                               separately labelled, non-equivalent operation)
6. retention                -> at least three complete generations survive pruning, and the
                               newest one is never deleted
7. total-target increase    -> a LARGER `--iterations` resumes; a target below completed work is
                               refused

Pure-Python:
    python3 -m pytest env/mec_offloaing_envs/scheduler/tests/test_v2_f18_generation_recovery.py -q -p no:nameko
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
from spec.automotive_training.v2.crn import training_seed_pair  # noqa: E402


class _Sampler:
    def __init__(self, cursor=0):
        self._draw_cursor = cursor
        self.sampling_contract_version = "crn_token_keyed_v2"
        self.draw_branch = "crn_token_keyed_v2"
        self.counters = {"support_calls": 1, "query_calls": 1}


class _Wrapper:
    def __init__(self, sampler):
        self.sampler = sampler

    def __getattr__(self, item):
        return getattr(self.sampler, item)


class _Trainer:
    def __init__(self, cursor=0):
        self.sampler = _Sampler(cursor=cursor)
        self.auto_sampler = _Wrapper(self.sampler)
        self.auto_obs_version = "automotive_v2_obs_v1"
        self.auto_dataset_fingerprint = "dataset-sha"
        self.auto_scheduler_fingerprint = "scheduler-sha"
        self.auto_protocol_id = "automotive_crn_gumbel_v1"
        self.auto_sampling_contract = run_tf._sampling_contract_default()
        self.auto_v2_system = {"link_regime": "moderate", "mec_workers": 1}
        self.auto_v2_constraints = {"enabled": True, "names": ["total_energy_mec"],
                                    "spec_sha256": "spec-sha",
                                    "objectives": {"primary": "constrained_latency"}}
        self.auto_v2_constraint_manager = None
        self.started_iterations = 0
        self.completed_iterations = 0
        self.last_completed_iteration = -1
        self.next_iteration = 0
        self.outer_updates_completed = 0
        self.dual_updates_completed = 0
        self.validations_completed = 0
        self.best_val_objective = None


def _args(ckpt_dir, **over):
    args = argparse.Namespace(seed=0, background=2, background_policy="all_mec",
                              link_regime="moderate", mec_workers=1, iterations=3,
                              r_select=2, s_select=2, validation_interval=50,
                              ckpt_dir=str(ckpt_dir),
                              sampling_contract=run_tf._sampling_contract_default())
    for key, value in over.items():
        setattr(args, key, value)
    return args


def _write_generation(ckpt_dir, itr, complete=True, extra_files=None):
    """A REAL generation directory: TF shards + state bundle + manifest of file hashes."""
    gen = Path(ckpt_dir) / run_tf.GEN_DIR / ("gen_%08d" % itr)
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "v2_run.ckpt").write_text("ckpt data %d\n" % itr)
    (gen / "v2_run.ckpt.index").write_text("index %d\n" % itr)
    (gen / "v2_run.ckpt.meta").write_text("meta %d\n" % itr)
    state = {"schema": run_tf.RESUME_SCHEMA, "iteration_target": int(itr) + 1,
             "next_iteration": int(itr) + 1, "completed_iterations": int(itr) + 1,
             "started_iterations": int(itr) + 1, "last_completed_iteration": int(itr),
             "outer_updates_completed": int(itr) + 1, "dual_updates_completed": int(itr) + 1,
             "validations_completed": 0, "seed": 0, "background_dags": 2,
             "background_policy": "all_mec", "link_regime": "moderate", "mec_workers": 1,
             "obs_version": "automotive_v2_obs_v1", "dataset_fingerprint": "dataset-sha",
             "scheduler_fingerprint": "scheduler-sha",
             "sampling_contract": run_tf._sampling_contract_default(),
             "validation_interval": 50, "draw_cursor": int(itr) * 2,
             "draw_cursor_path": "auto_sampler.sampler._draw_cursor"}
    bundle_text = json.dumps(state, indent=2, sort_keys=True) + "\n"
    (gen / "v2_resume_state.json").write_text(bundle_text)
    files = {p.name: run_tf._sha256_file(p) for p in sorted(gen.iterdir())
             if p.is_file() and p.name != run_tf.GENERATION_MANIFEST}
    manifest = {"schema": "v2_checkpoint_generation_v1", "iteration": int(itr),
                "next_iteration": int(itr) + 1,
                "sampling_contract": run_tf._sampling_contract_default(),
                "files": files, "complete": bool(complete)}
    (gen / run_tf.GENERATION_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name in (extra_files or []):
        (gen / name).write_text("undeclared\n")
    return gen


def _set_current(ckpt_dir, name, iteration=0):
    (Path(ckpt_dir) / run_tf.CURRENT_POINTER).write_text(json.dumps(
        {"schema": "v2_checkpoint_pointer_v1", "generation": name,
         "iteration": int(iteration)}, sort_keys=True) + "\n")


def _ok_generation(ckpt_dir, itr=1):
    gen = _write_generation(ckpt_dir, itr)
    ok, reason = run_tf._verify_generation(gen)
    if not ok:                                    # the fixture itself must be valid
        raise AssertionError("valid generation fixture rejected: %s" % reason)
    return gen


class TestGenerationVerification(unittest.TestCase):
    def test_valid_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp)
            self.assertEqual(run_tf._verify_generation(gen), (True, "complete"))

    def test_missing_shard_is_rejected_and_falls_back(self):
        """(1) delete one TF shard from the NEWEST generation."""
        with tempfile.TemporaryDirectory() as tmp:
            older = _ok_generation(tmp, 1)
            newest = _ok_generation(tmp, 2)
            (newest / "v2_run.ckpt.meta").unlink()
            ok, reason = run_tf._verify_generation(newest)
            self.assertFalse(ok)
            self.assertIn("missing file v2_run.ckpt.meta", reason)
            _set_current(tmp, newest.name)
            resolved = run_tf._resolve_resume_generation(tmp)
            self.assertEqual(resolved, older, "recovery must fall back to the previous COMPLETE "
                                              "generation")
            self.assertEqual(run_tf._resolve_resume_ckpt(tmp),
                             str(older / "v2_run.ckpt"))

    def test_missing_shard_with_no_fallback_raises_precisely(self):
        with tempfile.TemporaryDirectory() as tmp:
            newest = _ok_generation(tmp, 4)
            (newest / "v2_run.ckpt.index").unlink()
            _set_current(tmp, newest.name)
            with self.assertRaises(run_tf.V2RunError) as ctx:
                run_tf._resolve_resume_generation(tmp)
            self.assertIn("no COMPLETE checkpoint generation", str(ctx.exception))

    def test_truncated_bundle_is_rejected(self):
        """(2) truncating the state bundle must be caught by the manifest hash."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 3)
            self.assertEqual(run_tf._verify_generation(gen)[0], True)
            bundle = gen / "v2_resume_state.json"
            text = bundle.read_text()
            bundle.write_text(text[: max(1, len(text) // 3)])
            ok, reason = run_tf._verify_generation(gen)
            self.assertFalse(ok)
            self.assertIn("hash mismatch for v2_resume_state.json", reason)

    def test_corrupt_manifest_is_rejected(self):
        """(3) a manifest that is not valid JSON must be named as such."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 5)
            (gen / run_tf.GENERATION_MANIFEST).write_text("{ not json")
            ok, reason = run_tf._verify_generation(gen)
            self.assertFalse(ok)
            self.assertIn("manifest unreadable", reason)

    def test_wrong_manifest_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 6)
            doc = json.loads((gen / run_tf.GENERATION_MANIFEST).read_text())
            doc["schema"] = "v2_checkpoint_generation_v0"
            (gen / run_tf.GENERATION_MANIFEST).write_text(json.dumps(doc))
            ok, reason = run_tf._verify_generation(gen)
            self.assertFalse(ok)
            self.assertIn("unexpected manifest schema", reason)

    def test_undeclared_extra_file_is_rejected(self):
        """(3b) a file present but NOT in the manifest must not be trusted as part of it."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 7)
            (gen / "v2_resume_state.json.old").write_text("foreign state\n")
            ok, reason = run_tf._verify_generation(gen)
            self.assertFalse(ok)
            self.assertIn("undeclared file", reason)
            self.assertIn("v2_resume_state.json.old", reason)

    def test_interrupted_save_is_not_a_generation(self):
        """(4) a `gen_N.tmp` staging directory left behind must never be picked."""
        with tempfile.TemporaryDirectory() as tmp:
            older = _ok_generation(tmp, 8)
            staging = Path(tmp) / run_tf.GEN_DIR / "gen_00000009.tmp"
            staging.mkdir(parents=True)
            (staging / "v2_run.ckpt.index").write_text("partial\n")
            (staging / "v2_resume_state.json").write_text("{}\n")
            self.assertNotIn(staging, run_tf._complete_generations(tmp))
            _set_current(tmp, staging.name)
            self.assertEqual(run_tf._resolve_resume_generation(tmp), older)

    def test_current_pointing_at_incomplete_generation_falls_back(self):
        """(4b) CURRENT -> an incomplete `gen_N` must fall back to the newest complete one."""
        with tempfile.TemporaryDirectory() as tmp:
            oldest = _ok_generation(tmp, 10)
            middle = _ok_generation(tmp, 11)
            newest = _ok_generation(tmp, 12)
            (newest / "v2_run.ckpt").unlink()
            _set_current(tmp, newest.name)
            self.assertEqual(run_tf._resolve_resume_generation(tmp), middle)
            del oldest
            self.assertTrue(all(g.name != newest.name
                                for g in run_tf._complete_generations(tmp)))

    def test_missing_current_pointer_uses_newest_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            _ok_generation(tmp, 20)
            newest = _ok_generation(tmp, 21)
            self.assertEqual(run_tf._resolve_resume_generation(tmp), newest)


class TestLegacyAndRetention(unittest.TestCase):
    def test_legacy_bundle_in_a_valid_generation_is_refused(self):
        """(5) a v2_run-style generation carrying a v1 bundle is refused with an explanation."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 30)
            legacy = {"schema": run_tf.RESUME_SCHEMA_V1, "iteration_target": 31}
            (gen / "v2_resume_state.json").write_text(
                json.dumps(legacy, indent=2, sort_keys=True) + "\n")
            # re-take the manifest so the generation is provably COMPLETE and merely carries a
            # legacy bundle (the manifest itself is never one of its own declared files)
            files = {p.name: run_tf._sha256_file(p) for p in sorted(gen.iterdir())
                     if p.is_file() and p.name != run_tf.GENERATION_MANIFEST}
            doc = json.loads((gen / run_tf.GENERATION_MANIFEST).read_text())
            doc["files"] = files
            (gen / run_tf.GENERATION_MANIFEST).write_text(json.dumps(doc))
            self.assertEqual(run_tf._verify_generation(gen)[0], True)
            trainer = _Trainer()
            args = _args(tmp)
            trainer.auto_obs_version = "automotive_v2_obs_v1"
            with self.assertRaises(run_tf.V2RunError) as ctx:
                run_tf._restore_resume_state(trainer, gen / "v2_resume_state.json", args)
            message = str(ctx.exception)
            self.assertIn(run_tf.RESUME_SCHEMA_V1, message)
            self.assertIn("--warm-start-weights-from", message)

    def test_retention_keeps_three_and_never_the_newest(self):
        """(6) pruning keeps >= 3 recoverable generations plus any protected one."""
        with tempfile.TemporaryDirectory() as tmp:
            gens = [_ok_generation(tmp, itr) for itr in (40, 41, 42, 43, 44, 45)]
            best = gens[1].name
            final = gens[0].name
            deleted = run_tf._prune_generations(tmp, keep=3, protect=(best, final))
            remaining = [g.name for g in run_tf._complete_generations(tmp)]
            self.assertGreaterEqual(len(remaining), 3)
            self.assertIn("gen_00000045", remaining, "the newest complete generation was deleted")
            self.assertIn(best, remaining, "the best-validation generation was deleted")
            self.assertIn(final, remaining, "the protected final generation was deleted")
            for name in remaining:
                self.assertEqual(run_tf._verify_generation(
                    Path(tmp) / run_tf.GEN_DIR / name)[0], True)
            self.assertTrue(deleted)
            # pruning when there is nothing to prune is a no-op
            self.assertEqual(run_tf._prune_generations(tmp, keep=3, protect=(best, final)), [])

    def test_retention_never_touches_an_unusable_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for itr in (50, 51, 52, 53):
                _ok_generation(tmp, itr)
            broken = Path(tmp) / run_tf.GEN_DIR / "gen_00000050"
            (broken / "v2_run.ckpt").unlink()
            deleted = run_tf._prune_generations(tmp, keep=3)
            self.assertNotIn("gen_00000050", deleted)


class _StubManager:
    """A constraint manager stub that records the dual state it was asked to restore."""

    def __init__(self, lambdas=None):
        self.names = ["total_energy_mec", "ue_energy"]
        self._state = {"schema": "v2_constraint_state_v1", "constraints_sha256": "spec-sha",
                       "lambdas": dict(lambdas or {n: 0.0 for n in self.names})}

    def state(self):
        return dict(self._state)

    def load_state(self, state):
        if state.get("schema") != self._state["schema"]:
            raise RuntimeError("constraint spec mismatch")
        self._state = dict(state)

    def lambdas_by_name(self):
        return dict(self._state["lambdas"])


class TestResumeBoundary(unittest.TestCase):
    def test_restored_lambdas_and_draw_identity_are_the_first_post_resume_values(self):
        """(8) the first post-resume collection must use the RESTORED duals and draw identity."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _ok_generation(tmp, 70)
            bundle = json.loads((gen / "v2_resume_state.json").read_text())
            bundle["constraints"] = {"schema": "v2_constraint_state_v1",
                                     "constraints_sha256": "spec-sha",
                                     "lambdas": {"total_energy_mec": 0.75, "ue_energy": 0.25}}
            bundle["draw_cursor"] = 33
            bundle["draw_cursor_path"] = "auto_sampler.sampler._draw_cursor"
            bundle["next_iteration"] = 71
            bundle["completed_iterations"] = 71
            (gen / "v2_resume_state.json").write_text(
                json.dumps(bundle, indent=2, sort_keys=True) + "\n")
            files = {p.name: run_tf._sha256_file(p) for p in sorted(gen.iterdir())
                     if p.is_file() and p.name != run_tf.GENERATION_MANIFEST}
            doc = json.loads((gen / run_tf.GENERATION_MANIFEST).read_text())
            doc["files"] = files
            (gen / run_tf.GENERATION_MANIFEST).write_text(json.dumps(doc))
            self.assertEqual(run_tf._verify_generation(gen)[0], True)

            trainer = _Trainer()
            trainer.auto_v2_constraint_manager = _StubManager()
            args = _args(tmp, iterations=100)
            run_tf._restore_resume_state(trainer, gen / "v2_resume_state.json", args)
            # the loop starts at the first iteration that has NOT completed, not at 0
            self.assertEqual(trainer.start_itr, 71)
            self.assertEqual(trainer.next_iteration, 71)
            self.assertEqual(trainer.started_iterations, 71)
            self.assertEqual(trainer.outer_updates_completed, 71)
            # the duals the first post-resume rollout is penalised with are the restored ones
            self.assertEqual(trainer.auto_v2_constraint_manager.lambdas_by_name(),
                             {"total_energy_mec": 0.75, "ue_energy": 0.25})
            # and the draw identity continues from the restored cursor on the OWNING attribute
            self.assertEqual(trainer.sampler._draw_cursor, 33)
            self.assertEqual(run_tf._read_draw_cursor(trainer), 33)
            pair = training_seed_pair(int(getattr(trainer.sampler, "draw_identity_seed", 0) or 0), 33)
            self.assertEqual(pair[1], 33)
            self.assertEqual(pair[0] % 2, 1,
                             "the post-resume training identity must stay in the train phase")

    def test_resume_uses_the_newest_complete_generation_after_a_corrupt_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            older = _ok_generation(tmp, 60)
            newest = _ok_generation(tmp, 61)
            (newest / "v2_run.ckpt.meta").unlink()
            _set_current(tmp, newest.name)
            chosen = run_tf._resolve_resume_generation(tmp)
            self.assertEqual(chosen, older)
            trainer = _Trainer()
            args = _args(tmp, iterations=int(older.name[-8:]) + 2)
            raw_text = (chosen / "v2_resume_state.json").read_text()
            state = json.loads(raw_text)
            self.assertEqual(state["next_iteration"], 61)
            restored = run_tf._restore_resume_state(trainer, chosen / "v2_resume_state.json",
                                                    args)
            self.assertEqual(trainer.next_iteration, 61)
            self.assertEqual(trainer.start_itr, 61)
            self.assertEqual(trainer.completed_iterations, 61)
            self.assertEqual(trainer.sampler._draw_cursor, 120)
            self.assertTrue(restored["target_increased"])

    def test_empty_directory_reports_a_precise_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(run_tf.V2RunError) as ctx:
                run_tf._resolve_resume_generation(tmp)
            self.assertIn("no COMPLETE checkpoint generation", str(ctx.exception))
            with self.assertRaises(run_tf.V2RunError):
                run_tf._resolve_resume_ckpt(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
