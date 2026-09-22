#!/usr/bin/env python3
"""Regression tests for the phase4-eval audit feed (spec/branch_watch.py).

Lives next to the scheduler tests so the standard suite covers it. The bug these
guard against: `ls-remote` reveals a SHA without fetching its objects, so the
watcher reported "no new commits" and then advanced its marker, losing the commit
from the feed permanently. A second variant was a stale tracking ref making a
FAILED fetch look like "nothing new".

Everything runs against throwaway repositories under a tmp dir; nothing touches
the project's own git state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

WATCH = [
    sys.executable,
    "-m",
    "spec.branch_watch",
    "--remote",
    "{remote}",
    "--branch",
    "main",
    "--state",
    "{state}",
]


def _git(cwd, *args, check=True):
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise AssertionError("git %s failed: %s" % (" ".join(args), proc.stderr))
    return proc.stdout.strip()


@unittest.skipIf(shutil.which("git") is None, "git not available")
class TestBranchWatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bw_test_"))
        self.remote = self.tmp / "remote.git"
        self.work = self.tmp / "work"
        self.state = self.tmp / "state.json"
        _git(self.tmp, "init", "-q", "--bare", str(self.remote))
        _git(self.tmp, "init", "-q", str(self.work))
        _git(self.work, "config", "user.email", "t@example.com")
        _git(self.work, "config", "user.name", "tester")
        _git(self.work, "commit", "-q", "--allow-empty", "-m", "first")
        _git(self.work, "branch", "-M", "main")
        _git(self.work, "remote", "add", "origin", str(self.remote))
        _git(self.work, "push", "-q", "origin", "main")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, remote=None):
        argv = [
            arg.format(remote=remote or str(self.remote), state=str(self.state))
            for arg in WATCH
        ]
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        # run inside the scratch clone: the watcher shells out to git in its cwd,
        # and the project repository must not receive temp refs or objects
        return subprocess.run(argv, cwd=str(self.work), capture_output=True, text=True, env=env)

    def _report(self, proc):
        self.assertTrue(proc.stdout.strip(), "watcher printed nothing: %s" % proc.stderr)
        return json.loads(proc.stdout)

    def test_first_look_records_head(self):
        report = self._report(self._run())
        self.assertTrue(report["is_first_look"])
        self.assertTrue(report["head"])
        self.assertIn(report["head"], json.loads(self.state.read_text()).popitem()[1]["sha"])

    def test_new_commit_is_reported(self):
        """The original bug: a pushed commit must appear, not be swallowed."""
        self._run()
        _git(self.work, "commit", "-q", "--allow-empty", "-m", "second")
        _git(self.work, "push", "-q", "origin", "main")
        report = self._report(self._run())
        self.assertFalse(report["is_first_look"])
        subjects = [c["subject"] for c in report["new_commits"]]
        self.assertIn("second", subjects, report)
        self.assertFalse(report["history_rewritten"])

    def test_marker_only_advances_after_a_report(self):
        first = self._report(self._run())
        self._run()
        state = json.loads(self.state.read_text())
        self.assertEqual(list(state.values())[0]["sha"], first["head"])

    def test_force_push_is_flagged(self):
        self._run()
        _git(self.work, "commit", "-q", "--allow-empty", "-m", "second")
        _git(self.work, "push", "-q", "origin", "main")
        self._run()                                    # marker now at "second"
        _git(self.work, "reset", "-q", "--hard", "HEAD~1")
        _git(self.work, "commit", "-q", "--allow-empty", "-m", "rewritten")
        _git(self.work, "push", "-q", "-f", "origin", "main")
        report = self._report(self._run())
        self.assertTrue(report["history_rewritten"], report)
        self.assertIn("rewritten", [c["subject"] for c in report["new_commits"]])

    def test_failed_fetch_does_not_advance_the_marker(self):
        first = self._report(self._run())
        _git(self.work, "commit", "-q", "--allow-empty", "-m", "second")
        _git(self.work, "push", "-q", "origin", "main")
        missing = self.tmp / "gone.git"
        proc = self._run(remote=str(missing))
        self.assertNotEqual(proc.returncode, 0, "a failed fetch must not exit 0")
        self.assertIn("error", self._report(proc))
        state = json.loads(self.state.read_text())
        self.assertEqual(
            list(state.values())[0]["sha"],
            first["head"],
            "marker advanced although the fetch failed",
        )

    def test_stale_ref_plus_failed_fetch_is_still_an_error(self):
        """Second variant of the bug: a stale tracking ref must not look like 'no news'."""
        self._run()                                    # creates the tracking ref
        missing = self.tmp / "gone.git"
        proc = self._run(remote=str(missing))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("error", self._report(proc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
