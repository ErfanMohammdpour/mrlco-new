#!/usr/bin/env python3
"""Pull-based audit feed for a MARGO branch.

Prints commits that appeared on the remote branch since the last time this
script looked, then records the new head. It exists so an external review can
follow `phase4-eval` without a GitHub token, a webhook or CI.

    python -m spec.branch_watch                      # what is new?
    python -m spec.branch_watch --remote origin      # upstream MARGO instead
    python -m spec.branch_watch --watch 600          # poll every 10 min
    python -m spec.branch_watch --state /tmp/w.json  # where the marker lives

State is a JSON file with the last seen SHA per branch; the default lives under
`runs/branch_watch/`, which is run output, not source. Exit code is 0 whether or
not new commits exist -- parse the JSON if you need to branch on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATE = "runs/branch_watch/branch_watch.json"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="erfan")
    parser.add_argument("--branch", default="phase4-eval")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help="keep polling (this is a foreground loop, run it yourself)",
    )
    return parser.parse_args(argv)


def _git(*args, check=True):
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            "git %s failed: %s" % (" ".join(args), proc.stderr.strip())
        )
    return proc.stdout.strip()


def remote_head(remote, branch):
    """SHA of the remote branch, without fetching objects."""
    out = _git("ls-remote", "--heads", remote, branch)
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.strip() == "refs/heads/%s" % branch:
            return sha.strip()
    raise RuntimeError("branch %s not found on remote %s" % (branch, remote))


def load_state(path):
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def commits_since(sha, head):
    """(sha, subject) pairs on head that are not reachable from sha."""
    if sha == head:
        return []
    line = _git("log", "--oneline", "--no-decorate", "%s..%s" % (sha, head), check=False)
    out = []
    for entry in line.splitlines():
        parts = entry.split(" ", 1)
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def check(remote, branch, state, state_path):
    head = remote_head(remote, branch)
    key = "%s/%s" % (remote, branch)
    previous = (state.get(key) or {}).get("sha")
    # a SHA may be unknown locally (fresh clone or new remote): fetch just enough
    if previous:
        have = _git("cat-file", "-e", previous + "^{commit}", check=False)
        if have == "" and _git("rev-parse", "--verify", previous, check=False) == "":
            _git("fetch", "--quiet", remote, branch, check=False)

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "remote": remote,
        "branch": branch,
        "head": head,
        "previous": previous,
        "is_first_look": previous is None,
        "new_commits": [],
    }
    if previous is None:
        result["new_commits"] = [
            {"sha": sha, "subject": subj}
            for sha, subj in [
                (line.split(" ", 1)[0], line.split(" ", 1)[1])
                for line in _git(
                    "log", "--oneline", "--no-decorate", "-n", "5", head, check=False
                ).splitlines()
                if " " in line
            ]
        ]
        result["note"] = "first look: showing the last 5 commits as context"
    else:
        result["new_commits"] = [
            {"sha": sha, "subject": subj} for sha, subj in commits_since(previous, head)
        ]

    state[key] = {"sha": head, "checked_at": result["checked_at"]}
    save_state(state_path, state)

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    state_path = args.state
    while True:
        state = load_state(state_path)
        try:
            check(args.remote, args.branch, state, state_path)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            if args.watch is None:
                return 1
        if args.watch is None:
            return 0
        time.sleep(max(5.0, float(args.watch)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
