#!/usr/bin/env python3
"""Pull-based audit feed for a MARGO branch.

Prints commits that appeared on the remote branch since the last time this
script looked, then records the new head. It exists so an external review can
follow `phase4-eval` without a GitHub token, a webhook or CI.

    python -m spec.branch_watch                      # what is new?
    python -m spec.branch_watch --remote origin      # upstream MARGO instead
    python -m spec.branch_watch --watch 600          # poll every 10 min
    python -m spec.branch_watch --state /tmp/w.json  # where the marker lives

The head is always FETCHED into `refs/remotes/<remote>/<branch>` before the log
is computed: `ls-remote` only reveals a SHA, so logging it without fetching the
object would silently report "no new commits" and then advance the marker,
losing those commits forever.

State is a JSON file with the last seen SHA per branch; the default lives under
`runs/branch_watch/`, which is run output, not source. The marker is only
advanced after the report is computed successfully. Exit code is 0 whether or
not new commits exist -- parse the JSON if you need to branch on it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATE = "runs/branch_watch/branch_watch.json"
MAX_LISTED = 100


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
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), proc.stderr.strip()))
    return proc.stdout.strip()


def _ref_slug(remote):
    """A remote name usable inside a ref (remote may be a path or URL)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(remote)).strip("_") or "remote"


def tracking_ref(remote, branch):
    return "refs/remotes/%s/%s" % (_ref_slug(remote), branch)


def fetch_head(remote, branch):
    """Fetch the branch head into its tracking ref. Returns (head_sha, fetched).

    A failed fetch is fatal even when a STALE tracking ref still resolves: the
    stale SHA would compare equal to the recorded marker and the run would report
    "no new commits" while the remote moved on.
    """
    ref = tracking_ref(remote, branch)
    proc = subprocess.run(
        [
            "git",
            "fetch",
            "--quiet",
            "--no-tags",
            remote,
            "+refs/heads/%s:%s" % (branch, ref),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git fetch %s %s failed: %s" % (remote, branch, proc.stderr.strip())
        )
    head = _git("rev-parse", "--verify", ref + "^{commit}", check=False)
    if not head:
        raise RuntimeError(
            "fetch of %s %s succeeded but %s does not resolve" % (remote, branch, ref)
        )
    return head, True


def is_ancestor(ancestor, descendant):
    """True/False when known; None when git cannot decide (unknown object)."""
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def log_entries(rev_range=None, rev=None):
    args = ["log", "--oneline", "--no-decorate", "--max-count=%d" % (MAX_LISTED + 1)]
    if rev_range:
        args.append(rev_range)
    elif rev:
        args.append(rev)
    else:
        return []
    out = _git(*args, check=False)
    entries = []
    for line in out.splitlines():
        sha, _, subject = line.partition(" ")
        if sha:
            entries.append({"sha": sha, "subject": subject})
    return entries


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


def check(remote, branch, state, state_path):
    key = "%s/%s" % (remote, branch)
    previous = (state.get(key) or {}).get("sha")
    head, fetched = fetch_head(remote, branch)

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "remote": remote,
        "branch": branch,
        "head": head,
        "previous": previous,
        "objects_fetched": bool(fetched),
        "is_first_look": previous is None,
        "history_rewritten": False,
        "new_commits": [],
    }

    if previous is None:
        listing = log_entries(rev=head)
        result["new_commits"] = listing
        result["note"] = "first look: showing the current head history as context"
        result["truncated"] = len(listing) > MAX_LISTED
        result["new_commits"] = listing[:MAX_LISTED]
        save_state(state_path, _advanced(state, key, head, result["checked_at"]))
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if previous == head:
        save_state(state_path, _advanced(state, key, head, result["checked_at"]))
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    ancestor = is_ancestor(previous, head)
    if ancestor is False:
        # force-push / rebase: a range is still computable, but flag it loudly
        result["history_rewritten"] = True
        result["note"] = (
            "previous %s is NOT an ancestor of head %s (force-push or rebase)"
            % (previous, head)
        )
    listing = log_entries(rev_range="%s..%s" % (previous, head))
    result["truncated"] = len(listing) > MAX_LISTED
    result["new_commits"] = listing[:MAX_LISTED]
    save_state(state_path, _advanced(state, key, head, result["checked_at"]))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _advanced(state, key, head, checked_at):
    state = dict(state)
    state[key] = {"sha": head, "checked_at": checked_at}
    return state


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    while True:
        state = load_state(args.state)
        try:
            check(args.remote, args.branch, state, args.state)
        except RuntimeError as exc:
            # never advance the marker on failure: the next run retries
            print(json.dumps({"error": str(exc), "state_advanced": False}, indent=2))
            if args.watch is None:
                return 1
        if args.watch is None:
            return 0
        time.sleep(max(5.0, float(args.watch)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
