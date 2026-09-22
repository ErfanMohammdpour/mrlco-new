"""Stop-rule watchdog for the ⑥b mask sanity runs.

The 500-iteration runs are unattended for ~a day, so the agreed stop conditions
are enforced while they run instead of only at the end:

    every metric column present      every row the same field count as the header
    every metric finite              all five control rates exactly 0.0
    critic/value_abs_max < 1e3       no stall (row age below the threshold)
    the run did not disappear without writing its exit file

On a violation it writes `watchdog_failure.json` into the run directory, kills the
matching container and exits non-zero. It also keeps `watchdog_status.json`
updated, so a single `cat` answers "how far did it get".

    python -m spec.mask_run_watchdog --mode static --interval 60
    python -m spec.mask_run_watchdog --mode off --dry-run      # never kills

The CSV checks are pure functions so they can be tested without docker.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from spec.mask_sanity import METRIC_KEYS, VALUE_ABS_MAX_LIMIT, ZERO_RATE_KEYS, run_dir

IMAGE = "margo-phase4-tf115-nv2212"
DEFAULT_INTERVAL = 60.0
DEFAULT_STALL_MINUTES = 30.0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("off", "static"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default="runs/mask_sanity_v3")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--stall-minutes", type=float, default=DEFAULT_STALL_MINUTES)
    parser.add_argument("--dry-run", action="store_true", help="report, never kill")
    parser.add_argument("--once", action="store_true", help="single check")
    return parser.parse_args(argv)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def check_rows(header, rows):
    """Violations for one CSV snapshot (pure: no filesystem, no docker)."""
    violations = []
    if not header:
        # startup: the file exists but nothing has been logged yet. A run that
        # never logs anything is caught by the stall and disappearance rules.
        return []
    if len(set(header)) != len(header):
        violations.append({"check": "csv_header_duplicate_keys", "detail": header})
    if any(str(k) == "" for k in header):
        violations.append({"check": "csv_header_empty_key", "detail": header})
    for key in METRIC_KEYS:
        if key not in header:
            violations.append({"check": "metric_column_missing", "detail": key})
    if violations:
        return violations

    for index, row in enumerate(rows):
        if len(row) != len(header):
            violations.append(
                {
                    "check": "row_field_count",
                    "detail": {"row": index, "fields": len(row), "header": len(header)},
                }
            )
            continue
        values = dict(zip(header, row))
        for key in METRIC_KEYS:
            number = _finite(values.get(key))
            if number is None:
                violations.append(
                    {"check": "metric_not_finite", "detail": {"row": index, "key": key,
                                                              "value": values.get(key)}}
                )
                continue
            if key in ZERO_RATE_KEYS and number != 0.0:
                violations.append(
                    {"check": "control_rate_nonzero",
                     "detail": {"row": index, "key": key, "value": number}}
                )
            if key == "critic/value_abs_max" and number >= VALUE_ABS_MAX_LIMIT:
                violations.append(
                    {"check": "value_abs_max_too_large",
                     "detail": {"row": index, "value": number}}
                )
    return violations


def stall_violation(rows, age_seconds, stall_minutes):
    """A run that has logged at least one row and then gone quiet."""
    if rows <= 0:
        return None            # startup can legitimately take ~25 minutes
    if age_seconds is None:
        return None
    if age_seconds > float(stall_minutes) * 60.0:
        return {
            "check": "stalled",
            "detail": {"rows": rows, "age_seconds": round(age_seconds, 1),
                       "stall_minutes": stall_minutes},
        }
    return None


def disappeared_violation(rows, containers, finished, started_at, now, grace_minutes=15.0):
    """A run that vanished without an exit file.

    Startup may take ~25 minutes and the watchdog can start a moment before the
    container exists, so a run that has produced no rows gets a grace period
    measured from its own recorded start time.
    """
    if containers or finished:
        return None
    detail = {"csv_rows": rows}
    if rows > 0:
        return {"check": "run_disappeared", "detail": detail}
    if started_at is None:
        return None
    if (now - started_at) > float(grace_minutes) * 60.0:
        detail["grace_minutes"] = grace_minutes
        return {"check": "run_disappeared", "detail": detail}
    return None


def read_start_time(rd):
    """Epoch seconds of the run's recorded start, or None."""
    path = rd / "config.resolved.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    stamp = payload.get("started_at")
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_snapshot(csv_path):
    if not csv_path.is_file():
        return [], [], None
    mtime = csv_path.stat().st_mtime
    with csv_path.open() as handle:
        raw = list(csv.reader(handle))
    if not raw:
        return [], [], mtime
    return raw[0], raw[1:], mtime


def matches_mode(command_text, mode):
    """True when a container's argument string runs the given mask mode."""
    text = " ".join(str(command_text or "").split())
    return ("--mode %s" % mode) in text


def _container_args(cid):
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{join .Config.Cmd \" \"}}", cid],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def find_container(mode):
    """Container ids running spec.mask_sanity in `mode`.

    `docker ps --format {{.Command}}` shows the image ENTRYPOINT (the nvidia
    wrapper), so the mode has to be read from `.Config.Cmd` instead.
    """
    proc = subprocess.run(
        ["docker", "ps", "--filter", "ancestor=%s" % IMAGE, "--format", "{{.ID}}"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return []
    hits = []
    for cid in proc.stdout.split():
        if matches_mode(_container_args(cid), mode):
            hits.append(cid.strip())
    return hits


def kill_containers(ids):
    killed = []
    for cid in ids:
        proc = subprocess.run(["docker", "kill", cid], capture_output=True, text=True,
                              check=False)
        killed.append({"id": cid, "rc": proc.returncode, "stderr": proc.stderr.strip()})
    return killed


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def check_once(args):
    rd = run_dir(args.mode, args.seed, args.runs_root)
    csv_path = rd / "logs" / "progress.csv"
    exit_file = Path("/tmp/ts_%s.exit" % args.mode)

    header, rows, mtime = read_snapshot(csv_path)
    age = None if mtime is None else max(0.0, time.time() - mtime)
    violations = check_rows(header, rows)
    stalled = stall_violation(len(rows), age, args.stall_minutes)
    if stalled:
        violations.append(stalled)

    containers = find_container(args.mode)
    finished = exit_file.is_file()
    gone = disappeared_violation(
        len(rows), containers, finished, read_start_time(rd), time.time()
    )
    if gone:
        violations.append(gone)

    status = {
        "mode": args.mode,
        "checked_at": now_iso(),
        "csv_rows": len(rows),
        "csv_fields": len(header),
        "row_age_seconds": None if age is None else round(age, 1),
        "containers": containers,
        "exit_file": str(exit_file) if finished else None,
        "violations": violations,
    }
    if rows:
        last = dict(zip(header, rows[-1]))
        status["last_row"] = {k: last.get(k) for k in METRIC_KEYS if k in last}

    if not violations:
        status["verdict"] = "ok"
        if finished and not containers:
            status["verdict"] = "finished"
        write_json(rd / "watchdog_status.json", status)
        return status

    status["verdict"] = "violation"
    status["killed"] = []
    if not args.dry_run:
        status["killed"] = kill_containers(containers)
        write_json(rd / "watchdog_failure.json", status)
    write_json(rd / "watchdog_status.json", status)
    return status


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    while True:
        status = check_once(args)
        print(json.dumps(status, sort_keys=True, default=str), flush=True)
        if status["verdict"] == "violation":
            return 2
        if status["verdict"] == "finished" or args.once:
            return 0
        time.sleep(max(5.0, float(args.interval)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
