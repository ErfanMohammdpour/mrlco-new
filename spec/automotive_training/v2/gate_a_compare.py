#!/usr/bin/env python3
"""Gate A comparison: control vs fresh-process resume across a scheduled validation.

Reads the two run directories produced by `kish_gate_a.sh` and compares, iteration by iteration,
the per-iteration ledger the trainer writes (`<run_dir>/v2_iterations.jsonl`, where `run_dir` is
the PARENT of the checkpoint directory) plus the two run reports and the isolated evaluation of
the final post-update checkpoint.

    python3 gate_a_compare.py <gate_a_output_dir>

Writes `GATE_A_VERDICT.json` and prints the verdict. Exact match means every compared iteration
agrees on energy, penalty, duals, world realization and the completed counters, and both
processes reached the requested total target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _rows(run_dir: Path):
    path = run_dir / "v2_iterations.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _rel(a, b):
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _report(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv):
    out = Path(argv[1] if len(argv) > 1 else ".")
    control = _rows(out / "control")
    resume = _rows(out / "split")
    by_control = {int(r["iteration"]): r for r in control if "iteration" in r}
    by_resume = {int(r["iteration"]): r for r in resume if "iteration" in r}

    compared, worst = [], 0.0
    for itr in sorted(set(by_control) & set(by_resume)):
        c, s = by_control[itr], by_resume[itr]
        cd, sd = c.get("iteration_delta") or {}, s.get("iteration_delta") or {}
        row = {
            "iteration": int(itr),
            "control_system_joules": c.get("mean_system_joules"),
            "resume_system_joules": s.get("mean_system_joules"),
            "control_penalty_sum": cd.get("penalty_sum"),
            "resume_penalty_sum": sd.get("penalty_sum"),
            "lambdas_equal": bool(c.get("lambdas") == s.get("lambdas")),
            "world_equal": bool(c.get("world_realization") == s.get("world_realization")),
            "counters_equal": bool(
                c.get("outer_updates_completed") == s.get("outer_updates_completed")
                and c.get("validations_completed") == s.get("validations_completed")),
        }
        row["energy_rel_diff"] = _rel(row["control_system_joules"], row["resume_system_joules"])
        row["penalty_rel_diff"] = _rel(row["control_penalty_sum"], row["resume_penalty_sum"])
        for key in ("energy_rel_diff", "penalty_rel_diff"):
            if row[key] is not None:
                worst = max(worst, row[key])
        compared.append(row)

    cj = _report(out / "control.json")
    sj = _report(out / "split.json")
    ej = _report(out / "control_eval.json")
    control_target = int(cj.get("iterations_requested") or 0)
    resume_target = int(sj.get("iterations_requested") or 0)

    exact = bool(compared) and worst <= 1e-12 and all(
        r["lambdas_equal"] and r["world_equal"] and r["counters_equal"] for r in compared)
    reached = (int(cj.get("iterations_completed") or 0) == control_target == 6
               and int(sj.get("iterations_completed") or 0) == resume_target == 6)
    verdict = {
        "schema": "v2_gate_a_verdict_v1",
        "control_completed": cj.get("iterations_completed"),
        "resume_completed": sj.get("iterations_completed"),
        "control_target": cj.get("iterations_requested"),
        "resume_target": sj.get("iterations_requested"),
        "control_error": cj.get("error"),
        "resume_error": sj.get("error"),
        "compared_iterations": compared,
        "n_compared": len(compared),
        "worst_relative_difference": worst,
        "exact_match_all_compared_iterations": exact,
        "both_reached_total_target_6": reached,
        "isolated_eval_present": bool(ej),
        "isolated_eval_start_iteration": ej.get("start_iteration"),
        "isolated_eval_completed_iterations": ej.get("completed_iterations"),
        "isolated_eval_k3_latency_s": (ej.get("k3") or {}).get("query_mean_latency_s"),
        "PASS": bool(exact and reached and ej),
    }
    (out / "GATE_A_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print("GATE A: %s" % ("PASS" if verdict["PASS"] else "FAIL"))
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
