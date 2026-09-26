#!/usr/bin/env python3
"""P1/P3: merge the per-label checkpoint evaluations into one persisted verdict.

`spec/evaluate_checkpoint.py` runs ONE label per process (full graph isolation)
and writes one `checkpoint_eval_v1` JSON per label. This module is the pure,
re-runnable step that turns those files into a single
`checkpoint_eval_comparison.json` with the true-init row, the in-training itr-0
row taken from the committed Pilot A progress CSV, and the checkpoint rows, plus
an explicit verdict.

Verdict vocabulary (fixed):
  * BLOCKED_CHECKPOINT_EVALUATION           - a correctness check failed
  * READY_FOR_LONG_LATENCY_DIAGNOSTIC       - improved AND k3 adaptation healthy
  * LONG_RUN_ALLOWED_BUT_NO_VALIDATION_IMPROVEMENT_YET - everything else;
    `detail` always states the concrete reason (e.g. "improved but k3 >= k0").

Nothing here trains, adapts or samples; it only reads JSON/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCHEMA = "checkpoint_eval_comparison_v1"
BASELINE_TOLERANCE = 1e-6
LABELS = (
    "BLOCKED_CHECKPOINT_EVALUATION",
    "READY_FOR_LONG_LATENCY_DIAGNOSTIC",
    "LONG_RUN_ALLOWED_BUT_NO_VALIDATION_IMPROVEMENT_YET",
)

CSV_COLUMNS = {
    "itr": "Itr",
    "k0": "validation_query_mean_latency_k0",
    "k3": "validation_query_mean_latency_k3",
    "all_mec": "validation/validation_all_mec_latency",
    "greedy": "validation/validation_greedy_latency",
}


def _number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def label_row(label: str, doc: dict) -> dict:
    """One comparison row from one `checkpoint_eval_v1` document."""
    checkpoints = doc.get("checkpoints") or {}
    if label not in checkpoints:
        raise KeyError("label %r missing from the evaluation document" % label)
    entry = checkpoints[label]
    k0 = _number(entry.get("k0", {}).get("query_mean_latency"))
    k3 = _number(entry.get("k3", {}).get("query_mean_latency"))
    all_mec = _number(entry.get("k3", {}).get("query_all_mec_latency"))
    greedy = _number(entry.get("k3", {}).get("query_greedy_latency"))
    return {
        "label": label,
        "source": "checkpoint_eval",
        "checkpoint_sha256": entry.get("checkpoint_sha256"),
        "weights_changed": entry.get("weights_changed"),
        "deterministic_k0_fresh": doc.get("deterministic_k0_fresh"),
        "k0": k0,
        "k3": k3,
        "all_mec": all_mec,
        "greedy": greedy,
        "gap_to_all_mec": None if k0 is None or all_mec is None else k3 - all_mec,
        "gap_to_greedy": None if k3 is None or greedy is None else k3 - greedy,
        "k3_better_than_k0": None if k0 is None or k3 is None else k3 < k0,
    }


def itr0_row(csv_path: str | Path, label: str = "itr0_from_pilot_a") -> dict:
    """The in-training itr-0 validation row of the Pilot A progress CSV."""
    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if _number(row.get(CSV_COLUMNS["k3"])) is None:
            continue
        k0 = _number(row.get(CSV_COLUMNS["k0"]))
        k3 = _number(row.get(CSV_COLUMNS["k3"]))
        all_mec = _number(row.get(CSV_COLUMNS["all_mec"]))
        greedy = _number(row.get(CSV_COLUMNS["greedy"]))
        return {
            "label": label,
            "source": str(csv_path),
            "itr": _number(row.get(CSV_COLUMNS["itr"])),
            "checkpoint_sha256": None,
            "weights_changed": None,
            "deterministic_k0_fresh": None,
            "k0": k0,
            "k3": k3,
            "all_mec": all_mec,
            "greedy": greedy,
            "gap_to_all_mec": None if all_mec is None else k3 - all_mec,
            "gap_to_greedy": None if greedy is None else k3 - greedy,
            "k3_better_than_k0": None if k0 is None else k3 < k0,
        }
    raise ValueError("no validation row found in %s" % csv_path)


def _same(a, b) -> bool:
    if a is None or b is None:
        return True  # missing baselines cannot contradict
    return abs(float(a) - float(b)) <= BASELINE_TOLERANCE


def checks_and_verdict(rows: list[dict]) -> dict:
    """Correctness checks + the three-label verdict over the comparison rows."""
    by_label = {row["label"]: row for row in rows}
    init = by_label.get("true_init")
    candidates = [
        row for row in rows
        if row["label"] not in ("true_init",) and row["source"] == "checkpoint_eval"
    ]
    finite = all(
        row["k0"] is not None and row["k3"] is not None
        and math.isfinite(row["k0"]) and math.isfinite(row["k3"])
        for row in rows
    )
    weights = {row["label"]: row["weights_changed"] for row in candidates}
    determinism = {
        row["label"]: row["deterministic_k0_fresh"]
        for row in rows
        if row["source"] == "checkpoint_eval"
    }
    checkpoints = [row for row in rows if row["source"] == "checkpoint_eval"]
    reference = checkpoints[0] if checkpoints else None
    baselines_consistent = bool(reference) and all(
        _same(reference["all_mec"], row["all_mec"])
        and _same(reference["greedy"], row["greedy"])
        for row in checkpoints
    )
    checks = {
        "true_init_present": init is not None,
        "candidates_present": bool(candidates),
        "all_finite": bool(finite),
        "weights_changed": all(bool(v) for v in weights.values()) if weights else False,
        "weights_changed_by_label": weights,
        "deterministic_k0_fresh": all(determinism.values()) if determinism else False,
        "deterministic_k0_fresh_by_label": determinism,
        "baselines_consistent_across_labels": baselines_consistent,
    }
    correctness_ok = all(
        checks[key] for key in (
            "true_init_present", "candidates_present", "all_finite",
            "weights_changed", "deterministic_k0_fresh",
            "baselines_consistent_across_labels",
        )
    )

    best = None
    k3_health = None
    improved = False
    if init is not None and candidates:
        best = min(candidates, key=lambda row: row["k3"])
        improved = bool(
            init["k3"] is not None and best["k3"] is not None
            and best["k3"] < init["k3"]
            and init["gap_to_all_mec"] is not None
            and best["gap_to_all_mec"] is not None
            and best["gap_to_all_mec"] < init["gap_to_all_mec"]
        )
        k3_health = bool(best["k3_better_than_k0"])

    if not correctness_ok:
        verdict = LABELS[0]
        detail = "a correctness check failed: %s" % sorted(
            key for key in checks if checks[key] is False
        )
    elif improved and k3_health:
        verdict = LABELS[1]
        detail = "%s beats true-init on k3 and its gap, and k3 < k0" % best["label"]
    elif improved:
        verdict = LABELS[2]
        detail = (
            "%s improves on true-init (k3 %.4f vs %.4f) but its k3 adaptation is "
            "unhealthy (k3=%s >= k0=%s); a long latency-only diagnostic may run, "
            "but the adaptation claim is not supported"
            % (best["label"], best["k3"], init["k3"], best["k3"], best["k0"])
        )
    else:
        verdict = LABELS[2]
        detail = "no candidate beats true-init on both k3 and its gap to all-MEC"

    return {
        "checks": checks,
        "verdict": verdict,
        "verdict_detail": detail,
        "improved_over_true_init": improved,
        "best_candidate": None if best is None else best["label"],
        "k3_better_than_k0": k3_health,
        "gain_vs_true_init": None if best is None or init is None or init["k3"] is None
        or best["k3"] is None else init["k3"] - best["k3"],
    }


def build(docs: list[tuple[str, dict]], pilot_csv: str | Path | None = None,
          pilot_label: str = "itr0_from_pilot_a") -> dict:
    rows = [label_row(label, doc) for label, doc in docs]
    if pilot_csv is not None:
        rows.insert(1 if rows and rows[0]["label"] == "true_init" else 0,
                    itr0_row(pilot_csv, label=pilot_label))
    result = {
        "schema": SCHEMA,
        "rows": rows,
        "training_code_sha": _first(docs, "training_code_sha"),
        "evaluation_code_sha": _first(docs, "evaluation_code_sha"),
    }
    result.update(checks_and_verdict(rows))
    return result


def _first(docs: list[tuple[str, dict]], key: str):
    for _label, doc in docs:
        value = doc.get(key)
        if value:
            return value
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", action="append", default=[],
        help="LABEL=JSON of one per-label checkpoint_eval_v1 document",
    )
    parser.add_argument("--pilot-csv", default=None)
    parser.add_argument("--pilot-label", default="itr0_from_pilot_a")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    docs = []
    for spec in args.eval:
        if "=" not in spec:
            raise SystemExit("use LABEL=JSON, got %r" % spec)
        label, path = spec.split("=", 1)
        with open(path) as handle:
            docs.append((label, json.load(handle)))
    if not docs:
        raise SystemExit("at least one --eval LABEL=JSON is required")

    result = build(docs, pilot_csv=args.pilot_csv, pilot_label=args.pilot_label)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "detail": result["verdict_detail"],
        "checks": result["checks"],
    }, indent=1, sort_keys=True))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
