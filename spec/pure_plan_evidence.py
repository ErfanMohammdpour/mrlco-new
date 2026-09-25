#!/usr/bin/env python3
"""E4.1 pure-plan evidence: all-UE / all-MEC / all-HELPER / mixed.

CPU-only, deterministic, no data files. For every plan and graph it records the
scientific numbers the migration is about:

    latency | requester/mobile/system energy | MEC compute | MEC TX |
    primary scalar | constraint raw/status | scheduler fingerprint

and gates the contract identities before declaring the evidence valid:

  * boundary identity:  requester <= mobile <= system and
    system == mobile + mec_compute + mec_tx;
  * component identity: for each scope, the sum of the per-task scoped
    attribution equals the plan scalar;
  * timing invariance: switching ENERGY accounting (legacy vs physical) with the
    timing axis fixed does not move makespan, task start/finish or transfers;
  * finiteness: every recorded number is finite.

Usage:  python3 -m spec.pure_plan_evidence [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from env.mec_offloaing_envs.scheduler import (
    CanonicalDAG,
    CanonicalTask,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.constraints import (
    ConstraintSpec,
    measure_metrics,
)
from env.mec_offloaing_envs.scheduler.energy_api import (
    ReferenceRanges,
    attribute_scoped_energy_by_task,
)
from env.mec_offloaing_envs.scheduler.energy_cache import primary_scope_of
from env.mec_offloaing_envs.scheduler.energy_scope import (
    ENERGY_SCOPES,
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    energy_scalar,
)
from env.mec_offloaing_envs.scheduler.primary_config import (
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

PLANS = {
    "all_UE": [0, 0, 0],
    "all_MEC": [1, 1, 1],
    "all_HELPER": [2, 2, 2],
    "mixed": [0, 2, 1],
}

BOUNDARY_TOL = 1e-9


def _chain() -> CanonicalDAG:
    """Root -> middle -> sink chain; a MEC task therefore has a return hop."""
    tasks = [
        CanonicalTask(0, 2_097_152, 1_048_576, 2_097_152),
        CanonicalTask(1, 2_097_152, 1_048_576, 0),
        CanonicalTask(2, 2_097_152, 1_048_576, 0),
    ]
    return CanonicalDAG.from_records(
        tasks, [(0, 1, 1_048_576), (1, 2, 1_048_576)]
    )


def _fork() -> CanonicalDAG:
    """Root with two children so a mixed plan leaves UE/MEC/HELPER work alive."""
    tasks = [
        CanonicalTask(0, 1_048_576, 524_288, 1_048_576),
        CanonicalTask(1, 1_048_576, 524_288, 0),
        CanonicalTask(2, 1_048_576, 524_288, 0),
    ]
    return CanonicalDAG.from_records(
        tasks, [(0, 1, 524_288), (0, 2, 524_288)]
    )


GRAPHS = {"chain": _chain, "fork": _fork}


def _legacy_timing_resources() -> ResourceConfig:
    """Frozen yaml with legacy ENERGY accounting, for the timing-invariance gate."""
    return ResourceConfig.from_frozen_yaml()


def _canonical_refs(dag, order, resources) -> ReferenceRanges:
    """System-scoped reference ranges for a CanonicalDAG (no legacy adapter)."""
    metrics = {}
    for action in (0, 1, 2):
        out = schedule(dag, order, [action] * len(order), resources)
        metrics[action] = (out.makespan_seconds, energy_scalar(out, scope=SCOPE_SYSTEM))
    return ReferenceRanges(
        L_ue=metrics[0][0], L_mec=metrics[1][0], L_helper=metrics[2][0],
        E_ue=metrics[0][1], E_mec=metrics[1][1], E_helper=metrics[2][1],
        energy_scope=SCOPE_SYSTEM,
        scheduler_config_sha256=str(resolved_config_sha256(resources)),
    )


def plan_row(name, dag, order, actions, resources) -> dict:
    result = schedule(dag, order, actions, resources)
    primary_scope = primary_scope_of(resources)
    row = {
        "graph": name,
        "plan": None,
        "latency_s": float(result.makespan_seconds),
        "requester_joules": energy_scalar(result, scope=SCOPE_REQUESTER),
        "mobile_joules": energy_scalar(result, scope=SCOPE_MOBILE),
        "system_joules": energy_scalar(result, scope=SCOPE_SYSTEM),
        "mec_compute_joules": float(result.energy.mec_compute_joules_optional),
        "mec_tx_joules": float(result.energy.mec_tx_joules_optional),
        "primary_scope": primary_scope,
        "primary_joules": energy_scalar(result, scope=primary_scope),
        "scheduler_config_sha256": str(result.scheduler_config_sha256),
        "budget_status": None,
        "total_energy_raw_system_j": None,
        "total_energy_status": None,
    }
    metrics = measure_metrics(result)
    row["total_energy_raw_system_j"] = float(metrics.total_energy_j)
    configured = ConstraintSpec(mode="lagrangian", total_energy_budget_j=1.0)
    row["total_energy_status"] = configured.constraint_status()["total_energy"]
    unconfigured = ConstraintSpec(mode="lagrangian", ue_energy_budget_j=1.0)
    row["budget_status"] = unconfigured.constraint_status()["total_energy"]
    return row


def _component_identity(result, resources, scope, tolerance=BOUNDARY_TOL) -> bool:
    per_task = attribute_scoped_energy_by_task(result, resources, scope=scope)
    return abs(sum(per_task.values()) - energy_scalar(result, scope=scope)) <= tolerance


def _timing_invariance(dag, order, actions, physical, legacy) -> bool:
    a = schedule(dag, order, actions, physical)
    b = schedule(dag, order, actions, legacy)
    if abs(a.makespan_seconds - b.makespan_seconds) > 1e-12:
        return False
    for tid, rec in a.tasks.items():
        other = b.tasks[tid]
        if abs(rec.start - other.start) > 1e-12 or abs(rec.finish - other.finish) > 1e-12:
            return False
        if rec.location != other.location:
            return False
    if len(a.transfers) != len(b.transfers):
        return False
    for ta, tb in zip(a.transfers, b.transfers):
        if ta.hop != tb.hop or abs(ta.start - tb.start) > 1e-12 or abs(ta.end - tb.end) > 1e-12:
            return False
    return True


def _finite(row) -> bool:
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def run() -> dict:
    physical = resolved_primary_scheduler_config()
    legacy = _legacy_timing_resources()
    fingerprint = str(resolved_config_sha256(physical))
    rows = []
    gates = {
        "boundary_identity": True,
        "component_identity": True,
        "timing_invariance": True,
        "fingerprint_parity": True,
        "all_finite": True,
    }
    for graph_name, factory in GRAPHS.items():
        dag = factory()
        order = sorted(int(t) for t in dag.tasks)
        refs = _canonical_refs(dag, order, physical)
        for plan_name, template in PLANS.items():
            actions = [template[i % len(template)] for i in range(len(order))]
            row = plan_row(graph_name, dag, order, actions, physical)
            row["plan"] = plan_name
            row["refs_panel_min_system_j"] = refs.E_ref_min
            result = schedule(dag, order, actions, physical)
            # boundary + component identities
            if not (
                row["requester_joules"] <= row["mobile_joules"] + BOUNDARY_TOL
                and row["mobile_joules"] <= row["system_joules"] + BOUNDARY_TOL
                and abs(
                    row["system_joules"]
                    - (row["mobile_joules"] + row["mec_compute_joules"] + row["mec_tx_joules"])
                )
                <= BOUNDARY_TOL
            ):
                gates["boundary_identity"] = False
            for scope in ENERGY_SCOPES:
                if not _component_identity(result, physical, scope):
                    gates["component_identity"] = False
            if not _timing_invariance(dag, order, actions, physical, legacy):
                gates["timing_invariance"] = False
            if row["scheduler_config_sha256"] != fingerprint:
                gates["fingerprint_parity"] = False
            if not _finite(row):
                gates["all_finite"] = False
            rows.append(row)
    return {
        "schema": "pure_plan_evidence_v1",
        "primary_scope": primary_scope_of(physical),
        "scheduler_config_sha256": fingerprint,
        "refs_energy_scope": SCOPE_SYSTEM,
        "plans": list(PLANS),
        "graphs": list(GRAPHS),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "rows": rows,
    }


_COLUMNS = (
    ("graph", 6),
    ("plan", 10),
    ("latency_s", 12),
    ("requester_joules", 16),
    ("mobile_joules", 14),
    ("system_joules", 14),
    ("mec_compute_joules", 18),
    ("mec_tx_joules", 13),
    ("primary_scope", 13),
    ("primary_joules", 14),
    ("total_energy_raw_system_j", 24),
    ("total_energy_status", 18),
    ("budget_status", 16),
)


def print_table(evidence) -> None:
    header = " ".join("%*s" % (w, name) for name, w in _COLUMNS)
    print(header)
    print("-" * len(header))
    for row in evidence["rows"]:
        cells = []
        for name, width in _COLUMNS:
            value = row[name]
            text = "%.9g" % value if isinstance(value, float) else str(value)
            cells.append("%*s" % (width, text))
        print(" ".join(cells))
    print()
    for gate, ok in evidence["gates"].items():
        print("gate %-22s %s" % (gate, "PASS" if ok else "FAIL"))
    print("all_gates_pass: %s" % evidence["all_gates_pass"])


def main() -> int:
    parser = argparse.ArgumentParser()
    default_json = (
        Path(__file__).resolve().parents[1]
        / "reports" / "v0.3-audit" / "pure_plan_evidence.json"
    )
    parser.add_argument("--json", default=str(default_json))
    args = parser.parse_args()
    evidence = run()
    print_table(evidence)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % out)
    return 0 if evidence["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
