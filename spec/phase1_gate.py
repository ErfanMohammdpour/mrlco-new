#!/usr/bin/env python3
"""Phase 1 closure gate.

Exit 0 only when every required check PASSes. Does NOT create a git tag.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import yaml

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
FROZEN = SPEC / "frozen_experiment.yaml"
SCHEDULER = ROOT / "env" / "mec_offloaing_envs" / "scheduler"
ENV_PY = ROOT / "env" / "mec_offloaing_envs" / "offloading_env.py"
EVAL_PY = ROOT / "meta_evaluator.py"
STATUS_MD = SPEC / "PHASE1_STATUS.md"

FORBIDDEN_CALENDAR = (
    "FT_cloud",
    "FT_ws",
    "FT_wr",
    "FT_v2v_dl",
    "ws_avaliable_time",
)
LEGACY_REWARD = (
    "energy_proportions",
    "total_energy_score",
    "_compute_energy_bounds",
)


def block(reasons: list[str], msg: str) -> None:
    reasons.append(msg)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)


def check_phase0(reasons: list[str]) -> None:
    proc = run_capture([sys.executable, str(SPEC / "phase0_gate.py")])
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        block(reasons, f"Phase 0 gate failed (rc={proc.returncode})")


def check_pydotplus(reasons: list[str]) -> None:
    try:
        import pydotplus.graphviz  # noqa: F401
    except Exception as exc:
        block(reasons, f"required dependency pydotplus missing/unusable: {exc}")


def check_scheduler_tests(reasons: list[str]) -> None:
    proc = run_capture(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "env/mec_offloaing_envs/scheduler/tests",
            "-v",
        ]
    )
    out = proc.stdout + "\n" + proc.stderr
    # Print a compact summary; full logs are long.
    for line in out.splitlines():
        if line.startswith(("test_", "OK", "FAILED", "ERROR", "Ran ", "=")):
            print(line)
        if "SKIP" in line or "skipped" in line.lower():
            # Required .gv integration must not skip under the gate.
            if "TestRealGvIntegration" in line or "parser_adapter_schedule" in line:
                block(reasons, f"required .gv integration skipped: {line.strip()}")
    if proc.returncode != 0:
        block(reasons, f"scheduler unittest suite failed (rc={proc.returncode})")
    if "Ran " not in out:
        block(reasons, "scheduler unittest produced no Ran summary")


def check_oracles(reasons: list[str]) -> None:
    proc = run_capture(
        [
            sys.executable,
            str(SPEC / "toy_oracles" / "oracle_checker.py"),
            "--engine",
            "production",
        ]
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or "ALL toy oracles PASS" not in proc.stdout:
        block(reasons, "production toy oracles did not all PASS")


def check_architecture(reasons: list[str]) -> None:
    env_text = ENV_PY.read_text()
    env_body = env_text.split("class OffloadingEnvironment", 1)[1]
    eval_text = EVAL_PY.read_text()
    for label, text in (("OffloadingEnvironment", env_body), ("meta_evaluator", eval_text)):
        for token in FORBIDDEN_CALENDAR:
            if token in text:
                block(reasons, f"{label} still contains legacy calendar token {token}")
    for token in LEGACY_REWARD:
        if token in env_body:
            block(reasons, f"legacy reward wiring still present: {token}")
    if "telescoping_token_rewards" not in env_body:
        block(reasons, "env reward path does not call telescoping_token_rewards")

    # Scheduling math outside scheduler package: forbid new calendar tokens in env tree
    # except watermark fields on Resources.
    for path in (ROOT / "env").rglob("*.py"):
        if SCHEDULER in path.parents or path.name.startswith("test_"):
            continue
        text = path.read_text()
        if path.name == "offloading_env.py":
            text = text.split("class OffloadingEnvironment", 1)[1]
        for token in ("FT_cloud", "FT_ws", "FT_wr", "FT_v2v_dl"):
            if token in text:
                block(reasons, f"scheduling calendar token {token} found in {path.relative_to(ROOT)}")


def check_publication_weights(reasons: list[str]) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from env.mec_offloaing_envs.scheduler.energy_api import (
        ENERGY_WEIGHT,
        LATENCY_WEIGHT,
        frozen_objective_weights,
        require_publication_weights,
    )

    try:
        lw, ew = frozen_objective_weights(FROZEN)
    except Exception as exc:
        block(reasons, f"frozen_objective_weights failed: {exc}")
        return
    if abs(lw - 0.5) > 1e-12 or abs(ew - 0.5) > 1e-12:
        block(reasons, f"frozen weights must be 0.5/0.5, got {lw}/{ew}")
    try:
        require_publication_weights(LATENCY_WEIGHT, ENERGY_WEIGHT)
    except Exception as exc:
        block(reasons, f"require_publication_weights(0.5,0.5) failed: {exc}")
    try:
        require_publication_weights(0.7, 0.3)
        block(reasons, "require_publication_weights failed to reject 0.7/0.3")
    except ValueError:
        pass

    doc = yaml.safe_load(FROZEN.read_text())
    w = doc["energy"]["objective_weights"]
    if abs(float(w["latency_weight"]) - 0.5) > 1e-12 or abs(float(w["energy_weight"]) - 0.5) > 1e-12:
        block(reasons, "frozen_experiment.yaml objective_weights not 0.5/0.5")


def check_reward_batch_shape(reasons: list[str]) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Stub optional imports before pulling env helpers via scheduler only.
    for name in ("gym", "gym.core", "graphviz"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["gym.core"].Env = type("Env", (), {})
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz")).Digraph = type("Digraph", (), {})

    from env.mec_offloaing_envs.scheduler import ResourceConfig, telescoping_token_rewards

    class _T:
        def __init__(self, proc, tx):
            self.processing_data_size = proc
            self.transmission_data_size = tx

    class _TG:
        def __init__(self, n=20):
            self.task_number = n
            self.task_list = [_T(1_048_576, 458_752) for _ in range(n)]
            self.prioritize_sequence = list(range(n))
            self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
            self.edge_set = [[i, 0, 0, 458_752, i + 1, 0, 0] for i in range(n - 1)]

    cfg = ResourceConfig.from_frozen_yaml()
    batch = []
    for actions in ([0] * 20, [1] * 20, [2] * 20):
        tg = _TG(20)
        plan = list(zip(range(20), actions))
        out = telescoping_token_rewards(tg, plan, cfg, compute_j_report=False)
        if len(out.rewards) != 20:
            block(reasons, f"reward length {len(out.rewards)} != 20")
        if out.j_report_value is not None:
            block(reasons, "training path must not compute j_report by default")
        if not all(math.isfinite(r) for r in out.rewards):
            block(reasons, "non-finite reward value")
        batch.append(out.rewards)
    if len(batch) != 3 or any(len(r) != 20 for r in batch):
        block(reasons, "reward batch shape is not (batch, 20)")


def check_status_doc(reasons: list[str], gate_pass: bool) -> None:
    text = STATUS_MD.read_text()
    if gate_pass:
        if not re.search(r"^Status:\s*CLOSED\b", text, re.M):
            block(reasons, "PHASE1_STATUS.md must say Status: CLOSED after gate PASS")
    else:
        # While blocked, status may still say IN PROGRESS — OK.
        pass


def main() -> int:
    reasons: list[str] = []
    print("=== Phase 1 closure gate ===")
    check_phase0(reasons)
    check_pydotplus(reasons)
    check_scheduler_tests(reasons)
    check_oracles(reasons)
    check_architecture(reasons)
    check_publication_weights(reasons)
    check_reward_batch_shape(reasons)

    gate_pass = not reasons
    # Status file is updated by the human/agent commit after a green gate; verify when PASS.
    if gate_pass:
        # Soft expectation: if already CLOSED, good; if still IN PROGRESS, remind but
        # do not fail the computational gate — caller updates status in same commit.
        text = STATUS_MD.read_text()
        if "Status: CLOSED" not in text and "Status: IN PROGRESS" in text:
            print("NOTE: set PHASE1_STATUS.md to CLOSED in the closure commit.")

    print("\nGATE SNAPSHOT")
    print(f"Phase 0 gate:                 {'PASS' if not any('Phase 0' in r for r in reasons) else 'FAIL'}")
    print(f"Scheduler tests:              checked")
    print(f"Production oracles:           checked")
    print(f"Architecture / reward wiring: checked")
    print(f"Publication weights 0.5/0.5:  checked")
    print(f"Reward shape (batch, 20):     checked")

    if reasons:
        print("\nPhase 1 closure: BLOCKED")
        for r in reasons:
            print(f"  - {r}")
        return 1

    print("\nPhase 1 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
