#!/usr/bin/env python3
"""Phase 3 learning contract gate.

Does NOT close Phase 3. Does NOT train. Does NOT use GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
STATUS = SPEC / "PHASE3_STATUS.md"
MRLCO = ROOT / "meta_algos" / "MRLCO.py"
TRAINER = ROOT / "meta_trainer.py"
EVAL = ROOT / "meta_evaluator.py"
FROZEN = SPEC / "frozen_experiment.yaml"


def block(reasons: list[str], msg: str) -> None:
    reasons.append(msg)


def check_status(reasons: list[str]) -> None:
    text = STATUS.read_text()
    if not re.search(r"^Status:\s*IN PROGRESS\b", text, re.M):
        block(reasons, "PHASE3_STATUS.md must say Status: IN PROGRESS until Phase 3 closes")
    if "phase2-freeze-v0.1" not in text:
        block(reasons, "PHASE3_STATUS.md must name phase2-freeze-v0.1")
    if "Do not move or rewrite" not in text:
        block(reasons, "PHASE3_STATUS.md must forbid rewriting Phase 1/2 tags")
    if "no scientific GPU training" not in text:
        block(reasons, "PHASE3_STATUS.md must forbid GPU training")


def check_sources(reasons: list[str]) -> None:
    mrlco = MRLCO.read_text()
    trainer = TRAINER.read_text()
    evaluator = EVAL.read_text()
    frozen = FROZEN.read_text()
    if "mrlco_first_order_mean_pseudogradient" not in frozen:
        block(reasons, "frozen_experiment.yaml missing outer method")
    if "self.old_v[i] + tf.clip_by_value" not in mrlco:
        block(reasons, "MRLCO value clip must be v_old + clip(v_new - v_old)")
    if "self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i]" in mrlco:
        block(reasons, "legacy value-clip formula still present in MRLCO")
    if "mean_pseudogradient(" not in mrlco:
        block(reasons, "MRLCO must call mean_pseudogradient")
    if "/ self.meta_batch_size / self.update_numbers" in mrlco:
        block(reasons, "MRLCO still scales outer grads by update_numbers")
    if "def reset_inner_optimizer" not in mrlco:
        block(reasons, "inner Adam must be reset per meta-task")
    if "meta_train_graph_prefixes" not in trainer:
        block(reasons, "meta_trainer.py must load split_loader meta_train prefixes")
    if "num_inner_grad_steps=1" in trainer:
        block(reasons, "meta_trainer.py still declares k_steps=1")
    if "inner_batch_size=10" in trainer:
        block(reasons, "meta_trainer.py still uses batch_size=10")
    if "one-step update" in trainer:
        block(reasons, "forbidden phrase 'one-step update' in trainer")
    if "CUDA_VISIBLE_DEVICES" not in trainer:
        block(reasons, "trainer must default-disable GPU until Phase 3 closes")
    if "meta_test_support" not in evaluator or "meta_test_query" not in evaluator:
        block(reasons, "evaluator must wire support/query from the manifest")
    if "batch_size=500" in evaluator:
        block(reasons, "evaluator still uses batch_size=500")


def check_tests(reasons: list[str]) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "env.mec_offloaing_envs.scheduler.tests.test_phase3_learning",
            "-v",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    for line in out.splitlines():
        if line.startswith(("test_", "OK", "FAILED", "ERROR", "Ran ", "=")):
            print(line)
        if re.search(r"\bSKIP(?:PED)?\b", line):
            block(reasons, f"phase 3 test skipped: {line.strip()}")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        block(reasons, f"Phase 3 unittest failed (rc={proc.returncode})")


def main() -> int:
    reasons: list[str] = []
    print("=== Phase 3 learning gate ===")
    check_status(reasons)
    check_sources(reasons)
    check_tests(reasons)
    print("\nGATE SNAPSHOT")
    print("PHASE3_STATUS.md:   checked")
    print("Source contracts:   checked")
    print("Phase 3 unit tests: checked")
    if reasons:
        print("\nPhase 3 learning: BLOCKED")
        for item in reasons:
            print(f"  - {item}")
        return 1
    print("\nPhase 3 learning: PASS")
    print("Phase 3 closure: NOT CLAIMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
