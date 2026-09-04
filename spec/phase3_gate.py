#!/usr/bin/env python3
"""Phase 3 learning closure gate.

Exit 0 when learning tests + wiring checks PASS and PHASE3_STATUS.md is CLOSED.
Does NOT create a tag. Does NOT train. Does NOT use GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
STATUS = SPEC / "PHASE3_STATUS.md"
EVIDENCE = SPEC / "phase3_tf_smoke_evidence.txt"
TESTED_SHA = "f17d264b49d0886b3f0e9f438d203af8ab0ef9af"
MRLCO = ROOT / "meta_algos" / "MRLCO.py"
PPO = ROOT / "meta_algos" / "ppo_offloading.py"
TRAINER = ROOT / "meta_trainer.py"
EVAL = ROOT / "meta_evaluator.py"
FROZEN = SPEC / "frozen_experiment.yaml"


def block(reasons: list[str], msg: str) -> None:
    reasons.append(msg)


def check_status(reasons: list[str]) -> None:
    text = STATUS.read_text()
    if not re.search(r"^Status:\s*CLOSED\b", text, re.M):
        block(reasons, "PHASE3_STATUS.md must say Status: CLOSED")
    if "phase2-freeze-v0.1" not in text:
        block(reasons, "PHASE3_STATUS.md must name phase2-freeze-v0.1")
    if "Do not move or rewrite" not in text:
        block(reasons, "PHASE3_STATUS.md must forbid rewriting Phase 1/2 tags")
    if TESTED_SHA not in text:
        block(reasons, "PHASE3_STATUS.md must pin tested SHA " + TESTED_SHA)
    if "Phase 3 closure does not imply paper results" not in text:
        block(reasons, "PHASE3_STATUS.md must state Phase 3 is not a paper-result claim")
    if not EVIDENCE.is_file():
        block(reasons, "spec/phase3_tf_smoke_evidence.txt missing")
        return
    evidence = EVIDENCE.read_text()
    if TESTED_SHA not in evidence:
        block(reasons, "smoke evidence must record tested_sha=" + TESTED_SHA)
    if "Python 3.7.17" not in evidence:
        block(reasons, "smoke evidence must record Python 3.7.17")
    if "1.15.5" not in evidence:
        block(reasons, "smoke evidence must record TensorFlow 1.15.5")
    if "Ran 41 tests" not in evidence:
        block(reasons, "smoke evidence must record 41/41 tests")
    if "Phase 3 learning: PASS" not in evidence:
        block(reasons, "smoke evidence must record Phase 3 learning: PASS")


def check_sources(reasons: list[str]) -> None:
    mrlco = MRLCO.read_text()
    ppo = PPO.read_text()
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
    if "inner_batch_size = 500" in trainer or "inner_batch_size=500" in trainer:
        block(reasons, "meta_trainer.py still defaults inner_batch_size=500")
    if "one-step update" in trainer:
        block(reasons, "forbidden phrase 'one-step update' in trainer")
    if "CUDA_VISIBLE_DEVICES" not in trainer:
        block(reasons, "trainer must default-disable GPU until Phase 3 closes")
    if "sync_task_policies_from_core" not in trainer:
        block(reasons, "trainer must sync task policies from core each outer iter")
    if "validation_interval" not in trainer or "HeldOutQueryEvaluator" not in trainer:
        block(reasons, "trainer must wire validation_interval held-out query eval")
    if "protocol_log_kvs" not in trainer:
        block(reasons, "trainer must log LEARNING_PROTOCOL §9 fields")
    if "support_query_tasks" not in evaluator:
        block(reasons, "evaluator must wire support/query from split_loader")
    if "batch_size=500" in evaluator:
        block(reasons, "evaluator still uses batch_size=500")
    if "set_task(0)" in evaluator:
        block(reasons, "evaluator set_task(0) leaks the full 100-graph pool")
    if "greedy_solution_for_current_task" not in evaluator:
        block(reasons, "evaluator greedy must stay on the current query slice")
    if "k_steps=0" not in evaluator:
        block(reasons, "evaluator must report zero-shot k_steps=0")
    if "mpi4py" in ppo or "MpiAdamOptimizer" in ppo:
        block(reasons, "ppo_offloading.py still imports mpi4py / MpiAdamOptimizer")
    if "lr=1e-4" in ppo or "epsilon=1e-5" in ppo:
        block(reasons, "ppo_offloading.py still uses legacy lr/epsilon")
    if "def reset_inner_optimizer" not in ppo:
        block(reasons, "PPO must reset inner Adam per adapt")
    if "shuffled_minibatch_slices" not in ppo:
        block(reasons, "PPO must shuffle minibatches each inner epoch")
    tf_test = ROOT / "env" / "mec_offloaing_envs" / "scheduler" / "tests" / "test_phase3_learning_tf.py"
    if not tf_test.is_file():
        block(reasons, "test_phase3_learning_tf.py missing")
    if "Phase 3 learning smoke requires TensorFlow 1.15" not in tf_test.read_text():
        block(reasons, "TF smoke must fail when TensorFlow is missing")


def check_tests(reasons: list[str]) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "env.mec_offloaing_envs.scheduler.tests.test_phase3_learning",
            "env.mec_offloaing_envs.scheduler.tests.test_phase3_learning_tf",
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
        if re.search(r"\bSKIP(?:PED)?\b", line) or " ... skipped" in line.lower():
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
    print("Phase 3 TF smoke:   checked")
    if reasons:
        print("\nPhase 3 learning: BLOCKED")
        for item in reasons:
            print(f"  - {item}")
        return 1
    print("\nPhase 3 learning: PASS")
    print("Phase 3 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
