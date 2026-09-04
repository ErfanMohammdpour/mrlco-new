#!/usr/bin/env python3
"""Phase 4 campaign gate.

Exit 0 when campaign contract tests PASS and PHASE4_STATUS.md is IN PROGRESS.
Does NOT train. Does NOT use GPU. Does NOT close the phase.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
STATUS = SPEC / "PHASE4_STATUS.md"
YAML = SPEC / "phase4_campaign.yaml"
CAMPAIGN = SPEC / "phase4_campaign.py"
DRIVER = SPEC / "phase4_train_driver.py"
PARENT_SHA = "0c776924b49da6c66c511c12a8cde70be732e25d"


def block(reasons, msg):
    reasons.append(msg)


def check_status(reasons):
    text = STATUS.read_text()
    if not re.search(r"^Status:\s*IN PROGRESS\b", text, re.M):
        block(reasons, "PHASE4_STATUS.md must say Status: IN PROGRESS until GPU campaign artifacts exist")
    if "phase3-freeze-v0.1" not in text:
        block(reasons, "PHASE4_STATUS.md must name phase3-freeze-v0.1")
    if "Do not move or rewrite" not in text:
        block(reasons, "PHASE4_STATUS.md must forbid rewriting older freeze tags")
    if "--i-allow-gpu" not in text or "MARGO_ALLOW_GPU=1" not in text:
        block(reasons, "PHASE4_STATUS.md must document GPU locks")
    if "No paper figures" not in text:
        block(reasons, "PHASE4_STATUS.md must forbid paper figures before artifacts")
    yaml_text = YAML.read_text()
    if PARENT_SHA not in yaml_text and PARENT_SHA not in CAMPAIGN.read_text():
        block(reasons, "campaign must pin parent SHA " + PARENT_SHA)
    if "seeds: [0, 1, 2, 3, 4]" not in yaml_text:
        block(reasons, "campaign yaml must freeze five evaluation seeds")
    if "meta_test: [7, 12, 14, 20, 23]" not in yaml_text:
        block(reasons, "campaign yaml must list all five meta-test distributions")
    if "launching_gpu_without_human_approval" not in yaml_text:
        block(reasons, "campaign yaml must forbid GPU without human approval")
    driver = DRIVER.read_text()
    if "require_gpu_permission" not in driver:
        block(reasons, "phase4_train_driver.py must call require_gpu_permission")
    if "n_itr must be" not in driver:
        block(reasons, "train driver must refuse n_itr other than 3500")
    if "margo_v0.1_gpu_smoke" not in driver:
        block(reasons, "train driver must isolate GPU smoke from the primary 3500 run")
    if "paper_result" not in driver:
        block(reasons, "train driver must record paper_result=false")


def check_tests(reasons):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "env.mec_offloaing_envs.scheduler.tests.test_phase4_campaign",
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
            block(reasons, "phase 4 test skipped: %s" % line.strip())
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        block(reasons, "Phase 4 unittest failed (rc=%s)" % proc.returncode)


def main():
    reasons = []
    print("=== Phase 4 campaign gate ===")
    check_status(reasons)
    check_tests(reasons)
    print("\nGATE SNAPSHOT")
    print("PHASE4_STATUS.md:   checked")
    print("Campaign contract:  checked")
    print("Phase 4 unit tests: checked")
    if reasons:
        print("\nPhase 4 campaign: BLOCKED")
        for item in reasons:
            print("  - %s" % item)
        return 1
    print("\nPhase 4 campaign: PASS")
    print("Phase 4 closure: NOT CLAIMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
