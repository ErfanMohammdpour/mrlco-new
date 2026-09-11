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
    if "margo_v0.1_learning_probe" not in driver:
        block(reasons, "train driver must isolate learning probe from the primary 3500 run")
    if "margo_v0.1_diag_1k" not in driver:
        block(reasons, "train driver must isolate 1000-iter diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_200" not in driver:
        block(reasons, "train driver must isolate 200-iter diagnostic from the primary 3500 run")
    if "margo_v0.1_parallel_probe" not in driver:
        block(reasons, "train driver must isolate parallel env probe from the primary 3500 run")
    if "margo_v0.1_diag_500_parallel" not in driver:
        block(reasons, "train driver must isolate 500-iter parallel diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_latency_tmec" not in driver:
        block(reasons, "train driver must isolate latency-Tmec diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_pomo_tmec" not in driver:
        block(reasons, "train driver must isolate POMO-Tmec diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_bc_greedy_tmec" not in driver:
        block(reasons, "train driver must isolate BC greedy-from-MEC diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_bc_only_eval" not in driver:
        block(reasons, "train driver must isolate BC-only greedy eval from the primary 3500 run")
    if "margo_v0.1_diag_bc_continue" not in driver:
        block(reasons, "train driver must isolate BC-continue plateau from the primary 3500 run")
    if "margo_v0.1_diag_kl_bc_ppo" not in driver:
        block(reasons, "train driver must isolate KL-to-frozen-π_BC PPO from the primary 3500 run")
    if "margo_v0.1_diag_bc_unseen" not in driver:
        block(reasons, "train driver must isolate BC unseen greedy decode from the primary 3500 run")
    if "margo_v0.1_diag_bc_fewshot" not in driver:
        block(reasons, "train driver must isolate BC few-shot diagnostic from the primary 3500 run")
    if "margo_v0.1_diag_bc_scheduled" not in driver:
        block(reasons, "train driver must isolate scheduled-sampling BC from the primary 3500 run")
    if "margo_v0.1_diag_hamming2_expert" not in driver:
        block(reasons, "train driver must isolate Hamming-2 expert probe from the primary 3500 run")
    if "margo_v0.1_diag_motif_expert" not in driver:
        block(reasons, "train driver must isolate motif expert audit from the primary 3500 run")
    if "margo_v0.1_diag_2opt_expert" not in driver:
        block(reasons, "train driver must isolate iterative 2-opt teacher from the primary 3500 run")
    if "margo_v0.1_diag_bc_2opt" not in driver:
        block(reasons, "train driver must isolate BC-on-2opt from the primary 3500 run")
    if "margo_v0.1_diag_pair_ranker" not in driver:
        block(reasons, "train driver must isolate ΔT pair ranker from the primary 3500 run")
    if "margo_v0.1_diag_pair_seq" not in driver:
        block(reasons, "train driver must isolate sequential pair refine from the primary 3500 run")
    if "margo_v0.2_diag_cavia_frozen" not in driver:
        block(reasons, "train driver must isolate CAVIA-on-z energy-off from the primary 3500 run")
    if "margo_v0.2_diag_cavia_energy" not in driver:
        block(reasons, "train driver must isolate CAVIA-on-z energy-on from the primary 3500 run")
    if "margo_v0.2_diag_cavia_strong" not in driver:
        block(reasons, "train driver must isolate CAVIA strong-inner from the primary 3500 run")
    if "margo_v0.2_diag_pairsup" not in driver:
        block(reasons, "train driver must isolate pair-sup B vs A from the primary 3500 run")
    if "margo_v0.2_diag_cavia_bccont" not in driver:
        block(reasons, "train driver must isolate CAVIA-on-z from bc_continue from the primary 3500 run")
    if "margo_v0.2_diag_pairfrac_mec" not in driver:
        block(reasons, "train driver must isolate all-MEC pair-frac probe from the primary 3500 run")
    if "margo_v0.2_diag_rewrite_mec" not in driver:
        block(reasons, "train driver must isolate all-MEC rewrite from the primary 3500 run")
    if "margo_v0.2_diag_oracle_dist" not in driver:
        block(reasons, "train driver must isolate oracle dist_id embed from the primary 3500 run")
    if "margo_v0.2_diag_binary_lat" not in driver:
        block(reasons, "train driver must isolate binary no-V2V latency diagnostic from the primary 3500 run")
    if "margo_v0.3_diag_encoder" not in driver:
        block(reasons, "train driver must isolate v0.3 encoder ablation from the primary 3500 run")
    if "margo_v0.3_diag_bestofk" not in driver:
        block(reasons, "train driver must isolate v0.3 best-of-k inference from the primary 3500 run")
    if "margo_v0.3_diag_eas" not in driver:
        block(reasons, "train driver must isolate v0.3 EAS-on-φ adaptation from the primary 3500 run")
    if "margo_v0.3_diag_eas_inst" not in driver:
        block(reasons, "train driver must isolate v0.3 per-instance EAS from the primary 3500 run")
    camp = (ROOT / "spec" / "phase4_campaign.py").read_text()
    for mid, label in (
        ("margo_v0.3_expert_profiles", "expert_profiles"),
        ("margo_v0.3_bc_profiles", "bc_profiles"),
        ("margo_v0.3_eas_profiles", "eas_profiles"),
        ("margo_v0.3_ctx_profiles", "ctx_profiles"),
        ("margo_v0.3_bok_profiles", "bok_profiles"),
        ("margo_v0.3_expert_energy", "expert_energy"),
        ("margo_v0.3_bc_energy", "bc_energy"),
        ("margo_v0.3_eas_energy", "eas_energy"),
    ):
        if mid not in camp and mid not in driver:
            block(reasons, "campaign/driver must register v0.3 %s method id" % label)
    if "frozen primary remains 3500" not in driver:
        block(reasons, "learning probe must state frozen primary remains 3500")
    if "paper_result" not in driver:
        block(reasons, "train driver must record paper_result=false")


def check_tests(reasons):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "env.mec_offloaing_envs.scheduler.tests.test_phase4_campaign",
            "env.mec_offloaing_envs.scheduler.tests.test_train_audit",
            "env.mec_offloaing_envs.scheduler.tests.test_parallel_env_executor",
            "env.mec_offloaing_envs.scheduler.tests.test_cavia_objective",
            "env.mec_offloaing_envs.scheduler.tests.test_pair_sup",
            "env.mec_offloaing_envs.scheduler.tests.test_pair_frac",
            "env.mec_offloaing_envs.scheduler.tests.test_rewrite_mec",
            "env.mec_offloaing_envs.scheduler.tests.test_oracle_dist",
            "env.mec_offloaing_envs.scheduler.tests.test_eas_adapt",
            "env.mec_offloaing_envs.scheduler.tests.test_phase5_energy",
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
