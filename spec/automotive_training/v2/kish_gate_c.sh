#!/bin/bash
# ============================================================================
# Gate C -- production GPU smoke on the repaired sampling/unit/objective
#           contracts, at the APPROVED fractions, with observed kernel
#           placement.
#
# Settings (exactly the authorized production contract):
#   supported CUDA-11.8 TF1.15/contrib image | seed 0 | background 2 | all_mec
#   moderate links | total-energy fraction 0.5 (foreground all-MEC referenced)
#   UE fraction 1.0 | R/S 2/2 executed for real | validation interval 50
#   trace OFF | bounded GPU fraction | >= 3 completed outer iterations
#   transactional checkpoint every iteration, then an ISOLATED evaluation of the
#   final post-update checkpoint (--eval-only)
#
# Usage: bash kish_gate_c.sh <repo_dir> <out_dir> [iterations] [image]
# ============================================================================
set +e
REPO="${1:?usage: kish_gate_c.sh <repo_dir> <out_dir> [iterations] [image]}"
OUT="${2:?usage: kish_gate_c.sh <repo_dir> <out_dir> [iterations] [image]}"
ITERS="${3:-3}"
IMG="${4:-margo-phase4-tf115-nv2212:latest}"
mkdir -p "$OUT"
LOG="$OUT/gate_c.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "Gate C  start $(date -u +%FT%TZ)"
echo "repo $REPO HEAD=$(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
echo "image $IMG  iterations=$ITERS"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
echo "=============================================================="

COMMON=(--device gpu --seed 0
        --background 2 --background-policy all_mec --link-regime moderate
        --energy-budget-fraction 0.5 --ue-budget-fraction 1.0
        --r-select 2 --s-select 2
        --validation-interval 50 --checkpoint-interval 1
        --gpu-fraction 0.03 --intra-op 8 --inter-op 2
        --sampling-contract crn_token_keyed_v2)

t0=$(date -u +%s)
docker run --rm --gpus all \
  -e PYTHONPATH=/work -e TF_CPP_MIN_LOG_LEVEL=3 -e TF_FORCE_GPU_ALLOW_GROWTH=false \
  -e MARGO_OBS_VERSION=automotive_v2_obs_v1 \
  -v "$REPO":/work:ro -v "$OUT":/out -w /work --entrypoint python "$IMG" \
  -m spec.automotive_training.v2.run_tf \
  --iterations "$ITERS" --ckpt-dir /out/ckpt --json /out/run.json \
  --kernel-profile /out/kernel_profile.json \
  "${COMMON[@]}" 2>&1 | grep -vE "^WARNING:tensorflow" | tail -30
RUN_EXIT=${PIPESTATUS[0]}
echo "-------------------- [gate C train] EXIT=$RUN_EXIT elapsed=$(( $(date -u +%s) - t0 ))s"

echo ""
echo "-------------------- [gate C isolated evaluation of the final checkpoint]"
t1=$(date -u +%s)
docker run --rm --gpus all \
  -e PYTHONPATH=/work -e TF_CPP_MIN_LOG_LEVEL=3 -e TF_FORCE_GPU_ALLOW_GROWTH=false \
  -e MARGO_OBS_VERSION=automotive_v2_obs_v1 \
  -v "$REPO":/work:ro -v "$OUT":/out -w /work --entrypoint python "$IMG" \
  -m spec.automotive_training.v2.run_tf \
  --iterations "$ITERS" --ckpt-dir /out/ckpt --json /out/final_eval.json \
  --eval-only --resume-from /out/ckpt \
  "${COMMON[@]}" 2>&1 | grep -vE "^WARNING:tensorflow" | tail -15
EVAL_EXIT=${PIPESTATUS[0]}
echo "-------------------- [gate C eval-only] EXIT=$EVAL_EXIT elapsed=$(( $(date -u +%s) - t1 ))s"

echo ""
echo "=============================================================="
echo "Gate C verdict $(date -u +%FT%TZ)"
echo "=============================================================="
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
def rd(name):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else {}

run = rd("run.json")
evd = run.get("evidence") if isinstance(run.get("evidence"), dict) else {}
for _k, _v in (evd or {}).items():
    run.setdefault(_k, _v)
prof = rd("kernel_profile.json")
ev = rd("final_eval.json")
gens = sorted(p.name for p in (out / "ckpt" / "generations").glob("gen_*")) \
    if (out / "ckpt" / "generations").is_dir() else []

obs = prof.get("observed") or {}
checks = {
    "run_json_present": bool(run),
    "iterations_completed": run.get("iterations_completed"),
    "iterations_completed_ge_3": int(run.get("iterations_completed") or 0) >= 3,
    "outer_updates_completed": run.get("outer_updates_completed"),
    "outer_updates_equal_completed": run.get("outer_updates_completed") == run.get("iterations_completed"),
    "validations_completed": run.get("validations_completed"),
    "sampler_counters": run.get("sampler_counters"),
    "constraints_names": (run.get("constraints") or {}).get("names"),
    "constraint_lambdas": (run.get("constraints") or {}).get("lambdas"),
    "constraint_status": (run.get("constraints") or {}).get("status"),
    "energy_constraint": (run.get("report", {}).get("constraints") or {}).get("energy_constraint"),
    "state_finite": (run.get("trained_core_norms") or {}).get("all_finite"),
    "exit_ok": run.get("ok"),
    "error": run.get("error"),
    "complete_generations": gens,
    "generations_ge_3": len(gens) >= 3,
    "kernel_profile_ok": prof.get("ok"),
    "gpu_ops": obs.get("gpu_ops"),
    "cpu_ops": obs.get("cpu_ops"),
    "n_ops": obs.get("n_ops"),
    "gpu_share": prof.get("gpu_share"),
    "kernel_warning": prof.get("warning"),
    "kernel_profiled_run_number": prof.get("profiled_run_number"),
    "kernel_visible_devices": (prof.get("visible") or {}).get("physical_devices"),
    "final_eval_present": bool(ev),
    "final_eval_k3_latency_s": (ev.get("k3") or {}).get("query_mean_latency_s"),
    "final_eval_lambdas": ev.get("lambdas"),
    "final_eval_completed_iterations": ev.get("completed_iterations"),
}
verdict = {
    "schema": "v2_gate_c_verdict_v1",
    "checks": checks,
    "PASS": bool(checks["iterations_completed_ge_3"]
                 and checks["outer_updates_equal_completed"]
                 and checks["state_finite"]
                 and checks["generations_ge_3"]
                 and checks["kernel_profile_ok"]
                 and int(checks["gpu_ops"] or 0) > 0
                 and checks["final_eval_present"]),
    "note": ("PASS requires >=3 completed outer iterations with matching outer-update counts, "
             "finite state, >=3 complete checkpoint generations, an observed GPU kernel "
             "placement artifact with gpu_ops>0, and the isolated final-checkpoint evaluation."),
}
(out / "GATE_C_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
print("GATE C: %s" % ("PASS" if verdict["PASS"] else "FAIL"))
PY
echo ""
echo "exits: train=$RUN_EXIT eval-only=$EVAL_EXIT"
echo "Gate C done $(date -u +%FT%TZ)  log=$LOG"
