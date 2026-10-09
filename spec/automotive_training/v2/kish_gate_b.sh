#!/bin/bash
# ============================================================================
# Gate B -- crash-recovery fixtures, legacy refusal, and the REAL paired
#           R x S evaluator on the frozen contract.
#
# Checks (all executed, none inferred):
#   1. partial-generation recovery + legacy-schema refusal fixtures
#   2. the resume contract (schema/dataset/scheduler/sampling mismatch refusal,
#      target increase, restored lambdas + draw cursor)
#   3. the paired evaluator: identical-plan equality, panel/policy world pairing,
#      cell identities, deterministic vs sampled blocks, and the TF-gated
#      real-policy R x S grid
#
# Usage: bash kish_gate_b.sh <repo_dir_on_host> <out_dir> [image]
# ============================================================================
set +e
REPO="${1:?usage: kish_gate_b.sh <repo_dir> <out_dir> [image]}"
OUT="${2:?usage: kish_gate_b.sh <repo_dir> <out_dir> [image]}"
IMG="${3:-margo-phase4-tf115-nv2212:latest}"
mkdir -p "$OUT"
LOG="$OUT/gate_b.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "Gate B  start $(date -u +%FT%TZ)"
echo "repo $REPO HEAD=$(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
echo "image $IMG"
echo "=============================================================="

run_pattern() {
  pat="$1"; extra_env="$2"; label="$3"
  echo ""
  echo "########## $label ($pat) ##########"
  docker run --rm $extra_env \
    -e PYTHONPATH=/work -e TF_CPP_MIN_LOG_LEVEL=3 -e CUDA_VISIBLE_DEVICES= \
    -v "$REPO":/work -w /work --entrypoint python "$IMG" \
    -m unittest discover -s env/mec_offloaing_envs/scheduler/tests -p "$pat" -v 2>&1 \
    | grep -vE "^WARNING:tensorflow" | tail -40
  echo "########## EXIT=${PIPESTATUS[0]} $label ##########"
}

run_pattern 'test_v2_f18_generation_recovery.py' '' 'partial-generation recovery + legacy refusal'
run_pattern 'test_v2_f19_resume_contract.py'     '' 'resume contract + target increase'
run_pattern 'test_v2_f25_paired_worlds.py'       '' 'paired policy/panel worlds + identical-plan equality'
run_pattern 'test_v2_eval_loop.py'               '' 'real R x S cells, decoding blocks, fail-loud metrics'
run_pattern 'test_v2_constraints.py'             '' 'dual state isolation (validation must not mutate train duals)'
run_pattern 'test_v2_stack.py'                   '' 'TF-gated R x S grid on the REAL policy'

echo ""
echo "Gate B done $(date -u +%FT%TZ)  log=$LOG"
