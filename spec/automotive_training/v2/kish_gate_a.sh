#!/bin/bash
# ============================================================================
# Gate A -- CPU continuation across a scheduled validation boundary (V2).
#
# Design (acceptance spec):
#   1. UNINTERRUPTED control to 6 completed outer iterations.
#   2. A SEPARATE process to 2 completed outer iterations with a committed
#      transactional checkpoint generation.
#   3. A FRESH process resumes that checkpoint to the TOTAL target 6.
#   Validations are scheduled at indices 2 and 4 (V2 validation interval 2).
#   4. The final post-update checkpoint is additionally evaluated in an
#      ISOLATED evaluation context (--eval-only), i.e. outside the training
#      process that produced it.
#
# Everything runs in the supported TF1.15 image with zero visible GPUs.
# Usage: bash kish_gate_a.sh <repo_dir_on_host> <output_dir> [image]
# ============================================================================
set +e
REPO="${1:?usage: kish_gate_a.sh <repo_dir> <out_dir> [image]}"
OUT="${2:?usage: kish_gate_a.sh <repo_dir> <out_dir> [image]}"
IMG="${3:-margo-phase4-tf115-nv2212:latest}"

SEED=0
FRAC=0.05
R_SELECT=2
S_SELECT=2
VALIDATION_INTERVAL=2
CHECKPOINT_INTERVAL=2
TOTAL=6
SPLIT_AT=2

mkdir -p "$OUT"
LOG="$OUT/gate_a.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "Gate A  start $(date -u +%FT%TZ)"
echo "repo    $REPO   HEAD=$(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
echo "image   $IMG"
echo "outputs $OUT"
echo "=============================================================="

COMMON=(--device cpu --seed "$SEED" --background 2 --background-policy all_mec
        --link-regime moderate
        --energy-budget-fraction "$FRAC" --ue-budget-fraction "$FRAC"
        --r-select "$R_SELECT" --s-select "$S_SELECT"
        --validation-interval "$VALIDATION_INTERVAL"
        --checkpoint-interval "$CHECKPOINT_INTERVAL"
        --intra-op 1 --inter-op 1)

run_v2() {   # run_v2 <label> <iterations> <ckpt_dir> <json> [extra args...]
  local label="$1"; shift
  local iters="$1"; shift
  local ckpt="$1"; shift
  local json="$1"; shift
  echo ""
  echo "-------------------- [$label] iterations=$iters ckpt=$ckpt"
  local t0
  t0=$(date -u +%s)
  docker run --rm \
    -e PYTHONPATH=/work -e CUDA_VISIBLE_DEVICES= -e TF_CPP_MIN_LOG_LEVEL=3 \
    -v "$REPO":/work -v "$OUT":/out -w /work --entrypoint python "$IMG" \
    -m spec.automotive_training.v2.run_tf \
    --iterations "$iters" --ckpt-dir "$ckpt" --json "$json" \
    "${COMMON[@]}" "$@" 2>&1 | tail -25
  local code=${PIPESTATUS[0]}
  echo "-------------------- [$label] EXIT=$code elapsed=$(( $(date -u +%s) - t0 ))s"
  return $code
}

rm -rf "$OUT/control" "$OUT/split"
mkdir -p "$OUT/control" "$OUT/split"

# 1. uninterrupted control
run_v2 control "$TOTAL" /out/control/ckpt /out/control.json
CONTROL_EXIT=$?

# 2. split: first process to SPLIT_AT completed iterations
run_v2 split-part1 "$SPLIT_AT" /out/split/ckpt /out/split_part1.json
SPLIT1_EXIT=$?

echo ""
echo "generations after part 1:"
ls "$OUT/split/ckpt/generations" 2>/dev/null
echo "CURRENT: $(cat "$OUT/split/ckpt/generations/CURRENT" 2>/dev/null)"

# 3. fresh process resumes to the TOTAL target
run_v2 split-part2 "$TOTAL" /out/split/ckpt /out/split.json \
  --resume-from /out/split/ckpt
SPLIT2_EXIT=$?

# 4. isolated evaluation of the control's FINAL post-update checkpoint
run_v2 eval-only "$TOTAL" /out/control/ckpt /out/control_eval.json \
  --eval-only --resume-from /out/control/ckpt
EVAL_EXIT=$?

echo ""
echo "=============================================================="
echo "Gate A  comparison $(date -u +%FT%TZ)"
echo "=============================================================="
python3 "$(cd "$(dirname "$0")" && pwd)/gate_a_compare.py" "$OUT"
A_VERDICT=$?

echo ""
echo "exits: control=$CONTROL_EXIT split1=$SPLIT1_EXIT split2=$SPLIT2_EXIT evalonly=$EVAL_EXIT"
echo "Gate A done $(date -u +%FT%TZ)  log=$LOG"
