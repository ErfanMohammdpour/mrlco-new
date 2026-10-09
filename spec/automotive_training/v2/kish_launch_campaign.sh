#!/bin/bash
# ============================================================================
# Launch the authorized 1500-completed-outer-iteration V2 campaign (seed 0).
#
# Frozen production settings (see the acceptance report):
#   iterations 1500 (TOTAL completed outer iterations, not episodes/tokens)
#   seed 0 | background DAGs 2 | background policy all_mec | moderate links
#   total-energy fraction 0.5 (foreground all-MEC referenced), UE fraction 1.0
#   R/S 2/2 with REAL execution | validation interval 50 | checkpoint every 10
#   sampling contract crn_token_keyed_v2 | trace OFF | meta-test closed
#
# The source is mounted READ-ONLY (immutable tested revision) and every output goes to a
# unique persistent run directory that also holds the identity/fingerprint manifest.
#
# Usage: bash kish_launch_campaign.sh <repo_dir> <out_root> [iterations] [image]
# ============================================================================
set -euo pipefail
REPO="${1:?usage: kish_launch_campaign.sh <repo_dir> <out_root> [iterations] [image]}"
OUT_ROOT="${2:?usage: kish_launch_campaign.sh <repo_dir> <out_root> [iterations] [image]}"
ITERATIONS="${3:-1500}"
IMG="${4:-margo-phase4-tf115-nv2212:latest}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$OUT_ROOT/primary_${ITERATIONS}_seed0_${STAMP}"
NAME="margo-v2-${ITERATIONS}-seed0-${STAMP}"
mkdir -p "$RUN"

SHA="$(git -C "$REPO" rev-parse HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')"
IMG_ID="$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | awk -v i="$IMG" '$1==i {print $2}')"

# The v2 obs schema and the frozen contract ids this run pins.
ARGS=(
  -m spec.automotive_training.v2.run_tf
  --device gpu --iterations "$ITERATIONS" --seed 0
  --background 2 --background-policy all_mec --link-regime moderate
  --energy-budget-fraction 0.5 --ue-budget-fraction 1.0
  --r-select 2 --s-select 2
  --validation-interval 50 --checkpoint-interval 10
  --gpu-fraction 0.03 --intra-op 8 --inter-op 2
  --sampling-contract crn_token_keyed_v2
  --ckpt-dir /out/ckpt --json /out/run.json
)

MANIFEST="$RUN/RUN_MANIFEST.json"
cat > "$MANIFEST" <<EOF
{
  "schema": "v2_campaign_manifest_v1",
  "run_name": "$NAME",
  "started_utc": "$STAMP",
  "host": "$(hostname)",
  "repo": "$REPO",
  "source_sha": "$SHA",
  "source_dirty_files": $DIRTY,
  "image": "$IMG",
  "image_id": "$IMG_ID",
  "iterations_total_target": $ITERATIONS,
  "seed": 0,
  "background_dags": 2,
  "background_policy": "all_mec",
  "link_regime": "moderate",
  "total_energy_fraction": 0.5,
  "ue_energy_fraction": 1.0,
  "r_select": 2,
  "s_select": 2,
  "validation_interval": 50,
  "checkpoint_interval": 10,
  "sampling_contract": "crn_token_keyed_v2",
  "trace": false,
  "meta_test": "closed",
  "gpu_fraction": 0.03,
  "intra_op": 8,
  "inter_op": 2,
  "source_mount": "read-only",
  "run_dir": "$RUN",
  "stdout_log": "$RUN/stdout.log",
  "stderr_log": "$RUN/stderr.log",
  "container_name": "$NAME"
}
EOF

echo "manifest: $MANIFEST"
echo "source_sha=$SHA dirty=$DIRTY image=$IMG($IMG_ID)"

nohup setsid docker run --name "$NAME" --gpus all \
  -e PYTHONPATH=/work -e TF_CPP_MIN_LOG_LEVEL=3 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=false \
  -e MARGO_OBS_VERSION=automotive_v2_obs_v1 \
  -v "$REPO":/work:ro -v "$RUN":/out -w /work \
  --entrypoint python "$IMG" "${ARGS[@]}" \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log" &
LAUNCHER_PID=$!
sleep 5
CID="$(docker ps -q --filter "name=$NAME" | head -1)"
python3 - "$MANIFEST" "$LAUNCHER_PID" "$CID" <<'PY'
import json, sys
path, pid, cid = sys.argv[1], sys.argv[2], sys.argv[3]
doc = json.load(open(path))
doc["launcher_pid"] = int(pid)
doc["container_id"] = cid or None
json.dump(doc, open(path, "w"), indent=2, sort_keys=True)
print("launcher_pid=%s container_id=%s" % (pid, cid))
PY
echo "RUNNING dir=$RUN"
