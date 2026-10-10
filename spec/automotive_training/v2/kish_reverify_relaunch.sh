#!/bin/bash
# ============================================================================
# Re-verify after the scheduler repair and relaunch the authorized 1500-iteration
# campaign as a FRESH run (the aborted run's state is NOT resumed: the scheduler
# contract changed, so its 321 iterations are a different experiment).
#
# Usage: bash kish_reverify_relaunch.sh <bundle_on_host> <sha> <superseded_run_dir> [tag]
# ============================================================================
set +e
BUNDLE="${1:?usage: kish_reverify_relaunch.sh <bundle> <sha> <superseded_run_dir> [tag]}"
SHA="${2:?usage: ...}"
SUPERSEDED="${3:?usage: ...}"
TAG="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"
SRC=/opt/margo/mrlco-v2-src
GATES=/opt/margo/gates_$TAG
OUT=/opt/margo/acceptance_$TAG
LOGDIR=/opt/margo/logs
mkdir -p "$GATES" "$LOGDIR"

LOG="$LOGDIR/reverify_$TAG.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== reverify+relaunch $TAG  sha=$SHA  supersedes=$SUPERSEDED  $(date -u +%FT%TZ)"

rm -rf "$SRC"
git clone -q "$BUNDLE" "$SRC" || { echo "CLONE FAILED"; exit 2; }
cd "$SRC" || exit 2
git checkout -q "$SHA" || { echo "CHECKOUT FAILED"; exit 2; }
echo "checkout HEAD=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"

for f in kish_gate_a.sh kish_gate_b.sh kish_gate_c.sh kish_launch_campaign.sh \
         kish_monitor_campaign.sh gate_a_compare.py; do
  cp "$SRC/spec/automotive_training/v2/$f" "$GATES/$f"
done

# --- Gate A (continuation contract) ---
bash "$GATES/kish_gate_a.sh" "$SRC" "$OUT/gateA"
python3 "$GATES/gate_a_compare.py" "$OUT/gateA"
A=$?
echo "GATE_A_EXIT=$A"
[ "$A" -ne 0 ] && { echo "STOP: Gate A failed after the fix"; exit 3; }

# --- Gate B (recovery + paired) ---
bash "$GATES/kish_gate_b.sh" "$SRC" "$OUT/gateB"
grep -q "EXIT=[^0]" "$OUT/gateB/gate_b.log" && { echo "STOP: Gate B failing module"; exit 4; }
echo "GATE_B_OK"

# --- Gate C (production GPU smoke + observed kernels) ---
bash "$GATES/kish_gate_c.sh" "$SRC" "$OUT/gateC" 3
C=$(python3 -c "
import json
try: print(1 if json.load(open('$OUT/gateC/GATE_C_VERDICT.json'))['PASS'] else 0)
except Exception: print(0)")
echo "GATE_C_PASS=$C"
[ "$C" -ne 1 ] && { echo "STOP: Gate C failed after the fix"; exit 5; }

# --- fresh authorized campaign (new run dir; the aborted run is NOT resumed) ---
echo "ALL GATES PASSED -> launching the FRESH authorized 1500-iteration campaign"
bash "$GATES/kish_launch_campaign.sh" "$SRC" /opt/margo/campaigns 1500
RUN=$(ls -td /opt/margo/campaigns/primary_1500_seed0_* | head -1)
python3 - "$RUN" "$SUPERSEDED" "$SHA" <<'PY'
import json, sys, os
run, superseded, sha = sys.argv[1], sys.argv[2], sys.argv[3]
doc = {
    "schema": "v2_relaunch_context_v1",
    "supersedes_run": superseded,
    "supersede_reason": (
        "the earlier run ABORTED at iteration 321/1500 on a real scheduler defect "
        "(V2ScheduleError: a HELPER->UE relay committed outside the helper contact window). "
        "The planner was repaired, which changes scheduling semantics, so the earlier state is "
        "NOT a compatible continuation and was deliberately not resumed; this is a FRESH run."),
    "previous_run_completed_iterations": 321,
    "new_source_sha": sha,
    "fresh_initialization": True,
    "warm_start_used": False,
}
p = os.path.join(run, "RELAUNCH_CONTEXT.json")
json.dump(doc, open(p, "w"), indent=2, sort_keys=True)
print("wrote", p)
PY
nohup setsid bash "$GATES/kish_monitor_campaign.sh" "$RUN" 60 45 > "$RUN/monitor.log" 2>&1 &
echo "monitor started for $RUN"
echo "===== reverify+relaunch $TAG done $(date -u +%FT%TZ)"
