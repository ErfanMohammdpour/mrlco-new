#!/bin/bash
# ============================================================================
# Continue the authorized acceptance chain ONCE the GPU host is reachable.
#
# Order (each stage must pass before the next):
#   1. sync the frozen tested revision into an isolated checkout
#   2. Gate A  (CPU continuation across a scheduled validation boundary)
#   3. Gate B  (recovery fixtures, legacy refusal, paired R x S)
#   4. Gate C  (production GPU smoke at fraction 0.5 + isolated final eval)
#   5. ONLY if A, B and C all pass: launch the 1500-iteration campaign detached
#
# It never asks for approval and never invents a pass: a failing stage stops the chain and
# leaves its log/verdict on disk.
#
# Usage: bash kish_continue.sh <bundle_path_on_host> <sha> [attempt_tag]
# ============================================================================
set +e
BUNDLE="${1:?usage: kish_continue.sh <bundle> <sha> [tag]}"
SHA="${2:?usage: kish_continue.sh <bundle> <sha> [tag]}"
TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
SRC=/opt/margo/mrlco-v2-src
LOGDIR=/opt/margo/logs
OUTROOT=/opt/margo/acceptance_$TAG
GATES=/opt/margo/gates_$TAG
mkdir -p "$LOGDIR" "$GATES"

CHAIN_LOG="$LOGDIR/continue_$TAG.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "===== continue chain $TAG start $(date -u +%FT%TZ) sha=$SHA"

rm -rf "$SRC"
git clone -q "$BUNDLE" "$SRC" || { echo "CLONE FAILED"; exit 2; }
cd "$SRC" || exit 2
git checkout -q "$SHA" || { echo "CHECKOUT FAILED"; exit 2; }
echo "checkout HEAD=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
cp "$SRC/spec/automotive_training/v2/kish_gate_a.sh" "$GATES/kish_gate_a.sh"
cp "$SRC/spec/automotive_training/v2/kish_gate_b.sh" "$GATES/kish_gate_b.sh"
cp "$SRC/spec/automotive_training/v2/kish_gate_c.sh" "$GATES/kish_gate_c.sh"
cp "$SRC/spec/automotive_training/v2/kish_launch_campaign.sh" "$GATES/kish_launch_campaign.sh"
cp "$SRC/spec/automotive_training/v2/kish_monitor_campaign.sh" "$GATES/kish_monitor_campaign.sh"
cp "$SRC/spec/automotive_training/v2/gate_a_compare.py" "$GATES/gate_a_compare.py"

rc=0
bash "$GATES/kish_gate_a.sh" "$SRC" "$OUTROOT/gateA" || rc=$?
python3 "$GATES/gate_a_compare.py" "$OUTROOT/gateA"
A=$?
echo "GATE_A_EXIT=$A"
[ "$A" -ne 0 ] && { echo "CHAIN STOPPED: Gate A failed"; exit 3; }

bash "$GATES/kish_gate_b.sh" "$SRC" "$OUTROOT/gateB"
grep -q "EXIT=[^0]" "$OUTROOT/gateB/gate_b.log" && { echo "CHAIN STOPPED: Gate B had a failing module"; exit 4; }
echo "GATE_B_OK"

bash "$GATES/kish_gate_c.sh" "$SRC" "$OUTROOT/gateC" 3
C=$(python3 -c "
import json,sys
try: print(1 if json.load(open('$OUTROOT/gateC/GATE_C_VERDICT.json'))['PASS'] else 0)
except Exception: print(0)")
echo "GATE_C_PASS=$C"
[ "$C" -ne 1 ] && { echo "CHAIN STOPPED: Gate C failed"; exit 5; }

echo "ALL GATES PASSED -> launching the authorized 1500-iteration campaign"
bash "$GATES/kish_launch_campaign.sh" "$SRC" /opt/margo/campaigns 1500
RUN=$(ls -td /opt/margo/campaigns/primary_1500_seed0_* 2>/dev/null | head -1)
echo "RUN=$RUN"
nohup setsid bash "$GATES/kish_monitor_campaign.sh" "$RUN" 60 45 > "$RUN/monitor.log" 2>&1 &
echo "monitor started for $RUN"
echo "===== continue chain $TAG done $(date -u +%FT%TZ)"
