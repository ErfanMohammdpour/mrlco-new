#!/bin/bash
# ============================================================================
# Detached monitor / heartbeat for the V2 campaign (or any v2 run directory).
#
# Records, without ever restarting the run:
#   * progress: completed/1500, next iteration, current phase, last progress timestamp
#   * validation/checkpoint counts and the newest COMPLETE generation
#   * termination: launcher exit or container exit code (137 = OOM-killed)
#   * a STALLED run (progress file unchanged for > --stall-minutes)
#   * non-finite telemetry in the progress ledger
# On any of those it writes a final verdict to the heartbeat and EXITS (no restart loop).
#
# Usage: bash kish_monitor_campaign.sh <run_dir> [interval_s] [stall_minutes]
# ============================================================================
set -uo pipefail
RUN="${1:?usage: kish_monitor_campaign.sh <run_dir> [interval_s] [stall_minutes]}"
INTERVAL="${2:-60}"
STALL_MIN="${3:-25}"

HB="$RUN/heartbeat.json"
PROGRESS="$RUN/v2_progress.json"
[ -f "$PROGRESS" ] || PROGRESS="$RUN/ckpt/v2_progress.json"
MANIFEST="$RUN/RUN_MANIFEST.json"
PIDFILE="$RUN/monitor.pid"
echo $$ > "$PIDFILE"

json_get() { python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
except Exception:
    print(''); raise SystemExit
cur=d
for k in '$2'.split('.'):
    if isinstance(cur, dict):
        cur=cur.get(k)
    else:
        cur=None; break
print('' if cur is None else cur)
" 2>/dev/null; }

atomic_write() {  # atomic_write <path> <json-string>
  python3 -c "
import os, sys
p=sys.argv[1]
tmp=p+'.tmp'
open(tmp,'w').write(sys.argv[2]+'\n')
os.replace(tmp,p)
" "$1" "$2"
}

last_progress=""
stall_since=$(date -u +%s)
state="RUNNING"
stop_reason=""

while :; do
  now=$(date -u +%s)
  now_iso=$(date -u +%FT%TZ)
  launcher=$(json_get "$MANIFEST" launcher_pid)
  cname=$(json_get "$MANIFEST" container_name)
  container_alive="no"
  container_status=""
  if [ -n "${cname:-}" ]; then
    if docker ps -q --filter "name=$cname" | grep -q .; then
      container_alive="yes"
    else
      container_status=$(docker inspect "$cname" --format '{{.State.ExitCode}}' 2>/dev/null || echo "")
    fi
  fi
  launcher_alive="no"
  [ -n "${launcher:-}" ] && kill -0 "$launcher" 2>/dev/null && launcher_alive="yes"

  completed=$(json_get "$PROGRESS" completed_iterations)
  target=$(json_get "$PROGRESS" iteration_target)
  next_itr=$(json_get "$PROGRESS" next_iteration)
  phase=$(json_get "$PROGRESS" phase_of_last_audit)
  updated=$(json_get "$PROGRESS" updated_utc)
  validations=$(json_get "$PROGRESS" validations_completed)
  lambdas=$(json_get "$PROGRESS" lambdas)
  mean_J=$(json_get "$PROGRESS" mean_system_joules)
  mean_viol=$(json_get "$PROGRESS" mean_violation)

  gens=""
  [ -d "$RUN/ckpt/generations" ] && gens=$(ls "$RUN/ckpt/generations" 2>/dev/null | grep -c '^gen_')

  finite="yes"
  case "$mean_J$mean_viol" in
    *nan*|*NaN*|*inf*|*Inf*) finite="no";;
  esac

  if [ "$finite" = "no" ]; then
    state="FAILED"; stop_reason="non-finite telemetry in the progress ledger"
  elif [ "$container_alive" = "no" ] && [ "$launcher_alive" = "no" ]; then
    if [ -n "$completed" ] && [ -n "$target" ] && [ "$completed" -ge "$target" ] 2>/dev/null; then
      state="COMPLETED"; stop_reason="target reached"
    else
      state="TERMINATED"; stop_reason="process exited (container exit code ${container_status:-unknown})"
    fi
  fi

  if [ -n "$updated" ] && [ "$updated" != "$last_progress" ]; then
    last_progress="$updated"; stall_since=$now
  fi
  stalled_for=$(( (now - stall_since) / 60 ))
  if [ "$state" = "RUNNING" ] && [ "$stalled_for" -ge "$STALL_MIN" ] && [ -n "$completed" ]; then
    state="STALLED"; stop_reason="no progress for ${stalled_for} minutes"
  fi

  atomic_write "$HB" "$(python3 - "$state" "$stop_reason" "$now_iso" "$completed" "$target" \
      "$next_itr" "$phase" "$updated" "$validations" "$gens" "$container_alive" \
      "$launcher_alive" "$container_status" "$stalled_for" "$mean_J" "$mean_viol" \
      "$lambdas" <<'PY'
import json, sys
(a, reason, now, completed, target, nxt, phase, updated, validations, gens,
 alive, lalive, cstatus, stalled, mj, mv, lambdas) = sys.argv[1:18]
def num(x):
    try:
        return float(x)
    except Exception:
        return None
def integer(x):
    try:
        return int(x)
    except Exception:
        return None
doc = {
    "schema": "v2_campaign_heartbeat_v1",
    "state": a,
    "stop_reason": reason or None,
    "updated_utc": now,
    "completed_iterations": integer(completed),
    "iteration_target": integer(target),
    "next_iteration": integer(nxt),
    "current_phase": phase or None,
    "last_progress_timestamp": updated or None,
    "validations_completed": integer(validations),
    "complete_generations": integer(gens),
    "container_alive": alive == "yes",
    "launcher_alive": lalive == "yes",
    "container_exit_code": integer(cstatus) if cstatus not in ("", None) else None,
    "stalled_minutes": integer(stalled),
    "mean_system_joules": num(mj),
    "mean_violation": num(mv),
    "lambdas": lambdas or None,
}
print(json.dumps(doc, indent=2, sort_keys=True))
PY
)"
  echo "[$(date -u +%FT%TZ)] state=$state completed=${completed:-?}/${target:-?} gens=${gens:-?} phase=${phase:-?} ${stop_reason:+($stop_reason)}"

  if [ "$state" != "RUNNING" ]; then
    echo "monitor exiting: $state ($stop_reason)"
    break
  fi
  sleep "$INTERVAL"
done
