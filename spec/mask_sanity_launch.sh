#!/bin/bash
# Launch the ⑥b mask sanity runs and their stop-rule watchdogs, detached.
#
#   bash spec/mask_sanity_launch.sh 500 off static     # parallel, the default ask
#   bash spec/mask_sanity_launch.sh 1 static           # single mode
#
# Each mode gets its own container, its own run directory and its own watchdog.
# Everything is started with setsid+nohup, so an ssh disconnect cannot kill it:
# a plain `docker run` client dies on SIGHUP and takes the container with it.
#
# Exit files (/tmp/ts_<mode>.exit) are what the watchdog and any poller read.
set -u
ROOT="${MARGO_ROOT:-/opt/margo/mrlco-new-6b}"
ITR="${1:-500}"
shift || true
if [ "$#" -eq 0 ]; then set -- off static; fi

cd "$ROOT" || exit 1

for MODE in "$@"; do
  LOG="/tmp/ts_${MODE}.log"
  WLOG="/tmp/watchdog_${MODE}.log"
  rm -f "/tmp/ts_${MODE}.exit"
  setsid nohup bash -c "cd $ROOT && MARGO_ROOT=$ROOT bash spec/kish_gpu.sh mask-sanity-${ITR}-${MODE} > $LOG 2>&1; echo exit=\$? > /tmp/ts_${MODE}.exit" \
    </dev/null >/dev/null 2>&1 &
  setsid nohup bash -c "cd $ROOT && python3 -m spec.mask_run_watchdog --mode $MODE --seed 0 --runs-root $ROOT/runs/mask_sanity_v3 --interval 60 > $WLOG 2>&1" \
    </dev/null >/dev/null 2>&1 &
  echo "launched mode=$MODE itr=$ITR log=$LOG watchdog=$WLOG"
done

sleep 15
echo "--- containers ---"
docker ps --filter ancestor=margo-phase4-tf115-nv2212 --format '{{.Names}} {{.Status}} {{.Command}}' | head -4
