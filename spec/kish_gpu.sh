#!/bin/bash
# kish-ai helpers. Do not run on Apple Silicon.
set -euo pipefail
IMAGE="${MARGO_GPU_IMAGE:-margo-phase4-tf115-nv2212}"
ROOT="${MARGO_ROOT:-/opt/margo/mrlco-new}"
cd "$ROOT"

gpu_run() {
  docker run --rm --gpus all \
    -e MARGO_ALLOW_GPU=1 \
    -e PYTHONPATH=/work \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$ROOT":/work -w /work \
    "$IMAGE" \
    "$@"
}

gpu_run_v2() {
  docker run --rm --gpus all \
    -e MARGO_ALLOW_GPU=1 \
    -e MARGO_OBS_VERSION=v2 \
    -e PYTHONPATH=/work \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$ROOT":/work -w /work \
    "$IMAGE" \
    "$@"
}

cpu_run() {
  docker run --rm \
    -e PYTHONPATH=/work \
    -e PYTHONUNBUFFERED=1 \
    -v "$ROOT":/work -w /work \
    "$IMAGE" \
    "$@"
}

case "${1:-}" in
  probe)
    gpu_run python -c 'import tensorflow as tf
print("python_ok")
print("tf", tf.__version__)
print("contrib", hasattr(tf, "contrib"))
print("gpu_available", tf.test.is_gpu_available())
from tensorflow.python.client import device_lib
for d in device_lib.list_local_devices():
    print(d.device_type, d.name)
'
    ;;
  smoke)
    gpu_run python spec/phase4_campaign.py --gpu-smoke --seed 0 --i-allow-gpu
    ;;
  audit5)
    gpu_run python spec/phase4_campaign.py --learning-probe --seed 0 --i-allow-gpu
    ;;
  diag1k)
    gpu_run python spec/phase4_campaign.py --diagnostic-1k --seed 0 --i-allow-gpu
    ;;
  diag200)
    gpu_run python spec/phase4_campaign.py --diagnostic-200 --seed 0 --i-allow-gpu
    ;;
  par5)
    gpu_run python spec/phase4_campaign.py --parallel-probe --seed 0 --i-allow-gpu
    ;;
  par500)
    gpu_run python spec/phase4_campaign.py --diagnostic-500 --seed 0 --i-allow-gpu
    ;;
  lat50)
    gpu_run python spec/phase4_campaign.py --diagnostic-latency-tmec --seed 0 --i-allow-gpu
    ;;
  pomo50)
    gpu_run python spec/phase4_campaign.py --diagnostic-pomo-tmec --seed 0 --i-allow-gpu
    ;;
  bc50)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-greedy-tmec --seed 0 --i-allow-gpu
    ;;
  bconly)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-only-eval --seed 0 --i-allow-gpu
    ;;
  bccont)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-continue --seed 0 --i-allow-gpu
    ;;
  klppo50)
    gpu_run python spec/phase4_campaign.py --diagnostic-kl-bc-ppo --seed 0 --i-allow-gpu
    ;;
  bcunseen)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-unseen --seed 0 --i-allow-gpu
    ;;
  bcfew)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-fewshot --seed 0 --i-allow-gpu
    ;;
  bcss)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-scheduled --seed 0 --i-allow-gpu
    ;;
  h2)
    cpu_run python spec/phase4_campaign.py --diagnostic-hamming2-expert --seed 0
    ;;
  motif)
    cpu_run python spec/phase4_campaign.py --diagnostic-motif-expert --seed 0
    ;;
  twopt)
    cpu_run python spec/phase4_campaign.py --diagnostic-2opt-expert --seed 0
    ;;
  pairfrac)
    cpu_run python spec/phase4_campaign.py --diagnostic-pairfrac-mec --seed 0
    ;;
  bc2opt)
    gpu_run python spec/phase4_campaign.py --diagnostic-bc-2opt --seed 0 --i-allow-gpu
    ;;
  pairhead)
    gpu_run python spec/phase4_campaign.py --diagnostic-pair-head --seed 0 --i-allow-gpu
    ;;
  pairksweep)
    gpu_run python spec/phase4_campaign.py --diagnostic-pair-ksweep --seed 0 --i-allow-gpu
    ;;
  pairranker)
    gpu_run python spec/phase4_campaign.py --diagnostic-pair-ranker --seed 0 --i-allow-gpu
    ;;
  pairseq)
    gpu_run python spec/phase4_campaign.py --diagnostic-pair-seq --seed 0 --i-allow-gpu
    ;;
  cavia)
    gpu_run python spec/phase4_campaign.py --diagnostic-cavia --seed 0 --i-allow-gpu
    ;;
  caviaE)
    gpu_run python spec/phase4_campaign.py --diagnostic-cavia-energy --seed 0 --i-allow-gpu
    ;;
  caviaS)
    gpu_run python spec/phase4_campaign.py --diagnostic-cavia-strong --seed 0 --i-allow-gpu
    ;;
  pairsup)
    gpu_run python spec/phase4_campaign.py --diagnostic-pairsup --seed 0 --i-allow-gpu
    ;;
  caviaB)
    gpu_run python spec/phase4_campaign.py --diagnostic-cavia-bccont --seed 0 --i-allow-gpu
    ;;
  rewrite)
    gpu_run python spec/phase4_campaign.py --diagnostic-rewrite-mec --seed 0 --i-allow-gpu
    ;;
  oracle)
    gpu_run python spec/phase4_campaign.py --diagnostic-oracle-dist --seed 0 --i-allow-gpu
    ;;
  binary)
    gpu_run python spec/phase4_campaign.py --diagnostic-binary-lat --seed 0 --i-allow-gpu
    ;;
  encoder)
    ENC_TYPE="${2:?}"
    ENC_READ="${3:?}"
    ENC_SEED="${4:?}"
    SMOKE_FLAG=()
    if [ "${5:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run python spec/phase4_campaign.py --diagnostic-encoder --encoder-type "$ENC_TYPE" --readout-type "$ENC_READ" --seed "$ENC_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  bestofk)
    BOK_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run python spec/phase4_campaign.py --diagnostic-bestofk --seed "$BOK_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  eas)
    EAS_SEED="${2:?}"
    EAS_SUBSET="${3:-lastlayer}"
    EAS_LOSS="${4:-pg_il}"
    SMOKE_FLAG=()
    if [ "${5:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run python spec/phase4_campaign.py --diagnostic-eas --adapt-subset "$EAS_SUBSET" --loss "$EAS_LOSS" --seed "$EAS_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  easinst)
    EAS_SEED="${2:?}"
    EAS_BUDGET="${3:-32}"
    SMOKE_FLAG=()
    if [ "${4:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run python spec/phase4_campaign.py --diagnostic-eas-inst --adapt-subset lastlayer --loss pg --eas-budget "$EAS_BUDGET" --seed "$EAS_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  expertprof)
    EP_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    cpu_run python spec/phase4_campaign.py --diagnostic-expert-profiles --seed "$EP_SEED" "${SMOKE_FLAG[@]}"
    ;;
  expertprofval)
    EP_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    cpu_run python spec/phase4_campaign.py --diagnostic-expert-profiles --expert-split validation --seed "$EP_SEED" "${SMOKE_FLAG[@]}"
    ;;
  expertenergy)
    EE_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    cpu_run python spec/phase4_campaign.py --diagnostic-expert-energy --seed "$EE_SEED" "${SMOKE_FLAG[@]}"
    ;;
  expertenergyval)
    EE_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    cpu_run python spec/phase4_campaign.py --diagnostic-expert-energy --expert-split validation --seed "$EE_SEED" "${SMOKE_FLAG[@]}"
    ;;
  fitstatsv2)
    cpu_run python spec/fit_encoder_stats.py --obs-version v2
    ;;
  bcprof)
    BC_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run_v2 python spec/phase4_campaign.py --diagnostic-bc-profiles --seed "$BC_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  easprof)
    EAS_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run_v2 python spec/phase4_campaign.py --diagnostic-eas-profiles --seed "$EAS_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  ctxprof)
    CTX_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run_v2 python spec/phase4_campaign.py --diagnostic-ctx-profiles --seed "$CTX_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  bokprof)
    BOK_SEED="${2:?}"
    SMOKE_FLAG=()
    if [ "${3:-}" = "smoke" ]; then
      SMOKE_FLAG=(--smoke)
    fi
    gpu_run_v2 python spec/phase4_campaign.py --diagnostic-bok-profiles --seed "$BOK_SEED" --i-allow-gpu "${SMOKE_FLAG[@]}"
    ;;
  train0)
    gpu_run python spec/phase4_campaign.py --execute-train --seed 0 --i-allow-gpu
    ;;
  gate)
    gpu_run python spec/phase4_gate.py
    gpu_run python spec/phase3_gate.py
    ;;
  *)
    echo "usage: $0 probe|gate|smoke|train0|audit5|par5|par500|lat50|pomo50|bc50|bconly|bccont|klppo50|bcunseen|bcfew|bcss|h2|motif|twopt|pairfrac|bc2opt|pairhead|pairksweep|pairranker|pairseq|cavia|caviaE|caviaS|pairsup|caviaB|rewrite|oracle|binary|encoder|bestofk|eas|easinst|expertprof|expertprofval|expertenergy|expertenergyval|fitstatsv2|bcprof|easprof|ctxprof|bokprof|diag1k|diag200" >&2
    exit 2
    ;;
esac
