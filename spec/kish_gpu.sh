#!/bin/bash
# kish-ai helpers. Do not run on Apple Silicon.
set -euo pipefail
IMAGE="${MARGO_GPU_IMAGE:-margo-phase4-tf115-gpu}"
ROOT="${MARGO_ROOT:-/opt/margo/mrlco-new}"
cd "$ROOT"

gpu_run() {
  docker run --rm --gpus all \
    -e MARGO_ALLOW_GPU=1 \
    -e PYTHONUNBUFFERED=1 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
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
  train0)
    gpu_run python spec/phase4_campaign.py --execute-train --seed 0 --i-allow-gpu
    ;;
  gate)
    gpu_run python spec/phase4_gate.py
    gpu_run python spec/phase3_gate.py
    ;;
  *)
    echo "usage: $0 probe|gate|smoke|train0" >&2
    exit 2
    ;;
esac
