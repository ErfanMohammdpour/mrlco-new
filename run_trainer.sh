#!/bin/bash

# Script to run meta_trainer.py with different device configurations

echo "Usage: $0 [cpu|gpu|multi-gpu]"
echo ""

MODE=${1:-gpu}

case $MODE in
    cpu)
        echo "Running in CPU-only mode..."
        export CUDA_VISIBLE_DEVICES=""
        export TF_CPP_MIN_LOG_LEVEL=2
        python meta_trainer.py
        ;;
    gpu)
        echo "Running with single GPU..."
        export CUDA_VISIBLE_DEVICES="0"
        export TF_CPP_MIN_LOG_LEVEL=2
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        python meta_trainer.py
        ;;
    multi-gpu)
        echo "Running with all available GPUs..."
        unset CUDA_VISIBLE_DEVICES
        export TF_CPP_MIN_LOG_LEVEL=2
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        python meta_trainer.py
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Valid modes: cpu, gpu, multi-gpu"
        exit 1
        ;;
esac