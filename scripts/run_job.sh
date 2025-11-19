#!/bin/bash
set -e

if [ $# -ne 7 ]; then
    echo "Usage: $0 OPTIMIZER ARCH JOB_ID PHASE DEVICE CKPT_ROOT NUM_STEPS"
    echo "Example: $0 muon rms 003 sweep cuda checkpoints 3000"
    exit 1
fi

OPTIMIZER=$1
ARCH=$2
JOB_ID=$3
PHASE=$4
DEVICE=$5
CKPT_ROOT=$6
NUM_STEPS=$7

CONFIG="jobs/${OPTIMIZER}/${ARCH}/job_${JOB_ID}.json"

echo "Running config: $CONFIG"
python3 ../main.py \
    --config "$CONFIG" \
    --phase "$PHASE" \
    --num-steps "$NUM_STEPS" \
    --device "$DEVICE" \
    --ckpt-root "$CKPT_ROOT" \
    --job-id "$JOB_ID"