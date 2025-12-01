#!/bin/bash
set -e

if [ $# -ne 9 ]; then
    echo "Usage: $0 OPTIMIZER ARCH LIPS_FLAG JOB_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY"
    echo "Example: $0 muon rms lips 003 sweep cuda checkpoints 3000 200"
    exit 1
fi

OPTIMIZER=$1
ARCH=$2
LIPS_FLAG=$3   # "lips" or "nolips"
JOB_ID=$4
PHASE=$5
DEVICE=$6
CKPT_ROOT=$7
NUM_STEPS=$8
CHECKPOINT_EVERY=$9

if [[ "$LIPS_FLAG" != "lips" && "$LIPS_FLAG" != "nolips" ]]; then
    echo "Error: LIPS_FLAG must be 'lips' or 'nolips'"
    exit 1
fi

CONFIG="bluey/jobs/${OPTIMIZER}/${ARCH}/${LIPS_FLAG}/job_${JOB_ID}.json"

echo "Running config: $CONFIG"

python3 -m bluey.main \
    --config "$CONFIG" \
    --phase "$PHASE" \
    --num-steps "$NUM_STEPS" \
    --device "$DEVICE" \
    --ckpt-root "$CKPT_ROOT" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --job-id "$JOB_ID" \