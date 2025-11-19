#!/bin/bash
set -e

# Usage:
#   ./run_jobs_local.sh OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS
# Example:
#   ./run_jobs_local.sh muon rms 0 15 sweep cuda checkpoints 3000

if [ "$#" -ne 8 ]; then
  echo "Usage: $0 OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS"
  exit 1
fi

OPTIMIZER=$1    # e.g. muon
ARCH=$2         # e.g. rms
START=$3        # e.g. 0
END=$4          # e.g. 15
PHASE=$5        # e.g. sweep
DEVICE=$6       # cpu | cuda | tpu | auto
CKPT_ROOT=$7    # e.g. checkpoints
NUM_STEPS=$8    # e.g. 3000

for i in $(seq "$START" "$END"); do
  JOB_ID=$(printf "%03d" "$i")
  echo "=== Running job ${JOB_ID} (${OPTIMIZER}, ${ARCH}) ==="
  ./run_job.sh "$OPTIMIZER" "$ARCH" "$JOB_ID" "$PHASE" "$DEVICE" "$CKPT_ROOT" "$NUM_STEPS" $
done
