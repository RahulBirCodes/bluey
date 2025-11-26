#!/bin/bash
set -e

# Usage:
#   ./run_jobs_local.sh OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY [NUM_GPUS]
# Example:
#   ./run_jobs_local.sh AdamW rms 0 15 sweep cuda checkpoints 3000 200 4
#
# If NUM_GPUS is omitted, defaults to 1 (serial).

if [ "$#" -lt 9 ] || [ "$#" -gt 10 ]; then
  echo "Usage: $0 OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY [NUM_GPUS]"
  exit 1
fi

OPTIMIZER=$1    # e.g. AdamW
ARCH=$2         # e.g. rms
START=$3        # e.g. 0
END=$4          # e.g. 15
PHASE=$5        # e.g. sweep
DEVICE=$6       # cpu | cuda | tpu | auto
CKPT_ROOT=$7    # e.g. checkpoints
NUM_STEPS=$8    # e.g. 3000
CHECKPOINT_EVERY=$9
NUM_GPUS=${10:-1}   # Optional; default 1

echo "Using up to ${NUM_GPUS} GPU(s)"

# simple concurrency limiter
active_jobs=0

for i in $(seq "$START" "$END"); do
  JOB_ID=$(printf "%03d" "$i")
  GPU_ID=$(( i % NUM_GPUS ))   # round-robin assignment

  echo "=== Launching job ${JOB_ID} on GPU ${GPU_ID} (${OPTIMIZER}, ${ARCH}) ==="

  if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "auto" ]; then
    # Bind this job to one GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID \
      scripts/run_job.sh "$OPTIMIZER" "$ARCH" "$JOB_ID" "$PHASE" "cuda" "$CKPT_ROOT" "$NUM_STEPS" "$CHECKPOINT_EVERY" &
  else
    # CPU / TPU case: no CUDA_VISIBLE_DEVICES
    scripts/run_job.sh "$OPTIMIZER" "$ARCH" "$JOB_ID" "$PHASE" "$DEVICE" "$CKPT_ROOT" "$NUM_STEPS" "$CHECKPOINT_EVERY" &
  fi

  active_jobs=$((active_jobs + 1))

  # if we already have NUM_GPUS jobs running, wait for one to finish
  if [ "$active_jobs" -ge "$NUM_GPUS" ]; then
    # wait for any one background job to finish (bash 4.3+)
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

# wait for all remaining jobs
wait
echo "All jobs finished."
