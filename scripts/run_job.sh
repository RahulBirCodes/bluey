#!/bin/bash
set -e

OPTIMIZER=$1        # e.g. muon
ARCH=$2             # e.g. rms or standard
JOB_ID=$3           # integer like 014

CONFIG="jobs/${OPTIMIZER}/${ARCH}/job_${JOB_ID}.json"

echo "Running config: $CONFIG"
python3 scripts/train_from_config.py --config "$CONFIG"