#!/bin/bash
set -e

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 OPTIMIZER ARCH START_ID END_ID"
  echo "Example: $0 muon rms 0 15   # submits job_000..job_015 as separate SLURM jobs"
  exit 1
fi

OPTIMIZER=$1
ARCH=$2
START=$3
END=$4

for i in $(seq "$START" "$END"); do
  JOB_ID=$(printf "%03d" "$i")
  JOB_NAME="${OPTIMIZER}_${ARCH}_${JOB_ID}"

  echo "Submitting SLURM job $JOB_NAME"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}.out
#SBATCH --error=logs/${JOB_NAME}.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Activate env here if needed:
# source ~/.bashrc
# conda activate bluey-env

cd "$(pwd)"
./run_job.sh "${OPTIMIZER}" "${ARCH}" "${JOB_ID}"
EOF

done
