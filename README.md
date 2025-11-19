Call generate_configs.py with your desired experiment set-up

python scripts/generate_configs.py \
  --experiment-phase sweep \
  --xy-size 5 \
  --num-pairs 20 \
  --num-steps 100 \
  --checkpoint-every 20 \
  --device cuda \
  --project-name bluey-merdifold \
  --base-ckpt-dir checkpoints \
  --last-k 50 \
  --print-config

  On SLURM side: 

  ./run_jobs.sh AdamW rms 0 15

