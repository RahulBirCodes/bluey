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

## Hyperparameter sweep job generation (deterministic)
```
python3 scripts/sweep.py \
    --xy_size 5 \
    --num_pairs 48 \
    --project_name bluey-merdifold \
    --last_k 50 \
    --checkpoint_every 200 \
    --output_dir jobs
```

## Run specific job
```
run_job.sh OPTIMIZER ARCH JOB_ID PHASE DEVICE CKPT_ROOT NUM_STEPS
```

## Run multiple jobs
```
!python main.py \
    --optimizer AdamW \
    --arch rms \
    --start-id 0 \
    --end-id 15 \
    --phase sweep \
    --device cpu \
    --ckpt-root "$CKPT_ROOT" \
    --num-steps 100

```

Wandb auths the first time you do init call if not alr auth'd.
