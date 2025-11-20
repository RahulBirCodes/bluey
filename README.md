## Hyperparameter sweep job generation (deterministic)
```
python3 scripts/sweep.py \
    --xy_size 5 \
    --num_pairs 48 \
    --project_name bluey-merdifold \
    --last_k 50 \
    --output_dir jobs
```

## Run specific job

```

run_job.sh OPTIMIZER ARCH JOB_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY
```

## Run multiple jobs
```
chmod +x run_job.sh run_jobs.sh

run_jobs.sh OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY

scripts/run_jobs.sh AdamW rms 1 899 sweep cpu checkpoints 50000 500

```

Wandb auths the first time you do init call if not alr auth'd.
