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

Wandb auths the first time you do init call if not alr auth'd.