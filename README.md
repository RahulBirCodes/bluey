## Install the repository
```
git clone https://github.com/RahulBirCodes/bluey.git
cd bluey
```
## Environment set-up
```
uv lock
```
or if uv uninstalled
```
pip install .
```
## Hyperparameter sweep job generation (deterministic)
```
python3 scripts/sweep.py \
    --xy_size 5 \
    --project_name bluey-merdifold \
    --last_k 50 \
    --output_dir jobs
```

## Run specific job

```

run_job.sh OPTIMIZER ARCH JOB_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY
```

## Run multiple jobs
You can vary the number of devices by adding an optional number of GPU flag
```
chmod +x run_job.sh run_jobs.sh

run_jobs.sh OPTIMIZER ARCH START_ID END_ID PHASE DEVICE CKPT_ROOT NUM_STEPS CHECKPOINT_EVERY [NUM_GPU]

scripts/run_job.sh ManifoldMuonW none 1 sweep cpu checkpoints 200 200



```

Wandb auths the first time you do init call if not alr auth'd.
