import torch
import torch.nn.functional as F
import wandb
import os
import itertools
import hashlib
from collections import deque
import time
import wandb
import datetime
import re

# Optional TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

def resolve_device_and_saver(device_str: str):
    """
    Returns (torch.device-like, save_fn, optimizer_step_fn).
    """
    if device_str.lower() == "tpu":
        if not HAS_XLA:
            raise RuntimeError("TPU requested but torch_xla is not installed.")
        device = xm.xla_device()
        save_fn = xm.save

        def optimizer_step_fn(optimizer):
            xm.optimizer_step(optimizer)
            xm.mark_step()

    else:
        device = torch.device(device_str)
        save_fn = torch.save

        def optimizer_step_fn(optimizer):
            optimizer.step()

    return device, save_fn, optimizer_step_fn


def save_checkpoint(model, optimizer, step: int, path: str, scheduler=None, save_fn=torch.save):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    save_fn(ckpt, path)
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(model, optimizer, path: str, device="cuda", scheduler=None) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    torch.random.set_rng_state(ckpt["rng_state"])
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    print(f"[checkpoint] resumed from {path}")
    return ckpt["step"]


class WarmupConstantDecayLrScheduler:
    def __init__(self, optimizer, total_steps, warmup_ratio=0.02, decay_ratio=0.10):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps = int(total_steps * decay_ratio)
        self.decay_start = total_steps - self.decay_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.last_step = 0

    def state_dict(self):
        return {"last_step": self.last_step}

    def load_state_dict(self, state):
        self.last_step = state["last_step"]

    def step(self):
        step = self.last_step
        if step < self.warmup_steps and warmup_steps != 0:
            scale = step / self.warmup_steps
        elif step < self.decay_start:
            scale = 1.0
        else:
            remaining = max(1, self.total_steps - self.decay_start)
            scale = max(0.0, 1 - (step - self.decay_start) / remaining)

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale
        self.last_step += 1


def train(
    model,
    optimizer,
    logger,
    *,
    get_batch,
    batch_size=8,
    num_pairs=5,
    xy_size=5,
    num_steps=1000,
    device="cuda",
    verbose=True,
    print_interval=20,
    checkpoint_every=20,
    checkpoint_dir=None,
    resume_from: str | None = None,
    scheduler=None,
):
    device, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    model.to(device)
    model.train()

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    prev_step = 0
    if resume_from:
        prev_step = load_checkpoint(model, optimizer, resume_from, device=device, scheduler=scheduler)
       
    for step in range(prev_step, num_steps):
        iter_start = time.time()
        tokens, X, Y, W, x_token_indices = get_batch(
            batch_size=batch_size,
            num_pairs=num_pairs,
            xy_size=xy_size,
            device=device,
        )
        outputs = model(tokens)
        B, S, D = outputs.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1)
        y_pred = outputs[b_idx, x_token_indices, :]
        loss = torch.sum((y_pred-Y)**2, dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer_step_fn(optimizer)
        if scheduler is not None:
            scheduler.step()

        # --- Checkpointing ---
        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            now = datetime.datetime.now()  # or datetime.datetime.utcnow()
            timestamp = now.strftime("%Y%m%d-%H%M%S")  # e.g. 20251118-123045
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step+1}_{timestamp}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step + 1,
                path=ckpt_path,
                scheduler=scheduler,
                save_fn=save_fn,
            )
            if verbose:
                print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

        if verbose and (step % print_interval == 0):
            print(f"[Step {step}] loss = {loss.item():.6f}")

        if logger is not None:
            iter_sec = time.time() - iter_start
            logger.log({"train/loss": loss.item(), "step": step, "train/iter_sec": iter_sec})

    return model


class WandbLossLogger:
    """
    Wraps a wandb.Run-like object to:
      - forward logs to wandb
      - keep a rolling window of the last K 'loss' values
    """
    def __init__(self, run, last_k: int = 50):
        self.run = run
        self.last_k = deque(maxlen=last_k)
    
    def log(self, metrics: dict):
        if "loss" in metrics:
            self.last_k.append(metrics["loss"])
        self.run.log(metrics)
    
    def finish(self):
        self.run.finish()


def _short_hparam_str(hparams: dict, max_len: int =200) -> str:
    """
    Turn a small hyperparam dict into a compact, filesystem-safe string.
    Example: {'lr':1e-3,'wd':0.1} -> 'lr1e-3_wd0.1' (possibly truncated + hash).
    """
    parts = []
    for k, v in hparams.items():
        # Normalize floats for readability
        if isinstance(v, float):
            v_str = f"{v:.1e}" if (v < 0.01 or v > 1000) else str(v)
        else:
            v_str = str(v)
        parts.append(f"{k}{v_str}")
    base = "_".join(parts)
    if len(base) <= max_len:
        return base
    # Truncate and append hash so we keep uniqueness but stay short
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    return base[: max_len - 7] + "_" + h


def _iter_hparam_configs(hyperparam_grid: dict):
    """
    Given {"lr":[1e-4,1e-3], "wd":[0.0,0.1]}, yield:
        {"lr":1e-4,"wd":0.0}, {"lr":1e-4,"wd":0.1}, ...
    """
    keys = list(hyperparam_grid.keys())
    values = [hyperparam_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))



def find_checkpoint(ckpt_dir: str) -> str | None:
    """
    Look in `ckpt_dir` for checkpoint files named like:
      step_<STEP>_TIMESTAMP.pt
    and return the path to the one with the largest STEP.
    Returns None if no checkpoints found.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    best_step = -1
    best_path = None

    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue

        #filenames formatted "step_1234_20251118-120000.pt"
        m = re.search(r"step_(\d+)_", fname)
        if not m:
            continue
        step = int(m.group(1))

        if step > best_step:
            best_step = step
            best_path = os.path.join(ckpt_dir, fname)

    return best_path

def _wandb_run_is_complete(
    project_name: str,
    run_id: str,
    num_steps: int,
    *,
    entity: str | None = None,
):
    """
    Returns (is_complete, summary_dict).

    is_complete:
      True  -> run exists, is finished, and max logged step >= num_steps-1
      False -> run not found, not finished, or step < num_steps-1

    summary_dict:
      run.summary as a dict (or None if run missing / incomplete).
    """
    api = wandb.Api()
    # Entity: prefer env var if not passed
    if entity is None:
        entity = os.environ.get("WANDB_ENTITY")

    # Path format: "entity/project/run_id" OR "project/run_id" if no entity set
    if entity:
        path = f"{entity}/{project_name}/{run_id}"
    else:
        path = f"{project_name}/{run_id}"

    try:
        run = api.run(path)
    except Exception:
        # No such run or network issue → treat as incomplete
        return False, None

    # If you only want fully finished runs:
    if run.state != "finished":
        return False, None

    # Scan history for max `step`
    max_step = -1
    # keys=["step"] keeps it light; index=False avoids DF index column in older wandb versions
    for row in run.history(keys=["step"], pandas=False):
        step_val = row.get("step")
        if isinstance(step_val, (int, float)):
            if step_val > max_step:
                max_step = int(step_val)

    # In your code, steps go from 0..num_steps-1.
    # So if max_step >= num_steps-1, consider run "complete".
    if max_step >= num_steps - 1:
        # Grab summary values if you logged them earlier
        summary = dict(run.summary) if run.summary is not None else {}
        return True, summary

    return False, None

def _run_single_config(
    experiment_phase: str,
    arch_name: str,
    make_model,
    optimizer_name: str,
    optimizer_class,
    hparams: dict,
    get_batch,
    num_pairs: int,
    xy_size: int,
    *,
    num_steps: int,
    device: str,
    project_name: str,
    base_ckpt_dir: str,
    last_k: int,
    checkpoint_every: int,
    continue_checkpoint: bool=False,
    skip_complete: bool=False,
):
    """
    Run a single (arch, hyperparam config) training job.
    Returns a dict with summary stats.
    """
    hparam_str = _short_hparam_str(hparams)
    # base_ckpt_dir/phase/optimizer/arch/hparam_str/
    run_id = hparam_str[:128] 

    current_time = time.time()
    time_hash = hash(current_time)

    ckpt_dir = os.path.join(
        base_ckpt_dir,
        experiment_phase,
        optimizer_name,
        arch_name,
        hparam_str,
        
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if skip_complete:
        is_complete, wb_summary = _wandb_run_is_complete(
            project_name=project_name,
            run_id=run_id,
            num_steps=num_steps,
        )
        
        if is_complete:
            avg_last_k_loss = wb_summary.get("avg_last_k_train_loss", float("nan"))
            final_eval_loss = wb_summary.get("final_eval_loss", float("nan"))

            print(f"[SKIP] {arch_name} / {optimizer_name} / {hparam_str} already complete "
                f"(max step >= {num_steps-1}). Skipping train().")

            return {
                "experiment_phase": experiment_phase,
                "optimizer": optimizer_name,
                "arch": arch_name,
                "hparams": hparams,
                "ckpt_dir": ckpt_dir,
                "run_name": run_id,
                "group_name": f"{experiment_phase}/{optimizer_name}/{arch_name}",
                "final_eval_loss": final_eval_loss,
                "avg_last_k_train_loss": avg_last_k_loss,
            }
    

    model = make_model(arch_name)
    opt_kwargs = {}
    if "lr" in hparams:
        opt_kwargs["lr"] = hparams["lr"]
    if "weight_decay" in hparams:
        opt_kwargs["weight_decay"] = hparams["weight_decay"]
    if "beta1" in hparams and "beta2" in hparams:
        opt_kwargs["betas"] = (hparams["beta1"], hparams["beta2"])
    for k in ["momentum", "nesterov"]:
        if k in hparams:
            opt_kwargs[k] = hparams[k]
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)

    resume_from = None
    wandb_resume = "never"
    if continue_checkpoint:
        resume_from = find_checkpoint(ckpt_dir)
        wandb_resume = "allow"

    run_name = f"{hparam_str}"
    group_name = f"{experiment_phase}/{optimizer_name}/{arch_name}"

    run = wandb.init(
        id=run_id,
        project=project_name,
        name=run_id,  # wandb name limit
        group=group_name,
        config={
            "experiment_phase": experiment_phase,
            "optimizer": optimizer_name,
            "arch": arch_name,
            "num_steps": num_steps,
            "device": device,
            **hparams,
        },
        reinit=True,
        resume=wandb_resume
    )

    logger = WandbLossLogger(run, last_k=last_k)
    
    model = train(
        model=model,
        optimizer=optimizer,
        logger=logger,
        get_batch=get_batch,
        batch_size=hparams.get("batch_size", 8),
        num_pairs=num_pairs,
        xy_size=xy_size,
        num_steps=num_steps,
        device=device,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=ckpt_dir,
        resume_from=resume_from,
        verbose=True,
    )
    if len(logger.last_k) > 0:
        avg_last_k_loss = sum(logger.last_k) / len(logger.last_k)
    else:
        avg_last_k_loss = float("nan")
    run.log({"avg_last_k_train_loss": avg_last_k_loss})
    logger.finish()

    return {
        "experiment_phase": experiment_phase,
        "optimizer": optimizer_name,
        "arch": arch_name,
        "hparams": hparams,
        "ckpt_dir": ckpt_dir,
        "run_name": run_name,
        "group_name": group_name,
        "avg_last_k_train_loss": avg_last_k_loss,
    }

def hyperparameter_sweep(
    experiment_phase: str,
    model_architectures: list[str],
    make_model,
    optimizer_name: str,
    optimizer_class,
    hyperparam_grid: dict[str, list],
    checkpoint_every: int,
    xy_size: int = 5,
    num_pairs: int = 5,
    *,
    get_batch,
    num_steps: int,
    device: str,
    project_name: str = "bluey-merdifold",
    base_ckpt_dir: str = "checkpoints",
    last_k: int = 50,
    continue_checkpoint: bool=False,
    skip_complete: bool=False,
):
    """
    Run a grid search over hyperparameters *and* architectures
    for a given optimizer, with organized wandb + checkpoint structure.

    Returns:
        results: list of training run results
    """
    results = []
    
    for arch_name in model_architectures:
            for hparams in _iter_hparam_configs(hyperparam_grid):
                print(hparams)
                summary = _run_single_config(
                    experiment_phase,
                    arch_name,
                    make_model,
                    optimizer_name,
                    optimizer_class,
                    hparams,
                    checkpoint_every=checkpoint_every,
                    get_batch=get_batch,
                    num_pairs=num_pairs,
                    xy_size=xy_size,
                    num_steps=num_steps,
                    device=device,
                    project_name=project_name,
                    base_ckpt_dir=base_ckpt_dir,
                    last_k=last_k,
                    continue_checkpoint=continue_checkpoint,
                    skip_complete=skip_complete,
                )
                results.append(summary)

    return results

