import torch
import torch.nn.functional as F
import wandb
import os
import itertools
import hashlib
from collections import deque

import wandb

# Optional TPU support
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

def resolve_device_and_saver(device_str: str):
    """
    Returns (torch.device-like, save_fn, optimizer_step_fn).

    device_str:
      - "cpu"      → CPU
      - "cuda"     → current CUDA device
      - "cuda:0"   → specific CUDA device
      - "tpu"      → XLA device (requires torch_xla)
    """
    if device_str.lower() == "tpu":
        if not HAS_XLA:
            raise RuntimeError("TPU requested but torch_xla is not installed.")
        device = xm.xla_device()
        save_fn = xm.save

        def optimizer_step_fn(optimizer):
            xm.optimizer_step(optimizer)

    else:
        device = torch.device(device_str)
        save_fn = torch.save

        def optimizer_step_fn(optimizer):
            optimizer.step()

    return device, save_fn, optimizer_step_fn


def save_checkpoint(model, optimizer, step, epoch, path, scheduler=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    torch.save(ckpt, path)
    print(f"[checkpoint] saved to {path}")

def load_checkpoint(model, optimizer, path, device="cuda", scheduler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    torch.random.set_rng_state(ckpt["rng_state"])
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    print(f"[checkpoint] resumed from {path}")
    return ckpt["step"], ckpt["epoch"]




def train(
    model, #Should be a transformer model
    optimizer, #Could be AdamW, ManifoldMuonW, or MuonW
    logger, #Should be a wandb logger
    *,
    get_batch, #Function that returns tokens, X, Y, W, and indices of tokens that refer to X tokens
    batch_size=8,
    num_pairs=5,
    xy_size=5,
    num_steps=1000,
    device="cuda", #Could be "TPU", "cuda", "cpu"
    verbose=True,
    print_interval=20,
    checkpoint_every=20,
    checkpoint_dir=None,
):
    """
    Train model on synthetic least-squares data.

    Assumes:
      - get_batch(...) returns (tokens, X, Y, W, x_token_indices)
      - tokens: (B, T_seq, token_dim)
      - X, Y:  (B, num_pairs, xy_size)
      - model(tokens) -> (B, T_seq, xy_size)
      - x_token_indices: positions where *x* is input; outputs at these
        positions are interpreted as predictions of the next y.
    """
    device, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    model.to(device)
    model.train()

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for step in range(num_steps):
        # --- Generate fresh synthetic batch every step ---
        # Yes: for synthetic LS tasks, it's standard to resample each iteration.
        tokens, X, Y, W, x_token_indices = get_batch(
            batch_size=batch_size,
            num_pairs=num_pairs,
            xy_size=xy_size,
            device=device,
        )
        # tokens, X, Y, etc. should already be on the right device

        # FWD
        outputs = model(tokens)               # (B, T_seq, xy_size)

        B, S, D = outputs.shape
        # Gather predictions at x-token positions
        # x_token_indices: 1D tensor (num_pairs,) of time indices
        # We want outputs[:, x_token_indices, :] -> (B, num_pairs, xy_size)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S)

        y_pred = outputs[b_idx, x_token_indices, :]

        # Least-squares / MSE loss between predicted y’s and true Y
        loss = torch.sum((y_pred-Y)**2, dim=1).mean()

        # BWD
        optimizer.zero_grad()
        loss.backward()
        optimizer_step_fn(optimizer)

        # --- Checkpointing ---
        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step+1}.pt")
            state = {
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss.item(),
            }
            save_fn(state, ckpt_path)
            if verbose:
                print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

        # --- Print logging ---
        if verbose and (step % print_interval == 0):
            print(f"[Step {step}] loss = {loss.item():.6f}")

        # --- WandB logging ---
        if logger is not None:
            logger.log({"loss": loss.item(), "step": step})

    if logger is not None:
        logger.finish()

    return model


def load_checkpoint(model, optimizer, filepath, device="cuda"):
    """
    Device-aware checkpoint loading.
    """
    device_obj, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    checkpoint = torch.load(filepath, map_location=device_obj)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint["step"]
    loss = checkpoint["loss"]
    model.to(device_obj)
    return model, optimizer, step, loss


# Optional Ray support
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


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


def _short_hparam_str(hparams: dict, max_len: int = 30) -> str:
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


def _run_single_config(
    experiment_phase: str,
    arch_name: str,
    make_model,
    optimizer_name: str,
    optimizer_class,
    hparams: dict,
    *,
    get_batch,
    num_steps: int,
    device: str,
    project_name: str,
    base_ckpt_dir: str,
    last_k: int,
):
    """
    Run a single (arch, hyperparam config) training job.
    Returns a dict with summary stats.
    """
    # --- Build names/paths ---
    hparam_str = _short_hparam_str(hparams)  # e.g. 'lr1e-3_wd0.1'
    # Checkpoint dir tree:
    # base_ckpt_dir/phase/optimizer/arch/hparam_str/
    ckpt_dir = os.path.join(
        base_ckpt_dir,
        experiment_phase,
        optimizer_name,
        arch_name,
        hparam_str,
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # WandB run name & group
    run_name = f"{experiment_phase}-{optimizer_name}-{arch_name}-{hparam_str}"
    # Optionally use group to mirror phase/optimizer/arch
    group_name = f"{experiment_phase}/{optimizer_name}/{arch_name}"

    run = wandb.init(
        project=project_name,
        name=run_name[:128],  # wandb name limit
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
    )
    logger = WandbLossLogger(run, last_k=last_k)

    # --- Build model & optimizer ---
    model = make_model(arch_name)
    optimizer = optimizer_class(model.parameters(), **hparams)

    # --- Train ---
    model = train(
        model=model,
        optimizer=optimizer,
        logger=logger,
        get_batch=get_batch,
        batch_size=hparams.get("batch_size", 8),
        num_pairs=hparams.get("num_pairs", 5),
        xy_size=hparams.get("xy_size", 5),
        num_steps=num_steps,
        device=device,
        checkpoint_every=hparams.get("checkpoint_every", 50),
        checkpoint_dir=ckpt_dir,
        verbose=True,
    )

    # --- Summaries ---
    # Average of last K training losses
    if len(logger.last_k) > 0:
        avg_last_k_loss = sum(logger.last_k) / len(logger.last_k)
    else:
        avg_last_k_loss = float("nan")

    # Evaluate on a fresh synthetic batch as a "final loss"
    model.eval()
    with torch.no_grad():
        tokens, X, Y, W, x_token_indices = get_batch(
            batch_size=hparams.get("eval_batch_size", 8),
            num_pairs=hparams.get("num_pairs", 5),
            xy_size=hparams.get("xy_size", 5),
            device=device,
        )
        outputs = model(tokens)
        y_pred = outputs[:, x_token_indices, :]
        final_loss = torch.nn.functional.mse_loss(y_pred, Y).item()

    wandb.log(
        {
            "final_eval_loss": final_loss,
            "avg_last_k_train_loss": avg_last_k_loss,
        }
    )

    logger.finish()

    return {
        "experiment_phase": experiment_phase,
        "optimizer": optimizer_name,
        "arch": arch_name,
        "hparams": hparams,
        "ckpt_dir": ckpt_dir,
        "run_name": run_name,
        "group_name": group_name,
        "final_eval_loss": final_loss,
        "avg_last_k_train_loss": avg_last_k_loss,
    }


def hyperparameter_sweep(
    experiment_phase: str,           # "sweep", "exp1", etc.
    model_architectures: list[str],  # ["ln", "no_ln_resnet", ...]
    make_model,                      # callable arch_name -> nn.Module
    optimizer_name: str,             # "AdamW", "MuonW", "ManifoldMuonW"
    optimizer_class,                 # e.g. torch.optim.AdamW
    hyperparam_grid: dict[str, list],
    *,
    get_batch,
    num_steps: int,
    device: str,
    project_name: str = "bluey-merdifold",
    base_ckpt_dir: str = "checkpoints",
    last_k: int = 50,
    use_ray: bool = False,
):
    """
    Run a grid search over hyperparameters *and* architectures
    for a given optimizer, with organized wandb + checkpoint structure.

    Returns:
        results: list of dicts, each with keys:
            - experiment_phase, optimizer, arch
            - hparams (dict)
            - ckpt_dir
            - run_name, group_name
            - final_eval_loss
            - avg_last_k_train_loss
    """
    results = []

    if use_ray:
        if not HAS_RAY:
            raise RuntimeError("use_ray=True but Ray is not installed.")
        if not ray.is_initialized():
            ray.init()

        @ray.remote
        def _remote_run_single_config(*args, **kwargs):
            return _run_single_config(*args, **kwargs)

        ray_jobs = []
        for arch_name in model_architectures:
            for hparams in _iter_hparam_configs(hyperparam_grid):
                job = _remote_run_single_config.remote(
                    experiment_phase,
                    arch_name,
                    make_model,
                    optimizer_name,
                    optimizer_class,
                    hparams,
                    get_batch=get_batch,
                    num_steps=num_steps,
                    device=device,
                    project_name=project_name,
                    base_ckpt_dir=base_ckpt_dir,
                    last_k=last_k,
                )
                ray_jobs.append(job)

        results = ray.get(ray_jobs)

    else:
        for arch_name in model_architectures:
            for hparams in _iter_hparam_configs(hyperparam_grid):
                summary = _run_single_config(
                    experiment_phase,
                    arch_name,
                    make_model,
                    optimizer_name,
                    optimizer_class,
                    hparams,
                    get_batch=get_batch,
                    num_steps=num_steps,
                    device=device,
                    project_name=project_name,
                    base_ckpt_dir=base_ckpt_dir,
                    last_k=last_k,
                )
                results.append(summary)

    return results

