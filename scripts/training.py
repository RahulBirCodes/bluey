import torch
import torch.nn.functional as F
import wandb
import os
import glob
from collections import deque
import time
import torch_xla
import wandb
from optimizers.muonW1 import MuonW
from optimizers.manifold_muonW import ManifoldMuonW
from ..types.config_types import OptimizerKwargs, ExperimentConfig
from ..model.model import make_model
from dataset import get_batch as get_ols_batch

# Optional TPU support
try:
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

    else:
        device = torch.device(device_str)
        save_fn = torch.save

        def optimizer_step_fn(optimizer):
            optimizer.step()

    return device, save_fn, optimizer_step_fn


def save_checkpoint(model, optimizer, step, path, scheduler=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
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
    return ckpt["step"]


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not files:
        return None
    parsed = []
    for path in files:
        base = os.path.basename(path)  
        # expected pattern: step_<step>_time_<time>.pt
        parts = base.split("_")   # ['step', '50', 'time', '1731954200.1234.pt']
        if len(parts) < 4:
            continue
        if parts[0] != "step" or parts[2] != "time":
            continue
        try:
            step = int(parts[1])
            timestamp_str = parts[3].split(".pt")[0]
            timestamp = float(timestamp_str)
        except ValueError:
            continue
        parsed.append((path, step, timestamp))
    if not parsed:
        return None
    parsed.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return parsed[0][0]


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
        if step < self.warmup_steps and self.warmup_steps != 0:
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
    resume_from=None,
    scheduler=None
):
    device, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    model.to(device)
    model.train()

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    prev_step = 0
    if resume_from is not None:
        prev_step = load_checkpoint(model, optimizer, resume_from, scheduler=scheduler, device=device)

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

        if checkpoint_dir is not None and step % checkpoint_every == 0 and step != 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step+1}_time_{iter_start}.pt")
            state = {
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss.item(),
                "scheduler": scheduler.state_dict()
            }
            save_fn(state, ckpt_path)
            if verbose:
                print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

        if verbose and (step % print_interval == 0):
            print(f"[Step {step}] loss = {loss.item():.6f}")

        if logger is not None:
            iter_sec = time.time() - iter_start
            logger.log({"train/loss": loss.item(), "step": step, "train/iter_sec": iter_sec})

    if logger is not None:
        logger.finish()

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
    
    def get_last_k_loss(self):
        return sum(self.last_k) / len(self.last_k)
    
    def finish(self):
        self.run.finish()


OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "MuonW": MuonW,
    "ManifoldMuonW": ManifoldMuonW,
}


def run_from_config(config: ExperimentConfig):
    """
    Run a job from a given config.
    Returns a dict with summary stats.
    """
    experiment_phase: str = config["experiment_phase"]
    run_name: str = config["run_name"]
    arch_name: str = config["arch_name"]
    optimizer_name: str = config["optimizer_name"]
    optimizer_kwargs: OptimizerKwargs = config["optimizer_kwargs"]
    xy_size: int = config["xy_size"]
    num_pairs: int = config["num_pairs"]
    num_steps: int = config["num_steps"]
    batch_size: int = config["batch_size"]
    checkpoint_every: int = config["checkpoint_every"]
    device: str = config["device"]
    project_name: str = config["project_name"]
    base_ckpt_dir: str = config["base_ckpt_dir"]
    last_k: int = config["last_k"]

    ckpt_dir = os.path.join(
        base_ckpt_dir,
        experiment_phase,
        optimizer_name,
        arch_name,
        run_name,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_from = find_latest_checkpoint(ckpt_dir)

    group_name = f"{experiment_phase}/{optimizer_name}/{arch_name}"
    run = wandb.init(
        project=project_name,
        name=run_name[:128],  # wandb name limit
        group=group_name,
        config=config,
        reinit=True,
    )
    logger = WandbLossLogger(run, last_k=last_k)
    model = make_model(arch_name)
    optimizer_class = OPTIMIZER_REGISTRY[optimizer_name]
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    model = train(
        model=model,
        optimizer=optimizer,
        logger=logger,
        get_batch=get_ols_batch,
        batch_size=batch_size,
        num_pairs=num_pairs,
        xy_size=xy_size,
        num_steps=num_steps,
        device=device,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=ckpt_dir,
        verbose=True,
        resume_from=resume_from
    )
    avg_last_k_loss = logger.get_last_k_loss()
    logger.log({"avg_last_k_train_loss": avg_last_k_loss})
    logger.finish()

    return {"avg_last_k_train_loss": avg_last_k_loss}

