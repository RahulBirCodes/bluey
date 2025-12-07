import torch
import torch.nn.functional as F
import wandb
import os
import glob
from collections import deque
import time
from ..optimizers.muonW1 import MuonW
from ..optimizers.manifold_muonW import ManifoldMuon
from ..config_types.config_types import OptimizerKwargs, ExperimentConfig
from ..model.model import make_model
from ..model.model import orthogonal_init
from ..scripts.dataset import get_batch as get_ols_batch
import datetime
from typing import Optional
from ..scripts.monitors import MaxAbsActMonitor, RMSMonitor

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
    print("Loaded checkpoint keys:", ckpt.keys())

    # Handle old vs new key names
    model_key = "model" if "model" in ckpt else "model_state"
    optim_key = "optimizer" if "optimizer" in ckpt else "optimizer_state"

    model.load_state_dict(ckpt[model_key])
    optimizer.load_state_dict(ckpt[optim_key])

    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # RNG state is optional – only restore if present
    """ if "rng_state" in ckpt:
        torch.random.set_rng_state(ckpt["rng_state"])
    if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"]) """

    print(f"[checkpoint] resumed from {path}")
    return ckpt.get("step", 0)

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None

    files = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not files:
        return None

    best_path = None
    best_step = -1

    for path in files:
        base = os.path.basename(path)  # e.g. "step_2000_20251119-154210.pt"
        if not base.startswith("step_"):
            continue
        parts = base.split("_")
        # ["step", "<step>", "<timestamp>.pt"]
        if len(parts) < 3:
            continue

        step_str = parts[1]
        try:
            step = int(step_str)
        except ValueError:
            continue

        if step > best_step:
            best_step = step
            best_path = path

    return best_path


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


class JointOptimizer:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
    
    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


def train(
    model,
    optimizer,
    logger,
    *,
    get_batch,
    batch_size=8,
    num_pairs=5,
    xy_size=5,
    add_fake_dim=False,
    add_input_noise=False,
    num_steps=1000,
    device="cuda",
    verbose=False,
    print_interval=1000,
    checkpoint_every=20,
    checkpoint_dir=None,
    resume_from: str | None = None,
    scheduler=None,
    monitors=[]
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
            add_fake_dim=add_fake_dim,
            add_input_noise=add_input_noise,
        )
        outputs = model(tokens)
        B, S, D = outputs.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1)
        y_pred = outputs[b_idx, x_token_indices, :]
        loss = torch.sum((y_pred-Y)**2, dim=2).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Skipping step {step}: Loss is {loss.item()}")
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()

        if isinstance(optimizer, ManifoldMuon):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_step_fn(optimizer)
        if scheduler is not None:
            scheduler.step()

        if checkpoint_dir is not None and step % checkpoint_every == 0 and step != 0:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d-%H%M%S")
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step+1}_{timestamp}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step + 1,
                path=ckpt_path,
                scheduler=scheduler,
                save_fn=save_fn,
            )

            if logger is not None and hasattr(logger, "run"):
                artifact = wandb.Artifact(
                    name=f"ckpt_step_{step+1}",
                    type="model",
                    metadata={
                        "step": step + 1,
                        "arch": getattr(model, "__class__", type(model)).__name__,
                    },
                )
                artifact.add_file(ckpt_path)
                logger.run.log_artifact(artifact)
            
            if verbose:
                print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

        if verbose and (step % print_interval == 0):
            print(f"[Step {step}] loss = {loss.item():.6f}")

        if logger is not None:
            for monitor in monitors:
                monitor.log_to_wandb(logger, step)
            iter_sec = time.time() - iter_start
            logger.log({"train/loss": loss.item(), "step": step, "train/iter_sec": iter_sec})
        
        for monitor in monitors:
            monitor.reset()

    return model


class WandbLossLogger:
    """
    Wraps a wandb.Run-like object to:
      - forward logs to wandb
      - keep a rolling window of the last K 'loss' values
    """
    def __init__(self, run, last_k: int = 50):
        self.start_time = time.time()
        self.run = run
        self.last_k = deque(maxlen=last_k)
    
    def log(self, metrics: dict, step: int | None = None, commit: bool = True):
        if "train/loss" in metrics:
            self.last_k.append(metrics["train/loss"])
        self.run.log(metrics, step=step, commit=commit)
    
    def get_last_k_loss(self):
        return sum(self.last_k) / len(self.last_k)
    
    def finish(self):
        self.run.finish()


OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Muon": torch.optim.Muon,
    "ManifoldMuon": ManifoldMuon,
}

def create_optimizer_groups(model):
    std_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # run Adam on emb, unemb, and gain parameters
        if "unembedding" in name or "embedding" in name or param.ndim < 2:
            adam_params.append(param)
        else:
            std_params.append(param)
    return std_params, adam_params


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
    lips: bool = config["lips"]
    add_fake_dim: bool = config["add_fake_dim"]
    add_input_noise: bool = config["add_input_noise"]
    manifold_linear_gain_cap: Optional[float] = config.get("manifold_linear_gain_cap", None)

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
        entity="bluey-merdifold",
        project=project_name,
        name=run_name,
        group=group_name,
        config=config,
        reinit=True,
        resume="allow",
    )

    logger = WandbLossLogger(run, last_k=last_k)
    model = make_model(arch_name, lips=lips, xy_size=xy_size, add_fake_dim=add_fake_dim, manifold_linear_gain_cap=manifold_linear_gain_cap)
    max_abs_act_monitor = MaxAbsActMonitor(model)
    rms_monitor = RMSMonitor(model)
    max_abs_act_monitor.register_hook(model)
    rms_monitor.register_hook(model)

    optimizer_class = OPTIMIZER_REGISTRY[optimizer_name]
    if optimizer_name == "ManifoldMuon":
        std_params, adam_params = create_optimizer_groups(model)
        optimizer = JointOptimizer(
            optimizer_class(std_params, **optimizer_kwargs),
            torch.optim.AdamW(adam_params, lr=0.01, betas=(0.9, 0.98), weight_decay=0.01)
        ) 
    elif optimizer_name == "Muon":
        std_params, adam_params = create_optimizer_groups(model)
        optimizer = JointOptimizer(
            optimizer_class(std_params, **optimizer_kwargs),
            torch.optim.AdamW(adam_params, lr=optimizer_kwargs['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        )   
    else:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    scheduler = WarmupConstantDecayLrScheduler(optimizer, num_steps)
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
        resume_from=resume_from,
        verbose=True,
        scheduler=scheduler,
        add_fake_dim=add_fake_dim,
        add_input_noise=add_input_noise,
        monitors=[max_abs_act_monitor, rms_monitor]
    )

    avg_last_k_loss = logger.get_last_k_loss()
    logger.log({"avg_last_k_train_loss": avg_last_k_loss})
    logger.finish()

    return {"avg_last_k_train_loss": avg_last_k_loss}
