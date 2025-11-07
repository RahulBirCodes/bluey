#!/usr/bin/env python
# file: train_ols_xla.py
"""
TPU-ready training script for an OLS-style regression task on a handcrafted
<10M-param MLP. Supports architectural variants (RMSNorm/LayerNorm/none, with/without residuals),
wandb logging, checkpointing, and XLA-safe data pipeline. Works on CPU/GPU/TPU
but is tuned for PyTorch/XLA with PJRT on Cloud TPU.

Launch (single-host 8-core v5e/v4 TPU):
    PJRT_DEVICE=TPU python3 train_ols_xla.py \
        --model mlp3 --norm rms --residual yes \
        --dataset synthetic --n-train 20000 --n-val 5000 --n-dim 32 \
        --batch_size 128 --learning_rate 1e-3 --optimizer adam \
        --weight_decay 0.0 --momentum 0.9 --epochs 5 \
        --save_checkpoint_dir ./checkpoints --save_checkpoint_every 200 \
        --eval_every 100 --project tpu_ols_muon --notes "baseline rms+res"

Multi-host pods: prefer torch_xla.distributed.run or a cluster launcher.

References:
  - PJRT + XLA best practices (mark_step/optimizer_step): docs.pytorch.org/xla (see citations in chat)
  - FSDP on XLA exists but is overkill for <10M params.
  - RMSNorm: Zhang et al. 2019
  - Manifold Muon (ADMM variant): Buchanan 2025 (we wire a pluggable hook below)
"""

import argparse, math, os, time, json, uuid, pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# XLA imports are optional at import-time to allow CPU/GPU dev
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False

# ------------------------------- Data -------------------------------
class OLSSynthetic(Dataset):
    def __init__(self, n:int, d:int, noise:float=0.1, seed:int=42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, d, generator=g)
        w_true = torch.randn(d, 1, generator=g)
        y = self.X @ w_true + noise*torch.randn(n, 1, generator=g)
        self.y = y.squeeze(-1)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------- Modules ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-8, affine:bool=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    def forward(self, x):
        # x: [..., d]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x / rms
        if self.affine:
            y = y * self.weight
        return y

class MLPBlock(nn.Module):
    def __init__(self, dim:int, hidden:int, norm:str='none', residual:bool=True, dropout:float=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if norm == 'layer':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm == 'rms':
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
    def forward(self, x):
        h = self.norm1(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return x + h if self.residual else h

class OLSNet(nn.Module):
    def __init__(self, d:int, depth:int=3, width:int=1024, norm:str='none', residual:bool=True, dropout:float=0.0):
        super().__init__()
        self.inp = nn.Linear(d, width)
        blocks = [MLPBlock(width, width, norm=norm, residual=residual, dropout=dropout) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(width, 1)
    def forward(self, x):
        h = F.gelu(self.inp(x))
        h = self.blocks(h)
        y = self.out(h).squeeze(-1)
        return y

# ----------------------------- Metrics ------------------------------
@dataclass
class BatchMetrics:
    loss: float
    rmse: float
    r2: float

def regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = F.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(loss)
    # R^2
    var = torch.var(y_true, unbiased=False) + 1e-9
    r2 = 1.0 - (loss / (var))
    return loss, rmse, r2

# --------------------------- Optimizers -----------------------------
class MuonADMM(torch.optim.Optimizer):
    """
    Minimal pluggable skeleton for Manifold Muon via ADMM (Buchanan, 2025).
    This is a placeholder that performs a standard Adam step, followed by a
    post-update projection hook where you can add manifold constraints (e.g.,
    spectral/clipping/orthogonalization per-parameter) driven by `rho`.

    NOTE: Implementing the exact ADMM updates requires the proximal subproblem
    on matrix manifolds; wire your update in `_admm_post_update`.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho)
        super().__init__(params, defaults)
        self._adam = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._adam.step(closure)
        # ADMM-style post-update projection (stub)
        for group in self.param_groups:
            rho = group['rho']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim == 2:  # matrix weights only
                    self._admm_post_update(p, rho)
        return loss

    @staticmethod
    def _admm_post_update(W: torch.Tensor, rho: float):
        # Placeholder: clip spectral norm as a crude stability proxy.
        # Replace with ADMM updates from the blog for real experiments.
        # e.g., W <- argmin ||W - Z + U||^2 + (rho/2)*I subject to manifold constraint
        with torch.no_grad():
            # spectral norm clip to s_max
            s_max = 2.0  # conservative cap; adjust as needed
            if W.numel() == 0:
                return
            try:
                u, s, v = torch.linalg.svd(W, full_matrices=False)
                s_clipped = torch.clamp(s, max=s_max)
                W.copy_(u @ torch.diag(s_clipped) @ v)
            except Exception:
                # fallback: weight norm clip
                W.copy_(torch.clamp(W, -s_max, s_max))

# ------------------------------ Utils -------------------------------

def xla_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_master():
    if XLA_AVAILABLE:
        return xm.is_master_ordinal()
    return True


def setup_wandb(args, run_id: str):
    if args.no_wandb:
        return None
    import wandb
    run = wandb.init(
        project=args.project,
        name=args.name or run_id,
        config=vars(args),
        notes=args.notes,
        reinit=True,
        mode='online' if not args.wandb_offline else 'offline',
    )
    return run


def save_ckpt(model, optimizer, step:int, run_dir:pathlib.Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    state = { 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step }
    path = run_dir / f"iter{step:08d}.pt"
    if XLA_AVAILABLE:
        xm.save(state, str(path))
    else:
        torch.save(state, str(path))

# --------------------------- Train / Eval ---------------------------

def make_model(args, d:int) -> nn.Module:
    # Param budget: depth*2*width^2 roughly + IO layers. Keep <10M.
    depth = {'mlp2':2, 'mlp3':3, 'mlp4':4}.get(args.model, 3)
    width = args.width
    norm_map = {'none':'none','layer':'layer','rms':'rms'}
    m = OLSNet(d=d, depth=depth, width=width, norm=norm_map[args.norm], residual=(args.residual=='yes'), dropout=args.dropout)
    return m


def make_optimizer(args, model: nn.Module):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'muon_admm':
        return MuonADMM(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")


def run_eval(model, loader, device):
    model.eval()
    tot_loss = 0.0; tot_rmse = 0.0; tot_r2 = 0.0; n=0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            loss, rmse, r2 = regression_metrics(yp, yb)
            bsz = xb.size(0)
            tot_loss += loss.item()*bsz
            tot_rmse += rmse.item()*bsz
            tot_r2 += r2.item()*bsz
            n += bsz
    return {
        'loss': tot_loss/max(1,n),
        'rmse': tot_rmse/max(1,n),
        'r2': tot_r2/max(1,n)
    }


def train_loop(args):
    device = xla_device()
    if is_master():
        print(f"Device: {device} | XLA_AVAILABLE={XLA_AVAILABLE}")

    # Data
    if args.dataset == 'synthetic':
        ds_train = OLSSynthetic(args.n_train, args.n_dim, noise=args.noise, seed=args.seed)
        ds_val = OLSSynthetic(args.n_val, args.n_dim, noise=args.noise, seed=args.seed+1)
    else:
        raise NotImplementedError("Only --dataset synthetic implemented; plug your loader here.")

    sampler_train = torch.utils.data.distributed.DistributedSampler(ds_train, shuffle=True) if XLA_AVAILABLE else None
    sampler_val = torch.utils.data.distributed.DistributedSampler(ds_val, shuffle=False) if XLA_AVAILABLE else None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler_train is None), sampler=sampler_train, drop_last=True, num_workers=args.num_workers, pin_memory=not XLA_AVAILABLE)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, sampler=sampler_val, num_workers=args.num_workers, pin_memory=not XLA_AVAILABLE)

    # Model/opt
    model = make_model(args, d=args.n_dim).to(device)
    opt = make_optimizer(args, model)

    # XLA loader wrapper (prefetch/input pipeline)
    if XLA_AVAILABLE:
        train_pl = pl.MpDeviceLoader(dl_train, device)
        val_pl = pl.MpDeviceLoader(dl_val, device)
    else:
        train_pl = dl_train
        val_pl = dl_val

    # Run id / logging / ckpts
    run_id = f"{int(time.time())}_{args.model}_norm={args.norm}_res={args.residual}_d={args.n_dim}_w={args.width}_lr={args.learning_rate}_wd={args.weight_decay}_bsz={args.batch_size}_opt={args.optimizer}"
    run_dir = pathlib.Path(args.save_checkpoint_dir) / run_id
    wb = setup_wandb(args, run_id)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        for xb, yb in train_pl:
            xb = xb.to(device); yb = yb.to(device)
            yp = model(xb)
            loss, rmse, r2 = regression_metrics(yp, yb)
            opt.zero_grad()
            loss.backward()
            if XLA_AVAILABLE:
                xm.optimizer_step(opt, barrier=True)
                xm.mark_step()
            else:
                opt.step()

            if is_master() and (global_step % args.log_every == 0):
                log = { 'train/loss': loss.item(), 'train/rmse': rmse.item(), 'train/r2': r2.item(), 'step': global_step, 'epoch': epoch }
                if wb: wb.log(log, step=global_step)
                else: print(log)

            if is_master() and (args.save_checkpoint_every > 0) and (global_step % args.save_checkpoint_every == 0) and (global_step>0):
                save_ckpt(model, opt, global_step, run_dir)

            if (args.eval_every > 0) and (global_step % args.eval_every == 0) and (global_step>0):
                eval_stats = run_eval(model, val_pl, device)
                if is_master():
                    if wb: wb.log({f"val/{k}":v for k,v in eval_stats.items()}, step=global_step)
                    else: print({f"val/{k}":v for k,v in eval_stats.items()})

            global_step += 1
            if global_step >= args.max_steps > 0:
                break
        if global_step >= args.max_steps > 0:
            break

    # final save
    if is_master():
        save_ckpt(model, opt, global_step, run_dir)
        if wb: wb.finish()

# ------------------------------ CLI --------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument('--model', type=str, default='mlp3', choices=['mlp2','mlp3','mlp4'])
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--norm', type=str, default='none', choices=['none','layer','rms'])
    p.add_argument('--residual', type=str, default='yes', choices=['yes','no'])
    p.add_argument('--dropout', type=float, default=0.0)
    # Data
    p.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic'])
    p.add_argument('--n-train', dest='n_train', type=int, default=20000)
    p.add_argument('--n-val', dest='n_val', type=int, default=5000)
    p.add_argument('--n-dim', dest='n_dim', type=int, default=32)
    p.add_argument('--noise', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    # Optimization
    p.add_argument('--optimizer', type=str, default='adam', choices=['sgd','adam','muon_admm'])
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--rho', type=float, default=1.0, help='ADMM penalty for Manifold Muon (muon_admm)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--max_steps', type=int, default=-1, help='Stop early after N steps (for sweeps)')
    # Logging & ckpt
    p.add_argument('--project', type=str, default='tpu_ols_muon')
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--notes', type=str, default='')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--wandb_offline', action='store_true')
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--save_checkpoint_dir', type=str, default='./checkpoints')
    p.add_argument('--save_checkpoint_every', type=int, default=200)
    p.add_argument('--eval_every', type=int, default=100)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_loop(args)


# file: sweep_ols.py
"""
Hyperparameter sweep driver. Define architecture variants and HP grids using
nested dicts, and spawn the training script as a subprocess per trial.

Example:
    PJRT_DEVICE=TPU python3 sweep_ols.py --max_trials 50 --dry_run

Notes:
  * On TPU, prefer many short runs (set --max_steps) to iterate quickly.
  * For multi-host pods, run one sweep controller per host with disjoint seeds
    or use a simple queue (left as an exercise) to avoid duplicate trials.
"""
import argparse, itertools, json, os, random, shlex, subprocess, time, pathlib

TRAIN = "train_ols_xla.py"

ARCHES = {
    # 1) RMSNorm + residuals
    "rms+res": {
        "model": ["mlp3"],
        "norm": ["rms"],
        "residual": ["yes"],
    },
    # 2) LayerNorm (learnable affine) + residuals
    "ln+res": {
        "model": ["mlp3"],
        "norm": ["layer"],
        "residual": ["yes"],
    },
    # 3) No norm, residuals on
    "none+res": {
        "model": ["mlp3"],
        "norm": ["none"],
        "residual": ["yes"],
    },
    # 4) No norm, no residuals (harder optional)
    "none+nores": {
        "model": ["mlp3"],
        "norm": ["none"],
        "residual": ["no"],
    },
}

HYPERS = {
    # Common training knobs
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "batch_size": [32, 64, 128, 256],
    "optimizer": ["adam", "sgd", "muon_admm"],
    # SGD-only momentum (ignored by Adam/Muon)
    "momentum": [0.9],
    # Manifold Muon ADMM penalty; held relatively stable per blog
    "rho": [0.5, 1.0, 2.0],
    # Regularization/architecture
    "dropout": [0.0, 0.1],
    "width": [512, 768, 1024],  # keep total params < 10M
}

FIXED = {
    "dataset": "synthetic",
    "n_train": 20000,
    "n_val": 5000,
    "n_dim": 32,
    "epochs": 3,
    "eval_every": 100,
    "save_checkpoint_every": 200,
    "project": "tpu_ols_muon",
    "notes": "automated sweep",
    "max_steps": 2000,  # early cut for fast sweeps; adjust as needed
}


def grid(dict_of_lists):
    keys = list(dict_of_lists.keys())
    for vals in itertools.product(*[dict_of_lists[k] for k in keys]):
        yield dict(zip(keys, vals))


def trial_cmd(args, arch_name, arch_cfg, hp_cfg):
    cmd = ["python3", TRAIN]
    # arch
    for k,v in arch_cfg.items():
        cmd += [f"--{k}", str(v)] if isinstance(v, str) else sum([[f"--{k}", str(x)] for x in v], [])
    # hypers
    for k,v in hp_cfg.items():
        cmd += [f"--{k}", str(v)]
    # fixed
    for k,v in FIXED.items():
        cmd += [f"--{k}", str(v)]
    # book-keeping
    cmd += ["--save_checkpoint_dir", args.save_dir]
    cmd += ["--log_every", str(args.log_every)]
    if args.no_wandb: cmd += ["--no_wandb"]
    if args.wandb_offline: cmd += ["--wandb_offline"]
    if args.dry_run: cmd += ["--max_steps", "50"]
    # name
    override_name = f"sweep:{arch_name}"
    cmd += ["--name", override_name]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', type=str, default='./checkpoints')
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--max_trials', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--no_wandb', action='store_true')
    ap.add_argument('--wandb_offline', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)

    trials = []
    # build trials per-architecture
    for arch_name, arch_space in ARCHES.items():
        # resolve singletons (ARCHES stores lists but we want single values here)
        arch_cfg = {k:(v[0] if isinstance(v, list) and len(v)==1 else v) for k,v in arch_space.items()}
        for hp_cfg in grid(HYPERS):
            trials.append((arch_name, arch_cfg, hp_cfg))

    random.shuffle(trials)
    trials = trials[:args.max_trials]

    print(f"Planned trials: {len(trials)}")
    for i, (arch_name, arch_cfg, hp_cfg) in enumerate(trials):
        cmd = trial_cmd(args, arch_name, arch_cfg, hp_cfg)
        print(f"[{i+1}/{len(trials)}] RUN: {' '.join(shlex.quote(c) for c in cmd)}")
        env = os.environ.copy()
        # Ensure PJRT is set for TPU runs; on CPU/GPU you may unset
        env.setdefault('PJRT_DEVICE', os.environ.get('PJRT_DEVICE', 'TPU'))
        # Best practice: limit thread contention
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
        subprocess.run(cmd, env=env, check=False)

if __name__ == '__main__':
    main()
