
import argparse
import os
import json
import itertools
import hashlib
from typing import TypedDict
from ..types.config_types import OptimizerKwargs, ExperimentConfig


HYPERPARAM_GRID_ADAMW = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.95, 0.98, 0.999],
    "weight_decay": [0.0, 0.01, 0.1, 0.2],
    "batch_size": [32, 64, 128, 256, 512, 1024],
}

HYPERPARAM_GRID_MUON = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "momentum": [0.9, 0.95, 0.98],  # often you’ll just fix 0.95
    "weight_decay": [0.0, 0.01, 0.1],
    "batch_size": [32, 64, 128, 256, 512, 1024],
}


ARCHITECTURES = ["rms", "standard", "none"]

OPTIMIZER_HYPERPARAMS = {
    "AdamW": HYPERPARAM_GRID_ADAMW,
    "MuonW": HYPERPARAM_GRID_MUON,
    "ManifoldMuonW": HYPERPARAM_GRID_MUON,
}

def _short_hparam_str(hparams: dict, max_len: int = 40) -> str:
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

def _optimizer_kwargs_from_hparams(optimizer_name: str, hp: dict) -> OptimizerKwargs:
    if optimizer_name == "AdamW":
        return {
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "beta1": hp["beta1"],
            "beta2": hp["beta2"],
        }
    elif optimizer_name in ("MuonW", "ManifoldMuonW"):
        return {
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "momentum": hp["momentum"],
        }
    else:
        raise ValueError(f"Unknown optimizer for kwargs: {optimizer_name}")

def iter_experiment_configs(
    experiment_phase: str,
    xy_size: int,
    num_pairs: int,
    num_steps: int,
    checkpoint_every: int,
    device: str,
    project_name: str,
    base_ckpt_dir: str,
    last_k: int,
):
    """
    Yield fully-formed ExperimentConfig objects
    over all (architecture, optimizer, hyperparam combo).
    """
    for optimizer_name, hyper_grid in OPTIMIZER_HYPERPARAMS.items():
        for arch_name in ARCHITECTURES:
            for hp in _iter_hparam_configs(hyper_grid):
                opt_kwargs = _optimizer_kwargs_from_hparams(optimizer_name, hp)

                # short string describing hyperparams
                hparam_str = _short_hparam_str(hp)
                run_name = f"{experiment_phase}_{optimizer_name}_{arch_name}_{hparam_str}"

                cfg: ExperimentConfig = {
                    "experiment_phase": experiment_phase,
                    "run_name": run_name,
                    "arch_name": arch_name,
                    "optimizer_name": optimizer_name,
                    "optimizer_kwargs": opt_kwargs,
                    "xy_size": xy_size,
                    "num_pairs": num_pairs,
                    "num_steps": num_steps,
                    "batch_size": hp["batch_size"],
                    "checkpoint_every": checkpoint_every,
                    "device": device,
                    "project_name": project_name,
                    "base_ckpt_dir": base_ckpt_dir,
                    "last_k": last_k,
                }
                yield cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSON experiment configs for bluey OLS transformer sweeps."
    )

    parser.add_argument(
        "--experiment-phase",
        "--phase",
        dest="experiment_phase",
        default="sweep",
        help='Experiment phase label, e.g. "sweep", "exp1" (default: "sweep")',
    )

    parser.add_argument(
        "--xy-size",
        type=int,
        default=5,
        help="Dimensionality of x/y vectors (default: 5)",
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of (x, y) pairs per sequence (default: 30)",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=50_000,
        help="Number of training steps per run (default: 50000)",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help='Device string to put into configs, e.g. "cuda", "cpu", "tpu" (default: "cuda")',
    )

    parser.add_argument(
        "--project-name",
        default="bluey-merdifold",
        help="Weights & Biases project name (default: bluey-merdifold)",
    )

    parser.add_argument(
        "--base-ckpt-dir",
        default="checkpoints",
        help='Base directory for checkpoints in configs (default: "checkpoints")',
    )

    parser.add_argument(
        "--last-k",
        type=int,
        default=50,
        help="Rolling window size for last-k loss (default: 50)",
    )

    parser.add_argument(
        "--jobs-root",
        default="jobs",
        help='Root directory to write job JSONs into (default: "jobs")',
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved generator config before writing jobs.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.print_config:
        print("=== CONFIG GENERATOR SETTINGS ===")
        print(f"experiment_phase : {args.experiment_phase}")
        print(f"xy_size          : {args.xy_size}")
        print(f"num_pairs        : {args.num_pairs}")
        print(f"num_steps        : {args.num_steps}")
        print(f"checkpoint_every : {args.checkpoint_every}")
        print(f"device           : {args.device}")
        print(f"project_name     : {args.project_name}")
        print(f"base_ckpt_dir    : {args.base_ckpt_dir}")
        print(f"last_k           : {args.last_k}")
        print(f"jobs_root        : {args.jobs_root}")
        print("=================================")
        
    repo_root = os.path.dirname(os.path.abspath(__file__))
    jobs_root = os.path.join(repo_root, "jobs")
    os.makedirs(jobs_root, exist_ok=True)

    configs = list(
        iter_experiment_configs(
            experiment_phase=args.experiment_phase,
            xy_size=args.xy_size,
            num_pairs=args.num_pairs,
            num_steps=args.num_steps,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            project_name=args.project_name,
            base_ckpt_dir=args.base_ckpt_dir,
            last_k=args.last_k,
        )
    )

    print(f"Generating {len(configs)} configs under {jobs_root}/")

    for idx, cfg in enumerate(configs):
        
        opt = cfg["optimizer_name"]
        arch = cfg["arch_name"]
        job_id = f"{idx:03d}"

        #Make a path -- sweep_optimizer_arch_job_id
        cfg_dir = os.path.join(jobs_root, args.experiment_phase, opt, arch)
        os.makedirs(cfg_dir, exist_ok=True)

        cfg_path = os.path.join(cfg_dir, f"job_{job_id}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print("Done.")
    print("Example submission command:")
    print("  ./run_jobs.sh AdamW rms 0 15")


    