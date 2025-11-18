import torch
import argparse
from scripts import training as t
from scripts import dataset as d
from model.model import make_model

from optimizers.muonW1 import MuonW
from optimizers.manifold_muonW import ManifoldMuonW

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "MuonW": MuonW,
    "ManifoldMuonW": ManifoldMuonW,
}

HYPERPARAM_GRID_ADAMW = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.95, 0.98, 0.999],
    "weight_decay": [0.0, 0.01, 0.1, 0.2],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024],
}

HYPERPARAM_GRID_MUON = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], #According to Muon is Scalable for LLM Training, scaling Muon allows for reusing lr and weight decay tuned for AdamW
    "momentum": [0.9, 0.95, 0.98],  # often you’ll just fix 0.95
    "weight_decay": [0.0, 0.01, 0.1],
    "batch_size": [64, 128, 256, 512, 1024],
}

HYPERPARAM_GRID_LION = {
    "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.97, 0.99, 0.995],
    "weight_decay": [0.0, 0.01, 0.1],
    "batch_size": [64, 128, 256, 512, 1024],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep driver for bluey least-squares Transformer."
    )

    parser.add_argument(
        "--optimizer",
        "-o",
        choices=list(OPTIMIZERS.keys()),
        default="AdamW",
        help="Which optimizer to use (default: AdamW)",
    )

    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or "tpu" (default: auto)',
    )

    parser.add_argument(
        "--phase",
        "--experiment-phase",
        default="sweep",
        help='Experiment phase label, e.g. "sweep", "exp1" (default: "sweep")',
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=50_000,
        help="Number of training steps per run (default: 50000)",
    )

    parser.add_argument(
        "--architectures",
        "-a",
        default="rms",
        help='Comma-separated list of architectures, e.g. "rms,standard" (default: "rms,standard")',
    )

    parser.add_argument(
        "--project-name",
        default="bluey-merdifold",
        help="Weights & Biases project name (default: bluey-merdifold)",
    )

    parser.add_argument(
        "--base-ckpt-dir",
        default="checkpoints",
        help='Base directory for checkpoints (default: "checkpoints")',
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration before running sweeps.",
    )

    return parser.parse_args()


def _select_hyperparam_grid(optimizer_name: str) -> dict:
    if optimizer_name == "AdamW":
        return HYPERPARAM_GRID_ADAMW
    elif optimizer_name in ("MuonW", "ManifoldMuonW"):
        return HYPERPARAM_GRID_MUON
    # elif optimizer_name == "Lion":
    #     return HYPERPARAM_GRID_LION
    else:
        raise ValueError(f"No hyperparameter grid defined for optimizer {optimizer_name!r}")


def _resolve_device(device_arg: str) -> str:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return device_arg


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    # -----------------------------
    # 1. Choose optimizer + device
    # -----------------------------
    optimizer_name = args.optimizer
    optimizer_class = OPTIMIZERS[optimizer_name]

    device = _resolve_device(args.device)

    # -----------------------------
    # 2. Define hyperparameter grid
    # -----------------------------
    hyperparam_grid = _select_hyperparam_grid(optimizer_name)

    model_architectures = [
        arch.strip()
        for arch in args.architectures.split(",")
        if arch.strip()
    ]

    num_steps = args.num_steps

    if args.print_config:
        print("=== CONFIG ===")
        print(f"optimizer_name: {optimizer_name}")
        print(f"device:         {device}")
        print(f"phase:          {args.phase}")
        print(f"num_steps:      {num_steps}")
        print(f"architectures:  {model_architectures}")
        print(f"project_name:   {args.project_name}")
        print(f"base_ckpt_dir:  {args.base_ckpt_dir}")
        print("hyperparam_grid keys:", list(hyperparam_grid.keys()))
        print("===============")

    print(f"Starting hyperparameter sweep for optimizer = {optimizer_name}")
    print(f"Device: {device}")
    print(f"Architectures: {model_architectures}")
    print(f"Hyperparameter grid: {hyperparam_grid}")

    results = t.hyperparameter_sweep(
        experiment_phase=args.phase,           # e.g. "sweep", "exp1"
        model_architectures=model_architectures,
        make_model=make_model,                
        optimizer_name=optimizer_name,
        optimizer_class=optimizer_class,
        hyperparam_grid=hyperparam_grid,
        get_batch=d.get_batch,
        num_steps=num_steps,
        device=device,
        project_name=args.project_name,
        base_ckpt_dir=args.base_ckpt_dir,
        last_k=50,
    )

    # -----------------------------
    # 4. Summarize results
    # -----------------------------
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)

    if not results:
        print("No runs completed.")
        print("=" * 60)
        return

    best_run = min(results, key=lambda r: r["final_eval_loss"])
    best_hparams = best_run["hparams"]

    print(f"Best architecture:   {best_run['arch']}")
    print(f"Best optimizer:      {best_run['optimizer']}")
    print(f"Best final loss:     {best_run['final_eval_loss']:.6f}")
    print(f"Avg last-k loss:     {best_run['avg_last_k_train_loss']:.6f}")
    print(f"Best hyperparams:    {best_hparams}")
    print(f"Checkpoint dir:      {best_run['ckpt_dir']}")
    print()

    print("All runs:")
    for run in results:
        h = run["hparams"]
        print(
            f"- arch={run['arch']:<9} "
            f"lr={h.get('lr', 'NA')!s:<8} "
            f"wd={h.get('weight_decay', 'NA')!s:<6} "
            f"bs={h.get('batch_size', 'NA')!s:<5} "
            f"final_loss={run['final_eval_loss']:.6f}"
        )

    print("=" * 60)

if __name__ == "__main__":
    main()
