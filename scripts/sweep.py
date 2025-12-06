import hashlib
from itertools import product 
import argparse
import json
import os


HYPERPARAM_GRID_ADAMW_MUON = {
    "lr": [0.01, 1e-3, 1e-4],
    "beta1": [0.9],
    "beta2": [0.98],
    "weight_decay": [0.1],
    "batch_size": [64, 128, 256],
    "num_pairs": [64, 128, 256]
}

HYPERPARAM_GRID_MANIFOLD_MUON = {
    "lr": [0.2, 0.1, 0.05],
    "batch_size": [64, 128, 256],
    "num_pairs": [64, 128, 256]
}

OPTIMIZER_NAMES = ['AdamW', 'Muon', "ManifoldMuon"]

OPTIMIZER_GRID_REGISTRY = {
    "AdamW": HYPERPARAM_GRID_ADAMW_MUON,
    "Muon": HYPERPARAM_GRID_ADAMW_MUON,
    "ManifoldMuon": HYPERPARAM_GRID_MANIFOLD_MUON,
}

MODEL_ARCHS = ["rms", "none"]

def short_hparam_str(hparams: dict, max_len: int = 128) -> str:
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


def iter_hparam_configs(hyperparam_grid: dict):
    """
    Given {"lr":[1e-4,1e-3], "wd":[0.0,0.1]}, yield:
        {"lr":1e-4,"wd":0.0}, {"lr":1e-4,"wd":0.1}, ...
    """
    keys = list(hyperparam_grid.keys())
    values = [hyperparam_grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def main():
    parser = argparse.ArgumentParser(
        description="Generate hyperparameter sweep configuration files."
    )

    parser.add_argument(
        "--xy_size",
        type=int,
        required=True,
        help="Input feature dimensionality (D).",
    )

    parser.add_argument(
        "--add_fake_dim",
        type=bool,
        required=True,
        help="Add fake input to input.",
    )

    parser.add_argument(
        "--add_input_noise",
        type=bool,
        required=True,
        help="Add noise to the input.",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="WandB project name for all generated configs.",
    )

    parser.add_argument(
        "--last_k",
        type=int,
        required=True,
        help="Number of recent losses to average for run summary.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="jobs",
        help="Directory in which to save all generated config files.",
    )

    args = parser.parse_args()
    xy_size = args.xy_size
    project_name = args.project_name
    last_k = args.last_k
    root = args.output_dir
    add_input_noise = args.add_input_noise
    add_fake_dim = args.add_fake_dim

    os.makedirs(root, exist_ok=True)
    print("\n=== Generating sweep configs ===")
    for optimizer_name in OPTIMIZER_NAMES:
        opt_grid = OPTIMIZER_GRID_REGISTRY[optimizer_name]
        opt_dir = os.path.join(root, optimizer_name)
        os.makedirs(opt_dir, exist_ok=True)

        for lips in [True, False]:
            for arch_name in MODEL_ARCHS:
                arch_dir = os.path.join(opt_dir, arch_name, 'lips' if lips else 'nolips')
                os.makedirs(arch_dir, exist_ok=True)
                print(f"\nOptimizer: {optimizer_name}, Arch: {arch_name}, Lips: {lips}")
                hparam_dicts = list(iter_hparam_configs(opt_grid))
                for idx, hparams in enumerate(hparam_dicts):
                    batch_size = hparams["batch_size"]
                    num_pairs = hparams["num_pairs"]
                    optimizer_kwargs = {k: v for k, v in hparams.items() if k != "batch_size" and k != "num_pairs"}
                    hparam_str = short_hparam_str(hparams)
                    run_name = f"{optimizer_name}_norm_{arch_name}_{hparam_str}_{'lips' if lips else 'nolips'}"
                    spec = {
                        "run_name": run_name,
                        "arch_name": arch_name,
                        "lips": lips,
                        "optimizer_name": optimizer_name,
                        "optimizer_kwargs": optimizer_kwargs,
                        "xy_size": xy_size,
                        "num_pairs": num_pairs,
                        "batch_size": batch_size,
                        "project_name": project_name,
                        "last_k": last_k,
                        "add_input_noise": add_input_noise,
                        "add_fake_dim": add_fake_dim
                    }
                    # job_000.json naming
                    job_id = f"{idx:03d}"
                    out_path = os.path.join(arch_dir, f"job_{job_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(spec, f, indent=2)
    print("\n=== Sweep generation complete ===")


if __name__ == "__main__":
    main()