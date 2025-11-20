import json
import argparse
from typing import cast
from scripts.training import run_from_config
from loadtypes.config_types import ExperimentSpec, RunOptions, ExperimentConfig

def load_spec(path: str) -> ExperimentSpec:
    """Load only the experiment specification (static part) from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return cast(ExperimentSpec, data)


def main():
    parser = argparse.ArgumentParser(description="Run training from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON experiment spec file."
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        help="Experiment phase (e.g. sweep, exp1, ablation1)."
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of steps to train on."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run on: cpu, cuda, tpu, auto."
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        required=True,
        help="Base checkpoint directory."
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Id of the current job."
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        required=True,
        help="how often to checkpoint."
    )

    args = parser.parse_args()

    print(f"\n=== Loading experiment spec from {args.config} ===")
    spec = load_spec(args.config)
    run_options: RunOptions = {
        "experiment_phase": args.phase,
        "device": args.device,
        "base_ckpt_dir": args.ckpt_root,
        "num_steps": args.num_steps,
        "checkpoint_every": args.checkpoint_every,
    }

    config: ExperimentConfig = {**spec, **run_options}
    print("\n=== Starting training run ===")
    result = run_from_config(config)
    print("\n" + "=" * 60)
    print("TRAINING RUN COMPLETE")
    print("=" * 60)
    print(f"Run name:                 {spec['run_name']}")
    print(f"Experiment phase:         {run_options['experiment_phase']}")
    print(f"Optimizer:                {spec['optimizer_name']}")
    print(f"Architecture:             {spec['arch_name']}")
    print(f"Avg last-k train loss:    {result['avg_last_k_train_loss']:.6f}")
    print(f"Checkpoint directory:     {run_options['base_ckpt_dir']}")
    print(f"Checkpoint every:     {run_options['checkpoint_every']}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()