import json
import argparse
from typing import cast
from scripts.training import run_from_config
from loadtypes.config_types import ExperimentSpec, RunOptions, ExperimentConfig
import os

def load_spec(path: str) -> ExperimentSpec:
    """Load only the experiment specification (static part) from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return cast(ExperimentSpec, data)


def main():
    parser = argparse.ArgumentParser(description="Run training from a config file.")

    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--arch", required=True)

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

    args = parser.parse_args()

    for i in range(args.start_id, args.end_id + 1):
        job_id = f"{i:03d}"
        cfg_path = os.path.join("jobs", args.optimizer, args.arch,
                                f"job_{job_id}.json")
        if not os.path.exists(cfg_path):
            print(f"[skip] no config: {cfg_path}")
            continue

        spec = load_spec(cfg_path)

        run_options: RunOptions = {
            "experiment_phase": args.phase,
            "num_steps": args.num_steps,
            "device": args.device,
            "base_ckpt_dir": args.ckpt_root,
            "job_id": job_id,
        }

        config: ExperimentConfig = {**spec, **run_options}
        # Override some fields from CLI (so you can reuse configs)
        config["experiment_phase"] = args.phase
        config["device"] = args.device
        config["base_ckpt_dir"] = args.ckpt_root
        if args.num_steps is not None:
            config["num_steps"] = args.num_steps

        print(f"[run] {cfg_path} on device={config['device']}")
        # logs to wandb, handles resume_from, etc.
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
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()