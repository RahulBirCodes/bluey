import json
import argparse
from scripts.training import run_from_config

def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run training from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    print(f"\n=== Loading config from {args.config} ===")
    config = load_config(args.config)
    print("\n=== Starting training run ===")
    result = run_from_config(config)

    # print final results
    print("\n" + "="*60)
    print("TRAINING RUN COMPLETE")
    print("="*60)
    print(f"Run name:                 {config['run_name']}")
    print(f"Experiment phase:         {config['experiment_phase']}")
    print(f"Optimizer:                {config['optimizer_name']}")
    print(f"Architecture:             {config['arch_name']}")
    print(f"Avg last-k train loss:    {result['avg_last_k_train_loss']:.6f}")
    print(f"Checkpoint directory:     {config['base_ckpt_dir']}")
    print("="*60)
    print()

if __name__ == "__main__":
    main()