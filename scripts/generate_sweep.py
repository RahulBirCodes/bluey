#!/usr/bin/env python
import os
import itertools
from textwrap import dedent

MODELS = ["transformer_small"]          # or ["transformer_small", "transformer_big"]
OPTIMIZERS = ["manifold_muon"]         # or ["adamw", "manifold_muon", "muon"]
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
DROPOUTS = [0.0]
LAYERS = [12]
HIDDENS = [768]
SEEDS = [123, 456, 789]                # etc


def iter_configs():
    """
    Yield dicts of all hyperparameter combinations.
    """
    for (model, opt, lr, dropout, layers, hidden, seed) in itertools.product(
        MODELS, OPTIMIZERS, LRS, DROPOUTS, LAYERS, HIDDENS, SEEDS
    ):
        yield {
            "model": model,
            "optimizer": opt,
            "lr": lr,
            "dropout": dropout,
            "layers": layers,
            "hidden": hidden,
            "seed": seed,
        }


def make_run_id(idx: int, cfg: dict) -> str:
    """
    Short, stable ID for this run.
    E.g. "sweep_017_model=transformer_small_lr=3e-4"
    Feel free to simplify if you just want "sweep_017".
    """
    return f"sweep_{idx:03d}_{cfg['model']}_opt={cfg['optimizer']}_lr={cfg['lr']}"


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    jobs_dir = os.path.join(repo_root, "jobs")
    logs_dir = os.path.join(repo_root, "logs")

    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    configs = list(iter_configs())
    print(f"Generating {len(configs)} job scripts into {jobs_dir}/")

    for idx, cfg in enumerate(configs):
        run_id = make_run_id(idx, cfg)
        job_name = f"job_{idx:03d}.sh"
        job_path = os.path.join(jobs_dir, job_name)

        # Adjust SBATCH directives according to your cluster
        script = dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={run_id}
            #SBATCH --output=logs/{run_id}.out
            #SBATCH --error=logs/{run_id}.err
            #SBATCH --time=12:00:00
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=32G

            # Load environment (edit for your cluster)
            source ~/.bashrc
            # conda activate bluey-env  # or your venv

            # Go to repo root
            cd "{repo_root}"

            # Run the sweep job
            python main.py \\
              --model="{cfg['model']}" \\
              --optimizer="{cfg['optimizer']}" \\
              --lr={cfg['lr']} \\
              --dropout={cfg['dropout']} \\
              --layers={cfg['layers']} \\
              --hidden={cfg['hidden']} \\
              --seed={cfg['seed']} \\
              --run_id="{run_id}"
        """)

        with open(job_path, "w") as f:
            f.write(script)

        # Make executable
        os.chmod(job_path, 0o755)

    print("Done.")
    print("Example submission command:")
    print("  sbatch jobs/job_000.sh")


if __name__ == "__main__":
    main()
