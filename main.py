import torch

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


def main():
    # -----------------------------
    # 1. Choose optimizer + device
    # -----------------------------
    optimizer_name = "AdamW"  # change to "MuonW", "ManifoldMuonW", etc. as needed
    optimizer_class = OPTIMIZERS[optimizer_name]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"   # set "tpu" manually if you want to use XLA + your resolve_device logic

    # -----------------------------
    # 2. Define hyperparameter grid
    # -----------------------------
    hyperparam_grid = {
        "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "weight_decay": [0.8, 0.9, 0.95, 1.0],
        "batch_size": [2, 4, 8, 16, 32, 64, 128, 256],
        "num_pairs": [5],
        "xy_size": [5],
    }


    model_architectures = ["rms", "standard"]

    num_steps = 10_000  # training steps per run

    # -----------------------------
    # 3. Run sweep
    # -----------------------------
    print(f"Starting hyperparameter sweep for optimizer = {optimizer_name}")
    print(f"Device: {device}")
    print(f"Architectures: {model_architectures}")
    print(f"Hyperparameter grid: {hyperparam_grid}")

    results = t.hyperparameter_sweep(
        experiment_phase="sweep",          # "sweep", "exp1", etc.
        model_architectures=model_architectures,
        make_model=make_model,             # callable arch_name -> Transformer
        optimizer_name=optimizer_name,     # e.g. "AdamW"
        optimizer_class=optimizer_class,   # e.g. torch.optim.AdamW
        hyperparam_grid=hyperparam_grid,
        get_batch=d.get_batch,
        num_steps=num_steps,
        device=device,
        project_name="bluey-merdifold",
        base_ckpt_dir="checkpoints",
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

    # We defined hyperparameter_sweep to return a list of dicts like:
    # {
    #   "experiment_phase", "optimizer", "arch",
    #   "hparams", "ckpt_dir",
    #   "run_name", "group_name",
    #   "final_eval_loss", "avg_last_k_train_loss",
    # }
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
