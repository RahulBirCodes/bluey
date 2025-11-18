import torch
from scripts import training as t

# TODO: Replace these with the actual model factory and dataloader functions
# Update the imports and function names to match what your teammate created
from model.model import Transformer
from data import get_loaders   # or whatever the real function is

def make_model(arch_name: str,
               hidden_size=256,
               n_heads=8,
               n_layers=15,
               xy_size=5) -> Transformer:
    """
    arch_name options (you can extend):
      - "rms_res"      : RMSNorm + residuals (default)
      - "rms_nores"    : RMSNorm + no residuals
      - "ln_res"       : LayerNorm + residuals
      - "ln_nores"     : LayerNorm + no residuals
    """
    name = arch_name.lower()

    if name == "rms_res":
        norm_type = "rms"
        use_residual = True
    elif name == "rms_nores":
        norm_type = "rms"
        use_residual = False
    elif name == "ln_res":
        norm_type = "layernorm"
        use_residual = True
    elif name == "ln_nores":
        norm_type = "layernorm"
        use_residual = False
    else:
        raise ValueError(f"Unknown arch_name: {arch_name}")

    return Transformer(
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        xy_size=xy_size,
        norm_type=norm_type,
        use_residual=use_residual,
    )


def main():
    # Detect device    
    # Prepare log-scale learning rate sweep values
    #May also need to vary weight decay to be off, so 1?
    hyper_parameter_grid = {"lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                 "weight_decay": [0.8, 0.9, 0.95, 1]
    }
    # Run hyperparameter sweep
    print("Starting hyperparameter sweep...")
    print(f"Testing hyperparameter values: {hyper_parameter_grid}")
    
    best_lr, results = t.hyperparameter_sweep(
        experiment_phase="sweep",      # "sweep", "exp1", etc.
    model_architectures= [["ln",], ["ln", "resnet", ] ],  # ["ln should specify what kind of norm", "no_ln_resnet", ...]
    make_model=          ,            # callable arch_name -> nn.Module
    optimizer_name= "AdamW",             # "AdamW", "MuonW", "ManifoldMuonW"
    optimizer_class,                 # e.g. torch.optim.AdamW
    hyperparam_grid: dict[str, list],
    *,
    get_batch,
    num_steps: int,
    device: str,
    project_name: str = "bluey-merdifold",
    base_ckpt_dir: str = "checkpoints",
    last_k: int = 50,

    )
    
    # Print results
    print("\n" + "="*50)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*50)
    print(f"Best learning rate: {best_lr}")
    if results:
        print(f"Best validation accuracy: {max(r['best_val_acc'] for r in results):.2f}%")
        print("\nDetailed results:")
        for result in results:
            print(f"  LR={result['value']:.2e}: Best Val Acc={result['best_val_acc']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
