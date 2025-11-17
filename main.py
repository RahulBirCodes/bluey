import torch
from training import hyperparameter_sweep

# TODO: Replace these with the actual model factory and dataloader functions
# Update the imports and function names to match what your teammate created
from model import Transformer
from data import get_loaders   # or whatever the real function is


def model_factory():
    """Returns a fresh model instance for each training run."""
    return Transformer()  # Update this to match the actual function/class name


def get_dataloaders():
    """Returns (train_loader, val_loader) tuple."""
    return get_loaders()  # Update this to match the actual function name

def main():
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Prepare log-scale learning rate sweep values
    lr_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders()
    
    # Run hyperparameter sweep
    print("Starting hyperparameter sweep...")
    print(f"Testing learning rates: {lr_values}")
    
    best_lr, results = hyperparameter_sweep(
        model_factory=model_factory,
        optimizer_class=torch.optim.Adam,
        hyperparameter_name="lr",
        hyperparameter_values=lr_values,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        device=device,
        project_name="bluey-merdifold"
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
