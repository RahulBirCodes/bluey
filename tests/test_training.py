import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.training import train, hyperparameter_sweep


class DummyModel(nn.Module):
    """Simple dummy model for testing: single linear layer."""
    def __init__(self, input_dim=32, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # Handle both (batch, features) and (batch, seq_len, features)
        if x.dim() == 3:
            # Flatten sequence dimension: (batch, seq_len, features) -> (batch * seq_len, features)
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)
            out = self.linear(x)
            # Reshape back: (batch * seq_len, num_classes) -> (batch, seq_len, num_classes)
            return out.view(batch_size, seq_len, -1)
        else:
            # Standard: (batch, features) -> (batch, num_classes)
            return self.linear(x)


def create_dummy_dataloaders(batch_size=8, num_samples=32, input_dim=32, num_classes=10):
    """Create dummy train and validation dataloaders with random data."""
    # Create random input data
    train_inputs = torch.randn(num_samples, input_dim)
    train_targets = torch.randint(0, num_classes, (num_samples,))
    
    val_inputs = torch.randn(num_samples // 2, input_dim)
    val_targets = torch.randint(0, num_classes, (num_samples // 2,))
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def test_train_model():
    """Test train_model() function."""
    print("="*60)
    print("Testing train_model()...")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(input_dim=32, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, val_loader = create_dummy_dataloaders()
    
    # Initialize wandb for this test
    import wandb
    wandb.init(
        project="bluey-merdifold",
        name="test_train_model",
        config={"test": True, "num_epochs": 2}
    )
    
    try:
        history = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            device=device,
            log_every=1
        )
        
        print("✓ train_model() completed successfully!")
        print(f"  Train losses: {history['train_loss']}")
        print(f"  Val losses: {history['val_loss']}")
        print(f"  Val accuracies: {history['val_acc']}")
        
    finally:
        wandb.finish()
    
    return history


def test_hyperparameter_sweep():
    """Test hyperparameter_sweep() function."""
    print("\n" + "="*60)
    print("Testing hyperparameter_sweep()...")
    print("="*60)
    
    device = "cpu"
    train_loader, val_loader = create_dummy_dataloaders()
    
    # Model factory function
    def model_factory():
        return DummyModel(input_dim=32, num_classes=10)
    
    # Test with 2 learning rates
    lr_values = [1e-4, 1e-3]
    
    try:
        best_lr, results = hyperparameter_sweep(
            model_factory=model_factory,
            optimizer_class=torch.optim.Adam,
            hyperparameter_name="lr",
            hyperparameter_values=lr_values,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            device=device,
            project_name="bluey-merdifold"
        )
        
        print("✓ hyperparameter_sweep() completed successfully!")
        print(f"  Best learning rate: {best_lr}")
        print(f"  Results summary:")
        for result in results:
            print(f"    LR={result['value']:.2e}: Best Val Acc={result['best_val_acc']:.2f}%")
        
    except Exception as e:
        print(f"✗ hyperparameter_sweep() failed: {e}")
        raise
    
    return best_lr, results


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING TRAINING.PY SETUP")
    print("="*60)
    print("\nThis script tests train_model() and hyperparameter_sweep()")
    print("with dummy data to verify the training pipeline works.\n")
    
    try:
        # Test train_model
        test_train_model()
        
        # Test hyperparameter_sweep
        test_hyperparameter_sweep()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nCheck your wandb project 'bluey-merdifold' to verify the runs logged correctly.")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

