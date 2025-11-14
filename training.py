import torch
import torch.nn.functional as F
import wandb


def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, log_every=1):
    """
    Train a model with logging to wandb.
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        device: Device to train on (e.g., 'cuda' or 'cpu')
        log_every: Log batch loss every N batches (default: 1)
    
    Returns:
        Dictionary with keys:
            - "train_loss": list of average training loss per epoch
            - "val_loss": list of average validation loss per epoch
            - "val_acc": list of validation accuracy per epoch
    """
    model = model.to(device)
    model.train()
    
    # Watch model for gradient and parameter logging
    wandb.watch(model, log="all")
    
    # History storage
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss (cross-entropy for classification)
            # Handle different output shapes (e.g., if outputs are (batch, seq_len, vocab_size))
            # and targets are (batch, seq_len) or (batch,)
            if outputs.dim() == 3:
                # Sequence model: (batch, seq_len, vocab_size)
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)
            elif outputs.dim() == 2:
                # Standard classification: (batch, num_classes)
                pass
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_train_loss += loss.item()
            num_train_batches += 1
            
            # Log batch loss periodically
            if (batch_idx + 1) % log_every == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch
                })
        
        # Average training loss for this epoch
        avg_train_loss = running_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Handle different output shapes
                if outputs.dim() == 3:
                    batch_size, seq_len, vocab_size = outputs.shape
                    outputs_flat = outputs.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                elif outputs.dim() == 2:
                    outputs_flat = outputs
                    targets_flat = targets
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
                
                # Compute validation loss
                val_loss = F.cross_entropy(outputs_flat, targets_flat)
                running_val_loss += val_loss.item()
                num_val_batches += 1
                
                # Compute accuracy
                _, predicted = torch.max(outputs_flat.data, 1)
                total += targets_flat.size(0)
                correct += (predicted == targets_flat).sum().item()
        
        # Average validation metrics
        avg_val_loss = running_val_loss / num_val_batches
        val_acc = 100.0 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
    
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs
    }


def hyperparameter_sweep(model_factory, optimizer_class, hyperparameter_name, hyperparameter_values,
                         train_loader, val_loader, num_epochs, device, project_name="bluey-merdifold"):
    """
    Perform a grid search over a single hyperparameter.
    
    Args:
        model_factory: Callable that returns a fresh model instance for each run
        optimizer_class: Optimizer class (e.g., torch.optim.Adam), not an instance
        hyperparameter_name: Name of hyperparameter (e.g., "lr")
        hyperparameter_values: List of values to try (e.g., [1e-5, 1e-4, 1e-3])
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs per run
        device: Device to train on
        project_name: wandb project name (default: "bluey-merdifold")
    
    Returns:
        Tuple of (best_value, results):
            - best_value: The hyperparameter value with highest best_val_acc
            - results: List of dictionaries, one per hyperparameter value, containing:
                - "value": the hyperparameter value
                - "best_val_acc": best validation accuracy achieved
                - "history": training history dictionary from train_model
    """
    results = []
    best_value = None
    best_val_acc = -1.0
    
    for value in hyperparameter_values:
        # Start a new wandb run
        run = wandb.init(
            project=project_name,
            name=f"{hyperparameter_name}_{value}",
            config={
                hyperparameter_name: value,
                "num_epochs": num_epochs,
            },
            reinit=True,
        )
        
        # Create a fresh model
        model = model_factory()
        
        # Create optimizer with the hyperparameter value
        optimizer = optimizer_class(model.parameters(), **{hyperparameter_name: value})
        
        # Train the model
        history = train_model(model, optimizer, train_loader, val_loader, num_epochs, device)
        
        # Find best validation accuracy
        best_val_acc_for_value = max(history["val_acc"])
        
        # Log best validation accuracy
        wandb.log({"best_val_acc": best_val_acc_for_value})
        
        # Finish the run
        run.finish()
        
        # Store results
        result = {
            "value": value,
            "best_val_acc": best_val_acc_for_value,
            "history": history,
        }
        results.append(result)
        
        # Track overall best
        if best_val_acc_for_value > best_val_acc:
            best_val_acc = best_val_acc_for_value
            best_value = value
    
    return (best_value, results)

