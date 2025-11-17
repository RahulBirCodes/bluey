import torch
import torch.nn.functional as F
import wandb
import os
import torch
import torch.nn.functional as F

# Optional TPU support
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

def resolve_device_and_saver(device_str: str):
    """
    Returns (torch.device-like, save_fn, optimizer_step_fn).

    device_str:
      - "cpu"      → CPU
      - "cuda"     → current CUDA device
      - "cuda:0"   → specific CUDA device
      - "tpu"      → XLA device (requires torch_xla)
    """
    if device_str.lower() == "tpu":
        if not HAS_XLA:
            raise RuntimeError("TPU requested but torch_xla is not installed.")
        device = xm.xla_device()
        save_fn = xm.save

        def optimizer_step_fn(optimizer):
            xm.optimizer_step(optimizer)

    else:
        device = torch.device(device_str)
        save_fn = torch.save

        def optimizer_step_fn(optimizer):
            optimizer.step()

    return device, save_fn, optimizer_step_fn


def train(
    model, #Should be a transformer model
    optimizer, #Could be AdamW, ManifoldMuonW, or MuonW
    logger, #Should be a wandb logger
    *,
    get_batch, #Function that returns tokens, X, Y, W, and indices of tokens that refer to X tokens
    batch_size=8,
    num_pairs=5,
    xy_size=5,
    num_steps=1000,
    device="cuda", #Could be "TPU", "cuda", "cpu"
    verbose=True,
    print_interval=20,
    checkpoint_every=20,
    checkpoint_dir=None,
):
    """
    Train model on synthetic least-squares data.

    Assumes:
      - get_batch(...) returns (tokens, X, Y, W, x_token_indices)
      - tokens: (B, T_seq, token_dim)
      - X, Y:  (B, num_pairs, xy_size)
      - model(tokens) -> (B, T_seq, xy_size)
      - x_token_indices: positions where *x* is input; outputs at these
        positions are interpreted as predictions of the next y.
    """
    device, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    model.to(device)
    model.train()

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for step in range(num_steps):
        # --- Generate fresh synthetic batch every step ---
        # Yes: for synthetic LS tasks, it's standard to resample each iteration.
        tokens, X, Y, W, x_token_indices = get_batch(
            batch_size=batch_size,
            num_pairs=num_pairs,
            xy_size=xy_size,
            device=device,
        )
        # tokens, X, Y, etc. should already be on the right device

        # FWD
        outputs = model(tokens)               # (B, T_seq, xy_size)

        B, S, D = outputs.shape
        # Gather predictions at x-token positions
        # x_token_indices: 1D tensor (num_pairs,) of time indices
        # We want outputs[:, x_token_indices, :] -> (B, num_pairs, xy_size)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S)

        y_pred = outputs[b_idx, x_token_indices, :]

        # Least-squares / MSE loss between predicted y’s and true Y
        loss = torch.sum((y_pred-Y)**2, dim=1).mean()

        # BWD
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_step_fn(optimizer)

        # --- Checkpointing ---
        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step+1}.pt")
            state = {
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss.item(),
            }
            save_fn(state, ckpt_path)
            if verbose:
                print(f"[Step {step}] Saved checkpoint to {ckpt_path}")

        # --- Print logging ---
        if verbose and (step % print_interval == 0):
            print(f"[Step {step}] loss = {loss.item():.6f}")

        # --- WandB logging ---
        if logger is not None:
            logger.log({"loss": loss.item(), "step": step})

    if logger is not None:
        logger.finish()

    return model


def load_checkpoint(model, optimizer, filepath, device="cuda"):
    """
    Device-aware checkpoint loading.
    """
    device_obj, save_fn, optimizer_step_fn = resolve_device_and_saver(device)
    checkpoint = torch.load(filepath, map_location=device_obj)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint["step"]
    loss = checkpoint["loss"]
    model.to(device_obj)
    return model, optimizer, step, loss


def hyperparameter_sweep(experiment_phase: str, #"sweep", "exp1", etc.
                         model_architecture: str[], #Should be like ["layer_norm"] or ["res_net"]
                         optimizer_class: None, 
                         hyerparameter_dict: dict{str:float}, #Dictionary 
                         get_batch,
                         num_epochs, 
                         device, 
                         roject_name="bluey-merdifold"):
    """
    Perform a grid search over a single hyperparameter.
    
    Args:
        model_architecture: maybe a string: "LN", "noLN" "noLNnoResNet" or list of strings?
        optimizer_class: Optimizer class (e.g., torch.optim.Adam), not an instance
        hyperparameter_name: Name of hyperparameter (e.g., "lr")
        hyperparameter_values: List of values to try (e.g., [1e-5, 1e-4, 1e-3])
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs per run
        device: Device to train on
        project_name: wandb project name (default: "bluey-merdifold")
        experiment_phase: str

    
    Returns:
        Tuple of (best_value, results):
            - best_value: The hyperparameter value with highest best_val_acc
            - results: List of dictionaries, one per hyperparameter value, containing:
                - "value": the hyperparameter value
                - "best_val_acc": best validation accuracy achieved
                - "history": training history dictionary from train_model

    In this code we want to be able to run multiple runs, preferably using Ray, and distributed computing
    and log the checkpoints in unique checkpoint directories and on wandb, making sure that each run has a 
    unique ID or way of distinguishing it quickly from other runs, so that wandb is organized and makes sense as well 
    as so that the checkpoint directories are not overlapping and we have organized wandb and checkpoint
    directories.

    Checkpointing: Eventually we're running this project in multiple phases.

    here's a breakdown of the file-structure that we should have

    Phase of experiment

    --sweep (prelim)
        Optimizer
        --AdamW
            Model architecture
            --resnets
                Hyperparameter set (as dictionary)
                --lr=0.0001_weightdecay=0.9_...
                    Iteration
                    --iteration100
                    --iteration200
                    --iteration300
                --lr=0.001_weightdecay=0.9_...
                --lr=0.01_weightdecay=0.9_...
                
            --no resnets
            --LN
            --no LN
        --ManifoldMuonW
        --MuonW
    --exp1 
    --exp2

    Wandb: Wandb should have the exact same project structure to keep things organized. 
    Is this possible?

    Also, though, we want to be easily able to compare within optimizers what archtecture is best.
    What's the best way to be able to compare later down the line? Perhaps you can add some dictionary
    that logs the average of the final 50 validation losses per run, and then saves them in this
    nested dictionary format, so that we can know what is the best performance across different categories.

    Or maybe a pandas would be better. Not sure. 

    Use the training script that we wrote and create a hyperparameter sweep loop that keeps track
    of a set of different hyperparameter_values, creating a fresh run ID, wandb logger, model according to 
    that set of hyerparameters, maybe some metadata to also be passed into the wandb to ensure easy
    identification, the device type, and then the checkpoint directory, which should be unique per run, 
    and hopefully under 30 char. 

    I think that actually we should have a dictionary, like this: {hyperparameter names: list of values}.

    Instead of train/val loader, we're just going to use the dataset generation method that we've 
    already defined in scripts/dataset.py. Instead of train_model we're going to use the train method in
    this file. We don't need to track the best validation loss/accuracy.
    
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



def train_model(
        model, 
        optimizer, 
        train_loader, 
        val_loader, 
        num_epochs, 
        device, 
        log_every=1,
        logger=None,
        checkpoint_every=50):
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
        
            if (batch_idx + 1) % checkpoint_every == 0:
                save_checkpoint(
                    model=model, 
                    optimizer=optimizer, 
                    epoch=batch_idx,
                    loss=loss.item()
                )
        
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


