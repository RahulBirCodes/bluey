import torch

def generate_well_conditioned_X(B, T, D, device=None, cond_threshold=1000.0):
    """
    Generates a batch of matrices X (B, T, D) where every matrix along batch dimension
    has a condition number less than `cond_threshold`.
    """
    X = torch.randn(B, T, D, device=device)
    conds = torch.linalg.cond(X) 
    bad_mask = conds > cond_threshold
    retries = 0
    while torch.any(bad_mask): 
        num_bad = bad_mask.sum().item()
        X[bad_mask] = torch.randn(num_bad, T, D, device=device)
        new_conds = torch.linalg.cond(X[bad_mask])
        still_bad_subset = new_conds > cond_threshold
        bad_mask[bad_mask.clone()] = still_bad_subset
        retries += 1
        
    return X, retries

def get_batch(
    batch_size: int = 8,
    num_pairs: int = 5,     # T
    xy_size: int = 5,       # D
    add_fake_dim: bool = False,
    add_input_noise: bool = False,
    device=None
):
    """
    Generates a fresh least-squares problem every call:
        X ~ N(0,1)           shape (B, T, D)
        W ~ N(0,1/D)         shape (B, D, D)
        Y = X @ W            shape (B, T, D)

    Returns tokens of shape:
        tokens: (B, 2T, 2*(D+1))

    Layout per token vector:
        [x_flag, y_flag, x_1..x_D, y_1..y_D]

    Sequence layout:
        x_1, y_1, ..., x_T,  y_T
        or
        y_1, x_1, ..., y_T,  x_T 
        (chosen randomly to remove any positional symmetry)

    Also returns:
        X: (B, T, D)
        Y: (B, T, D)
        W: (B, D, D)
        y_pos: (B, T)
    """

    B, T, D = batch_size, num_pairs, xy_size
    token_dim = 2 * (D + 1) + (1 if add_fake_dim else 0)   # [x_flag, y_flag, x_D, y_D, [OPTIONAL 1]]
    # X = torch.randn(B, T, D, device=device)
    X, retries = generate_well_conditioned_X(B, T, D, device=device)
    # if retries > 0:
    #     print(f"Generated well-conditioned input after {retries} retries.")
    W = torch.randn(B, D, D, device=device) / (D ** 0.5)
    Y = torch.einsum("btd,bdk->btk", X, W)

    base = torch.arange(2*T, device=device)
    swapped = base.view(T, 2).flip(1).reshape(-1)
    flip_mask = torch.randint(0, 2, (B,), device=device)
    pos = torch.where(
        flip_mask[:,None] == 0,
        base.unsqueeze(0).expand(B, 2*T),
        swapped.unsqueeze(0).expand(B, 2*T)
    )
    pos_matrix = pos.view(B, T, 2)
    x_pos = pos_matrix[:, :, 0]
    y_pos = pos_matrix[:, :, 1]

    tokens = torch.zeros(B, 2*T, token_dim, device=device)
    b_ind = torch.arange(B, device=device).unsqueeze(1)
    t_ind = torch.arange(2*T, device=device).unsqueeze(0)
    tokens[b_ind, x_pos, 0] = 1.0
    tokens[b_ind, x_pos, 2:2+xy_size] = X
    tokens[b_ind, y_pos, 1] = 1.0
    tokens[b_ind, y_pos, 2+xy_size:2+2*xy_size] = Y
    # add fake dim for bias
    if add_fake_dim:
        tokens[b_ind, t_ind, -1] = 1.0
    # add input noise during training
    if add_input_noise:
        noise = torch.randn(B, T, D, device=device) * 0.01
        tokens[b_ind, x_pos, 2:2+D] += noise
    # return x_pos since model outputs y_preds there
    return tokens, X, Y, W, x_pos