import torch


def get_batch(
    batch_size: int = 8,
    num_pairs: int = 5,     # T
    xy_size: int = 5,       # D
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
    token_dim = 2 * (D + 1)   # [x_flag, y_flag, x_D, y_D]
    X = torch.randn(B, T, D, device=device)
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
    tokens[b_ind, x_pos, 0] = 1.0
    tokens[b_ind, x_pos, 2:2+xy_size] = X
    tokens[b_ind, y_pos, 1] = 1.0
    tokens[b_ind, y_pos, 2+xy_size:2+2*xy_size] = Y
    # return x_pos since model outputs y_preds there

    
    return tokens, X, Y, W, x_pos

if __name__ == "__main__":
    tokens, X, Y, W, x_pos = get_batch(64, 48, 5, "cpu")

    X_norm = torch.norm(X)
    Y_norm = torch.norm(Y)

    print(f"X_norm: {X_norm} Y_norm: {Y_norm}")
    print(f"first ten tokens of one batch: {tokens[0, :10, :]}")