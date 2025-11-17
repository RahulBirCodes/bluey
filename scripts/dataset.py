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
        x_1, x_2, ..., x_T,  y_1, y_2, ..., y_T

    Also returns:
        X: (B, T, D)
        Y: (B, T, D)
        W: (B, D, D)
    """

    B, T, D = batch_size, num_pairs, xy_size
    token_dim = 2 * (D + 1)   # [x_flag, y_flag, x_D, y_D]
    X = torch.randn(B, T, D, device=device)
    W = torch.randn(B, D, D, device=device) / (D ** 0.5)
    Y = torch.einsum("btd,bdk->btk", X, W)
    tokens = torch.zeros(B, 2 * T, token_dim, device=device)
    # insert flags
    tokens[:, :T, 0] = 1.0
    tokens[:, T:, 1] = 1.0
    # insert X
    x_start = 2
    x_end   = 2 + D
    tokens[:, :T, x_start:x_end] = X
    # insert Y
    y_start = 2 + D
    y_end   = 2 + 2*D
    tokens[:, T:, y_start:y_end] = Y

    return tokens, X, Y, W