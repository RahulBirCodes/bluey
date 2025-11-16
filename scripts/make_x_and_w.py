import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(
    batch_size: int = 8,
    num_pairs: int = 5,          # T: number of (x, y) pairs per sequence
    d_model: int = 12,
    mu_x: float = 0.0,
    sigma_x: float = 1.0,
    mu_W: float = 0.0,
    sigma_W: float = 1.0,
    mu_b: float = 0.0,
    sigma_b: float = 1.0,
    device=None,
    normalize: bool = True,
):
    """
    Generate a least-squares in-context batch for a decoder-only transformer.

    For each batch element, we sample:
        x_t ∈ ℝ^5,   t = 1..T
        W ∈ ℝ^(5x5), b ∈ ℝ^5

    and set:
        y_t = x_t @ W^T + b

    Then we pack tokens as:
        [x_flag, y_flag, x1..x5, y1..y5] ∈ ℝ^12

    Sequence layout (seq_len = 2 * num_pairs):
        x_1, y_1, x_2, y_2, ..., x_T, y_T

    Returns:
        tokens:      (B, 2T, 12) float tensor
        token_types: (B, 2T) long tensor, 0 = x token, 1 = y token
        x:           (B, T, 5)
        y:           (B, T, 5)
        W:           (B, 5, 5)
        b:           (B, 1, 5)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert d_model == 12, "Layout assumes d_model == 12: [x_flag, y_flag, x5, y5]."

    B = batch_size
    T = num_pairs
    payload_dim = 5

    # -----------------------------
    # 1. Sample x, W, b and compute y
    # -----------------------------
    # x: (B, T, 5)
    x = mu_x + sigma_x * torch.randn(B, T, payload_dim, device=device)

    # W: (B, 5, 5)
    W = mu_W + sigma_W * torch.randn(B, payload_dim, payload_dim, device=device)

    # b: (B, 1, 5) – broadcast across T
    b = mu_b + sigma_b * torch.randn(B, 1, payload_dim, device=device)

    # y: (B, T, 5), y_t = x_t @ W^T + b
    # x: (B, T, 5), W: (B, 5, 5) ⇒ y: (B, T, 5)
    y = torch.einsum("btk,bkj->btj", x, W) + b

    # -----------------------------
    # 2. Build token sequence layout
    #    seq_len = 2T, alternating x_t, y_t
    # -----------------------------
    seq_len = 2 * T
    tokens = torch.zeros(B, seq_len, d_model, device=device)

    # token_types: 0 = x, 1 = y
    positions = torch.arange(seq_len, device=device)  # (2T,)
    token_types = (positions % 2).unsqueeze(0).expand(B, -1).long()  # (B, 2T)

    x_mask = (token_types == 0)  # (B, 2T)
    y_mask = ~x_mask             # (B, 2T)

    # Flags
    tokens[..., 0] = x_mask.float()  # x_flag
    tokens[..., 1] = y_mask.float()  # y_flag

    # Pair index for each token: 0,0,1,1,2,2,...,T-1,T-1
    pair_idx = (positions // 2).unsqueeze(0).expand(B, -1)  # (B, 2T)

    # Expand x, y to token positions via pair_idx
    # x_expanded: (B, 2T, 5), y_expanded: (B, 2T, 5)
    x_expanded = x.gather(
        dim=1,
        index=pair_idx.unsqueeze(-1).expand(B, seq_len, payload_dim),
    )
    y_expanded = y.gather(
        dim=1,
        index=pair_idx.unsqueeze(-1).expand(B, seq_len, payload_dim),
    )

    # Flatten for easier masked assignment
    B_, S, D = tokens.shape
    flat_tokens    = tokens.view(B_ * S, D)
    flat_x_mask    = x_mask.view(B_ * S)
    flat_y_mask    = y_mask.view(B_ * S)
    flat_x_payload = x_expanded.view(B_ * S, payload_dim)
    flat_y_payload = y_expanded.view(B_ * S, payload_dim)

    # For x tokens: payload goes into positions 2..6
    flat_tokens[flat_x_mask, 2 : 2 + payload_dim] = flat_x_payload[flat_x_mask]

    # For y tokens: payload goes into positions 7..11
    flat_tokens[flat_y_mask, 2 + payload_dim : 2 + 2 * payload_dim] = flat_y_payload[flat_y_mask]

    tokens = flat_tokens.view(B_, S, D)

    # -----------------------------
    # 3. RMS-normalize each token vector (optional)
    # -----------------------------
    if normalize:
        rms = tokens.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8  # (B, S, 1)
        tokens = tokens / rms

    return tokens, token_types, x, y, W, b


