HYPERPARAM_GRID_ADAMW = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.95, 0.98, 0.999],
    "weight_decay": [0.0, 0.01, 0.1, 0.2],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024],
}

HYPERPARAM_GRID_MUON = {
    "lr_matrix": [1e-3, 3e-3, 1e-2, 2e-2, 3e-2],
    "lr_scalar": [1e-4, 3e-4, 1e-3, 3e-3],
    "momentum": [0.9, 0.95, 0.98],  # often you’ll just fix 0.95
    "weight_decay": [0.0, 0.01, 0.1],
    "batch_size": [64, 128, 256, 512, 1024],
}

HYPERPARAM_GRID_LION = {
    "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.97, 0.99, 0.995],
    "weight_decay": [0.0, 0.01, 0.1],
    "batch_size": [64, 128, 256, 512, 1024],
}

def _short_hparam_str(hparams: dict, max_len: int = 40) -> str:
    """
    Turn a small hyperparam dict into a compact, filesystem-safe string.
    Example: {'lr':1e-3,'wd':0.1} -> 'lr1e-3_wd0.1' (possibly truncated + hash).
    """
    parts = []
    for k, v in hparams.items():
        # Normalize floats for readability
        if isinstance(v, float):
            v_str = f"{v:.1e}" if (v < 0.01 or v > 1000) else str(v)
        else:
            v_str = str(v)
        parts.append(f"{k}{v_str}")
    base = "_".join(parts)
    if len(base) <= max_len:
        return base
    # Truncate and append hash so we keep uniqueness but stay short
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    return base[: max_len - 7] + "_" + h


def _iter_hparam_configs(hyperparam_grid: dict):
    """
    Given {"lr":[1e-4,1e-3], "wd":[0.0,0.1]}, yield:
        {"lr":1e-4,"wd":0.0}, {"lr":1e-4,"wd":0.1}, ...
    """
    keys = list(hyperparam_grid.keys())
    values = [hyperparam_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))