from typing import TypedDict, Optional

class OptimizerKwargs(TypedDict, total=False):
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    momentum: float
    nesterov: bool


class ExperimentSpec(TypedDict):
    run_name: str
    arch_name: str
    lips: bool
    optimizer_name: str
    optimizer_kwargs: OptimizerKwargs
    xy_size: int
    num_pairs: int
    batch_size: int
    project_name: str
    last_k: int
    add_fake_dim: bool
    add_input_noise: bool
    manifold_linear_gain_cap: Optional[float]


class RunOptions(TypedDict):
    job_id: str
    experiment_phase: str
    device: str
    base_ckpt_dir: str
    num_steps: int
    checkpoint_every: int


class ExperimentConfig(ExperimentSpec, RunOptions):
    pass