from typing import TypedDict

class OptimizerKwargs(TypedDict, total=False):
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    momentum: float
    nesterov: bool
  

class ExperimentConfig(TypedDict):
    experiment_phase: str
    run_name: str
    arch_name: str
    optimizer_name: str
    optimizer_kwargs: OptimizerKwargs
    xy_size: int
    num_pairs: int
    num_steps: int
    batch_size: int
    checkpoint_every: int
    device: str
    project_name: str
    base_ckpt_dir: str
    last_k: int