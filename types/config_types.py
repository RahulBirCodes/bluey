from typing import TypedDict

class OptimizerKwargs(TypedDict, total=False):
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    momentum: float
    nesterov: bool


class ExperimentSpec(TypedDict):
    run_name: str
    arch_name: str
    optimizer_name: str
    optimizer_kwargs: OptimizerKwargs
    xy_size: int
    num_pairs: int
    batch_size: int
    project_name: str
    last_k: int
    checkpoint_every: int
  

class RunOptions(TypedDict):
    experiment_phase: str
    num_steps: int
    device: str
    base_ckpt_dir: str
    job_id: str


ExperimentConfig = ExperimentSpec & RunOptions