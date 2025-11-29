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
    

class RunOptions(TypedDict):
    job_id: str
    experiment_phase: str
    device: str
    base_ckpt_dir: str
    num_steps: int
    checkpoint_every: int

ExperimentConfig = ExperimentSpec | RunOptions