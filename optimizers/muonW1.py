import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer as OptimizerBase

#from . import LayerNormBase
#from .config import OptimizerType, SchedulerConfig, SchedulerType, TrainConfig
#from .torch_util import get_default_device, is_distributed

""" Simulate import from .torch_util """

import gc
import os
from typing import Optional, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


""" end of simulation """




__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "MuonW",
    "Scheduler",
    "CosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "ConstantScheduler",
    "CosLinearEnvelope",
    "BoltOnWarmupScheduler",
    "build_optimizer",
    "build_scheduler",
]


log = logging.getLogger(__name__)

class Optimizer(OptimizerBase):
    def __init__(self, *args, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates

    def _clean_param_name(self, name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self,
        global_step: int,
        collect_param_metrics: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Clips gradients for every group that has the field `max_grad_norm`.
        At the same time collect metrics for each parameter and its gradient.
        """
        self._collecting_metrics = collect_param_metrics
        device = get_default_device() if device is None else device

        # NOTE (epwalsh): during distributed training we're making an assumption that the order of
        # the param groups and the params within each group are the same across all ranks.
        # This is justified since we initialize the parameter groups in every rank by iterating over
        # `module.parameters()` or `module.named_modules()` / `module.named_parameters()`, each of which
        # provides a consistent order.
        #  For each parameter (with a gradient) we'll collect:
        # - min, max, avg, norm of the param itself
        # - min, max, avg, norm of the param's gradient
        # - min, max, avg, norm of any additional per-parameter optimizer state metrics returned from
        #   `self.get_state_for_param()`.
        # Afterwards we'll reduce these all over all ranks.
        per_param_min_metrics: List[torch.Tensor] = []
        per_param_max_metrics: List[torch.Tensor] = []
        per_param_sum_metrics: List[torch.Tensor] = []
        per_param_norm_metrics: List[torch.Tensor] = []
        per_param_numel_metrics: List[torch.Tensor] = []

        per_param_min_metric_names: List[str] = []
        per_param_max_metric_names: List[str] = []
        per_param_avg_metric_names: List[str] = []
        per_param_norm_metric_names: List[str] = []

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        #######################################################################
        # part 1: collect metrics locally
        #######################################################################
        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                tensors: List[Optional[torch.Tensor]] = [p.grad]
                prefixes: List[str] = [f"grad/{name}"]
                if collect_param_metrics:
                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])
                    tensors.extend([p] + [state[key] for key in sorted_state_keys])
                    prefixes.extend([f"param/{name}"] + [f"{key}/{name}" for key in sorted_state_keys])
                assert len(tensors) == len(prefixes)

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    if x is not None and x.numel() > 0:
                        if collect_param_metrics:
                            x_abs = x.abs()
                            per_param_min_metrics.append(x_abs.min().unsqueeze(0).to(dtype=torch.float32))
                            per_param_max_metrics.append(x_abs.max().unsqueeze(0).to(dtype=torch.float32))
                            per_param_sum_metrics.append(x.sum().unsqueeze(0).to(dtype=torch.float32))
                            per_param_numel_metrics.append(
                                torch.tensor([x.numel()], device=device, dtype=torch.float32)
                            )
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0)
                        )
                    else:
                        if collect_param_metrics:
                            per_param_min_metrics.append(
                                torch.tensor([float("inf")], device=device, dtype=torch.float32)
                            )
                            per_param_max_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_sum_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_numel_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                        per_param_norm_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                    if collect_param_metrics:
                        per_param_min_metric_names.append(f"{prefix}.min")
                        per_param_max_metric_names.append(f"{prefix}.max")
                        per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        assert (
            len(per_param_min_metrics)
            == len(per_param_min_metric_names)
            == len(per_param_max_metrics)
            == len(per_param_max_metric_names)
            == len(per_param_sum_metrics)
            == len(per_param_numel_metrics)
            == len(per_param_avg_metric_names)
        )
        assert len(per_param_norm_metrics) == len(per_param_norm_metric_names)

        def is_grad_norm_metric(metric_name: str) -> bool:
            return metric_name.startswith("grad/") and metric_name.endswith(".norm")

        #######################################################################
        # part 2: reduce metrics over ranks
        #######################################################################
        param_group_sharded = False
        for group in self.param_groups:
            param_group_sharded = param_group_sharded or group.get("sharded", False)

        total_grad_norm: torch.Tensor
        per_param_avg_metrics: List[torch.Tensor] = []
        if is_distributed() and param_group_sharded:
            # Reduce metrics across all ranks. Note that we can use a `reduce` for most cases
            # instead of an `all_reduce`, but we need `all_reduce` for norms so that all ranks
            # get the right value for gradient norms so they can clip correctly.
            # Reduce mins.
            if per_param_min_metrics:
                all_mins = torch.cat(per_param_min_metrics).to(device)
                dist.reduce(all_mins, dst_rank, op=dist.ReduceOp.MIN, group=process_group)
                per_param_min_metrics = all_mins.split(1)
            # Reduce maxs.
            if per_param_max_metrics:
                all_maxs = torch.cat(per_param_max_metrics).to(device)
                dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
                per_param_max_metrics = all_maxs.split(1)
            # Reduce sums or just norms.
            all_norms = torch.cat(per_param_norm_metrics).to(device) ** 2.0
            if per_param_sum_metrics and per_param_numel_metrics:
                all_sums = torch.cat(per_param_sum_metrics).to(device)
                all_numels = torch.cat(per_param_numel_metrics).to(device)
                all_sums_norms_numels = torch.cat(
                    [all_sums.unsqueeze(0), all_norms.unsqueeze(0), all_numels.unsqueeze(0)], dim=0
                )
                dist.all_reduce(all_sums_norms_numels, op=dist.ReduceOp.SUM, group=process_group)
                all_sums, all_norms, all_numels = all_sums_norms_numels.split(1)
                # Get averages.
                # NOTE: could get infs for non-rank0 processes but that's okay.
                per_param_avg_metrics = (all_sums / all_numels).squeeze(0).split(1)
            else:
                dist.all_reduce(all_norms, op=dist.ReduceOp.SUM, group=process_group)
            grad_norm_metric_mask = torch.tensor(
                [float(is_grad_norm_metric(n)) for n in per_param_norm_metric_names], device=all_norms.device
            )
            total_grad_norm = (all_norms * grad_norm_metric_mask).sum() ** 0.5
            per_param_norm_metrics = (all_norms ** (0.5)).squeeze(0).split(1)
        else:
            total_grad_norm = (
                torch.cat(
                    [
                        m
                        for m, n in zip(per_param_norm_metrics, per_param_norm_metric_names)
                        if is_grad_norm_metric(n)
                    ]
                )
                ** 2.0
            ).sum() ** 0.5
            per_param_avg_metrics = [x / n for x, n in zip(per_param_sum_metrics, per_param_numel_metrics)]

        assert len(per_param_avg_metrics) == len(per_param_avg_metric_names)

        # Collect all metrics into a single dict.
        all_metrics: Dict[str, torch.Tensor] = {}
        if collect_param_metrics:
            for metric_name, metric in zip(per_param_min_metric_names, per_param_min_metrics):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_max_metric_names, per_param_max_metrics):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_avg_metric_names, per_param_avg_metrics):
                all_metrics[metric_name] = metric.squeeze(0)

        for metric_name, metric in zip(per_param_norm_metric_names, per_param_norm_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        all_metrics["total_grad_norm"] = total_grad_norm

        #######################################################################
        # part 3: clip grads
        #######################################################################
        num_grads_clipped = 0
        num_eligible_grads = 0
        for group in self.param_groups:
            if (max_norm_ratio := group.get("max_grad_norm_ratio")) is not None:
                num_clipped = self._do_adaptive_clipping(
                    group, max_norm_ratio, global_step, all_metrics, collect_param_metrics=collect_param_metrics
                )
            elif (max_norm := group.get("max_grad_norm")) is not None:
                num_clipped = self._do_global_fixed_clipping(
                    group, max_norm, all_metrics, collect_param_metrics=collect_param_metrics
                )
            else:
                # No clipping needed.
                continue
            num_eligible_grads += len(group["params"])
            if num_clipped is not None:
                num_grads_clipped += num_clipped

        if collect_param_metrics:
            if num_eligible_grads > 0:
                clipping_rate = torch.tensor(num_grads_clipped / num_eligible_grads, device="cpu")
            else:
                clipping_rate = torch.tensor(0.0, device="cpu")
            all_metrics["clipping_rate"] = clipping_rate

        # total_grad_norm is computed at all steps, even when collect_param_metrics is set to False
        return all_metrics

    @torch.no_grad()
    def _do_adaptive_clipping(
        self,
        group: Dict[str, Any],
        max_norm_ratio: float,
        global_step: int,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do adaptive gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        num_grads_clipped = 0
        # We'll use the bigger of beta1 and beta2 to update the exponential average of the norm of
        # the gradient (a scalar), not to be confused with the exponential average of the gradient.
        # TODO (epwalsh): handle optimizers that don't have betas.
        beta1, beta2 = group["betas"]
        beta = max(beta1, beta2)
        for name, p in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            # Get or initialize the exponential average of grad norm.
            # TODO: The way we have it right now, every rank tracks the `grad_norm_exp_avg` of every parameter,
            # even parameters for which the corresponding local shard is empty. This has the potential to
            # cause some issues with the optimizer, as we ran into with https://github.com/allenai/LLM/pull/372.
            # So we should consider changing how we do this at some point so that we don't add any state
            # to parameters for which the local shard is empty. That would probably add extra distributed
            # communication, at least on steps where we have to log (i.e. when `collect_param_metrics=True`).
            state = self.state[p]
            grad_norm_exp_avg = state.get("grad_norm_exp_avg")
            if grad_norm_exp_avg is None:
                grad_norm_exp_avg = grad_norm.clone().to(device)
                # We don't want to add anything to `state` until `state` has been initialized, otherwise
                # this will crash some optimizers which rely on checking `len(state)`. The downside here
                # is that we won't start tracking `grad_norm_exp_avg` until the 2nd training step.
                if global_step > 1:
                    state["grad_norm_exp_avg"] = grad_norm_exp_avg

            max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
            clip_coef = max_allowed_norm / (grad_norm + 1e-6)

            # Clip the gradients and update the exponential average.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))

            # Update the exponential average of the norm of the gradient with the clipped norm of the gradient.
            grad_norm_exp_avg.lerp_((grad_norm * clip_coef_clamped).to(grad_norm_exp_avg.device), 1 - beta)
            # Alternative: update with the *unclipped* norm of the gradient.
            #  grad_norm_exp_avg.lerp_(grad_norm.to(grad_norm_exp_avg.device), 1 - beta)

            if collect_param_metrics:
                # Can't avoid host-device sync here.
                if clip_coef_clamped < 1.0:
                    num_grads_clipped += 1
                all_metrics[f"grad_norm_exp_avg/{name}"] = grad_norm_exp_avg
        return num_grads_clipped if collect_param_metrics else None

    @torch.no_grad()
    def _do_global_fixed_clipping(
        self,
        group: Dict[str, Any],
        max_norm: float,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do global fixed gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        total_grad_norm = all_metrics["total_grad_norm"]
        clip_coef = max_norm / (total_grad_norm.to(device) + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        num_grads_clipped: Optional[int] = None
        if collect_param_metrics:
            # Can't avoid host-device sync here.
            if clip_coef_clamped < 1.0:
                num_grads_clipped = len(group["params"])
        for p in group["params"]:
            # Clip the gradients.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))
        return num_grads_clipped

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        del module, process_group
        return {}

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}
    
class MuonW(Optimizer):
    """
    Distributed implementation of Muon optimizer with weight decay.

    Muon applies orthogonalization to matrix parameter(2D+) updates using
    Newton-Schulz  orthogonalization iterations to compute the zeroth power. For non-matrix
    parameters(embeddings, heads, bias), it uses AdamW as a backup.

    """

    def __init__(
        self,
        params,
        lr=0.01,
        betas=(0.95, 0.95),  # Muon uses single momentum param
        weight_decay=0.0,
        ns_steps=5,
        nesterov=True,
        eps=1e-8,  # For AdamW backup
        record_update_metrics=False,
        selective_updates=False,
        device=None,
    ):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            # User provided param groups
            for param_group in params:
                if 'use_muon' not in param_group:
                    param_group['use_muon'] = True
        else:
            # Convert single params list to a param group
            params = [{'params': params, 'use_muon': True}]

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
            eps=eps,
            use_muon=True,  # Default to using Muon
        )
        super().__init__(
            params,
            defaults,
            record_update_metrics=record_update_metrics,
            selective_updates=selective_updates
        )
        self._device = device
        self._update_norms = None
        self._update_maxs = None
        self._update_param_names = None

    def zeropower_via_newtonschulz5(self, G, steps: int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        """Return optimizer state for a parameter."""
        state = self.state[param]
        if not state:
            return {}

        result = {}
        if 'momentum_buffer' in state:
            result['momentum_buffer'] = state['momentum_buffer']
        if 'exp_avg' in state:
            result['exp_avg'] = state['exp_avg']
        if 'exp_avg_sq' in state:
            result['exp_avg_sq'] = state['exp_avg_sq']

        return result

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        device = get_default_device() if self._device is None else self._device
        update_norms = []
        update_maxs = []
        update_param_names = []

        collecting_metrics = self._collecting_metrics and self._record_update_metrics

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            eps = group['eps']
            use_muon = group['use_muon']

            if "param_names" not in group:
                group["param_names"] = ["hello" for i in range(len(group["params"]))]

            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Check if we're in FSDP mode
                is_fsdp = hasattr(p, '_is_sharded') and p._is_sharded

                if p.grad is None:
                    if collecting_metrics:
                        update_param_names.append(name)
                        update_norms.append(torch.tensor([0.0], device=device))
                        update_maxs.append(torch.tensor([0.0], device=device))
                    continue

                # Apply weight decay
                #mask = p.grad != 0 if self._selective_updates else 1
                mask = (p.grad != 0) if self._selective_updates else torch.ones_like(p, dtype=torch.bool)
                p.mul_(1 - mask * (lr * weight_decay))

                grad = p.grad
                state = self.state[p]

                # Determine whether to use Muon or AdamW for this parameter
                # We use Muon for matrix parameters unless explicitly disabled
                should_use_muon = use_muon and p.ndim >= 2 and not ('embed' in name.lower() or 'head' in name.lower())

                if should_use_muon:
                    # --- Muon Update Logic ---

                    # Initialize momentum buffer if needed
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    momentum_buffer = state['momentum_buffer']

                    # Update momentum
                    momentum_buffer.lerp_(grad, mask * (1 - beta1))

                    # Compute update
                    if nesterov:
                        update = momentum_buffer * beta1 + grad * (1 - beta1)
                    else:
                        update = momentum_buffer.clone()

                    if isinstance(mask, torch.Tensor):
                        update.mul_(mask)


                    if is_fsdp:
                        # For FSDP, we need to gather the full gradient/update across ranks
                        import torch.distributed as dist

                        # Get world size and rank
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()

                        # Gather update tensor from all ranks
                        update_list = [torch.empty_like(update) for _ in range(world_size)]
                        dist.all_gather(update_list, update)

                        # Concatenate to get full update
                        full_update = torch.cat(update_list, dim=0)  # Assuming sharding on dim 0

                        # Perform Newton-Schulz on full matrix
                        orig_shape = full_update.shape
                        if full_update.ndim == 4:
                            full_update = full_update.view(full_update.shape[0], -1)

                        full_update = self.zeropower_via_newtonschulz5(full_update, steps=ns_steps)
                        full_update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5

                        if len(orig_shape) == 4:
                            full_update = full_update.view(orig_shape)

                        # Extract this rank's shard from the orthogonalized update
                        shard_size = full_update.shape[0] // world_size
                        start_idx = rank * shard_size
                        end_idx = start_idx + shard_size
                        update = full_update[start_idx:end_idx]

                    else:
                        # Non-FSDP path (single GPU)
                        # Handle conv filters
                        orig_shape = update.shape
                        if update.ndim == 4:
                            update = update.view(update.shape[0], -1)

                        # Apply Newton-Schulz
                        update = self.zeropower_via_newtonschulz5(update, steps=ns_steps)

                        # Scale update the KIMI way!
                        #Make sure that we scale by 0.2, which is the typical AdamW RMS update factors
                        #And also make sure that we scale by sqrt(max(A,B))
                        update *= 0.2 * max(grad.size(-2) / grad.size(-1)) ** 0.5

                        # Reshape if needed
                        if len(orig_shape) == 4:
                            update = update.view(orig_shape)

                else:
                    # --- AdamW Update Logic ---

                    # Initialize momentum buffers if needed
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                        state['step'] = 0

                    # Update step count
                    state['step'] += 1
                    step = state['step']

                    # Update momentum buffers
                    state['exp_avg'].lerp_(grad, mask * (1 - beta1))
                    state['exp_avg_sq'].mul_(1 - mask * (1 - beta2)).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Compute AdamW update
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    update = state['exp_avg'] / bias_correction1 / denom

                    if isinstance(mask, torch.Tensor):
                        update.mul_(mask)

                # Apply update
                p.add_(update, alpha=-lr)

                # Collect metrics
                if collecting_metrics:
                    update_param_names.append(name)
                    update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32).unsqueeze(0))
                    update_maxs.append(update.abs().max().unsqueeze(0))

        # Store metrics
        if collecting_metrics:
            self._update_norms = update_norms
            self._update_maxs = update_maxs
            self._update_param_names = update_param_names

        return None

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        """Get metrics about the optimization step."""
        if not (self._record_update_metrics and self._collecting_metrics):
            return {}

        device = get_default_device() if self._device is None else self._device
        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        param_names = self._update_param_names
        update_norms = self._update_norms
        update_maxs = self._update_maxs

        if param_names is None or update_norms is None or update_maxs is None:
            return {}

        # Reduce metrics if needed
        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce norms
            all_norms = torch.cat(update_norms).to(device) ** 2.0
            dist.reduce(all_norms, dst_rank, op=dist.ReduceOp.SUM, group=process_group)
            update_norms = (all_norms ** (0.5)).squeeze(0).split(1)

            # Reduce maxs
            all_maxs = torch.cat(update_maxs).to(device)
            dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
            update_maxs = all_maxs.split(1)

        # Collect metrics
        metrics = {}
        for param_name, update_norm, update_max in zip(param_names, update_norms, update_maxs):
            metrics[f"update/{param_name}.norm"] = update_norm.squeeze(0)
            metrics[f"update/{param_name}.max"] = update_max.squeeze(0)

        # Reset stored metrics
        self._update_norms = None
        self._update_maxs = None
        self._update_param_names = None

        return metrics