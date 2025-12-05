import torch
from ..model.model import MultiHeadAttention, SwiGLU

class MaxAbsActMonitor:
    def __init__(self, model):
        self.hooks = []
        self.stats = {
            "global_max": 0.0
        }
        self.register_hook(model)

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            if not isinstance(output, torch.Tensor): return
            with torch.no_grad():
                val = output.abs().max().item()
                if val > self.stats["global_max"]:
                    self.stats["global_max"] = val
        return hook

    def register_hook(self, model):
        for name, module in model.named_modules():
            self.hooks.append(
                module.register_forward_hook(self._hook_fn(name))
            )
    
    def log_to_wandb(self, logger, step):
        logger.log({"max_abs_act/global_max": self.stats["global_max"]}, step=step, commit=False)
    
    def reset(self):
        self.stats = { "global_max": 0.0 }
  

class RMSMonitor:
    def __init__(self, model):
        self.hooks = []
        self.stats = {}
        self.register_hook(model)

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            if not isinstance(output, torch.Tensor): return
            with torch.no_grad():
                rms = output.pow(2).mean(dim=-1).sqrt().mean().item()
                self.stats[name] = rms
        return hook

    def register_hook(self, model):
        for name, module in model.named_modules():
            is_embedding = "embedding" in name.lower()
            if is_embedding or isinstance(module, (MultiHeadAttention, SwiGLU)):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)

    def log_to_wandb(self, logger, step):
        logs = {f"rms/{layer_name}": val for layer_name, val in self.stats.items()}
        logger.log(logs, step=step, commit=False)
  
    def reset(self):
        self.stats = {}