import torch
from torch.optim import Optimizer


class MuonW(Optimizer):
    """
    Wrapper that uses:
      - torch.optim.Muon for param_groups with type == "manifold"
      - torch.optim.AdamW for all other groups

    Usage:
        optimizer_class = MuonW
        param_groups = create_optimizer_groups(model, lr=..., weight_decay=...)
        optimizer = optimizer_class(param_groups, **opt_kwargs)
    """

    def __init__(self, param_groups, **optimizer_kwargs):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]

        param_groups = list(param_groups)

        # Error Checking
        if not param_groups:
            raise ValueError("MuonW got an empty parameter list")
        for g in param_groups:
            if not isinstance(g, dict):
                raise TypeError("Each param_group must be a dict")
            if "params" not in g:
                raise ValueError("Each param_group must have a 'params' key")
            if "type" not in g:
                raise ValueError("Each param_group must have a 'type' key")

        # Let base Optimizer manage param_groups/state
        super().__init__(param_groups, defaults={})

        # -------- Build kwargs for underlying optimizers --------
        adamw_kwargs: dict = {}
        muon_kwargs: dict = {}

        if "lr" in optimizer_kwargs:
            adamw_kwargs["lr"] = optimizer_kwargs["lr"]
            muon_kwargs["lr"] = optimizer_kwargs["lr"]

        if "weight_decay" in optimizer_kwargs:
            adamw_kwargs["weight_decay"] = optimizer_kwargs["weight_decay"]
            muon_kwargs["weight_decay"] = optimizer_kwargs["weight_decay"]

        # AdamW betas
        if "betas" in optimizer_kwargs:
            adamw_kwargs["betas"] = optimizer_kwargs["betas"]
        elif "beta1" in optimizer_kwargs and "beta2" in optimizer_kwargs:
            adamw_kwargs["betas"] = (
                optimizer_kwargs["beta1"],
                optimizer_kwargs["beta2"],
            )

        # Extra Muon args
        for k in ("momentum", "nesterov"):
            if k in optimizer_kwargs:
                muon_kwargs[k] = optimizer_kwargs[k]
 
        # -------- Split groups by "type" field --------
        manifold_groups = [g for g in self.param_groups if g.get("type") == "manifold"]
        other_groups = [g for g in self.param_groups if g.get("type") != "manifold"]

        # -------- Instantiate underlying optimizers --------
        self.muon = torch.optim.Muon(manifold_groups, **muon_kwargs)
        self.adamw = torch.optim.AdamW(other_groups, **adamw_kwargs)

    # -------- Standard optimizer API forwarded to both inner opts --------

    def zero_grad(self, set_to_none: bool = False):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

        return loss

    # -------- Checkpointing: pack/unpack inner optimizer states --------

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])