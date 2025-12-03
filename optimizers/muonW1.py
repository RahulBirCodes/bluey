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
        # -------- Normalize param_groups to a list of dicts with "params" --------
        if isinstance(param_groups, (torch.nn.Parameter, torch.Tensor)):
            param_groups = [param_groups]
        elif isinstance(param_groups, dict):
            param_groups = [param_groups]

        param_groups = list(param_groups)

        normalized_groups = []
        for g in param_groups:
            if isinstance(g, dict):
                params = g.get("params", [])
                if isinstance(params, (torch.nn.Parameter, torch.Tensor)):
                    params = [params]
                else:
                    params = list(params)
                if not params:
                    continue
                g = dict(g)          # shallow copy, so we don't mutate caller's dict
                g["params"] = params
                normalized_groups.append(g)
            else:
                # bare iterable / tensor
                params = g
                if isinstance(params, (torch.nn.Parameter, torch.Tensor)):
                    params = [params]
                else:
                    params = list(params)
                if not params:
                    continue
                normalized_groups.append({"params": params})

        if not normalized_groups:
            raise ValueError("MuonW got an empty parameter list")

        # Let the base Optimizer manage param_groups normally
        super().__init__(normalized_groups, defaults={})
        # self.param_groups is now this normalized list of dicts

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
        manifold_groups = []
        other_groups = []

        for group in self.param_groups:
            params = group.get("params", [])
            if not params:
                continue

            if group.get("type") == "manifold":
                manifold_groups.append(group)
            else:
                other_groups.append(group)

        # -------- Instantiate underlying optimizers --------
        self.muon = (torch.optim.Muon(manifold_groups, **muon_kwargs))
        self.adamw = (torch.optim.AdamW(other_groups, **adamw_kwargs))

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
