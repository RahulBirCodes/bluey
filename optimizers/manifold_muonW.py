import torch
from torch.optim import Optimizer
from optimizers.msign import msign
import math
import torch

def manifold_muon_step(
    W: torch.Tensor,
    G: torch.Tensor,
    lr: float,
    alpha: float = 0.01,
    steps: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """One manifold Muon update step keeping W on a Stiefel-like manifold."""
    orig_tall = True
    if W.shape[0] < W.shape[1]:
        # Make W tall
        W = W.transpose(-2, -1)
        G = G.transpose(-2, -1)
        orig_tall = False

    # Dual variable initialization
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    for k in range(steps):
        # Candidate direction in ambient space
        A = msign(G + 2 * W @ Lambda)

        # Measure tangent-space violation
        H = W.T @ A + A.T @ W
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break

        # Dual ascent step with simple annealing
        Lambda = Lambda - alpha * (1.0 - k / steps) * H

    # Primal descent step
    new_W = W - lr * A

    new_W = msign(new_W)

    if not orig_tall:
        new_W = new_W.transpose(-2, -1)
        
    return new_W

def manifold_muon_ADMM_step(
    W: torch.Tensor,
    G: torch.Tensor,
    lr: float,
    alpha: float = 0.01,
    steps: int = 50,
    rho: int = 4.0,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Implements GD on || G + W @ (L + L.mT) ||_* (c.f. the blog)"""
    # Ensure that W and G are both tall matrices
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T
    # Initialize the lagrangian, slack, and dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    X = G + 2 * W @ Lambda
    Omega = torch.zeros_like(X)
    # Solve the dual problem with ADMM to find the update direction A
    for step in range(steps):
        # Update for Lambda (orthonormal least-squares solve)
        P = W.mT @ (1 / rho * Omega + X - G)
        Lambda_upd = 0.25 * (P + P.mT)
        # Update for X (singular value thresholding)
        B = G + 2 * W @ Lambda_upd - 1 / rho * Omega
        eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
        P_pos = 0.5 * (eye + msign(B.mT @ B - 1 / rho**2 * eye))
        X_upd = (B - 1 / rho * msign(B)) @ P_pos
        # Update for Omega (dual ascent)
        Omega_upd = Omega + rho * (X_upd - 2 * W @ Lambda_upd - G)
        Lambda, X, Omega = Lambda_upd, X_upd, Omega_upd
    # Calculate A from final ADMM solution
    # (at convergence, G + 2 * W @ Lambda \approx X)
    A = msign(G + 2 * W @ Lambda)
    # Descend on the primal problem
    new_W = W - lr * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_transpose else new_W

class ManifoldMuonW(Optimizer):
    """
    Hybrid optimizer:
      - For param groups with group['manifold'] == True:
          use manifold_muon_step (Stiefel + spectral norm) with a Muon-style
          momentum buffer.
      - For all other params: plain AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.95, 0.95),     # [0] used as Muon-style momentum; [1] for AdamW's second moment
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        mm_steps: int = 50,
        mm_alpha: float = 0.01,
        mm_tol: float = 1e-6,
        ADMM: bool = False,
        mm_use_momentum: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            mm_steps=mm_steps,
            mm_alpha=mm_alpha,
            mm_tol=mm_tol,
            ADMM=ADMM,
            mm_use_momentum=mm_use_momentum,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            mm_steps = group["mm_steps"]
            mm_alpha = group["mm_alpha"]
            mm_tol = group["mm_tol"]
            mm_use_momentum = group.get("mm_use_momentum", False)
            ADMM = group.get("ADMM", False)
            use_manifold = group.get("manifold", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state = self.state[p]

                # Initialize state lazily
                if len(state) == 0:
                    state["step"] = 0
                    # AdamW stats
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Muon-style momentum for manifold params
                    state["muon_m"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg, exp_avg_sq, muon_m = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["muon_m"],
                )

                # AdamW moments always maintained (even if not used)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if use_manifold and p.ndim >= 2:
                    print("using manifold!")
                    # ---- Manifold Muon branch: only use on matrix weights (e.g. Q/K/V) ----
                    # Use a Muon-style momentum as the "effective gradient"
                    if mm_use_momentum:
                        muon_m.lerp_(grad, 1.0 - beta1)   # simple EMA; could tweak
                        G_eff = muon_m
                    else:
                        # No momentum: use raw grad
                        G_eff = grad

                    W = p.data

                    if ADMM:
                        new_W = manifold_muon_ADMM_step(
                            W,
                            G_eff,
                            lr=lr,
                            alpha=mm_alpha,
                            steps=mm_steps,
                            tol=mm_tol,
                        )
                    else:
                        new_W = manifold_muon_step(
                            W,
                            G_eff,
                            lr=lr,
                            alpha=mm_alpha,
                            steps=mm_steps,
                            tol=mm_tol,
                        )
                    p.data.copy_(new_W)

                else:
                    # ---- AdamW branch ----
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
