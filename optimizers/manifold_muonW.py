import torch
from torch.optim import Optimizer
from ..optimizers.msign import msign
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
        #if step % 10 == 0:
            #print(f"W: {W} and step: {step}")
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

def manifold_muon_step_online(
    W: torch.Tensor,
    G: torch.Tensor,
    Lambda: torch.Tensor, # Passed in from state
    lr: float,
    alpha: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a single step of Online Dual Ascent.
    Returns: (New Weights, New Lambda)
    """
    orig_tall = True
    if W.shape[0] < W.shape[1]:
        W = W.transpose(-2, -1)
        G = G.transpose(-2, -1)
        # Lambda must also be transposed conceptually, but it is symmetric square
        # We handle Lambda management in the main loop to keep shapes consistent
        orig_tall = False

    # 1. Update Dual Variable (Lambda) - Single step of ascent
    # We measure the violation of the tangent space constraint
    # Candidate direction A is based on current Lambda
    A_candidate = msign(G + 2 * W @ Lambda)
    
    # Violation H
    H = W.T @ A_candidate + A_candidate.T @ W
    
    # Update Lambda (Ascent)
    Lambda = Lambda - alpha * H

    # 2. Compute Final Direction with updated Lambda
    A = msign(G + 2 * W @ Lambda)

    # 3. Primal Descent (Update Weights)
    new_W = W - lr * A

    # 4. Retraction (Project back to Stiefel Manifold)
    # This ensures strict orthogonality is maintained periodically
    new_W = msign(new_W)

    if not orig_tall:
        new_W = new_W.transpose(-2, -1)

    return new_W, Lambda


class ManifoldMuon(Optimizer):
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
        mm_alpha: float = 0.1,
        mm_tol: float = 1e-6,
        ADMM: bool = True,
        mm_use_momentum: bool = True,
        use_online: bool = False,
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
            #print(f"group: {group}, group['params'] {group['params']}")
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            mm_steps = group["mm_steps"]
            mm_alpha = group["mm_alpha"]
            mm_tol = group["mm_tol"]
            mm_use_momentum = group.get("mm_use_momentum", False)
            ADMM = group.get("ADMM", True)
            type = group.get("type", "standard")

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
                    
                    if type == "manifold" and p.ndim >= 2:
                        # Lambda initialization (Square matrix of size min(rows, cols))
                        dim = min(p.shape[0], p.shape[1])
                        state["lambda"] = torch.zeros((dim, dim), device=p.device, dtype=p.dtype)

                state["step"] += 1
                exp_avg, exp_avg_sq, muon_m = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["muon_m"],
                )

                # AdamW moments always maintained (even if not used)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if type == "manifold" and p.ndim >= 2:
                    # --- MANIFOLD MUON BRANCH ---
                    
                    # 1. Momentum handling
                    if mm_use_momentum:
                        # Update Muon momentum buffer
                        state["muon_m"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        G_eff = state["muon_m"]
                    else:
                        G_eff = grad

                    # 2. Perform Online Manifold Step
                    # Note: We pass the persistent Lambda
                    new_W, new_Lambda = manifold_muon_step_online(
                        p.data,
                        G_eff,
                        state["lambda"],
                        lr=lr,
                        alpha=mm_alpha
                    )
                    
                    # 3. Apply updates
                    p.data.copy_(new_W)
                    state["lambda"].copy_(new_Lambda)
                
                else:
                    # ---- AdamW branch ----
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = 1e-3 / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
