import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MultiHeadAttention
from optimizers.muonW1 import MuonW
from optimizers.manifold_muonW import ManifoldMuon
from model import Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### check if model can learn to output initial token, given input [A, B, C, D, E], MHA should output [A, A, A, A, A]
def get_batch(batch_size=8, seq_len=5, d_model=12):
    return torch.randn(batch_size, seq_len, d_model, device=device)

def test_identity(model, steps=40, batch_size=8, seq_len=5, lr=1e-3):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        #x = get_batch(batch_size, seq_len)   
        target = x
        out = model(x)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")


### check if model with RoPE can learn to output shifted left token, given input [A, B, C, D, E], MHA should output [A, A, B, C, D]
### both NoPE and RoPE can learn this, so we look at the attn probabilities to see that RoPE learns to attend to previous token
### additionally we alternate sequences so each token isn't unique to ensure the model isn't just creating 1-to-1 relations
def test_shift_left(model, steps=100, batch_size=1, seq_len=5, lr=1e-3, d_model=128):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # create TWO unique vectors (A and B)
    # we fix the seed to ensure they are distinct
    torch.manual_seed(42)
    vec_A = torch.randn(1, d_model, device=device)
    vec_B = torch.randn(1, d_model, device=device) # distinct from A
    # construct the Sequence [A, B, A, B, A]
    x = torch.cat([vec_A, vec_B, vec_A, vec_B, vec_A], dim=0).unsqueeze(0)
    target = x.clone()
    target[:, 1:, :] = x[:, :-1, :]
    for step in range(steps):
        out, probs = model(x)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 60 == 0 and step != 0:
            print("Attn Probs:", probs)
            print(f"Shift Test | Step {step:3d} | Loss: {loss.item():.6f}")


def test_muonW(model, steps=40, batch_size=8, seq_len=5, lr=1e-3):
    optim = MuonW(model.parameters(), lr=1e-3,)
    for step in range(steps):
        x = get_batch(batch_size, seq_len)   
        target = x
        out = model(x)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")

    print(f"Step {step:3d} | Loss: {loss.item():.6f}")

def test_manifold_muonW(model, steps=40, batch_size=8, seq_len=5, lr=1e-3):
    qkv_params = []
    rest_params = []

    for name, p in model.named_parameters():
        
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (
            ("qkv.weight" in lname)
            or ("out.weight" in lname)
            or ("fc1.weight" in lname)
            or ("fc2.weight" in lname)
        ):
            qkv_params.append(p)
        else:
            rest_params.append(p)

        optimizer = ManifoldMuon(
        [
            {"params": qkv_params, "manifold": True},   # Q/K/V and linear layers on Stiefel via manifold Muon
            {"params": rest_params, "manifold": False}, # everything else AdamW
        ],
        lr=3e-4,
        betas=(0.95, 0.98),
        weight_decay=0.01,
        mm_steps=20,
        mm_alpha=0.01,
        mm_tol=1e-4,
    )

    optim = MuonW(model.parameters(), lr=1e-3,)
    
    for step in range(steps):
        x = get_batch(batch_size, seq_len)   
        target = x
        out = model(x)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")
    
    print(f"Step {step:3d} | Loss: {loss.item():.6f}")
    
def test_Transformer():
    model = Transformer().to(device)
    test_manifold_muonW(model)


test_Transformer()