import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MultiHeadAttention
from optimizers.muonW1 import MuonW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### check if model can learn to output initial token, given input [A, B, C, D, E], MHA should output [A, A, A, A, A]
def get_batch(batch_size=8, seq_len=5, d_model=256):
    return torch.randn(batch_size, seq_len, d_model, device=device)

def test_identity(model, steps=200, batch_size=8, seq_len=5, lr=1e-3):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
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

def test_muonW(model, steps=200, batch_size=8, seq_len=5, lr=1e-3):
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
    
model = MultiHeadAttention().to(device)
test_muonW(model)