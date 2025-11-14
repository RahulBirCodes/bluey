import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
  pass


class RMSNorm(nn.Module):
  def __init__(self, num_features, eps=1e-5, learnable=True):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.learnable = learnable
    if self.learnable:
      self.scale = nn.Parameter(torch.ones(num_features))
  
  def forward(self, x):
    x_norm = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
    if self.learnable:
      return x_norm * self.scale
    return x_norm


class LayerNorm(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(num_features))
    self.bias = nn.Parameter(torch.zeros(num_features))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) * torch.rsqrt(variance + self.eps)
    return x_norm * self.scale + self.bias


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model = 256, n_heads = 8):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_k = d_model // n_heads
    self.qkv = nn.Linear(d_model, 3 * d_model)
    self.out = nn.Linear(d_model, d_model)

  def sdpa(self, Q, K, V):
    B, H, T, D = Q.shape
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    attn_scores = attn_scores / self.d_k ** 0.5
    mask = torch.tril(torch.ones(T, T, device=Q.device))
    attn_scores= attn_scores.masked_fill(mask == 0, -float("inf"))
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, V)
    return out

  def split_heads(self, x):
    b, t, d = x.shape
    return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    b, _, t, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, self.d_model)
  
  def forward(self, x):
    b, t, d = x.shape
    qkv = self.qkv(x)
    q = qkv[:, :, :self.d_model].contiguous()
    k = qkv[:, :, self.d_model:2*self.d_model].contiguous()
    v = qkv[:, :, 2*self.d_model:].contiguous()
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    attn_out = self.sdpa(q, k, v)
    output = self.out(self.combine_heads(attn_out))
    return output 


class AttentionBlock(nn.Module):
  pass

class SwiGLU(nn.Module):
  pass

class TransformerBlock(nn.Module):
  pass

class Transformer(nn.Module):
  pass
