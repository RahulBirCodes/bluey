import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
  def __init__(self, head_dim, base=10000):
    super().__init__()
    self.head_dim = head_dim
    self.base = base
  
  def calc_inv_freqs(self):
    inv_freqs = -2 * torch.arange(self.head_dim // 2) / self.head_dim
    inv_freqs = self.base ** inv_freqs
    return inv_freqs
  
  def calc_cos_sin(self, num_tokens):
    inv_freqs = self.calc_inv_freqs()
    t = torch.arange(num_tokens)
    freqs = torch.einsum("i,j->ij", t, inv_freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin
  
  def apply_rotary_emb(self, x, cos, sin):
    # t, d/2 = cos.shape
    # t, d/2 = sin.shape
    # b, h, t, d = x.shape
    x1, x2 = torch.chunk(x, 2, dim=-1)
    # b, h, t, d/2 = x1.shape
    # b, h, t, d/2 = x2.shape
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    # absolute position of rotated features doesn't matter as long as it's consistent in q and k in dot prod
    return torch.cat([o1, o2], dim=-1)

  def forward(self, q, k):
    num_tokens = q.shape[2]
    cos, sin = self.calc_cos_sin(num_tokens)
    q = self.apply_rotary_emb(q, cos, sin)
    k = self.apply_rotary_emb(k, cos, sin)
    return q, k
    

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
    self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
    self.out = nn.Linear(d_model, d_model, bias=False)
    self.rope = RotaryEmbedding(self.d_k)

  def sdpa(self, Q, K, V):
    B, H, T, D = Q.shape
    Q, K = self.rope(Q, K)
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
  def __init__(self, hidden_size=256, n_heads=8):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.norm = RMSNorm(hidden_size, learnable=False)
    self.mha = MultiHeadAttention(hidden_size, n_heads)
  
  def forward(self, x):
    t = self.norm(x)
    t = self.mha(t)
    return t + x

def swiglu(x):
  

class SwiGLU(nn.Module):
  pass

class TransformerBlock(nn.Module):
  pass

class Transformer(nn.Module):
  pass
