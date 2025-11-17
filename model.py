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
    cos, sin = cos.to(q.device), sin.to(q.device)
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
    return out, attn_probs

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
    attn_out, attn_probs = self.sdpa(q, k, v)
    output = self.out(self.combine_heads(attn_out))
    return output, attn_probs


class AttentionBlock(nn.Module):
  def __init__(self, hidden_size=256, n_heads=8):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.norm = RMSNorm(hidden_size, learnable=False)
    self.mha = MultiHeadAttention(hidden_size, n_heads)
  
  def forward(self, x):
    t = self.norm(x)
    t, _ = self.mha(t)
    return t + x


class SwiGLU(nn.Module):
  def __init__(self, hidden_size=256):
      super().__init__()
      self.fc1 = nn.Linear(hidden_size, 2 * 2 * hidden_size, bias=False)
      self.fc2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
      self.beta = nn.Parameter(torch.tensor(1.0))

  def forward(self, x):
      x_proj = self.fc1(x)
      x_main, x_gate = x_proj.chunk(2, dim=-1)
      gate = x_gate * torch.sigmoid(self.beta * x_gate)
      x = x_main * gate
      return self.fc2(x)


class MLP(nn.Module):
  def __init__(self, hidden_size=256):
    super().__init__()
    self.hidden_size = hidden_size
    self.norm = RMSNorm(hidden_size, learnable=False)
    self.swiglu = SwiGLU(hidden_size)

  def forward(self, x):
    t = self.norm(x)
    t = self.swiglu(t)
    return t + x


class TransformerBlock(nn.Module):
  def __init__(self, hidden_size=256, n_heads=8):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.attn = AttentionBlock(hidden_size, n_heads)
    self.mlp = MLP(hidden_size)
  
  def forward(self, x):
    x = self.attn(x)
    x = self.mlp(x)
    return x


class Transformer(nn.Module):
  def __init__(self, hidden_size=256, n_heads=8, n_layers=12, xy_size=5):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.xy_size = xy_size
    self.blocks = nn.ModuleList([TransformerBlock(hidden_size, n_heads) for _ in range(n_layers)])
    self.embedding = nn.Linear(2 * (xy_size + 1), hidden_size, bias=False)
    # emb should NOT use standard Xavier initialization
    # we can calculate and see that we need to scale by (xy_size + 1)**-0.5 to get the activation rms norm to be 1
    nn.init.normal_(self.embedding.weight, mean=0.0, std=(xy_size + 1)**-0.5)
    self.norm = RMSNorm(hidden_size, learnable=False)
    self.unembedding = nn.Linear(hidden_size, 1, bias=False)
  
  def forward(self, x):
    x = self.embedding(x)
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    x = self.unembedding(x)
    return x
