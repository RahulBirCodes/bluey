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
    x1, x2 = torch.chunk(x, 2, dim=-1)
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
  def __init__(self, num_features, learnable=True, eps=1e-5):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.learnable = learnable
    self.scale = nn.Parameter(torch.ones(num_features))
    self.bias = nn.Parameter(torch.zeros(num_features))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) * torch.rsqrt(variance + self.eps)
    return x_norm * self.scale + self.bias

class ManifoldLinearGain(nn.Module):
    """
    Linear layer that multiples a capped diagonal gain matrix for manifold updates.
    """
    def __init__(self, in_features, out_features, max_gain=10.0):
        super().__init__()
        self.max_gain = max_gain
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.gain = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # clip gain
        if self.max_gain is not None:
            with torch.no_grad():
                self.gain.clamp_(min=0.001, max=self.max_gain)

        out = self.linear(x)
        out = out * self.gain
        return out   

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_heads=8, lips=False, manifold_linear_gain_cap=None):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_k = d_model // n_heads
    if manifold_linear_gain_cap is not None:
      self.qkv = ManifoldLinearGain(d_model, 3 * d_model, max_gain=manifold_linear_gain_cap)
      self.out = ManifoldLinearGain(d_model, d_model, max_gain=manifold_linear_gain_cap)
    else:
      self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
      self.out = nn.Linear(d_model, d_model, bias=False)
    self.rope = RotaryEmbedding(self.d_k)
    self.lips = lips

  def sdpa(self, Q, K, V):
    B, H, T, D = Q.shape
    Q, K = self.rope(Q, K)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    attn_scores = attn_scores / (self.d_k if self.lips else self.d_k ** 0.5)
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
  def __init__(self, n_layers=15, hidden_size=256, n_heads=8, norm_fn=None, lips=False, manifold_linear_gain_cap=None):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.has_norm = norm_fn is not None
    if self.has_norm:
      self.norm = norm_fn(hidden_size)
    self.mha = MultiHeadAttention(hidden_size, n_heads, lips=lips, manifold_linear_gain_cap=manifold_linear_gain_cap)
    self.lips = lips
  
  def forward(self, x):
    if self.has_norm:
      t = self.norm(x)
    else:
      t = x
    t, _ = self.mha(t)
    if self.lips:
      out = t / self.n_layers + x * (self.n_layers - 1) / self.n_layers
    else:
      out = t + x
    return out


class SwiGLU(nn.Module):
  def __init__(self, hidden_size=256, manifold_linear_gain_cap=None):
      super().__init__()
      if manifold_linear_gain_cap is not None:
        self.fc1 = ManifoldLinearGain(hidden_size, 2 * 2 * hidden_size, max_gain=manifold_linear_gain_cap)
        self.fc2 = ManifoldLinearGain(2 * hidden_size, hidden_size, max_gain=manifold_linear_gain_cap)
      else:
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
  def __init__(self, n_layers=15, hidden_size=256, norm_fn=None, lips=False, manifold_linear_gain_cap=None):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.has_norm = norm_fn is not None
    if self.has_norm:
      self.norm = norm_fn(hidden_size)
    self.swiglu = SwiGLU(hidden_size, manifold_linear_gain_cap=manifold_linear_gain_cap)
    self.lips = lips

  def forward(self, x):
    if self.has_norm:
      t = self.norm(x)
    else:
      t = x
    t = self.swiglu(t)
    if self.lips:
      out = t / self.n_layers + x * (self.n_layers - 1) / self.n_layers
    else:
      out = t + x
    return out


class TransformerBlock(nn.Module):
  def __init__(self, n_layers=15, hidden_size=256, n_heads=8, norm_fn=None, lips=False, manifold_linear_gain_cap=None):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.attn = AttentionBlock(n_layers, hidden_size, n_heads, norm_fn=norm_fn, lips=lips, manifold_linear_gain_cap=manifold_linear_gain_cap)
    self.mlp = MLP(n_layers, hidden_size, norm_fn=norm_fn, lips=lips, manifold_linear_gain_cap=manifold_linear_gain_cap)
  
  def forward(self, x):
    x = self.attn(x)
    x = self.mlp(x)
    return x


class LinearEmbedding(nn.Linear):
    """
    Linear embedding layer that enforces per-row output emb RMS = 1
    """
    def __init__(self, in_features: int, out_features: int, xy_size: int):
        super().__init__(in_features, out_features, bias=False)
        self.target_rms = 1.0 / (xy_size + 1) ** 0.5

    @torch.no_grad()
    def _renorm_rows(self):
        W = self.weight
        row_rms = W.pow(2).mean(dim=1, keepdim=True).sqrt()
        W *= self.target_rms / (row_rms + 1e-12)

    def forward(self, x):
        self._renorm_rows()
        return F.linear(x, self.weight, self.bias)


class Transformer(nn.Module):
  def __init__(self,
                hidden_size=256, 
                n_heads=8, 
                n_layers=15, 
                xy_size=5, 
                lips=False,
                add_fake_dim=False,
                manifold_linear_gain_cap=None,
                norm_fn=lambda d: RMSNorm(d, learnable=False)):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.xy_size = xy_size
    self.blocks = nn.ModuleList([TransformerBlock(n_layers, hidden_size, n_heads, norm_fn=norm_fn, lips=lips, manifold_linear_gain_cap=manifold_linear_gain_cap) for _ in range(n_layers)])
    input_dim = 2 * (xy_size + 1) + (1 if add_fake_dim else 0)
    # we remove the input embedding linear layer for now as it plateaus loss too early and provides no additional stability
    # if lips:
    #   self.embedding = LinearEmbedding(input_dim, hidden_size, xy_size)
    # else:
    #   self.embedding = nn.Linear(input_dim, hidden_size, bias=False)
    self.embedding = nn.Linear(input_dim, hidden_size, bias=False)
    self.embedding.is_input_embedding = True
    # emb should NOT use standard Xavier initialization
    # we can calculate and see that we need to scale by (xy_size + 1)**-0.5 to get the activation rms norm to be 1
    nn.init.normal_(self.embedding.weight, mean=0.0, std=(xy_size + 1 + (1 if add_fake_dim else 0))**-0.5)
    self.has_norm = norm_fn is not None
    if self.has_norm:
      self.norm = norm_fn(hidden_size)
    self.unembedding = nn.Linear(hidden_size, xy_size, bias=False)
    self.unembedding.is_unembedding = True
  
  def forward(self, x):
    x = self.embedding(x)
    for block in self.blocks:
      x = block(x)
    if self.has_norm:
      x = self.norm(x)
    x = self.unembedding(x)
    return x

@torch.no_grad()
def orthogonal_init(m):
    if getattr(m, "is_input_embedding", False):
        return
    if isinstance(m, nn.Linear):
        w = m.weight
        if w.shape[0] >= w.shape[1]:
            nn.init.orthogonal_(w)
        else:
            # orthogonalize on the transposed shape then transpose back
            tmp = torch.empty(w.shape[1], w.shape[0], device=w.device, dtype=w.dtype)
            nn.init.orthogonal_(tmp)
            m.weight.copy_(tmp.t())
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def make_model(arch_name, lips=False, xy_size=5, add_fake_dim=False, manifold_linear_gain_cap=None):
    if arch_name == "rms":
      ln = lambda d: RMSNorm(d, learnable=False)
    elif arch_name == "ln":
      ln = lambda d: LayerNorm(d, learnable=True)
    else:
      ln = None
    transformer = Transformer(hidden_size=256, n_heads=8, n_layers=15, xy_size=xy_size, norm_fn=ln, lips=lips, add_fake_dim=add_fake_dim, manifold_linear_gain_cap=manifold_linear_gain_cap)
    if lips:
      transformer.apply(orthogonal_init)
    return transformer
