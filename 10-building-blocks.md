# 10 · Building Blocks — RMSNorm, SwiGLU, and the Modern Stack

> **TL;DR** A modern transformer block is **pre-norm + RMSNorm + GQA attention with RoPE → residual → pre-norm + SwiGLU FFN → residual**. Plus tied or untied embeddings, a final RMSNorm before the LM head, and (optionally) QK-norm. This template is shared by Llama 3, Qwen 3, Mistral, DeepSeek, Gemma, and basically every open model in 2026.

## 1. The reference block (the one you'll copy)

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.attn = GQAttention(cfg.D, cfg.H_q, cfg.H_kv, cfg.head_dim,
                                rope_base=cfg.rope_base, max_T=cfg.max_T)
        self.ffn_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.D, cfg.F)

    def forward(self, x, past_kv=None):
        a, new_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv
```

That's the whole block. Twenty-something layers of this, plus an embedding and a head, and you have an LLM.

---

## 2. Residual connections

Every sub-layer (attention, FFN) computes a delta, not a replacement:

```
x_{l+1} = x_l + sublayer(norm(x_l))
```

This is the **residual stream**. It started as a fix for very deep networks (He et al. 2015 ResNet) but has a beautiful interpretation in transformers: **each sublayer reads from a shared communication channel and writes back its update**. The residual stream's information flows from input embedding to LM head with each layer pushing in its contributions.

Practical consequences:

- The first and last layers tend to do the most work; middle layers are more redundant.
- If you ablate a single residual addition, the model breaks catastrophically.
- "Logit lens" — projecting any layer's residual through the LM head — gives surprisingly readable predictions.

Don't skip residuals.

---

## 3. LayerNorm vs RMSNorm

Attention and FFN both want their inputs in a stable distribution. The classic answer is **LayerNorm** (Ba et al. 2016):

```
LN(x) = γ · (x - μ) / sqrt(σ² + ε)  + β
```

It centers and scales each token's vector independently. Two learnable params per dim: `γ` (scale) and `β` (shift).

**RMSNorm** (Zhang & Sennrich 2019) drops the mean centering and the bias:

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

Why? Empirically, the centering does almost nothing. RMSNorm has fewer ops, fewer params, and trains as well or better. **Every major 2024-2026 LLM uses RMSNorm.**

```python
class RMSNorm(nn.Module):
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.eps = eps
    def forward(self, x):
        # cast to fp32 for stable variance, then back
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)
```

PyTorch 2.4+ has `nn.RMSNorm` built in. Use it.

A subtle implementation detail: **always compute the variance in fp32** even if your activations are bf16/fp16. You'll get NaN otherwise on aggressive training runs.

---

## 4. Pre-norm vs post-norm

The ordering matters more than people think.

- **Post-norm** (original Vaswani 2017): `x = LN(x + sublayer(x))`. Norm *after* the residual.
- **Pre-norm** (used in modern LLMs): `x = x + sublayer(LN(x))`. Norm *before* the sublayer; residual stream stays raw.

Pre-norm trains more stably at depth — you don't need warmup in the same fragile way, and gradients flow cleaner through the residual path. **Every modern LLM is pre-norm.**

There are micro-variants:

- **Sandwich-norm** (some 2024 models): pre-norm and an extra post-sublayer norm. Helps stabilize very deep nets.
- **DeepNorm**: a scaling on the residual to make 1000-layer training feasible. Niche.

For 99% of cases, pre-norm + RMSNorm is correct.

---

## 5. The FFN — and why SwiGLU won

The feed-forward network in each block is two linear layers with a non-linearity:

```
FFN(x) = W_2 · activation(W_1 · x)
```

`W_1` projects from `D` to `F`; `W_2` projects back from `F` to `D`. Original transformer used ReLU and `F = 4D`.

### GLU variants

A **gated linear unit** (Dauphin 2017) splits the FFN into two parallel projections, multiplies them elementwise, and projects back. Three projections:

```
SwiGLU(x) = W_3 · ( silu(W_1 · x)  ⊙  (W_2 · x) )
```

`silu(x) = x · sigmoid(x)`, the smooth ReLU. The `(W_1 · x)` branch is "gated" by the elementwise multiply with `(W_2 · x)`.

Empirically (Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU beats ReLU FFN by a noticeable margin and beats GeGLU (using GELU instead of SiLU) by a hair.

To keep parameter count comparable to the old `F = 4D` ReLU FFN, you set `F = (4D × 2/3)` because SwiGLU has 3 matrices instead of 2: `3 × D × F = 2 × D × 4D` solves to `F ≈ 2.67 D`. In Llama 3-8B `D = 4096` → `F = 14336`. Almost exactly that ratio.

Implementation:

```python
class SwiGLU(nn.Module):
    def __init__(self, D, F):
        super().__init__()
        self.gate_proj = nn.Linear(D, F, bias=False)
        self.up_proj   = nn.Linear(D, F, bias=False)
        self.down_proj = nn.Linear(F, D, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Many implementations use a single fused `gate_up_proj: nn.Linear(D, 2*F)` and split, which is slightly faster.

`F.silu` is `x * sigmoid(x)`. You can also call it `nn.SiLU()` or `swish` — same function.

---

## 6. Activation choices: a quick map

| Activation | Formula | Used in |
|------------|---------|---------|
| ReLU | `max(0, x)` | old transformers, some MoE |
| GELU | `x · Φ(x)` | BERT, GPT-2, T5 |
| SiLU / Swish | `x · sigmoid(x)` | the gate inside SwiGLU; also dense models |
| GeGLU | `gelu(W₁x) ⊙ W₂x` | PaLM, some Gemma variants |
| SwiGLU | `silu(W₁x) ⊙ W₂x` | **Llama, Qwen, Mistral, DeepSeek (default)** |
| ReGLU | `relu(W₁x) ⊙ W₂x` | rare |

Defaults that work in 2026: **SwiGLU FFN with `F ≈ 2.67 D`**.

---

## 7. QK-norm (the optional stability trick)

When you train at long context with bf16, attention logits can explode. The fix used in OLMoE, Gemma 3, Llama 3.5-Int4, and quite a few research models is **QK-norm**:

```
q = RMSNorm(q)
k = RMSNorm(k)
scores = q @ k.T / sqrt(d)
```

After RoPE, but before the matmul. Each head gets its own RMSNorm. Cheap, ~free quality, much more stable.

Some implementations use a `LayerNorm` instead — both work. Some normalize per-head; others use a shared norm across heads.

If you're training your own model and seeing intermittent loss spikes at long context, **add QK-norm**.

---

## 8. Logit soft-cap

Used by Gemma 2 to keep final logits in a sensible range:

```
logits = soft_cap × tanh(logits / soft_cap)        # soft_cap ≈ 30
```

The same trick is sometimes applied inside attention scores. This was a fix for an old training instability and is no longer essential when QK-norm is present. Mostly historical.

---

## 9. Embedding and LM head

Putting it all together at the model level:

```python
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.V, cfg.D)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.L)])
        self.final_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.D, cfg.V, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, ids, kv_cache=None):
        x = self.embed(ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            past = kv_cache[i] if kv_cache else None
            x, new_kv = layer(x, past)
            new_cache.append(new_kv)
        x = self.final_norm(x)
        return self.lm_head(x), new_cache
```

A few details:

- The **final RMSNorm** before the head — every modern model has it. Without it, gradients on the last layer behave badly.
- **Tied embeddings** for small models (≤1B), untied for larger.
- Bias is `False` everywhere — adding biases hurts modern models slightly and saves a few params.

---

## 10. Initialization

What numbers do all those weights start at?

- **Embeddings:** `nn.init.normal_(std=0.02)`. Standard since GPT-2.
- **Linear weights** (general): `std = 0.02` works fine. Some teams scale by `1/sqrt(2 × L)` on residual projections (the `o_proj` of attention and `down_proj` of FFN) to keep activation variance constant with depth — the GPT-2 / nanoGPT trick.
- **Norm weights** (`γ`): `1.0`.
- **Biases**: don't have any.

Pseudocode:

```python
def init_weights(self):
    for name, p in self.named_parameters():
        if 'embed' in name:
            nn.init.normal_(p, mean=0, std=0.02)
        elif p.dim() == 2 and 'norm' not in name:
            nn.init.normal_(p, mean=0, std=0.02)
            if 'o_proj' in name or 'down_proj' in name:
                p.data /= math.sqrt(2 * len(self.layers))
        elif 'norm' in name:
            nn.init.ones_(p)
```

---

## 11. The full small-model recipe (architecturally)

A **canonical 2026 small LLM** in one specification:

```python
@dataclass
class Config:
    V:        int = 50_257     # vocab size
    D:        int = 1_024      # model dim
    L:        int = 24         # layers
    H_q:      int = 16         # query heads
    H_kv:     int = 4          # KV heads (GQA group size 4)
    head_dim: int = 64
    F:        int = 2_752      # FFN inner dim, ≈ 2.67 D
    max_T:    int = 4_096
    rope_base:float = 500_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
```

Every component:

1. Token embedding `(V, D)`.
2. `L` blocks, each:
   - RMSNorm
   - GQA attention with RoPE
   - RMSNorm
   - SwiGLU FFN
3. Final RMSNorm.
4. LM head (tied to embedding).

That's it. Add training (chapter 14), data (chapter 4), tokenization (chapter 6), and you have a working LLM. We'll glue it together in **[chapter 11](./11-building-qwen-from-scratch.md)**.

---

## 12. Checklist when reading other models' code

When you look at someone else's transformer (HF, Llama, Qwen), check:

- [ ] Pre-norm or post-norm?
- [ ] LayerNorm or RMSNorm?
- [ ] Bias on linears?
- [ ] FFN: ReLU / GeGLU / SwiGLU?
- [ ] FFN intermediate ratio (F / D)?
- [ ] GQA group size?
- [ ] RoPE base?
- [ ] QK-norm? logit soft-cap?
- [ ] Tied embeddings?
- [ ] Final norm before head?

Almost all 2026 models answer this list as: pre, RMS, no, SwiGLU, ~2.67, 4 or 8, 500k-1M, sometimes, often-yes-for-small-models, yes.

Spotting the small differences (Qwen 2.5 vs DeepSeek vs Llama 3) is mostly hyperparameter shifts on this template.

---

## Going deeper

- Zhang & Sennrich 2019 — RMSNorm.
- Shazeer 2020 — "GLU Variants Improve Transformer."
- He et al. 2015 — original ResNet, where residuals come from.
- Karpathy's nanoGPT and the Llama-from-scratch tutorials — the easiest references for this template.

Next: **[11-building-qwen-from-scratch.md](./11-building-qwen-from-scratch.md)** — assembling a full Qwen-style model.
