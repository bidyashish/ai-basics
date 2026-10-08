# 10 · Building Blocks — RMSNorm, SwiGLU, and the Modern Stack

> **TL;DR** A modern transformer block is **pre-norm + RMSNorm + GQA attention with RoPE → residual → pre-norm + SwiGLU FFN → residual**. Plus tied or untied embeddings, a final RMSNorm before the LM head, and (optionally) QK-norm. This template is shared by Llama 3/4, Qwen3, DeepSeek-V3, Gemma 3/4, gpt-oss, and basically every open model in 2026. The one frontier-only addition is **depth recurrence** (looped blocks), reported for GPT-6 Astra and covered in §11.

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

## 3. RMSNorm (the only norm you need)

Attention and FFN both want their inputs in a stable distribution. The original answer was **LayerNorm** (center, scale, learned shift). **RMSNorm** (Zhang & Sennrich 2019) drops the centering and the shift:

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

Empirically the centering does nothing useful; RMSNorm has fewer ops, fewer params, and trains as well or better. **Every current LLM uses RMSNorm.** LayerNorm survives only in vision encoders and legacy code.

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

PyTorch has `nn.RMSNorm` built in. Use it.

A subtle implementation detail: **always compute the variance in fp32** even if your activations are bf16/fp16. You'll get NaN otherwise on aggressive training runs.

---

## 4. Pre-norm (plus an optional second norm)

Every current LLM is **pre-norm**: `x = x + sublayer(RMSNorm(x))`. The norm sits *before* the sublayer and the residual stream stays raw, so gradients flow cleanly through depth and warmup is far less fragile. Post-norm (`x = LN(x + sublayer(x))`, the original 2017 layout) is legacy.

The one live variant: Gemma 3/4 and OLMo 2 add a **second RMSNorm on the sublayer output** before the residual add (sometimes called sandwich norm). It costs almost nothing and removes most loss spikes in big runs. Start with plain pre-norm; add the second norm if your training is unstable.

---

## 5. The FFN — and why SwiGLU won

The feed-forward network in each block is two linear layers with a non-linearity:

```
FFN(x) = W_2 · activation(W_1 · x)
```

`W_1` projects from `D` to `F`; `W_2` projects back from `F` to `D`. The 2017 transformer used ReLU and `F = 4D`; nobody does now.

### GLU variants

A **gated linear unit** (Dauphin 2017) splits the FFN into two parallel projections, multiplies them elementwise, and projects back. Three projections:

```
SwiGLU(x) = W_3 · ( silu(W_1 · x)  ⊙  (W_2 · x) )
```

`silu(x) = x · sigmoid(x)`, the smooth ReLU. The `(W_1 · x)` branch is "gated" by the elementwise multiply with `(W_2 · x)`.

Empirically (Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU beats ReLU FFN by a noticeable margin and beats GeGLU (using GELU instead of SiLU) by a hair.

To keep parameter count comparable to the old `F = 4D` ReLU FFN, you set `F = (4D × 2/3)` because SwiGLU has 3 matrices instead of 2: `3 × D × F = 2 × D × 4D` solves to `F ≈ 2.67 D`. Real models round up: Llama 3-8B uses `F = 14336` at `D = 4096` (3.5×) and Qwen3-8B uses `F = 12288` (3.0×); the 2.67 figure is the parameter-matched floor, not a law.

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
| SwiGLU | `silu(W₁x) ⊙ W₂x` | **Llama, Qwen3, DeepSeek-V3, gpt-oss, Kimi K2 (the default)** |
| GeGLU | `gelu(W₁x) ⊙ W₂x` | Gemma 3/4 |
| SiLU / Swish | `x · sigmoid(x)` | the gate inside SwiGLU |

ReLU and GELU FFNs (BERT, GPT-2) are legacy; you will only meet them in old checkpoints. The default that works in 2026: **SwiGLU FFN with `F ≈ 2.67-3.5 D`**.

---

## 7. QK-norm (the optional stability trick)

When you train at long context with bf16, attention logits can explode. The fix used in Qwen3, OLMo 2, Gemma 3, and most 2025-2026 releases is **QK-norm**:

```
q = RMSNorm(q)
k = RMSNorm(k)
scores = q @ k.T / sqrt(d)
```

After RoPE, but before the matmul. Each head gets its own RMSNorm. Cheap, ~free quality, much more stable.

Some implementations use a `LayerNorm` instead — both work. Some normalize per-head; others use a shared norm across heads.

If you're training your own model and seeing intermittent loss spikes at long context, **add QK-norm**.

---

## 8. Embedding and LM head

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

## 9. Initialization

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

## 10. The full small-model recipe (architecturally)

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

## 11. Looped / recurrent-depth blocks (the GPT-6 pattern)

Everything above runs each block **once**. A **looped transformer** (also "recurrent depth") runs the *same* stack of blocks several times, feeding the output back in as input. Parameters stay fixed; effective depth multiplies.

```
x = prelude(x)                     # a few ordinary blocks
for pass in range(n_loops):        # same weights every pass
    x = core(x)                    # shared stack of blocks
x = coda(x)                        # a few ordinary blocks + final norm
```

Why 2026 cares:

- **The Information reported (Sept 2026) that OpenAI's GPT-6 Astra uses recurrent depth.** A Microsoft Foundry catalog note for GPT-6.1 Sol, circulated as screenshots on 6 Oct 2026, reads: "GPT-6.1-Sol uses the same base model weights as GPT-6 Sol with two inference passes instead of three." OpenAI has not published the architecture, so treat this as reported, not confirmed. OpenAI chief scientist Jakub Pachocki did say in September that the compute-graph depth of its frontier models, "including Astra, is within a factor of two of GPT-4."
- **Published open results are real.** Geiping et al. 2025 train a 3.5B recurrent-depth model whose 4 shared blocks can be unrolled 4-32 times at test time; Ouro (ByteDance, 2025) ships 1.4B/2.6B models with 4 recurrent steps that match 12B dense models; Nanbeige4.2-3B runs 22 blocks twice; Mixture-of-Recursions lets a router pick a loop count per token.
- **What you save is parameters, not compute.** `n_loops × C` blocks of compute and KV cache, with only `C` blocks of weights. A looped model is cheaper to store, download, and train per parameter; it is *not* cheaper to serve than a conventional model of the same effective depth.

A minimal implementation on top of this chapter's `TransformerBlock`:

```python
class LoopedGPT(nn.Module):
    """prelude -> (shared core) x n_loops -> coda, Geiping et al. 2025 style."""
    def __init__(self, cfg, n_prelude=2, n_core=4, n_coda=2, n_loops=4):
        super().__init__()
        self.embed   = nn.Embedding(cfg.V, cfg.D)
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_prelude)])
        self.core    = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_core)])
        self.coda    = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_coda)])
        self.inject  = nn.Linear(2 * cfg.D, cfg.D, bias=False)   # mixes state with input each pass
        self.final_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.D, cfg.V, bias=False)
        self.n_loops = n_loops

    def forward(self, ids, n_loops=None):
        n_loops = n_loops or self.n_loops          # more passes at test time = more latent 'thinking'
        e = self.embed(ids)
        for blk in self.prelude:
            e, _ = blk(e)
        s = torch.zeros_like(e)                     # recurrent state
        for _ in range(n_loops):                    # weights are reused, activations are not
            s = self.inject(torch.cat([s, e], dim=-1))
            for blk in self.core:
                s, _ = blk(s)
        for blk in self.coda:
            s, _ = blk(s)
        return self.lm_head(self.final_norm(s))
```

Training notes from the papers: sample `n_loops` randomly per batch (Geiping uses a log-normal Poisson around 32) so the model works at any depth; backprop only through the last few passes to keep memory flat; add a learned **exit gate** (Ouro, Universal Transformers) if you want the model to decide how many passes a token needs. KV cache: each pass of `core` needs its own cache entries, so a serving engine treats a 3-pass model as `P + 3C + K` layers.

When to use it: you are parameter-bound (memory, download size, training data per parameter) and can spend compute. If you are serving-cost-bound, a conventional deeper model is simpler and equally fast.

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
- [ ] QK-norm?
- [ ] One pass per block, or a shared stack looped several times?
- [ ] Tied embeddings?
- [ ] Final norm before head?

Almost all 2026 open models answer this list as: pre, RMS, no, SwiGLU, ~3, 4 or 8, 500k-1M, usually, one pass, often-yes-for-small-models, yes.

Spotting the small differences (Qwen3 vs DeepSeek-V3 vs Llama 4) is mostly hyperparameter shifts on this template.

---

## Going deeper

- Zhang & Sennrich 2019 — RMSNorm.
- Shazeer 2020 — "GLU Variants Improve Transformer."
- He et al. 2015 — original ResNet, where residuals come from.
- Karpathy's **nanochat** (2025) and HuggingFace's `modeling_qwen3.py` — the easiest readable references for this template.
- Geiping et al. 2025 — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171). The cleanest recurrent-depth recipe.
- Zhu et al. 2025 — Ouro, "Scaling Latent Reasoning via Looped Language Models" (arXiv 2510.25741); Bae et al. 2025 — Mixture-of-Recursions (arXiv 2507.10524).
- Sebastian Raschka, "GPT-6 Astra, Looped Transformers, and Hidden Reasoning" (Sept 2026) — https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and

Next: **[11-building-qwen-from-scratch.md](./11-building-qwen-from-scratch.md)** — assembling a full Qwen-style model.
