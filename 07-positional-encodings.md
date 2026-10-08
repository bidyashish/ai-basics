# 07 · Positional Encodings — RoPE and the Modern Alternatives

> **TL;DR** Attention by itself is **permutation-invariant**: it doesn't know which token came first. We fix that by encoding position into the queries and keys. **In 2026, RoPE (Rotary Position Embedding) is the universal default**, with **YaRN / NTK-aware scaling** when you want to extend the context window. Older schemes (sinusoidal, learned absolute, ALiBi) are legacy and get one paragraph; **NoPE on some layers** (Llama 4, SmolLM3) is the one live alternative.

## 1. Why positions need encoding

The attention score between query `q` at position `i` and key `k` at position `j` is `q · k`. Nothing in that dot product depends on `i` or `j`. Shuffle your sentence — same attention pattern, same output. The model needs help knowing the order.

Three families of fixes exist:

1. **Absolute positional encoding (APE).** Add a position-dependent vector to each token's embedding. Sinusoidal (vanilla transformer) or learned (GPT-2, BERT). Legacy.
2. **Relative positional encoding (RPE).** Inject the *distance* between tokens directly into the attention logits. ALiBi, T5-bias. Legacy.
3. **Rotary position encoding (RoPE).** Rotate query and key vectors by an angle proportional to their position so that `(q · k)` only depends on `i - j`. **The dominant choice in 2026.**

---

## 2. The legacy schemes, in one paragraph

Three older schemes still show up in old checkpoints and papers. **Sinusoidal** (Vaswani 2017) added fixed `sin`/`cos` waves to the embeddings; **learned absolute** (GPT-2, BERT) added an `nn.Embedding(max_T, D)` lookup; **ALiBi** (BLOOM) skipped position vectors and subtracted a per-head slope times `|i - j|` from the attention logits. None of them is used in a 2026 frontier model: the first two cannot extend past their training length, and ALiBi's extrapolation advantage disappeared once YaRN made RoPE extend cleanly. You will not need to implement them; you only need to recognize them when reading legacy code.

---

## 3. RoPE: the rotation trick

**Rotary Position Embedding** (Su et al. 2021) is now ubiquitous: Llama 3/4, Qwen3 / Qwen3.6, DeepSeek-V3, Gemma 3/4, gpt-oss, Kimi K2 — every current open model uses it.

### The idea

Group each query/key vector's `head_dim` into pairs `(x_0, x_1), (x_2, x_3), …`. Treat each pair as a 2D vector and **rotate it by an angle `θ_d × position`**, where `θ_d` is a frequency that decreases with the dim index.

After rotating both `q_i` and `k_j`, their dot product depends only on `i - j` (the *relative* position). The math:

```
For a 2D vector (x, y) and angle α:
  R(α) · (x, y) = (x cos α - y sin α,  x sin α + y cos α)

If we rotate q at position i and k at position j by α_i and α_j respectively:
  Rq · Rk  = some function of (α_i - α_j)
```

So a rotated dot product naturally encodes relative distance.

### The frequencies

```
θ_d = base^(-2d / D)        for d in [0, D/2)
```

`base = 10000` is the original choice. `base = 500_000` (Llama 3) or `base = 1_000_000` (Qwen3, long-context configs) extends the effective range — long-context models tune `base` upward.

Pair index 0 spins fastest (high frequency, captures fine local position). Pair index `D/2 - 1` spins slowest (low frequency, captures coarse "where in the document" info).

### The implementation

```python
import math, torch

def precompute_rope(head_dim, max_T, base=10000.0, device='cpu', dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)             # (T, head_dim/2)
    cos, sin = freqs.cos(), freqs.sin()
    return cos.to(dtype), sin.to(dtype)          # each (T, head_dim/2)

def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim) — head_dim must be even
    # cos, sin: (T, head_dim/2) — broadcast-friendly
    x1 = x[..., 0::2]                            # (B, H, T, head_dim/2)
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_x1
    out[..., 1::2] = rot_x2
    return out
```

That's RoPE. ~10 lines. You apply it to `q` and `k` only — not `v`, not the residual stream. Inside attention:

```python
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
```

### A subtle implementation detail: pair layouts

Some implementations interleave pairs as `(x_0, x_1, x_2, x_3, …)` — the **interleaved** layout above. Others split the head_dim in half — the **half-rotated** layout used by HuggingFace Llama / Qwen:

```python
def apply_rope_half(x, cos, sin):
    # x: (B, H, T, head_dim)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[..., :half]; sin = sin[..., :half]    # adjust shape per impl
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

Both produce equivalent attention scores **if you train and infer with the same layout**. Mix them and your model outputs garbage. Always check which one your loaded weights expect.

---

## 4. Extending RoPE to longer contexts

RoPE generalizes nicely *within* the trained context. Beyond it, the high-frequency rotations alias and quality drops. Several recipes exist:

### Position interpolation (PI)

Pretend the new positions are old positions divided by a stretch factor `s`:

```
position_used = position_actual / s
```

So extending 4k → 16k uses `s = 4`. Cheap, requires light fine-tuning.

### NTK-aware scaling

PI compresses *all* frequencies. NTK-aware (a.k.a. NTK by parts) scales the **base** instead of the positions, which preserves high-frequency detail and only stretches low-frequency dims:

```
base_new = base * s^(D / (D - 2))
```

Often works zero-shot (no fine-tune), at the cost of some quality.

### YaRN (Peng et al. 2023)

The 2026 favorite. Combines NTK-aware scaling with a frequency-dependent ramp: stretch low frequencies aggressively, leave high frequencies near-original. Add a small attention temperature correction. With ~100M tokens of fine-tuning, you can extend a model from 8k → 128k+ context.

```python
# config.json snippet Qwen3 ships to go from 32k native to 131k context
{
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

The `transformers` library applies this when building cosine/sine tables; `beta_fast`, `beta_slow`, and `attn_factor` have good defaults you rarely touch. vLLM and SGLang read the same keys.

### Llama-3-style "by-parts" scaling

Llama 3.1 introduced an explicit by-parts piecewise scaling: high-freq untouched, low-freq fully PI-scaled, medium-freq smoothly transitioned. Cleaner than NTK-aware in practice. Used by Llama 3.1 through Llama 4 (`"rope_type": "llama3"` in `config.json`).

### Putting it together

For your own model, pick context based on:

- **Train at 4k-8k** for compute efficiency.
- **Anneal at 32k-128k** with a YaRN-rescaled RoPE for the last few B tokens of pretraining.
- For inference at 1M context (Qwen3.6: 262k native, YaRN factor 4 to 1M), apply **YaRN** plus a separate long-context fine-tune. Frontier APIs (the GPT-6 family serves a 1.05M window) ship this built in.

---

## 5. NoPE: do you even need positions?

A **causal** transformer with no positional encoding at all (NoPE) can still learn position implicitly from the causal mask. In 2026 this is used *per layer*, not model-wide: Llama 4 interleaves NoPE global-attention layers between RoPE layers (iRoPE), and SmolLM3 drops RoPE in every 4th layer. The NoPE layers extrapolate better to unseen lengths; the RoPE layers keep short-range precision.

For your own model: RoPE everywhere by default. Try NoPE on a quarter of the layers only if you are chasing long-context extrapolation and can afford the ablation.

---

## 6. RoPE in attention — the full mini-block

```python
import torch, torch.nn as nn, math

class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, max_T):
        super().__init__()
        self.H = H
        self.head_dim = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        cos, sin = precompute_rope(self.head_dim, max_T, base=500_000.0)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x):                                 # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                             # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.cos[:T]; sin = self.sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # scaled-dot-product attention with causal mask
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)
```

This is 95% of every modern attention block. The extra 5% (KV cache, GQA) we cover next chapter.

---

## 7. Comparison cheat sheet

| Scheme | Year | Where it lives | Extrapolation | Used by (2026) |
|--------|------|----------------|---------------|----------------|
| **RoPE** | 2021 | rotates Q & K | OK; great with YaRN | **every current open model** |
| RoPE + YaRN | 2023 | rotates Q & K, scaled | excellent | Qwen3 / Qwen3.6, Llama 3.1+, DeepSeek-V3, Gemma 3 |
| NoPE (some layers) | 2023 | nothing | best | Llama 4 iRoPE, SmolLM3 |
| Sinusoidal / learned / ALiBi | 2017-2021 | embeddings or logit bias | poor / poor / good | legacy only |

---

## 8. A 2026 cheat sheet

- **Use RoPE.** Pick `base = 500k-1M` for training contexts ≥ 8k. The default `10k` is for 2k-context relics.
- **YaRN** to extend at inference / fine-tune time. Set `factor = target_T / train_T`.
- **Apply RoPE to Q and K only**, not V, not the residual stream.
- **Watch the pair layout** — interleaved vs split-half. Match the checkpoint you load.
- Test extrapolation early: train at 1k, eval at 2k. If perplexity collapses, RoPE base is too small.

---

## Going deeper

- Su et al. 2021 — original RoPE paper. Clean derivation.
- Peng et al. 2023 — YaRN paper.
- "Extending Context Window of LLMs via Position Interpolation," Chen et al. 2023.
- The HuggingFace `modeling_llama.py` — read `apply_rotary_pos_emb` once and the whole abstraction collapses.
- Llama 4 release blog (iRoPE) and the SmolLM3 blog post — how NoPE layers are mixed into RoPE models in practice.
- Eleuther.ai blog post on NTK-aware scaling — best intuitive write-up.

Next: **[08-attention-mechanisms.md](./08-attention-mechanisms.md)** — the heart of the transformer.
