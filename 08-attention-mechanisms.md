# 08 · Attention Mechanisms — The Heart of the Transformer

> **TL;DR** Attention is a soft, differentiable lookup. Each **query** vector decides how much to read from each **key/value** pair using a softmax. In a self-attention layer every token is both a query and a key/value, so tokens "talk" to each other. Multi-head attention does this in parallel with several smaller heads. Causal masking forces tokens to only see the past. **In 2026, you call `F.scaled_dot_product_attention` (which uses Flash Attention 2/3 under the hood) and never write the inner loop yourself.**

## 1. The intuition

Imagine a meeting where everyone has:
- A **question** they want answered (`q`).
- A **subject they're an expert in** (`k`).
- An **answer they can give if asked** (`v`).

Each person looks at everyone else's "subject," weights them by how relevant they are to their own "question," and combines their "answers" weighted accordingly. That's attention.

Mathematically, with `Q, K, V` matrices (one row per token):

```
attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d) ) V
```

- `Q Kᵀ` is the matrix of relevance scores: `score[i, j]` = "how much should query `i` care about key `j`?"
- Divide by `sqrt(d)` to keep scores in a sensible range (otherwise softmax saturates).
- Softmax row-wise → weights that sum to 1.
- Multiply by `V` → weighted sum of values for each query.

That's the whole thing. Everything else is window dressing.

---

## 2. Self vs cross attention

- **Self-attention:** `Q`, `K`, `V` come from the **same** sequence. Used in every transformer LLM today.
- **Cross-attention:** `Q` comes from one sequence (decoder), `K, V` from another (encoder). Used in encoder-decoder models (T5, mBART, Whisper) and in image-conditioned text models. Modern decoder-only LLMs don't use it.

Both are the same operation; only the source of QKV differs.

---

## 3. From dimension to multi-head

A single attention head with dim `D` is wasteful — it must encode every kind of relationship in one place. **Multi-head attention** splits `D` into `H` heads, each of dim `head_dim = D / H`, and runs attention separately on each, then concatenates. Each head can specialize.

Shapes:

```
input x:        (B, T, D)
Q, K, V:        (B, T, D)            after linear projections
reshape:        (B, T, H, head_dim)
transpose:      (B, H, T, head_dim)
attention out:  (B, H, T, head_dim)
back-shape:     (B, T, D)
output proj:    (B, T, D)
```

**Why heads help:** different heads end up tracking different patterns (positional, syntactic, semantic, copy-style, etc.). Empirically, 8-128 heads is sweet-spot territory; head_dim of 64 or 128 is best for Tensor Cores.

---

## 4. The causal mask

For language modeling we predict the next token, so position `i` must not see positions `> i`. We achieve this by **adding `-∞` to the upper triangle of the attention logits before softmax**. After softmax those entries become zero — token `i` ignores the future entirely.

```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
```

Equivalently, `F.scaled_dot_product_attention(..., is_causal=True)` does it in the kernel.

In encoder-only models (BERT) you use no mask. In encoder-decoder you have a causal mask only on the decoder side.

---

## 5. The simplest implementation (don't ship this)

```python
import torch, torch.nn as nn, math

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H, self.D = H, D
        self.head_dim = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x):                                # x: (B, T, D)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)           # each (B, T, D)

        # split into heads
        q = q.view(B, T, self.H, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.H, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.H, self.head_dim).transpose(1, 2)

        # attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = attn @ v                                    # (B, H, T, d)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.D)
        return self.proj(out)
```

Read it twice. Trace the shapes in your head. **If you can write this from scratch, you understand 80% of transformers.**

But — `O(T²)` memory and `O(T²)` compute. For `T=8192`, the score matrix is 256 MB per head per sample in fp32. Don't ship this.

---

## 6. Flash Attention (the 2026 default)

Flash Attention (Dao et al. 2022) and its successors (FA2 2023, FA3 2024) are **mathematically the same operation** but use a tiled, IO-aware kernel that:

- Never materializes the full `(T, T)` matrix in HBM.
- Uses online softmax (a streaming-friendly normalization).
- Fuses softmax + dropout + masking + matmul into one kernel.

Memory becomes `O(T)`. Speed: ~2-3× over naive. **For long context (32k+), FA is the only option.**

You don't write FA yourself. You call:

```python
out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,                          # (B, H, T, head_dim)
    is_causal=True,                   # for LM
    dropout_p=0.0,
    enable_gqa=True,                  # auto-handles GQA (chapter 9)
)
```

PyTorch dispatches to FA2/FA3 when shapes and dtypes allow. To force the FA backend:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

For more flexibility (custom masks, sliding windows, sparse patterns), use **`torch.nn.attention.flex_attention`** (PyTorch 2.5+) — a JIT-compiled API that compiles your custom score-modifier into a Flash-style kernel:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=T, KV_LEN=T)
out = flex_attention(q, k, v, block_mask=block_mask)
```

Flex Attention is a 2025 superpower. If you've ever written a `+ -1e9 * mask` line and watched memory blow up, it solves your problem.

---

## 7. Sliding-window attention (Mistral-style)

Some models bound attention to a window of the past `W` tokens:

```
position i attends to positions [max(0, i - W + 1), i]
```

Used by Mistral 7B (W=4096), Gemma 2/3 (mixed local + global), Llama 4 long-context layers. Saves compute at long context, with little quality loss because most useful information is local.

In Flex Attention:

```python
def sliding(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < W)
```

A common 2025-2026 architecture pattern: **interleave** sliding-window layers with full-context layers (e.g., Gemma 2 uses 1:1 alternating local/global). You get long-range connectivity without quadratic blow-up across all layers.

---

## 8. Multi-query (MQA) and grouped-query (GQA): preview

To save inference memory, **MQA** uses one shared K/V per layer (with H queries), and **GQA** shares one K/V per group of `H_q / H_kv` queries. This is so important for inference that it gets its own chapter — see **[09-kv-cache-mqa-gqa.md](./09-kv-cache-mqa-gqa.md)**.

In code, just use fewer K/V heads:

```python
self.q_proj = nn.Linear(D, H_q * head_dim, bias=False)
self.k_proj = nn.Linear(D, H_kv * head_dim, bias=False)
self.v_proj = nn.Linear(D, H_kv * head_dim, bias=False)
# H_q = 32, H_kv = 8 means GQA with group size 4
```

Then call `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` — PyTorch tiles K/V across the query heads automatically.

---

## 9. Multi-Head Latent Attention (DeepSeek-V2/V3)

A 2024 invention that has become a 2026 architecture standard for top-tier models. **MLA** projects K and V into a tiny shared latent space (`d_kv ≈ 512` regardless of `H`), stores only the latent, and reconstructs K/V on the fly:

- **KV cache size** drops by ~90% versus MHA.
- Quality is on par with MHA, often better than GQA at the same KV-cache budget.
- More complex to implement; numerical stability needs care.

Used by DeepSeek-V2/V3, MiniMax-Text-01. Worth knowing exists; for hand-rolled small LLMs, GQA is simpler. We touch it briefly in chapter 9.

---

## 10. Attention with biases / temperatures (optional)

Some 2025 models add small wrinkles inside attention:

- **QK-norm** (Llama-3.5-Int4, OLMoE, Gemma 3): apply RMSNorm to Q and K *after* RoPE, before the dot product. Stabilizes long-context training.
- **Attention temperature** (e.g., Gemma 2 logit soft-cap): replace softmax(`x`) with `softmax(soft_cap * tanh(x / soft_cap))`. Caps attention scores to prevent runaway logits.
- **Attention sinks** (StreamingLLM): always keep tokens 0-3 in K/V cache to anchor softmax at long context.

These are 1-2 line tweaks on top of the standard attention block.

---

## 11. Sanity-check your attention

A correctness test for any attention implementation:

```python
B, H, T, d = 2, 4, 8, 16
q = torch.randn(B, H, T, d)
k = torch.randn(B, H, T, d)
v = torch.randn(B, H, T, d)

# reference (eager)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
ref = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(mask, float('-inf')).softmax(-1) @ v

fast = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(ref, fast, atol=1e-5))     # should be True
```

This is your canary. If a refactor breaks it, stop.

---

## 12. Attention's compute and memory cost

| Resource | Standard (MHA) | Flash | Sliding (W) |
|----------|----------------|-------|-------------|
| Compute | `O(B H T² d)` | `O(B H T² d)` | `O(B H T W d)` |
| Memory (act'n) | `O(B H T²)` | `O(B H T)` | `O(B H T W)` |
| KV cache (inference) | `O(L H T d)` | same | same |

Flash Attention shrinks **memory**, not compute. Sliding shrinks both — at long context. MLA shrinks **KV cache memory** dramatically.

For inference, the KV cache (chapter 9) often dominates memory by orders of magnitude. **That's why GQA / MQA / MLA exist.**

---

## 13. The 2026 cheat sheet

- **Default to `F.scaled_dot_product_attention`** for both training and inference.
- **Use `is_causal=True`** for LM; let the kernel build the mask.
- **Use `enable_gqa=True`** so the same kernel handles MQA/GQA.
- **Use FlexAttention** when you need sliding windows, document boundaries, ALiBi-like biases, or sparse patterns.
- **head_dim = 64 or 128**; `D` divisible by `H`.
- **Apply RoPE inside attention**, after Q/K linear projections, before the dot product.
- **At inference, GQA ratio 4-8** is standard. If you really need long context, look at MLA.
- **Soft-cap logits or QK-norm** if you see attention divergence at long context.

---

## Going deeper

- Vaswani et al. 2017 — the original "Attention Is All You Need" paper.
- Dao et al. 2022 / 2023 / 2024 — Flash Attention, FA2, FA3.
- Karpathy's `nanoGPT` — `model.py` has the cleanest hand-rolled attention you'll find.
- DeepSeek-V2 paper — readable derivation of MLA.
- PyTorch 2.5+ docs on `torch.nn.attention.flex_attention` — the future of custom attention.

Next: **[09-kv-cache-mqa-gqa.md](./09-kv-cache-mqa-gqa.md)** — the inference optimizations that matter most.
