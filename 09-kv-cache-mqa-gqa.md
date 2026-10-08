# 09 · KV Cache, MQA, GQA — and Why Inference Is Memory-Bound

> **TL;DR** When you generate text token by token, the keys and values of all past tokens never change. Caching them — the **KV cache** — turns generation from O(T²) per step into O(T) per step. The cache itself becomes the *biggest* tensor in memory at long context. **MQA, GQA, MLA, paged attention, and KV-cache quantization** all exist to attack that one number.

## 1. Why generation is slow without a cache

A naive generation loop:

```
for t in range(T):
    logits = model(tokens[:t+1])    # full forward pass over the whole prefix
    next_tok = sample(logits[-1])
```

Each step recomputes attention over all `t+1` tokens. Total work = `O(T²)` (or `O(T³)` if you count the matmul inside attention — but let's keep it simple).

But notice: at step `t+1` the K and V tensors at positions `0..t` are the *same* as they were at step `t`. So caching them gives:

```
for t in range(T):
    logits = model(tokens[t:t+1], past_kv=cache)
    cache = updated cache
    next_tok = sample(logits[-1])
```

Now each step is `O(t)` work — we only attend the new query against all cached keys. Total `O(T²)` instead of `O(T³)`. **This is the KV cache, and every modern inference engine uses it.**

---

## 2. The shape of the cache

Per layer, per head:

```
K_cache: (B, H_kv, T, head_dim)
V_cache: (B, H_kv, T, head_dim)
```

Total memory per token (across all layers, both K and V):

```
bytes_per_token = 2 (K, V) × L × H_kv × head_dim × bytes_per_element
```

For **Llama-3-8B** (32 layers, H_kv = 8 with GQA, head_dim = 128, fp16):

```
2 × 32 × 8 × 128 × 2 = 131 072 bytes ≈ 128 KB / token
```

For a 32k-context conversation: 128 KB × 32 000 = **4.2 GB just for the cache**.
Add the **8 GB of weights** and you're at ~12 GB — fine on a 24 GB GPU.

For **Llama-3-70B** (80 layers, H_kv = 8, head_dim = 128, fp16): ~330 KB/token, ~10 GB at 32k context. Plus 140 GB of weights. Now you need MQA or quantization or model sharding.

For **Mistral-7B** (originally MHA with H_kv = 32): ~512 KB/token. Quadruple the cache cost. This is exactly why Mistral switched to **GQA** in later versions.

KV cache is the single largest variable cost at inference. Cut it and everything gets cheaper — more concurrent users, longer contexts, smaller GPUs.

---

## 3. From MHA → MQA → GQA

| Variant | # K/V heads | KV cache size | Quality |
|---------|-------------|---------------|---------|
| MHA (Vaswani 2017) | `H_kv = H_q` | full | best |
| MQA (Shazeer 2019) | `H_kv = 1` | `1/H_q` | small drop |
| GQA (Ainslie 2023) | `H_kv` shared by groups of `H_q / H_kv` | `H_kv / H_q` | almost no drop |
| MLA (DeepSeek 2024) | latent dim ~ 512 | ~10× smaller still | best for given budget |

**GQA is the 2026 default for small/medium open models.** Llama 2-70B, Llama 3, Mistral 7B v0.2+, Qwen 2.5 — all use GQA, typically with `H_q = 32, H_kv = 8` (group size 4) or `H_q = 28, H_kv = 4` (Qwen).

In code:

```python
class GQAttention(nn.Module):
    def __init__(self, D, H_q, H_kv):
        super().__init__()
        assert H_q % H_kv == 0
        self.H_q, self.H_kv = H_q, H_kv
        self.head_dim = D // H_q
        self.q_proj = nn.Linear(D, H_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(D, H_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(D, H_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(H_q * self.head_dim, D, bias=False)

    def forward(self, x, past_kv=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.H_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.H_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.H_kv, self.head_dim).transpose(1, 2)
        # apply RoPE here (omitted for brevity)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        # repeat K/V to match Q heads — or use enable_gqa
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None),
                                             enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv
```

`enable_gqa=True` (PyTorch 2.5+) makes Flash Attention itself broadcast the `H_kv` keys across the `H_q` queries inside the kernel, no `repeat_interleave` needed. Saves memory and compute.

---

## 4. Multi-head Latent Attention (MLA), briefly

DeepSeek-V2/V3 went further. MLA projects K and V to a small **latent vector** (e.g., `d_c = 512` for `D = 7168`), caches only that, and reconstructs full K/V on the fly during attention via two more linear layers.

```
hidden  → c_kv (B, T, d_c)         # tiny shared latent — this is what we cache
        → k = W_uk(c_kv)  → (B, H_kv, T, head_dim)
        → v = W_uv(c_kv)  → (B, H_kv, T, head_dim)
```

KV cache memory drops to `~d_c × L × bytes` per token (no `H` or `head_dim` factor) — typically **~10× smaller** than GQA.

Implementation gotchas: RoPE doesn't compose nicely with MLA's down-projection, so DeepSeek uses a **decoupled-RoPE** trick (small extra rotation-only K). Worth reading the V2 paper if you intend to use MLA. For 2026, MLA is making its way into open-source frameworks (vLLM, SGLang both support it).

For your own small model, **GQA is simpler and almost as good per byte at small scales.**

---

## 5. The two phases of inference

LLM inference has two distinct phases with very different cost profiles:

### Prefill (prompt processing)

You process the entire prompt at once: `(B, T_prompt, D)`. Compute is a big matmul → **compute-bound**, Tensor Cores happy. Cheap per token.

### Decode (generation)

You process one new token at a time: `(B, 1, D)`. The matmuls are small but you still load all the model weights and the entire KV cache from HBM. **Memory-bound**, expensive per token.

Numbers for Llama-3-8B fp16 on H100:

- Prefill: ~50 000 tokens/sec.
- Decode: ~120 tokens/sec/user (single batch).

Decode is ~400× slower per token. Why frameworks bend over backwards to **batch** decode (more users sharing one weight load) and to shrink the KV cache (less to load).

---

## 6. Continuous / dynamic batching

Older inference servers used **static batching**: launch B requests together, wait for them all to finish, then start the next batch. Wasteful — short requests sit idle waiting for long ones.

**Continuous batching** (Yu et al. 2022, popularized by vLLM 2023) inserts new requests into a running batch every step:

- Keeps GPU utilization at 80%+ even under varying request lengths.
- 2-10× higher throughput vs static batching for chat workloads.

You don't implement this yourself — you use vLLM, SGLang, TGI, or TensorRT-LLM.

---

## 7. PagedAttention (the trick behind vLLM)

vLLM (Kwon et al. 2023) treats the KV cache like virtual memory: split it into fixed-size **blocks** (e.g., 16 tokens each) and store metadata pointing to the blocks for each sequence. Benefits:

- No per-sequence pre-allocation (handles variable lengths efficiently).
- Easy to **share prefix** caches across requests with the same prompt prefix (system prompts!).
- Memory waste drops from ~60% (static padding) to ~4%.

This is now the production default for open-source inference. SGLang's **RadixAttention** generalizes to share *any* common prefix across requests, not just the system prompt.

---

## 8. KV-cache quantization

The KV cache lives in fp16/bf16 by default. You can store it in **int8** or **int4** instead:

- int8: ~free quality, 2× memory savings.
- int4: small quality drop (a few % on benchmarks), 4× savings.

vLLM supports `kv_cache_dtype = "fp8"` (Hopper+) and "int8" today. SGLang has experimental int4. For very long context (1M+ tokens), this is essential.

```bash
# vLLM example
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype fp8
```

You quantize on-the-fly: per-token (or per-block) scales chosen at write time. Decode reads the int8 K/V, dequantizes inside the attention kernel.

---

## 9. Speculative decoding

Decode is memory-bound; the GPU is mostly idle while waiting on KV-cache reads. **Speculative decoding** uses a small "draft" model to propose ~4 tokens at once, then the big model verifies them in a single batched forward pass:

- If draft was right, you got 4 tokens for the cost of 1 big-model forward.
- If wrong, you discard suffix and restart — minor cost.

Typical speedup: **2-3× wall-clock for chat workloads**, with no quality loss (the big model still chooses every emitted token). Implementations: vLLM `--speculative-config`, SGLang Eagle, TGI Medusa.

A 2026 variant: **EAGLE-2 / EAGLE-3** uses a small auxiliary head trained on the big model's hidden states; near-zero overhead, ~3× speedup.

---

## 10. Long-context tricks

For 1M+ tokens you stack:

- **GQA / MLA** — small KV cache.
- **YaRN** RoPE scaling (chapter 7).
- **KV quantization** (fp8/int4).
- **Sliding window attention** in some layers (Mistral, Gemma).
- **Attention sinks** — always keep tokens 0-3 in the cache (StreamingLLM).
- **Prefix caching / RadixAttention** — share KV for repeated system prompts.
- **Dynamic compression** — drop low-importance KV entries (`H2O`, `SnapKV`, `KIVI`).

A single-GPU inference of a 7B model at 1M context is now feasible (Qwen 2.5-1M, Gemma 3) using a stack like the above.

---

## 11. Implementing a KV cache from scratch

Here's a minimal `generate` with a manual cache:

```python
@torch.inference_mode()
def generate(model, tokens, max_new_tokens=200):
    cache = [None] * model.num_layers          # per-layer KV
    out_ids = list(tokens)
    x = tokens.unsqueeze(0)                    # (1, T_prompt)

    for step in range(max_new_tokens):
        logits, cache = model(x, kv_cache=cache)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        out_ids.append(next_id.item())
        x = next_id                             # next iter only feeds 1 token
    return out_ids
```

The model must accept and update the cache:

```python
class TransformerLayer(nn.Module):
    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(x, past_kv=past_kv)
        ...
        return out, new_kv
```

For batched generation, watch out for **left-padding** (prompts of different lengths) or use vLLM-style continuous batching.

---

## 12. Inference servers in 2026

You almost never serve raw PyTorch in production. Pick one:

| Server | Strengths | Quirks |
|--------|-----------|--------|
| **vLLM** | Best general throughput, PagedAttention, huge community | OpenAI-compatible API; Python heavy |
| **SGLang** | RadixAttention prefix cache, fast structured-output | Newer, fewer corners covered |
| **TensorRT-LLM** | Lowest latency on NVIDIA, best for production | NVIDIA-only, build step is painful |
| **TGI (HuggingFace)** | Drop-in for HF stack | Slower than vLLM |
| **llama.cpp** | CPU + Apple Silicon + tiny GPUs | Quantized only |
| **MLX** | Apple Silicon native | Mac-only |

For self-hosted GPU inference, **vLLM** is the safe default in 2026. For laptops, **llama.cpp** (or Ollama, which wraps it) or **MLX**.

```bash
# vLLM in three lines
pip install vllm
vllm serve Qwen/Qwen3-4B-Instruct --gpu-memory-utilization 0.9 --max-model-len 32768
# now hit http://localhost:8000/v1/chat/completions
```

---

## 13. The 2026 cheat sheet

- **Always use a KV cache.** Without it, decode is unusable.
- **Use GQA** in your model (group size 4 is canonical).
- **Use Flash Attention** with `enable_gqa=True`.
- **Quantize the KV cache** for very long context (`fp8` cheap, `int4` aggressive).
- **For production, use vLLM (or SGLang).** Don't roll your own server.
- **Speculative decoding** for ~2-3× chat speedup.
- **PagedAttention + prefix caching** when you have repeated system prompts.
- **MLA** if you're chasing the memory-quality frontier.

---

## Going deeper

- Shazeer 2019 — original MQA paper.
- Ainslie et al. 2023 — GQA paper.
- Kwon et al. 2023 — vLLM / PagedAttention paper.
- DeepSeek-V2 paper — best derivation of MLA.
- The vLLM source — surprisingly readable, especially `vllm/core/scheduler.py` and `attention/`.

Next: **[10-building-blocks.md](./10-building-blocks.md)** — RMSNorm, SwiGLU, residuals.
