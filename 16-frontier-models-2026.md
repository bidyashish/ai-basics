# 16 · Frontier Models in 2026 — Gemma 4 and Qwen 3.6

> **TL;DR** Two open-weights families are setting the 2026 small-/mid-model bar: **Gemma 4** (Google, multimodal, includes a 31B dense and a 26B-A4B MoE) and **Qwen 3.6** (Alibaba, hybrid Gated-DeltaNet + Attention, includes a 27B dense and a 35B-A3B MoE). Both are Apache 2.0, both natively support **128k-262k context**, both ship with strong vision, and both rewrite parts of the standard transformer template you saw in chapter 11. This file shows what's new, why it matters, and the exact configs.

---

## 1. The 2026 picture in one paragraph

The simple "stack 32 attention blocks, top with an LM head" template still wins at a few-billion-param size. But at the **dense-30B** and **MoE-30B-active-3B** classes, the frontier has moved on. The new shared themes are:

1. **Hybrid attention.** Replace most attention layers with **linear-attention** variants (Gated DeltaNet, Mamba-2, RWKV-7) that don't quadratically blow up at long context. Keep a few full-attention layers to retain quality.
2. **Smaller KV caches.** Either via **shared-KV** between layers (Gemma 4) or **MLA-style latent K/V** (DeepSeek-V3) or **linear attention** that has no KV at all (Qwen 3.6 DeltaNet layers).
3. **Per-Layer Embeddings.** Inject extra learned signals into every decoder block on top of the residual (Gemma 4 PLE).
4. **Big native context out of the box.** 128k-262k is the new normal. Extension to 1M via YaRN is built in.
5. **Multimodality is standard.** Vision-text is table stakes; audio is rising fast.
6. **Multi-Token Prediction (MTP)** as a free training-time signal that doubles as a speculative-decoding draft head at inference.
7. **Thinking modes** with a toggle (`enable_thinking=True`) for reasoning workloads.

You'll see all seven across both families.

---

## 2. Gemma 4 (Google, 2026)

### The family

| Variant | Total params | "Effective" (active) | Modalities | Context |
|---------|-------------|----------------------|------------|---------|
| **Gemma 4 E2B** | 5.1B (2.3B effective) | 2.3B | text, image, video, audio | 128k |
| **Gemma 4 E4B** | 8B (4.5B effective) | 4.5B | text, image, video, audio | 128k |
| **Gemma 4 26B-A4B** (MoE) | 26B | 4B active per token | text, image, video | 256k |
| **Gemma 4 31B** (Dense) | 31B | 31B | text, image, video | 256k |

Both base and instruction-tuned (IT) checkpoints. License: **Apache 2.0** (truly open, including for commercial fine-tunes).

The "E" in E2B/E4B = "Edge" — designed for on-device, with **Per-Layer Embeddings** stored separately and loaded on demand so the *runtime* memory footprint is much smaller than the total checkpoint size. E2B reports a 2.3 B effective active parameter count even though the file is 5.1 B.

### What's architecturally new in Gemma 4

#### Alternating sliding-window / global attention

Half the layers attend within a 512-token (small models) or 1024-token (big models) **sliding window**. The other half are full-context **global** layers. Cheaper compute and memory at long context, with the global layers keeping cross-document connectivity. (This pattern started in Gemma 2 and matured in Gemma 4.)

In code:

```python
# pseudo: alternating local / global pattern
for i, layer in enumerate(self.layers):
    if i % 2 == 0:
        x = layer(x, mask=local_window_mask(W))      # sliding window
    else:
        x = layer(x, mask=causal_mask)                # full context
```

Combine with `torch.nn.attention.flex_attention` (chapter 8) — one kernel, custom mask per layer.

#### Dual-RoPE (standard + "pruned" RoPE)

- **Local layers** use a standard RoPE base (within the small window, all frequencies are useful).
- **Global layers** use a **"pruned" RoPE** — high-frequency components dropped — so the model can extrapolate cleanly to 128k-256k without the high-freq aliasing that breaks extension.

You don't need to redesign anything; you just precompute two `(cos, sin)` tables and pick per layer.

#### Per-Layer Embeddings (PLE)

The signature Gemma 4 trick. In addition to the usual token embedding fed to the residual stream, **every decoder layer reads a small, separate embedding** specific to that layer:

```
PLE_l[token_id] = combine(
    layer_token_table_l[token_id],          # token-identity component
    learned_proj_l(token_context_features)   # context-aware component
)
```

`PLE_l` is added to the layer's input. The PLE dim is much smaller than `D` (think `D_PLE ≈ 256`), so the parameter cost is bounded.

Why it works: each layer gets a small, layer-specific "tag" of the token's identity right next to whatever the residual stream already carries. This lets layers specialize without losing token-level grounding. For multimodal inputs, the PLE for image patch tokens is computed before the soft-token merge using the pad-token ID as a neutral signal.

Effect: better depth efficiency. A 31B Gemma 4 punches above its weight largely because of PLE.

#### Shared KV cache

The last `num_kv_shared_layers` layers **don't compute their own K and V projections**. They reuse the K and V of the most recent non-shared layer of the same attention type (local or global). The Q projection is still per-layer.

```
K, V cache stored for: layer_0, layer_2, ... up to layer_(L - num_shared)
Layers L-N..L:        Q is fresh, K/V is borrowed
```

Effect: dramatic reduction in KV cache size for long-context inference, which is the dominant memory cost on edge devices. Quality cost: nearly zero (the deeper layers are operating on similar K/V representations anyway).

The combination of **shared KV + sliding window** is what lets Gemma 4 E4B run a 128k-context conversation on a phone.

#### Vision encoder

- 2D learned positional embeddings.
- **Multidimensional RoPE** in the vision tower.
- Native support for **variable aspect ratios** (no center crop).
- Image tokens are configurable: **70 / 140 / 280 / 560 / 1120** per image. Pick at inference time based on detail vs cost.

#### Audio encoder (E variants only)

USM-style conformer encoder shared with Gemma-3n. ~30 ms per audio frame. CoVoST and FLEURS scores in the model card.

### Headline benchmarks (text-only, IT)

| Benchmark | Gemma 4 31B | Gemma 4 26B-A4B | Gemma 4 E4B | Gemma 3 27B (prev gen) |
|-----------|-------------|-----------------|-------------|-------------------------|
| MMLU Pro | 85.2 | 82.6 | 69.4 | 67.6 |
| AIME 2026 | 89.2 | 88.3 | 42.5 | 20.8 |
| GPQA Diamond | 84.3 | 82.3 | 58.6 | 42.4 |
| BigBench Extra Hard | 74.4 | 64.8 | 33.1 | 19.3 |
| LiveCodeBench v6 | 80.0 | 77.1 | 52.0 | — |
| Codeforces ELO | 2150 | 1718 | 940 | — |
| MMMU Pro (vision) | 76.9 | 73.8 | 52.6 | — |
| MRCR-v2 (long context) | 66.4 | 44.1 | 25.4 | — |
| LMArena (text only) | 1452 | 1441 | — | — |

Translation: the 31B Dense is roughly the **strongest open multimodal model under 50B** at release, including being competitive with much larger proprietary models on math and reasoning. The 26B-A4B MoE matches it at ~6× cheaper inference.

### Inference notes

- HuggingFace `transformers` day-zero support, `bitsandbytes` and `PEFT` integrate cleanly.
- `llama.cpp` GGUF available — run on a laptop, including via Ollama and LM Studio.
- `MLX` Apple Silicon build with **TurboQuant** (Apple's W4A16 + smarter calibration).
- `mistral.rs` Rust runtime with full multimodal + tool-calling.
- 4-bit quantization plus **3.5-bit KV cache** is the recommended on-device combination.

### When to use which Gemma 4

- **E2B / E4B** — phones, browsers, edge devices, anywhere RAM is tight.
- **26B-A4B MoE** — high-throughput chat servers; lots of users, MoE keeps active params at 4B per token.
- **31B Dense** — best per-token quality, reasoning, and code; choose this when you don't have many requests in flight (smaller batch) and want the best answer.

---

## 3. Qwen 3.6 (Alibaba, April 2026)

### The family

| Variant | Total params | Active per token | Modalities | Context |
|---------|-------------|--------------------|------------|---------|
| **Qwen 3.6-27B** (Dense) | 27B | 27B | text + vision | 262k native, ~1M with YaRN |
| **Qwen 3.6-35B-A3B** (MoE) | 35B | ~3B | text + vision | 262k native, ~1M with YaRN |
| Qwen 3.6 Max Preview | (closed) | (closed) | text + vision | very long |

License: **Apache 2.0**. Same situation as Gemma 4 — the open release is genuinely permissive.

### The architectural headline: hybrid Gated DeltaNet + Gated Attention

Qwen 3.5 introduced this and Qwen 3.6 doubles down. Most layers are **Gated DeltaNet** — a *linear-attention* variant that:

- Has **no KV cache that grows with sequence length** (constant-size per layer).
- Compute is `O(T)` per layer instead of `O(T²)`.
- "Delta rule" lets it act as a fast read-write key-value memory.
- Quality at long context is essentially as good as full attention.

A small fraction of layers are still full **Gated Attention** (basically GQA with a learned gate on the output). They retain the precise long-range retrieval that pure linear attention is weak on.

The pattern is **3 DeltaNet : 1 Attention**:

```
27B Dense:  16 × (3 × DeltaNet + 1 × Attention)  →  64 layers total
35B-A3B:    10 × (3 × DeltaNet + 1 × Attention)  →  40 layers total
            (every block ends with an MoE FFN)
```

For long context (32k+), this means the *attention* memory and compute curve flattens dramatically — you only pay quadratic cost in the 25% of layers that are full attention.

### Qwen 3.6-27B Dense — exact config

| Field | Value |
|-------|-------|
| `vocab_size` | 248,320 (padded for tensor parallelism) |
| `hidden_size` (D) | 5,120 |
| `num_hidden_layers` (L) | 64 — 48 DeltaNet + 16 Gated Attention |
| `num_attention_heads` (Q) | 24 (in attention layers) |
| `num_key_value_heads` (KV) | 4 (GQA group 6) |
| `head_dim` (attention) | 256 |
| `head_dim` (DeltaNet) | 128, 48 V heads, 16 QK heads |
| `intermediate_size` (F) | 17,408 |
| `rotary_pct` (RoPE dim per head) | 64 / 256 |
| `max_position_embeddings` | 262,144 native |
| `rope_theta` | 10,000,000 (very high — built for long context) |
| Vision encoder | yes |
| Tensor type | bf16 |

Active params per forward = full 27B. Trained with **MTP** (multi-step token prediction) for both quality and speculative decoding.

### Qwen 3.6-35B-A3B MoE — exact config

| Field | Value |
|-------|-------|
| `vocab_size` | 248,320 |
| `hidden_size` | 2,048 |
| `num_hidden_layers` | 40 — 30 DeltaNet + 10 Gated Attention |
| `num_attention_heads` (Q) | 16 |
| `num_key_value_heads` | 2 (GQA group 8) |
| `head_dim` | 256 (attention), 128 (DeltaNet) |
| `num_experts` | 256 |
| `num_experts_per_tok` | 9 (8 routed + 1 shared) |
| `expert_intermediate_size` | 512 (each expert is small — fine-grained MoE in DeepSeek-V3 style) |
| `max_position_embeddings` | 262,144 native |
| `rope_theta` | 10,000,000 |
| Vision encoder | yes |
| Tensor type | bf16 |

Active params per token ≈ **3 B**: 9 small experts × 512 × 2 + DeltaNet/Attention compute. The total memory footprint is 35 B, but each token's compute matches a 3 B dense model. Fast inference, large knowledge.

### Multi-Token Prediction (MTP)

Both models train an extra small head that predicts not only token `t+1` but also `t+2` (and sometimes `t+3`). At inference time, this auxiliary head produces draft tokens for **speculative decoding** at near-zero cost — vLLM and SGLang exploit it for ~1.8-2× decode speedups with no quality loss.

### "Thinking Preservation"

Reasoning models often emit hundreds of tokens of `<think>...</think>` chain-of-thought before their final answer. In multi-turn chat, naively keeping all that thinking in context blows up the prompt and confuses later turns. Qwen 3.6 introduces **Thinking Preservation**: per-turn thinking can be stored compactly and only re-injected when the model judges it useful, instead of being concatenated verbatim. Toggle with `"preserve_thinking": True` in the chat template. Combined with `enable_thinking=True/False`, you get fine control over reasoning behavior.

### Long context

Both models are trained at 262k context natively. The combination of:

- **rope_theta = 10⁷** (very long-period RoPE),
- **3:1 DeltaNet:Attention** (memory and compute friendly at long T),
- **YaRN scaling** built into the config,

extends to **~1M tokens** at inference with a small fine-tuning anneal, and the Qwen team ships a `Qwen3.6-1M` variant that has done that anneal for you.

### Headline benchmarks

The Qwen 3.6-27B Dense surprised everyone by **beating Qwen 3.5-397B-A17B** (the previous flagship MoE) on agentic coding:

| Benchmark | Qwen 3.6-27B | Qwen 3.6-35B-A3B | Qwen 3.5-397B-A17B (prev) | Claude 4.5 Opus |
|-----------|--------------|------------------|----------------------------|------------------|
| SWE-bench Verified | **77.2** | (lower) | 76.2 | 80.9 |
| LiveCodeBench v6 | 83.9 | — | — | — |
| SWE-bench Pro | 53.5 | — | — | — |
| Terminal-Bench 2.0 | 59.3 | — | — | — |
| MMLU-Pro | 86.2 | — | — | — |

The story: a **27B dense** Apache-licensed open model, runnable on a single 80 GB GPU at full bf16, is now within striking distance of frontier closed models at agentic coding. That's the genuinely surprising result of 2026.

### Implementation notes

- HuggingFace `transformers` ≥ 4.55 has the Qwen 3.6 model class with custom Gated DeltaNet kernels.
- vLLM ≥ 0.7.x supports the hybrid stack natively (vLLM team co-designed the Triton kernel for Gated DeltaNet).
- The DeltaNet kernel is harder to write than standard attention; if you're rolling your own, use the `flash-linear-attention` library for reference implementations.
- For MoE-A3B, set `--enable-expert-parallel` in vLLM to spread the 256 experts across multiple GPUs; otherwise memory will blow up on 1 GPU.

### When to use which Qwen 3.6

- **27B Dense** — agentic coding, IDE assistants, anywhere quality > throughput. The current open SOTA on SWE-bench.
- **35B-A3B MoE** — high-throughput chat, RAG, agents that need speed; same architecture and tokenizer as 27B but inference compute of a 3B model.
- **1M variant** — repository-scale code analysis, long document reasoning.

---

## 4. The new architecture cheat sheet

Compared to the chapter 10 / 11 template (Llama-3 / Qwen 2.5 style), what changed in 2026?

| Component | Llama 3 / Qwen 2.5 | Gemma 4 | Qwen 3.6 |
|-----------|---------------------|---------|----------|
| Attention | full GQA | sliding + global GQA, alternating | 3 × Gated DeltaNet : 1 × Gated Attention |
| KV cache | full | shared-KV last N layers | tiny — only on attention layers |
| Position | RoPE | dual-RoPE (std + pruned) | RoPE θ=10⁷ |
| FFN | SwiGLU dense | SwiGLU dense (or MoE on A4B) | dense or fine-grained MoE 256-expert |
| Norm | RMSNorm | RMSNorm + per-layer-emb | RMSNorm |
| Extras | — | Per-Layer Embeddings, vision, audio | MTP head, Thinking-Preservation, vision |
| Native ctx | 8-32k | 128-256k | 262k native, 1M extensible |

If you only learn one architectural idea from each: **Gemma 4 → Per-Layer Embeddings**, **Qwen 3.6 → Hybrid linear/full attention**.

Both make the standard transformer template feel a bit dated. Expect later 2026 / 2027 models to combine them: PLE + DeltaNet + MoE + MLA. The pieces are already lying around.

---

## 5. Fine-tuning these models

The good news: the chapter 14 SFT / DPO / GRPO recipes work as-is, with two caveats.

1. **Mind the chat template.** Both families use distinct special tokens for thinking, tool calls, and roles. Always go through `tokenizer.apply_chat_template(...)`. Manual concatenation will break the model.
2. **Mind the linear-attention kernels.** Backward through Gated DeltaNet at long sequence length is fragile in some implementations. Train at 8k-16k context first, anneal to long context, and verify gradient norms.

QLoRA (chapter 12) works on both. For Gemma 4 PLE, freeze the per-layer embedding tables during LoRA training — they're already well-fit, and unfreezing them often hurts.

---

## 6. Decision tree for picking a model in 2026

```
Need vision + audio + on-device?
   → Gemma 4 E2B / E4B

Need top-tier coding agent?
   → Qwen 3.6-27B Dense

Need cheap high-throughput chat?
   → Qwen 3.6-35B-A3B MoE  or  Gemma 4 26B-A4B MoE

Need best reasoning / math under 50B?
   → Gemma 4 31B Dense  (AIME 2026: 89.2)

Need a 1M-context analyst?
   → Qwen 3.6-1M

Need to fine-tune cheaply on 1 GPU?
   → Gemma 4 E4B or Qwen 3.6-7B (if released) with QLoRA
```

---

## 7. Going deeper

- **HuggingFace blog: Welcome Gemma 4** — the official launch post with multimodal demos and code snippets.
- **Qwen3.6-27B blog** at qwen.ai — architecture, benchmarks, why dense beat MoE for them.
- **HuggingFace `Qwen/Qwen3.6-35B-A3B`** model card — config.json, MoE specifics.
- **HuggingFace `Qwen/Qwen3.6-27B`** model card — config.json, hybrid layout.
- **DeepSeek-V3 paper** — origin of the fine-grained MoE + shared-expert pattern that Qwen 3.6-MoE inherits.
- **Gated DeltaNet paper** (Yang et al. 2024) and **`flash-linear-attention`** repo — the linear attention math + kernels.
- **Multi-Token Prediction (Gloeckle et al. 2024)** — DeepSeek-V3 and Qwen 3.6 both use this technique.

---

## Sources

- [Welcome Gemma 4 — HuggingFace blog](https://huggingface.co/blog/gemma4)
- [Qwen3.6-35B-A3B — HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-27B — HuggingFace](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6-27B blog — qwen.ai](https://qwen.ai/blog?id=qwen3.6-27b)
- [Alibaba Qwen Team Releases Qwen3.6-27B — MarkTechPost (2026-04-22)](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/)
- [Qwen 3.6 Complete Guide — InsiderLLM](https://insiderllm.com/guides/qwen-3-6-local-ai-guide/)
- [Qwen3.6-27B VRAM Requirements — Will It Run AI](https://willitrunai.com/blog/qwen-3-6-27b-vram-requirements)
- [Qwen 3.6-27B vs 35B-A3B — AIMadeTools](https://www.aimadetools.com/blog/qwen-3-6-27b-vs-35b-a3b/)

Next: **[17-production-inference.md](./17-production-inference.md)** — how to actually serve any of these to 10,000 customers.
