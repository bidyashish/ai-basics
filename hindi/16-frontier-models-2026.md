# 16 · 2026 में Frontier Models — Gemma 4 और Qwen 3.6

> **TL;DR** दो open-weights families 2026 small-/mid-model bar set कर रही हैं: **Gemma 4** (Google, multimodal, इसमें 31B dense और 26B-A4B MoE शामिल हैं) और **Qwen 3.6** (Alibaba, hybrid Gated-DeltaNet + Attention, इसमें 27B dense और 35B-A3B MoE शामिल हैं)। दोनों Apache 2.0 हैं, दोनों natively **128k-262k context** support करते हैं, दोनों strong vision के साथ ship करते हैं, और दोनों standard transformer template के parts को rewrite करते हैं जो आपने chapter 11 में देखा। ये file दिखाती है कि क्या नया है, क्यों matter करता है, और exact configs।

---

## 1. एक Paragraph में 2026 Picture

Simple "32 attention blocks stack करो, LM head पर top करो" template अभी भी few-billion-param size पर जीतता है। लेकिन **dense-30B** और **MoE-30B-active-3B** classes पर, frontier आगे बढ़ चुका है। नए shared themes हैं:

1. **Hybrid attention।** ज़्यादातर attention layers को **linear-attention** variants (Gated DeltaNet, Mamba-2, RWKV-7) से replace करो जो long context पर quadratically blow up नहीं करते। कुछ full-attention layers retain करो quality रखने के लिए।
2. **Smaller KV caches।** या तो layers के बीच **shared-KV** के via (Gemma 4) या **MLA-style latent K/V** (DeepSeek-V3) या **linear attention** जिसमें KV नहीं है (Qwen 3.6 DeltaNet layers)।
3. **Per-Layer Embeddings।** हर decoder block में extra learned signals inject करो residual के top पर (Gemma 4 PLE)।
4. **Box के out big native context।** 128k-262k नया normal है। YaRN के via 1M तक extension built in है।
5. **Multimodality standard है।** Vision-text table stakes है; audio fast rise कर रहा है।
6. **Multi-Token Prediction (MTP)** एक free training-time signal के रूप में जो inference पर speculative-decoding draft head के रूप में double करता है।
7. **Thinking modes** एक toggle (`enable_thinking=True`) के साथ reasoning workloads के लिए।

आप दोनों families में सातों देखेंगे।

---

## 2. Gemma 4 (Google, 2026)

### Family

| Variant | Total params | "Effective" (active) | Modalities | Context |
|---------|-------------|----------------------|------------|---------|
| **Gemma 4 E2B** | 5.1B (2.3B effective) | 2.3B | text, image, video, audio | 128k |
| **Gemma 4 E4B** | 8B (4.5B effective) | 4.5B | text, image, video, audio | 128k |
| **Gemma 4 26B-A4B** (MoE) | 26B | 4B active per token | text, image, video | 256k |
| **Gemma 4 31B** (Dense) | 31B | 31B | text, image, video | 256k |

Base और instruction-tuned (IT) checkpoints दोनों। License: **Apache 2.0** (truly open, including commercial fine-tunes के लिए)।

E2B/E4B में "E" = "Edge" — on-device के लिए designed, **Per-Layer Embeddings** के साथ separately stored और on demand loaded ताकि *runtime* memory footprint total checkpoint size से much smaller हो। E2B 2.3 B effective active parameter count report करता है even though file 5.1 B है।

### Gemma 4 में Architecturally क्या नया है

#### Alternating Sliding-window / Global Attention

आधी layers एक 512-token (small models) या 1024-token (big models) **sliding window** के अंदर attend करती हैं। दूसरी आधी full-context **global** layers हैं। Long context पर cheaper compute और memory, global layers cross-document connectivity रखते हुए। (ये pattern Gemma 2 में start हुआ था और Gemma 4 में matured हो गया है।)

Code में:

```python
# pseudo: alternating local / global pattern
for i, layer in enumerate(self.layers):
    if i % 2 == 0:
        x = layer(x, mask=local_window_mask(W))      # sliding window
    else:
        x = layer(x, mask=causal_mask)                # full context
```

`torch.nn.attention.flex_attention` (chapter 8) के साथ combine करो — एक kernel, per layer custom mask।

#### Dual-RoPE (standard + "pruned" RoPE)

- **Local layers** एक standard RoPE base use करती हैं (small window के अंदर, सारी frequencies useful हैं)।
- **Global layers** एक **"pruned" RoPE** use करती हैं — high-frequency components dropped — ताकि model 128k-256k तक cleanly extrapolate कर सके बिना उन high-freq aliasing से जो extension break करते हैं।

आपको कुछ redesign नहीं करना; आप बस दो `(cos, sin)` tables precompute करते हो और per layer pick करते हो।

#### Per-Layer Embeddings (PLE)

Signature Gemma 4 trick। Usual token embedding जो residual stream को feed होती है उसके अलावा, **हर decoder layer एक small, separate embedding read करती है** उस layer के specific:

```
PLE_l[token_id] = combine(
    layer_token_table_l[token_id],          # token-identity component
    learned_proj_l(token_context_features)   # context-aware component
)
```

`PLE_l` layer के input में add होती है। PLE dim `D` से much smaller है (`D_PLE ≈ 256` सोचो), तो parameter cost bounded है।

ये क्यों काम करता है: हर layer को token की identity का एक small, layer-specific "tag" मिलता है जो residual stream जो भी पहले से carry कर रहा है उसके just next में। ये layers को token-level grounding खोए बिना specialize करने देता है। Multimodal inputs के लिए, image patch tokens के लिए PLE soft-token merge से पहले pad-token ID को neutral signal के रूप में use करते हुए compute होता है।

Effect: better depth efficiency। 31B Gemma 4 अपने weight से ज़्यादा punch करता है largely PLE की वजह से।

#### Shared KV Cache

Last `num_kv_shared_layers` layers **अपना khud का K और V projection compute नहीं करते**। वो most recent non-shared layer of same attention type (local या global) का K और V reuse करते हैं। Q projection अभी भी per-layer है।

```
K, V cache stored for: layer_0, layer_2, ... up to layer_(L - num_shared)
Layers L-N..L:        Q fresh है, K/V borrowed है
```

Effect: long-context inference के लिए KV cache size में dramatic reduction, जो edge devices पर dominant memory cost है। Quality cost: nearly zero (deeper layers anyway similar K/V representations पर operate कर रहे हैं)।

**Shared KV + sliding window** का combination वो है जो Gemma 4 E4B को phone पर 128k-context conversation run करने देता है।

#### Vision Encoder

- 2D learned positional embeddings।
- Vision tower में **Multidimensional RoPE**।
- **Variable aspect ratios** के लिए native support (no center crop)।
- Image tokens configurable हैं: per image **70 / 140 / 280 / 560 / 1120**। Inference time पर detail vs cost के basis पर pick करो।

#### Audio Encoder (E variants only)

USM-style conformer encoder Gemma-3n के साथ shared। ~30 ms per audio frame। CoVoST और FLEURS scores model card में हैं।

### Headline Benchmarks (text-only, IT)

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

Translation: 31B Dense release पर roughly **strongest open multimodal model under 50B** है, including math और reasoning पर much larger proprietary models के साथ competitive होना। 26B-A4B MoE इसे ~6× cheaper inference पर match करता है।

### Inference Notes

- HuggingFace `transformers` day-zero support, `bitsandbytes` और `PEFT` cleanly integrate करते हैं।
- `llama.cpp` GGUF available — laptop पर run करो, including Ollama और LM Studio के via।
- `MLX` Apple Silicon build **TurboQuant** (Apple का W4A16 + smarter calibration) के साथ।
- `mistral.rs` Rust runtime full multimodal + tool-calling के साथ।
- 4-bit quantization plus **3.5-bit KV cache** recommended on-device combination है।

### कौन सा Gemma 4 कब Use करें

- **E2B / E4B** — phones, browsers, edge devices, anywhere RAM tight हो।
- **26B-A4B MoE** — high-throughput chat servers; lots of users, MoE active params को 4B per token रखता है।
- **31B Dense** — best per-token quality, reasoning, और code; इसे choose करो जब आपके पास flight में many requests न हों (smaller batch) और best answer चाहो।

---

## 3. Qwen 3.6 (Alibaba, April 2026)

### Family

| Variant | Total params | Active per token | Modalities | Context |
|---------|-------------|--------------------|------------|---------|
| **Qwen 3.6-27B** (Dense) | 27B | 27B | text + vision | 262k native, ~1M with YaRN |
| **Qwen 3.6-35B-A3B** (MoE) | 35B | ~3B | text + vision | 262k native, ~1M with YaRN |
| Qwen 3.6 Max Preview | (closed) | (closed) | text + vision | very long |

License: **Apache 2.0**। Same situation Gemma 4 जैसी — open release genuinely permissive है।

### Architectural Headline: Hybrid Gated DeltaNet + Gated Attention

Qwen 3.5 ने इसे introduce किया और Qwen 3.6 इस पर doubles down करता है। ज़्यादातर layers **Gated DeltaNet** हैं — एक *linear-attention* variant जो:

- **Sequence length के साथ grow होने वाला KV cache नहीं** है (per layer constant-size)।
- Compute per layer `O(T)` है instead of `O(T²)`।
- "Delta rule" इसे fast read-write key-value memory की तरह act करने देता है।
- Long context पर quality essentially full attention जितनी अच्छी है।

Layers का small fraction अभी भी full **Gated Attention** (basically GQA output पर एक learned gate के साथ) है। वो precise long-range retrieval retain करते हैं जिसमें pure linear attention weak है।

Pattern है **3 DeltaNet : 1 Attention**:

```
27B Dense:  16 × (3 × DeltaNet + 1 × Attention)  →  64 layers total
35B-A3B:    10 × (3 × DeltaNet + 1 × Attention)  →  40 layers total
            (every block ends with an MoE FFN)
```

Long context (32k+) के लिए, इसका मतलब है *attention* memory और compute curve dramatically flatten हो जाती है — आप सिर्फ़ 25% layers में quadratic cost pay करते हो जो full attention हैं।

### Qwen 3.6-27B Dense — Exact Config

| Field | Value |
|-------|-------|
| `vocab_size` | 248,320 (tensor parallelism के लिए padded) |
| `hidden_size` (D) | 5,120 |
| `num_hidden_layers` (L) | 64 — 48 DeltaNet + 16 Gated Attention |
| `num_attention_heads` (Q) | 24 (attention layers में) |
| `num_key_value_heads` (KV) | 4 (GQA group 6) |
| `head_dim` (attention) | 256 |
| `head_dim` (DeltaNet) | 128, 48 V heads, 16 QK heads |
| `intermediate_size` (F) | 17,408 |
| `rotary_pct` (RoPE dim per head) | 64 / 256 |
| `max_position_embeddings` | 262,144 native |
| `rope_theta` | 10,000,000 (very high — long context के लिए built) |
| Vision encoder | yes |
| Tensor type | bf16 |

Per forward active params = full 27B। Quality और speculative decoding दोनों के लिए **MTP** (multi-step token prediction) के साथ trained।

### Qwen 3.6-35B-A3B MoE — Exact Config

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
| `expert_intermediate_size` | 512 (हर expert small है — DeepSeek-V3 style fine-grained MoE) |
| `max_position_embeddings` | 262,144 native |
| `rope_theta` | 10,000,000 |
| Vision encoder | yes |
| Tensor type | bf16 |

Active params per token ≈ **3 B**: 9 small experts × 512 × 2 + DeltaNet/Attention compute। Total memory footprint 35 B है, लेकिन हर token का compute 3 B dense model match करता है। Fast inference, large knowledge।

### Multi-Token Prediction (MTP)

दोनों models एक extra small head train करते हैं जो सिर्फ़ token `t+1` नहीं बल्कि `t+2` (और sometimes `t+3`) भी predict करता है। Inference time पर, ये auxiliary head **speculative decoding के लिए draft tokens** produce करता है near-zero cost पर — vLLM और SGLang इसे ~1.8-2× decode speedups के लिए exploit करते हैं no quality loss के साथ।

### "Thinking Preservation"

Reasoning models often final answer से पहले `<think>...</think>` chain-of-thought के hundreds of tokens emit करते हैं। Multi-turn chat में, उस सारे thinking को context में naively keep करना prompt blow up करता है और later turns को confuse करता है। Qwen 3.6 **Thinking Preservation** introduce करता है: per-turn thinking को compactly stored किया जा सकता है और सिर्फ़ तब re-injected किया जा सकता है जब model judge करे कि useful है, instead of verbatim concatenated होने के। Chat template में `"preserve_thinking": True` से toggle करो। `enable_thinking=True/False` के साथ combined, आप reasoning behavior पर fine control पाते हो।

### Long Context

दोनों models 262k context पर natively trained हैं। Combination:

- **rope_theta = 10⁷** (very long-period RoPE),
- **3:1 DeltaNet:Attention** (long T पर memory और compute friendly),
- **YaRN scaling** config में built,

inference पर **~1M tokens** तक extends एक small fine-tuning anneal के साथ, और Qwen team एक `Qwen3.6-1M` variant ship करती है जिसने वो anneal आपके लिए कर दिया है।

### Headline Benchmarks

Qwen 3.6-27B Dense ने सबको surprise किया **Qwen 3.5-397B-A17B** (previous flagship MoE) को agentic coding पर beat करके:

| Benchmark | Qwen 3.6-27B | Qwen 3.6-35B-A3B | Qwen 3.5-397B-A17B (prev) | Claude 4.5 Opus |
|-----------|--------------|------------------|----------------------------|------------------|
| SWE-bench Verified | **77.2** | (lower) | 76.2 | 80.9 |
| LiveCodeBench v6 | 83.9 | — | — | — |
| SWE-bench Pro | 53.5 | — | — | — |
| Terminal-Bench 2.0 | 59.3 | — | — | — |
| MMLU-Pro | 86.2 | — | — | — |

Story: एक **27B dense** Apache-licensed open model, single 80 GB GPU पर full bf16 में runnable, अब agentic coding पर frontier closed models के striking distance के अंदर है। ये 2026 का genuinely surprising result है।

### Implementation Notes

- HuggingFace `transformers` ≥ 4.55 में custom Gated DeltaNet kernels के साथ Qwen 3.6 model class है।
- vLLM ≥ 0.7.x natively hybrid stack support करता है (vLLM team ने Gated DeltaNet के लिए Triton kernel co-designed किया)।
- DeltaNet kernel standard attention से लिखने में harder है; अगर आप अपना खुद का roll कर रहे हो, reference implementations के लिए `flash-linear-attention` library use करो।
- MoE-A3B के लिए, vLLM में `--enable-expert-parallel` set करो ताकि 256 experts multiple GPUs के across spread हों; otherwise 1 GPU पर memory blow करेगा।

### कौन सा Qwen 3.6 कब Use करें

- **27B Dense** — agentic coding, IDE assistants, anywhere quality > throughput। SWE-bench पर current open SOTA।
- **35B-A3B MoE** — high-throughput chat, RAG, agents जिन्हें speed चाहिए; 27B के साथ same architecture और tokenizer लेकिन 3B model का inference compute।
- **1M variant** — repository-scale code analysis, long document reasoning।

---

## 4. New Architecture Cheat Sheet

Chapter 10 / 11 template (Llama-3 / Qwen 2.5 style) से compare करते हुए, 2026 में क्या change हुआ?

| Component | Llama 3 / Qwen 2.5 | Gemma 4 | Qwen 3.6 |
|-----------|---------------------|---------|----------|
| Attention | full GQA | sliding + global GQA, alternating | 3 × Gated DeltaNet : 1 × Gated Attention |
| KV cache | full | shared-KV last N layers | tiny — सिर्फ़ attention layers पर |
| Position | RoPE | dual-RoPE (std + pruned) | RoPE θ=10⁷ |
| FFN | SwiGLU dense | SwiGLU dense (या A4B पर MoE) | dense या fine-grained MoE 256-expert |
| Norm | RMSNorm | RMSNorm + per-layer-emb | RMSNorm |
| Extras | — | Per-Layer Embeddings, vision, audio | MTP head, Thinking-Preservation, vision |
| Native ctx | 8-32k | 128-256k | 262k native, 1M extensible |

अगर आप हर एक से सिर्फ़ एक architectural idea सीखो: **Gemma 4 → Per-Layer Embeddings**, **Qwen 3.6 → Hybrid linear/full attention**।

दोनों standard transformer template को थोड़ा dated बनाते हैं। 2026 / 2027 के बाद वाले models से expect करो उन्हें combine करना: PLE + DeltaNet + MoE + MLA। Pieces already lying around हैं।

---

## 5. इन Models को Fine-tuning

Good news: chapter 14 SFT / DPO / GRPO recipes as-is काम करते हैं, two caveats के साथ।

1. **Chat template का mind रखो।** दोनों families thinking, tool calls, और roles के लिए distinct special tokens use करते हैं। हमेशा `tokenizer.apply_chat_template(...)` के through जाओ। Manual concatenation model break करेगा।
2. **Linear-attention kernels का mind रखो।** Long sequence length पर Gated DeltaNet के through backward कुछ implementations में fragile है। 8k-16k context पर पहले train करो, long context पर anneal करो, और gradient norms verify करो।

QLoRA (chapter 12) दोनों पर काम करता है। Gemma 4 PLE के लिए, LoRA training के दौरान per-layer embedding tables freeze करो — वो already well-fit हैं, और unfreezing often hurts।

---

## 6. 2026 में Model Pick करने का Decision Tree

```
Vision + audio + on-device चाहिए?
   → Gemma 4 E2B / E4B

Top-tier coding agent चाहिए?
   → Qwen 3.6-27B Dense

Cheap high-throughput chat चाहिए?
   → Qwen 3.6-35B-A3B MoE  या  Gemma 4 26B-A4B MoE

50B के नीचे best reasoning / math चाहिए?
   → Gemma 4 31B Dense  (AIME 2026: 89.2)

1M-context analyst चाहिए?
   → Qwen 3.6-1M

1 GPU पर cheaply fine-tune करना चाहते हो?
   → Gemma 4 E4B या Qwen 3.6-7B (अगर release हो) QLoRA के साथ
```

---

## 7. और गहराई से

- **HuggingFace blog: Welcome Gemma 4** — multimodal demos और code snippets के साथ official launch post।
- **Qwen3.6-27B blog** at qwen.ai — architecture, benchmarks, उनके लिए dense MoE को क्यों beat किया।
- **HuggingFace `Qwen/Qwen3.6-35B-A3B`** model card — config.json, MoE specifics।
- **HuggingFace `Qwen/Qwen3.6-27B`** model card — config.json, hybrid layout।
- **DeepSeek-V3 paper** — Qwen 3.6-MoE जो inherit करता है उस fine-grained MoE + shared-expert pattern का origin।
- **Gated DeltaNet paper** (Yang et al. 2024) और **`flash-linear-attention`** repo — linear attention math + kernels।
- **Multi-Token Prediction (Gloeckle et al. 2024)** — DeepSeek-V3 और Qwen 3.6 दोनों इस technique को use करते हैं।

---

## Sources

- [Welcome Gemma 4 — HuggingFace blog](https://huggingface.co/blog/gemma4)
- [Qwen3.6-35B-A3B — HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-27B — HuggingFace](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6-27B blog — qwen.ai](https://qwen.ai/blog?id=qwen3.6-27b)
- [Alibaba Qwen Team Releases Qwen3.6-27B — MarkTechPost (2026-04-22)](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/)
- [Qwen 3.6 Complete Guide — InsiderLLM](https://insiderllm.com/guides/qwen-3-6-local-ai-guide/)

Next: **[17-production-inference.md](./17-production-inference.md)** — इन में से किसी को 10,000 customers तक actually कैसे serve करें।
