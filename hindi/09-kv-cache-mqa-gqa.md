# 09 · KV Cache, MQA, GQA — और Inference Memory-bound क्यों है

> **TL;DR** जब आप token-by-token text generate करते हो, सारे past tokens के keys और values कभी change नहीं होते। उन्हें cache करना — **KV cache** — generation को per step O(T²) से per step O(T) में बदल देता है। Cache खुद long context पर memory में *सबसे बड़ा* tensor बन जाता है। **MQA, GQA, MLA, paged attention, और KV-cache quantization** सब उस एक number को attack करने के लिए exist करते हैं।

## 1. Cache के बिना Generation क्यों Slow है

एक naive generation loop:

```
for t in range(T):
    logits = model(tokens[:t+1])    # पूरे prefix पर full forward pass
    next_tok = sample(logits[-1])
```

हर step सारे `t+1` tokens पर attention recompute करता है। Total work = `O(T²)` (या `O(T³)` अगर आप attention के अंदर matmul count करो — लेकिन simple रखें)।

लेकिन notice करो: step `t+1` पर positions `0..t` पर K और V tensors *same* हैं जैसे step `t` पर थे। तो उन्हें cache करना देता है:

```
for t in range(T):
    logits = model(tokens[t:t+1], past_kv=cache)
    cache = updated cache
    next_tok = sample(logits[-1])
```

अब हर step `O(t)` work है — हम सिर्फ़ नई query को सारी cached keys के against attend करते हैं। `O(T³)` के बजाय Total `O(T²)`। **ये है KV cache, और हर modern inference engine इसे use करता है।**

---

## 2. Cache का Shape

Per layer, per head:

```
K_cache: (B, H_kv, T, head_dim)
V_cache: (B, H_kv, T, head_dim)
```

Per token total memory (सारे layers के across, K और V दोनों):

```
bytes_per_token = 2 (K, V) × L × H_kv × head_dim × bytes_per_element
```

**Llama-3-8B** के लिए (32 layers, GQA के साथ H_kv = 8, head_dim = 128, fp16):

```
2 × 32 × 8 × 128 × 2 = 131 072 bytes ≈ 128 KB / token
```

32k-context conversation के लिए: 128 KB × 32 000 = **सिर्फ़ cache के लिए 4.2 GB**।
**8 GB weights** add करो और आप ~12 GB पर हो — 24 GB GPU पर fine।

**Llama-3-70B** के लिए (80 layers, H_kv = 8, head_dim = 128, fp16): ~330 KB/token, 32k context पर ~10 GB। Plus 140 GB weights। अब आपको MQA या quantization या model sharding चाहिए।

**Mistral-7B** के लिए (originally MHA H_kv = 32 के साथ): ~512 KB/token। Cache cost चौगुनी। बिल्कुल इसलिए Mistral बाद के versions में **GQA** पर switched।

KV cache inference पर single largest variable cost है। इसे cut करो और सब कुछ cheaper हो जाता है — more concurrent users, longer contexts, smaller GPUs।

---

## 3. MHA → MQA → GQA

| Variant | # K/V heads | KV cache size | Quality |
|---------|-------------|---------------|---------|
| MHA (Vaswani 2017) | `H_kv = H_q` | full | best |
| MQA (Shazeer 2019) | `H_kv = 1` | `1/H_q` | small drop |
| GQA (Ainslie 2023) | `H_q / H_kv` के groups द्वारा share `H_kv` | `H_kv / H_q` | almost no drop |
| MLA (DeepSeek 2024) | latent dim ~ 512 | अभी ~10× smaller | given budget के लिए best |

**GQA 2026 का default है small/medium open models के लिए।** Llama 2-70B, Llama 3, Mistral 7B v0.2+, Qwen 2.5 — सब GQA use करते हैं, typically `H_q = 32, H_kv = 8` (group size 4) या `H_q = 28, H_kv = 4` (Qwen) के साथ।

Code में:

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

`enable_gqa=True` (PyTorch 2.5+) Flash Attention को खुद `H_kv` keys को `H_q` queries पर kernel के अंदर broadcast करता है, no `repeat_interleave` needed। Memory और compute बचाता है।

---

## 4. Multi-head Latent Attention (MLA), briefly

DeepSeek-V2/V3 ने और आगे जाया। MLA K और V को एक small **latent vector** में project करता है (e.g., `D = 7168` के लिए `d_c = 512`), सिर्फ़ उसको cache करता है, और attention के दौरान दो more linear layers से full K/V को on the fly reconstruct करता है।

```
hidden  → c_kv (B, T, d_c)         # tiny shared latent — यही हम cache करते हैं
        → k = W_uk(c_kv)  → (B, H_kv, T, head_dim)
        → v = W_uv(c_kv)  → (B, H_kv, T, head_dim)
```

KV cache memory per token `~d_c × L × bytes` तक drop करती है (no `H` या `head_dim` factor) — typically GQA से **~10× smaller**।

Implementation gotchas: RoPE MLA के down-projection के साथ nicely compose नहीं होता, इसलिए DeepSeek **decoupled-RoPE** trick use करता है (small extra rotation-only K)। अगर आप MLA use करना चाहो तो V2 paper पढ़ने worth है। 2026 के लिए, MLA open-source frameworks (vLLM, SGLang दोनों इसे support करते हैं) में अपनी जगह बना रहा है।

अपने khud के small model के लिए, **GQA simpler है और small scales पर per byte almost as good**।

---

## 5. Inference के Two Phases

LLM inference के दो distinct phases हैं very different cost profiles के साथ:

### Prefill (Prompt Processing)

आप entire prompt को एक साथ process करते हो: `(B, T_prompt, D)`। Compute एक big matmul है → **compute-bound**, Tensor Cores happy। Per token cheap।

### Decode (Generation)

आप एक नया token at a time process करते हो: `(B, 1, D)`। Matmuls small हैं लेकिन आप अभी भी सारे model weights और entire KV cache HBM से load करते हो। **Memory-bound**, per token expensive।

H100 पर Llama-3-8B fp16 के लिए numbers:

- Prefill: ~50 000 tokens/sec।
- Decode: ~120 tokens/sec/user (single batch)।

Decode per token ~400× slower है। इसलिए frameworks **batch** decode पर bend over backwards करते हैं (more users एक weight load share करते हैं) और KV cache shrink करने के लिए (load करने के लिए कम)।

---

## 6. Continuous / Dynamic Batching

Older inference servers **static batching** use करते थे: B requests को साथ launch करो, उनके सबको finish होने का wait करो, फिर next batch start करो। Wasteful — short requests long ones के wait करते बैठे रहते हैं।

**Continuous batching** (Yu et al. 2022, vLLM 2023 ने popularized) हर step पर running batch में नए requests insert करता है:

- Varying request lengths के तहत भी GPU utilization 80%+ रखता है।
- Static batching vs chat workloads पर 2-10× higher throughput।

आप ये खुद implement नहीं करते — आप vLLM, SGLang, TGI, या TensorRT-LLM use करते हो।

---

## 7. PagedAttention (vLLM के पीछे का Trick)

vLLM (Kwon et al. 2023) KV cache को virtual memory की तरह treat करता है: इसे fixed-size **blocks** में split करता है (e.g., 16 tokens each) और हर sequence के लिए blocks point करने वाला metadata store करता है। Benefits:

- No per-sequence pre-allocation (variable lengths efficiently handle करता है)।
- Same prompt prefix वाले requests के across **prefix** caches share करना easy (system prompts!)।
- Memory waste ~60% (static padding) से ~4% तक drop करती है।

ये अब open-source inference के लिए production default है। SGLang का **RadixAttention** generalize करता है requests के across किसी भी common prefix share करने के लिए, सिर्फ़ system prompt नहीं।

---

## 8. KV-cache Quantization

KV cache by default fp16/bf16 में रहता है। आप इसे **int8** या **int4** में store कर सकते हो:

- int8: ~free quality, 2× memory savings।
- int4: small quality drop (benchmarks पर few %), 4× savings।

vLLM `kv_cache_dtype = "fp8"` (Hopper+) और "int8" आज support करता है। SGLang में experimental int4 है। Very long context (1M+ tokens) के लिए, ये essential है।

```bash
# vLLM example
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype fp8
```

आप on-the-fly quantize करते हो: per-token (या per-block) scales write time पर chosen। Decode int8 K/V read करता है, attention kernel के अंदर dequantize करता है।

---

## 9. Speculative Decoding

Decode memory-bound है; GPU mostly idle रहता है KV-cache reads के wait करते हुए। **Speculative decoding** एक small "draft" model use करता है ~4 tokens propose करने के लिए, फिर big model उन्हें single batched forward pass में verify करता है:

- अगर draft सही था, आपको 1 big-model forward के cost पर 4 tokens मिले।
- अगर ग़लत, आप suffix discard करो और restart करो — minor cost।

Typical speedup: chat workloads के लिए **2-3× wall-clock**, no quality loss के साथ (big model अभी भी हर emitted token choose करता है)। Implementations: vLLM `--speculative-config`, SGLang Eagle, TGI Medusa।

A 2026 variant: **EAGLE-2 / EAGLE-3** big model के hidden states पर trained एक small auxiliary head use करता है; near-zero overhead, ~3× speedup।

---

## 10. Long-context Tricks

1M+ tokens के लिए आप stack करते हो:

- **GQA / MLA** — small KV cache।
- **YaRN** RoPE scaling (chapter 7)।
- **KV quantization** (fp8/int4)।
- कुछ layers में **Sliding window attention** (Mistral, Gemma)।
- **Attention sinks** — हमेशा cache में tokens 0-3 रखो (StreamingLLM)।
- **Prefix caching / RadixAttention** — repeated system prompts के लिए KV share करो।
- **Dynamic compression** — low-importance KV entries drop करो (`H2O`, `SnapKV`, `KIVI`)।

A 7B model का 1M context पर single-GPU inference अब feasible है (Qwen 2.5-1M, Gemma 3) ऊपर stack जैसे का use करते हुए।

---

## 11. Scratch से एक KV Cache Implement करना

यहां एक minimal `generate` एक manual cache के साथ:

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

Model को cache accept और update करना चाहिए:

```python
class TransformerLayer(nn.Module):
    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(x, past_kv=past_kv)
        ...
        return out, new_kv
```

Batched generation के लिए, **left-padding** (different lengths वाले prompts) को watch out करो या vLLM-style continuous batching use करो।

---

## 12. 2026 में Inference Servers

आप production में almost कभी raw PyTorch serve नहीं करते। एक pick करो:

| Server | Strengths | Quirks |
|--------|-----------|--------|
| **vLLM** | Best general throughput, PagedAttention, huge community | OpenAI-compatible API; Python heavy |
| **SGLang** | RadixAttention prefix cache, fast structured-output | Newer, fewer corners covered |
| **TensorRT-LLM** | NVIDIA पर lowest latency, production के लिए best | NVIDIA-only, build step painful है |
| **TGI (HuggingFace)** | HF stack के लिए drop-in | vLLM से slower |
| **llama.cpp** | CPU + Apple Silicon + tiny GPUs | Quantized only |
| **MLX** | Apple Silicon native | Mac-only |

Self-hosted GPU inference के लिए, **vLLM 2026 में safe default है**। Laptops के लिए, **llama.cpp** (या Ollama, जो इसे wrap करता है) या **MLX**।

```bash
# तीन lines में vLLM
pip install vllm
vllm serve Qwen/Qwen3-4B-Instruct --gpu-memory-utilization 0.9 --max-model-len 32768
# अब http://localhost:8000/v1/chat/completions hit करो
```

---

## 13. 2026 Cheat Sheet

- **हमेशा KV cache use करो।** उसके बिना, decode unusable है।
- अपने model में **GQA use करो** (group size 4 canonical है)।
- **`enable_gqa=True`** के साथ Flash Attention use करो।
- Very long context के लिए **KV cache quantize करो** (`fp8` cheap, `int4` aggressive)।
- **Production के लिए, vLLM (या SGLang) use करो।** Apna server मत बनाओ।
- ~2-3× chat speedup के लिए **Speculative decoding**।
- जब आपके पास repeated system prompts हों तो **PagedAttention + prefix caching**।
- अगर आप memory-quality frontier chase कर रहे हो तो **MLA**।

---

## और गहराई से

- Shazeer 2019 — original MQA paper।
- Ainslie et al. 2023 — GQA paper।
- Kwon et al. 2023 — vLLM / PagedAttention paper।
- DeepSeek-V2 paper — MLA का best derivation।
- vLLM source — surprisingly readable, especially `vllm/core/scheduler.py` और `attention/`।

Next: **[10-building-blocks.md](./10-building-blocks.md)** — RMSNorm, SwiGLU, residuals।
