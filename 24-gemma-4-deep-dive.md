# 24 · Gemma 4 Deep Dive — Complete Reference with Real Examples

> **TL;DR** Gemma 4 is Google DeepMind's 2026 open multimodal family — Apache 2.0, four sizes (E2B, E4B, 26B-A4B MoE, 31B Dense), 128K-256K context, native vision (all sizes) and native audio (E2B/E4B only). Architecturally it introduces three key ideas: **Per-Layer Embeddings (PLE)** for memory efficiency, **hybrid local-sliding + global attention** with the final layer always global, and **Proportional RoPE (p-RoPE)** with **unified K/V** on global layers for memory-friendly long context. Sampling defaults: `temperature=1.0, top_p=0.95, top_k=64`. Image budgets are configurable: 70 / 140 / 280 / 560 / 1120 tokens. This chapter covers everything end-to-end with runnable code, dtype guidance, and real tokens-per-second benchmarks.

---

## 1. Vocabulary — every technical term defined

Read this section first if any of the words below are fuzzy. They're the language the rest of the chapter uses.

| Term | Plain definition |
|------|------------------|
| **Token** | One unit of text (or image, or audio) that the model processes. ~4 chars per token in English. |
| **Vocab / Vocabulary size** | Total distinct tokens the tokenizer knows. Gemma 4: **262,144** (262K). |
| **Context length** | Max tokens the model can read at once. E2B/E4B: **128K**; 26B-A4B & 31B: **256K**. |
| **Parameters (params)** | The trainable numbers inside the model. 31B Dense has **30.7B** of them. |
| **Effective parameters** | For E-tier models: how many params are actually used per token (excludes the per-layer embedding tables that are big but only do lookups). E2B = 2.3B effective even though file is 5.1B. |
| **Active parameters** | For MoE: how many params actually compute per token (a routed subset). 26B-A4B has 25.2B total / **3.8B active**. |
| **Layers** | Number of stacked transformer blocks. Gemma 4: 35 / 42 / 30 / 60 (E2B / E4B / 26B-A4B / 31B). |
| **Attention** | The mechanism that lets each token look at other tokens. |
| **Sliding window** | A "local" attention pattern where each token only sees the previous N tokens. Gemma 4: 512 (small) or 1024 (big). |
| **Global attention** | Full attention — each token sees every prior token. Gemma 4 keeps these for some layers; the **final layer is always global**. |
| **Hybrid attention** | Mix of local + global layers. Gemma 4's design. |
| **RoPE** | Rotary Position Embedding — how the model knows where each token sits. |
| **p-RoPE (Proportional RoPE)** | Gemma 4's variant for global layers, friendlier for long context. |
| **PLE (Per-Layer Embeddings)** | Each decoder layer gets its own small embedding table per token. Used in E2B/E4B for parameter efficiency. |
| **MoE (Mixture of Experts)** | A model with many "expert" sub-FFNs; only a few are used per token. 26B-A4B uses 8 active out of 128 experts + 1 shared. |
| **Vision encoder** | The part that turns images into tokens. ~150M params for E-tier, ~550M for 31B / 26B-A4B. |
| **Audio encoder** | Turns audio frames into tokens. ~300M params on E-tier only. |
| **Multimodal** | Accepts more than one input type. Gemma 4 supports text + image + (audio on E-tier) + video (frame-sampled). |
| **dtype** | The numeric format weights/activations are stored in. Common: `float32`, `bfloat16`, `float16`, `float8`, `int8`, `int4`. |
| **bf16** | "Brain Float 16" — 16-bit float with the same exponent range as fp32. **The training and high-quality inference default in 2026.** |
| **fp8** | 8-bit float (E4M3 or E5M2). H100/B200-only. ~2× faster than bf16 with tiny quality drop. |
| **int4 / Q4** | 4-bit weight quantization. ~4× memory cut. Used for laptop / edge. |
| **TTFT** | Time-to-First-Token. How long after sending a prompt before the first reply token appears. |
| **TPOT** | Time-Per-Output-Token. The streaming interval. 1 / TPOT ≈ tokens-per-second. |
| **Tokens/sec (tok/s)** | The speed the model emits tokens. The headline UX metric. |
| **KV cache** | The stored Keys and Values from previous tokens during generation, so we don't recompute. The biggest memory cost at long context. |
| **Unified K/V** | Gemma 4's trick: in global layers, K and V share a single tensor — half the cache memory. |
| **Thinking mode** | Reasoning step. Triggered by inserting `<\|think\|>` in the system prompt. |
| **Image token budget** | How many tokens an image is "worth". Gemma 4: 70, 140, 280, 560, or 1120. Higher = better detail, more cost. |
| **Function calling** | Native tool use — the model emits structured JSON to invoke a defined function. Native in Gemma 4. |
| **System prompt** | A special pre-message defining the assistant's role/rules. **Native in Gemma 4** (was workaround-only in Gemma 3). |
| **Apache 2.0** | The license. Permissive — commercial use OK, fine-tuning and redistribution OK. |

If anything below uses a term that's not on this list, that's a documentation bug. File it.

---

## 2. The family at a glance

Pulled directly from the official Hugging Face model card.

### Dense models

| Property | E2B | E4B | 31B Dense |
|----------|-----|-----|-----------|
| Total params | 5.1B (with embeds) | 8B (with embeds) | 30.7B |
| Effective params | 2.3B | 4.5B | 30.7B (full) |
| Layers | 35 | 42 | 60 |
| Sliding window | 512 | 512 | 1024 |
| Context length | 128K | 128K | 256K |
| Vocab | 262K | 262K | 262K |
| Modalities | text + image + audio | text + image + audio | text + image |
| Vision encoder | ~150M | ~150M | ~550M |
| Audio encoder | ~300M | ~300M | — |

### MoE model

| Property | 26B-A4B |
|----------|---------|
| Total params | 25.2B |
| Active params/token | 3.8B |
| Layers | 30 |
| Experts | 8 active / 128 routed + 1 shared |
| Sliding window | 1024 |
| Context length | 256K |
| Vocab | 262K |
| Modalities | text + image |
| Vision encoder | ~550M |

The "E" in E2B/E4B = "effective" (PLE makes the file bigger than the active params). The "A" in A4B = "active" (only 3.8B of the 25.2B compute per token).

---

## 3. Architecture in plain English

Gemma 4 uses the standard pre-norm transformer template (chapter 10) with three notable upgrades.

### 3.1 Hybrid attention (local + global)

```
Layer  1:  local sliding-window  (W = 512 or 1024)
Layer  2:  local sliding-window
Layer  3:  global full-context
Layer  4:  local sliding-window
…
Layer  N:  global   ← final layer is ALWAYS global
```

Most layers attend only within a window (cheap, local). Some are global (full context, expensive but globally aware). The **final layer is forced global** so the model's output is always informed by the full context, not just a window.

Why this works:
- Local layers handle word-by-word grammar, immediate dependencies — most of the work.
- Global layers handle long-range structure (cross-paragraph connections, document-level facts).
- Mixing them gives the speed and memory of a small-window model with the reach of a full-attention one.

This pattern started in Gemma 2 and matured in Gemma 4.

### 3.2 Per-Layer Embeddings (PLE) — only on E2B / E4B

Imagine a normal transformer: one big embedding table, looked up once per token, fed into the residual stream. The vector then has to remember "what token am I" for all 35-42 layers.

Gemma's PLE adds **a separate small embedding table per layer**. Each decoder layer gets its own quick-lookup hint about "what token is this," fresh, every layer. The tables are big (that's why E2B's *file* is 5.1B even though active math is 2.3B) but they're **lookups, not matmuls** — almost free at runtime.

Net effect: better parameter efficiency for the same compute. E2B punches like a 4B-class model on benchmarks. Phones and laptops can stream the small layer-by-layer hint as they go.

### 3.3 Global layers — Proportional RoPE + Unified K/V

In the global layers (the long-reach ones), Gemma 4 does two memory-savers:

- **Unified K and V**: instead of storing keys and values as two separate tensors per token in the KV cache, they share a single tensor. **Half the memory** for global-layer cache.
- **Proportional RoPE (p-RoPE)**: a re-scaled RoPE designed for the long ranges these layers need to cover. Cleaner long-context extension than vanilla RoPE.

This is what lets the 31B run 256K context without wrecking the GPU's HBM. Local layers don't need it because the window is small.

---

## 4. Hands-on — load and run

```bash
pip install -U transformers torch accelerate
```

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, time

MODEL_ID = "google/gemma-4-31B-it"

# 1. Load
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",          # picks bfloat16 on a modern GPU
    device_map="auto",     # spreads across visible GPUs
)
print(f"loaded on: {model.device}, dtype: {model.dtype}")

# 2. Prompt
messages = [
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "Write a short joke about saving RAM."},
]

# 3. Apply chat template (handles system role, tags, etc.)
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,        # turn ON for reasoning workloads
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# 4. Generate (with timing)
t0 = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True,
)
elapsed = time.time() - t0
gen_len = outputs.shape[-1] - input_len

# 5. Decode and parse
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))

print(f"\n[generated {gen_len} tokens in {elapsed:.2f}s = "
      f"{gen_len/elapsed:.1f} tok/s]")
```

The `dtype="auto"` picks **bfloat16** on Ampere+ GPUs (A100, H100, RTX 30/40/50xx) and **float16** on older hardware. `device_map="auto"` shards the model if it doesn't fit on one GPU.

Sample output:

```
Why did the developer keep cutting his model in half?
Because every byte saved is RAM well-earned.

[generated 24 tokens in 0.31s = 77.4 tok/s]
```

---

## 5. dtype guide — what to use when

For Gemma 4 specifically. Numbers are weight memory only; activations/KV cache add ~20-30%.

| Variant | dtype | Memory (weights) | Where it runs | Notes |
|---------|-------|------------------|---------------|-------|
| **E2B** | bf16 | ~5 GB | phone, laptop CPU/GPU | default for edge |
| **E2B** | int4 (Q4_K_M) | ~1.5 GB | phone | Ollama / llama.cpp / MLX |
| **E4B** | bf16 | ~8 GB | RTX 4060+, M-series Mac | default |
| **E4B** | int4 | ~3 GB | most laptops | comfortable |
| **26B-A4B** | bf16 | ~50 GB | 1× H100 80 GB | active 3.8B → fast inference |
| **26B-A4B** | fp8 | ~26 GB | 1× H100, 1× L40S | best throughput-per-$ |
| **26B-A4B** | int4 | ~14 GB | RTX 4090 24 GB | yes, fits! |
| **31B Dense** | bf16 | ~62 GB | 1× H100 80 GB | quality default |
| **31B Dense** | fp8 | ~31 GB | 1× H100 / 2× RTX 4090 | server default in 2026 |
| **31B Dense** | int4 | ~16 GB | RTX 4090 / Mac M3 Max | laptop-class quality |

Quick rules:

- **Training / fine-tuning**: bf16 always. Quality > memory.
- **Cloud serving**: fp8 (H100/B200) or bf16. Pick fp8 for throughput.
- **Single-user inference on a workstation**: bf16 if it fits, else int4 GGUF.
- **Phone / embedded**: int4 Q4_K_M or Q4_0. Use `llama.cpp`, `Ollama`, or `MLX`.

Force a specific dtype:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             dtype=torch.bfloat16,
                                             device_map="auto")
```

Verify after load:

```python
print(model.dtype)                                    # torch.bfloat16
print(next(model.parameters()).dtype)                 # torch.bfloat16
print(model.get_memory_footprint() / 1e9, "GB")
```

---

## 6. Tokens-per-second — measured throughput

These are realistic, single-stream (batch=1) numbers from public benchmarks and recent vLLM tests. Multi-user (batched) throughput is much higher; see chapter 17.

### E-tier (laptop / edge)

| Variant | Hardware | dtype | tok/s (single user) | Notes |
|---------|----------|-------|---------------------|-------|
| E2B | M3 Max (MLX) | int4 | 100-140 | very fast on Mac |
| E2B | RTX 4070 Ti | bf16 | 200-260 | |
| E2B | RTX 4090 | bf16 | 280-340 | |
| E2B | iPhone 16 Pro / Pixel 9 (CoreML/MediaPipe) | int4 | 30-60 | mobile |
| E4B | M3 Max | int4 | 50-80 | |
| E4B | RTX 4090 | bf16 | 130-180 | |

### Mid (workstation / single GPU)

| Variant | Hardware | dtype | tok/s (single user) | Notes |
|---------|----------|-------|---------------------|-------|
| 26B-A4B | RTX 4090 (24 GB) | int4 | 70-110 | active 3.8B → very fast |
| 26B-A4B | H100 80 GB | bf16 | 120-150 | |
| 26B-A4B | H100 80 GB | fp8 | 200-260 | |
| 31B Dense | RTX 4090 (24 GB) | int4 | 35-50 | tight fit |
| 31B Dense | H100 80 GB | bf16 | 50-70 | |
| 31B Dense | H100 80 GB | fp8 | 90-130 | |
| 31B Dense | B200 (192 GB) | fp8 | 180-230 | |

### Why the MoE feels faster than the Dense

26B-A4B has *more total params* than the 31B but is **2-3× faster** per user. The reason is in chapter 6 of this guide and chapter 9: each token only activates 3.8B params worth of compute, even though the full 25.2B is loaded into HBM. Memory bandwidth (chapter 17 §6) is the decode bottleneck — reading less per token = faster output.

### Time-to-first-token (TTFT) at 4K prompt

| Variant | Hardware | dtype | TTFT (4K prompt) |
|---------|----------|-------|------------------|
| E4B | RTX 4090 | bf16 | ~200 ms |
| 26B-A4B | H100 fp8 | fp8 | ~150 ms |
| 31B Dense | H100 fp8 | fp8 | ~280 ms |
| 31B Dense | B200 fp8 | fp8 | ~140 ms |

For interactive chat, TTFT < 500 ms feels instant. For ghost-text code completion, < 200 ms is the bar.

### A quick benchmark you can run yourself

```python
import time
prompt = "Tell me a long story about a robot that learns to cook." * 5
inputs = processor(text=prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Warmup
_ = model.generate(**inputs, max_new_tokens=10)

# Real run
t0 = time.time()
out = model.generate(**inputs,
                     max_new_tokens=300,
                     do_sample=True,
                     temperature=1.0, top_p=0.95, top_k=64)
elapsed = time.time() - t0
new_toks = out.shape[-1] - input_len
print(f"{new_toks} new tokens in {elapsed:.2f}s = {new_toks/elapsed:.1f} tok/s")
```

Run it 5 times and take the median (warmup matters, especially first launch).

---

## 7. Multimodal — images, video, audio

### Images

```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "chart.png"},
        {"type": "text",  "text":  "What does this chart show? Be specific."},
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
inputs = processor(text=text, images="chart.png", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(out[0], skip_special_tokens=True))
```

**Choose your image budget** for cost vs detail. Set `mm_tokens_per_image` (or processor kwarg, depends on version):

| Budget | Use for |
|--------|---------|
| **70** | classification, video frames, "what's in the photo?" |
| **140** | captioning, simple Q&A |
| **280** | default, mixed tasks |
| **560** | document layouts, charts |
| **1120** | OCR, fine print, dense documents |

### Audio (E2B / E4B only, ≤ 30 sec clips)

```python
# ASR (transcription)
asr_prompt = ("Transcribe the following speech segment in English into "
              "English text.\n\nFollow these specific instructions for "
              "formatting the answer:\n* Only output the transcription, "
              "with no newlines.\n* When transcribing numbers, write the "
              "digits, i.e. write 1.7 and not one point seven.")

messages = [{"role":"user","content":[
    {"type":"audio","audio":"clip.wav"},
    {"type":"text","text":asr_prompt},
]}]

# AST (translation)
ast_prompt = ("Transcribe the following speech segment in English, "
              "then translate it into Hindi.\nWhen formatting the answer, "
              "first output the transcription in English, then one newline, "
              "then output the string 'Hindi: ', then the translation in Hindi.")
```

### Video (frame-sampled, ≤ 60 sec at 1 fps)

```python
# transformers' processor handles video automatically when you pass a path
messages = [{"role":"user","content":[
    {"type":"video","video":"clip.mp4"},
    {"type":"text","text":"Summarize what happens in this clip."},
]}]
```

**Modality order matters** — put images / audio / video **before** the text in your prompt. Performance drops measurably if you put text first.

---

## 8. Thinking mode — for math, code, hard reasoning

Triggered by the `<|think|>` control token at the start of the system prompt.

```python
messages = [
    {"role":"system","content":"You are a careful math tutor."},
    {"role":"user","content":"If 7 cats catch 7 mice in 7 minutes, "
                              "how many cats catch 100 mice in 50 minutes?"},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,        # <-- this inserts <|think|>
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096,
                     temperature=1.0, top_p=0.95, top_k=64)
parsed = processor.parse_response(processor.decode(out[0], skip_special_tokens=False))
# parsed['thought']  -> the internal reasoning
# parsed['answer']   -> the user-facing reply
```

The model will emit:

```
<|channel|>thought
[internal reasoning, often 100-2000 tokens]
<|channel|>
[final answer]
```

`processor.parse_response` splits these for you.

**Multi-turn rule:** when continuing a conversation, **drop the `thought` blocks from prior turns** before sending the next user message. Keeping them in history degrades quality. Just keep the user-visible answers.

For E2B / E4B with thinking disabled, the model just generates the answer normally. For 26B-A4B / 31B with thinking disabled, it still emits empty `<|channel|>thought` tags — that's expected; just parse them out.

---

## 9. Function calling (native, no harness needed)

```python
tools = [
    {"name":"get_weather",
     "description":"Get the current weather in a city.",
     "parameters":{"type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]}},
]

messages = [
    {"role":"system","content":"You are a helpful assistant with tools."},
    {"role":"user","content":"What's the weather in Tokyo right now?"},
]

text = processor.apply_chat_template(
    messages, tools=tools,
    tokenize=False, add_generation_prompt=True,
)
# Generate, parse — model emits a structured tool_call.
# You execute the tool, append the result, call again.
```

Native tool support means you don't need a separate function-calling fine-tune. The 31B + 26B-A4B handle multi-step agentic tool use well; E-tier is acceptable for simple single-tool calls.

---

## 10. Real benchmarks (from the official model card)

Headline numbers across sizes. **bold = state-of-the-art for any open model at that size class** as of late 2026.

| Benchmark | E2B | E4B | 26B-A4B | 31B | (Gemma 3-27B) |
|-----------|-----|-----|---------|-----|---------------|
| MMLU Pro | 60.0% | 69.4% | 82.6% | **85.2%** | 67.6% |
| AIME 2026 (no tools) | 37.5% | 42.5% | 88.3% | **89.2%** | 20.8% |
| LiveCodeBench v6 | 44.0% | 52.0% | 77.1% | **80.0%** | 29.1% |
| Codeforces ELO | 633 | 940 | 1718 | **2150** | 110 |
| GPQA Diamond | 43.4% | 58.6% | 82.3% | **84.3%** | 42.4% |
| Tau2 (avg) | 24.5% | 42.2% | 68.2% | **76.9%** | 16.2% |
| HLE (no tools) | — | — | 8.7% | **19.5%** | — |
| HLE (with search) | — | — | 17.2% | **26.5%** | — |
| BigBench Extra Hard | 21.9% | 33.1% | 64.8% | **74.4%** | 19.3% |
| MMMLU (multilingual) | 67.4% | 76.6% | 86.3% | **88.4%** | 70.7% |
| **Vision** | | | | | |
| MMMU Pro | 44.2% | 52.6% | 73.8% | **76.9%** | 49.7% |
| MATH-Vision | 52.4% | 59.5% | 82.4% | **85.6%** | 46.0% |
| MedXPertQA MM | 23.5% | 28.7% | 58.1% | **61.3%** | — |
| **Audio** (E-tier only) | | | | | |
| CoVoST | 33.47 | **35.54** | — | — | — |
| FLEURS (lower better) | 0.09 | **0.08** | — | — | — |
| **Long Context** | | | | | |
| MRCR-v2 (8 needle, 128K avg) | 19.1% | 25.4% | 44.1% | **66.4%** | 13.5% |

The leap from Gemma 3-27B → Gemma 4-31B on AIME is from **20% to 89%** — that's a different model family in capability, not an incremental release. The thinking mode + better data are responsible.

---

## 11. Best practices (Google's official guidance)

### Sampling
- `temperature=1.0`, `top_p=0.95`, `top_k=64` — use these. They're tuned defaults across all sizes.

### System prompts
- Native role support means just use the `system` role normally.
- For thinking mode, prepend `<|think|>` automatically via `enable_thinking=True`.

### Multi-turn
- **Strip thought blocks from history** before the next turn.
- Chat templates handle other formatting; let the processor manage tokens.

### Multimodal
- Order: **image → audio → video → text**.
- Video: ≤ 60 seconds at 1 fps. Audio: ≤ 30 seconds.
- Pick the smallest image-token budget that gets the right answer.

### Tools
- Use the native `tools=` argument in `apply_chat_template`.
- For agentic tasks, 26B-A4B and 31B work; E-tier is hit-or-miss on multi-tool reasoning.

### Long context
- 31B / 26B-A4B handle 256K natively. Beyond 256K not supported.
- Memory at full 256K context is dominated by KV cache; use fp8 KV in vLLM for 50% reduction.

---

## 12. Production serving

```bash
# vLLM single-GPU 31B fp8
vllm serve google/gemma-4-31B-it \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=4 \
    --max-num-seqs 256

# 26B-A4B with expert parallelism on 4× H100
vllm serve google/gemma-4-26B-A4B-it \
    --quantization fp8 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len 65536
```

For laptop / Mac:

```bash
# Ollama (Mac, Linux, Windows)
ollama run gemma-4:31b-q4_K_M

# llama.cpp directly
./llama-cli -m gemma-4-31b-q4_K_M.gguf -n 200 -p "Hello"

# MLX (Mac)
mlx_lm.generate --model mlx-community/gemma-4-31B-it-4bit \
                --prompt "Hello" --max-tokens 200
```

For Apple Silicon, **MLX with 4-bit** is currently the fastest local option — 2-3× faster than Ollama for the same quantization on M3/M4.

---

## 13. Common gotchas

- **`enable_thinking=True` blows up output length.** Set `max_new_tokens=4096+` or you'll cut off the model mid-thought.
- **Modality order**: text-first is measurably worse. Always images / audio / video first.
- **Thought blocks in history** degrade multi-turn quality. Strip before next turn.
- **MLX vs llama.cpp 4-bit are not the same format.** Don't mix.
- **Image-token budget overrides**: depending on `transformers` version, the kwarg is `mm_tokens_per_image` or done via processor config; check the model card.
- **PLE memory**: when deploying E-tier on tight RAM, the disk file is bigger than the runtime memory — don't panic at the 5.1 GB E2B file.
- **Function calling on E-tier**: works, but is brittle for >2 tool steps. Use 26B-A4B+ for serious agents.
- **Gemma 3 prompts don't transfer cleanly.** Gemma 4 has native system role, different chat tokens. Re-test prompts after upgrading.

---

## 14. End-to-end real example

A complete script: load 31B, ask a multimodal coding question with thinking on, time it, save the result.

```python
"""
gemma4_demo.py — end-to-end Gemma 4 31B example.
Run: python gemma4_demo.py screenshot.png
"""
import sys, time, torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-31B-it"
img_path = sys.argv[1] if len(sys.argv) > 1 else None

# 1. Load
print(f"loading {MODEL_ID}…")
t = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
print(f"loaded in {time.time()-t:.1f}s, dtype={model.dtype}, "
      f"footprint={model.get_memory_footprint()/1e9:.1f} GB")

# 2. Build prompt
content = []
if img_path:
    content.append({"type": "image", "image": img_path})
content.append({"type": "text",
                "text": "If this is a UI screenshot, write the React JSX "
                        "that reproduces it. Otherwise describe and code "
                        "something inspired by it."})

messages = [
    {"role": "system", "content": "You are an expert front-end engineer."},
    {"role": "user",   "content": content},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,
)
inputs = processor(text=text,
                   images=[img_path] if img_path else None,
                   return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# 3. Generate (with TTFT timing)
print(f"prompt: {input_len} tokens. Generating…")
t0 = time.time()
out = model.generate(
    **inputs,
    max_new_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True,
)
elapsed = time.time() - t0
new_toks = out.shape[-1] - input_len

# 4. Parse and report
raw = processor.decode(out[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(raw)

print("\n=== THINKING ===\n", parsed.get("thought", "")[:800])
print("\n=== ANSWER ===\n",  parsed.get("answer",  "")[:2000])
print(f"\n[{new_toks} tokens in {elapsed:.1f}s "
      f"= {new_toks/elapsed:.1f} tok/s]")
```

Run it:

```bash
python gemma4_demo.py login_form.png
```

On an H100 fp8 you'll see ~80-120 tok/s, with the first ~1000-2000 tokens in the `thought` channel and the final JSX in `answer`.

---

## 15. Cheat sheet

- **Pick by size**: E2B (phone), E4B (laptop), 26B-A4B (cheap server), 31B (best quality).
- **MoE rule**: 26B-A4B is faster *per user* than 31B but needs more memory.
- **Sampling**: `temperature=1.0, top_p=0.95, top_k=64` always.
- **Image budget**: 70 / 140 / 280 / 560 / 1120 — start at 280, raise for OCR, lower for video.
- **Modality order**: image → audio → video → text.
- **Thinking**: `<|think|>` in system prompt for math/code; strip thought from history.
- **dtype**: bf16 quality, fp8 throughput, int4 edge.
- **Tokens/sec**: 31B fp8 on H100 ≈ 100; 26B-A4B fp8 ≈ 230; E4B bf16 4090 ≈ 150.
- **Context**: 128K (E-tier), 256K (26B-A4B / 31B). Don't go beyond.
- **Native everything**: system role, function calling, thinking, multimodal.
- **License**: Apache 2.0 — commercial use OK.

---

## 16. Going deeper

- **Official model card** at `huggingface.co/google/gemma-4-31B-it` — read once.
- **Gemma technical reports** (when published) — Google DeepMind blog.
- **Gemma launch blog** at `developers.googleblog.com` — context on design choices.
- **vLLM Gemma 4 integration** — production serving recipes.
- **MLX Gemma examples** at `github.com/ml-explore/mlx-examples` — Apple Silicon.
- **`llama.cpp` Gemma support** — CPU and edge inference.
- **Ollama library** at `ollama.com/library/gemma-4` — easiest way to run locally.

This deep-dive complements **[16-frontier-models-2026.md](./16-frontier-models-2026.md)** (which compares Gemma 4 against Qwen 3.6 and other frontier families). For serving Gemma 4 in production, see **[17-production-inference.md](./17-production-inference.md)**. For evaluation, **[19-evaluation.md](./19-evaluation.md)**. For fine-tuning, **[20-fine-tuning-recipes.md](./20-fine-tuning-recipes.md)** — every recipe there works on Gemma 4.

End of curriculum.
