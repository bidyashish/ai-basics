# 24 · Gemma 4 Deep Dive — Real Examples के साथ Complete Reference

> **TL;DR** Gemma 4 Google DeepMind का 2026 open multimodal family है — Apache 2.0, चार sizes (E2B, E4B, 26B-A4B MoE, 31B Dense), 128K-256K context, native vision (सब sizes) और native audio (सिर्फ़ E2B/E4B)। Architecturally ये तीन key ideas introduce करता है: memory efficiency के लिए **Per-Layer Embeddings (PLE)**, **hybrid local-sliding + global attention** with final layer हमेशा global, और memory-friendly long context के लिए **Proportional RoPE (p-RoPE)** with **unified K/V** on global layers। Sampling defaults: `temperature=1.0, top_p=0.95, top_k=64`। Image budgets configurable हैं: 70 / 140 / 280 / 560 / 1120 tokens। ये chapter end-to-end सब cover करता है runnable code, dtype guidance, और real tokens-per-second benchmarks के साथ।

---

## 1. Vocabulary — हर technical term defined

ये section पहले पढ़ो अगर नीचे के words fuzzy हैं। बाकी chapter इसी language को use करता है।

| Term | Plain definition |
|------|------------------|
| **Token** | Text (या image, या audio) की एक unit जो model process करता है। English में ~4 chars per token। |
| **Vocab / Vocabulary size** | Tokenizer कुल कितने distinct tokens जानता है। Gemma 4: **262,144** (262K)। |
| **Context length** | Max tokens जो model एक बार में पढ़ सकता है। E2B/E4B: **128K**; 26B-A4B & 31B: **256K**। |
| **Parameters (params)** | Model के अंदर trainable numbers। 31B Dense में **30.7B** हैं। |
| **Effective parameters** | E-tier models के लिए: per token actually कितने params use होते हैं (PLE की big embedding tables exclude जो सिर्फ़ lookups करती हैं)। E2B = 2.3B effective जबकि file 5.1B है। |
| **Active parameters** | MoE के लिए: per token actually कितने params compute करते हैं (एक routed subset)। 26B-A4B: 25.2B total / **3.8B active**। |
| **Layers** | Stacked transformer blocks की number। Gemma 4: 35 / 42 / 30 / 60 (E2B / E4B / 26B-A4B / 31B)। |
| **Attention** | वो mechanism जो हर token को बाकी tokens देखने देता है। |
| **Sliding window** | एक "local" attention pattern जहां हर token सिर्फ़ पिछले N tokens देखता है। Gemma 4: 512 (small) या 1024 (big)। |
| **Global attention** | Full attention — हर token सारे prior tokens देखता है। Gemma 4 कुछ layers में रखता है; **final layer हमेशा global है**। |
| **Hybrid attention** | Local + global layers का mix। Gemma 4 का design। |
| **RoPE** | Rotary Position Embedding — model कैसे जाने हर token कहां है। |
| **p-RoPE (Proportional RoPE)** | Gemma 4 का variant global layers के लिए, long context के लिए friendlier। |
| **PLE (Per-Layer Embeddings)** | हर decoder layer को per token अपनी छोटी embedding table मिलती है। E2B/E4B में parameter efficiency के लिए used। |
| **MoE (Mixture of Experts)** | कई "expert" sub-FFNs वाला model; per token सिर्फ़ कुछ use होते हैं। 26B-A4B 128 में से 8 active + 1 shared use करता है। |
| **Vision encoder** | वो part जो images को tokens में बदलता है। E-tier के लिए ~150M params, 31B / 26B-A4B के लिए ~550M। |
| **Audio encoder** | Audio frames को tokens में बदलता है। सिर्फ़ E-tier पर ~300M params। |
| **Multimodal** | एक से ज़्यादा input type accept करता है। Gemma 4 text + image + (E-tier पर audio) + video (frame-sampled) support करता है। |
| **dtype** | Numeric format जिसमें weights/activations stored हैं। Common: `float32`, `bfloat16`, `float16`, `float8`, `int8`, `int4`। |
| **bf16** | "Brain Float 16" — fp32 जैसी same exponent range वाला 16-bit float। **2026 में training और high-quality inference default।** |
| **fp8** | 8-bit float (E4M3 या E5M2)। H100/B200-only। bf16 से ~2× faster tiny quality drop के साथ। |
| **int4 / Q4** | 4-bit weight quantization। ~4× memory cut। Laptop / edge के लिए used। |
| **TTFT** | Time-to-First-Token। Prompt भेजने के बाद first reply token appear होने तक कितना समय। |
| **TPOT** | Time-Per-Output-Token। Streaming interval। 1 / TPOT ≈ tokens-per-second। |
| **Tokens/sec (tok/s)** | Speed जिस पर model tokens emit करता है। Headline UX metric। |
| **KV cache** | Generation के दौरान previous tokens से stored Keys और Values, ताकि हम recompute न करें। Long context पर biggest memory cost। |
| **Unified K/V** | Gemma 4 का trick: global layers में, K और V एक single tensor share करते हैं — global-layer cache memory आधी। |
| **Thinking mode** | Reasoning step। System prompt में `<\|think\|>` insert करके triggered। |
| **Image token budget** | एक image कितने tokens "worth" है। Gemma 4: 70, 140, 280, 560, या 1120। Higher = better detail, more cost। |
| **Function calling** | Native tool use — model एक defined function invoke करने के लिए structured JSON emit करता है। Gemma 4 में native। |
| **System prompt** | एक special pre-message जो assistant का role/rules define करता है। **Gemma 4 में native** (Gemma 3 में workaround-only था)। |
| **Apache 2.0** | License। Permissive — commercial use OK, fine-tuning और redistribution OK। |

अगर नीचे कोई term इस list पर नहीं है, ये documentation bug है। File करो।

---

## 2. Family at a Glance

Official Hugging Face model card से directly pulled।

### Dense Models

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

### MoE Model

| Property | 26B-A4B |
|----------|---------|
| Total params | 25.2B |
| Active params/token | 3.8B |
| Layers | 30 |
| Experts | 128 routed में से 8 active + 1 shared |
| Sliding window | 1024 |
| Context length | 256K |
| Vocab | 262K |
| Modalities | text + image |
| Vision encoder | ~550M |

E2B/E4B में "E" = "effective" (PLE file को active params से bigger बनाता है)। A4B में "A" = "active" (25.2B में से सिर्फ़ 3.8B per token compute करते हैं)।

---

## 3. Architecture Plain English में

Gemma 4 standard pre-norm transformer template (chapter 10) use करता है तीन notable upgrades के साथ।

### 3.1 Hybrid Attention (Local + Global)

```
Layer  1:  local sliding-window  (W = 512 या 1024)
Layer  2:  local sliding-window
Layer  3:  global full-context
Layer  4:  local sliding-window
…
Layer  N:  global   ← final layer ALWAYS global
```

Most layers एक window के अंदर attend करते हैं (cheap, local)। कुछ global हैं (full context, expensive लेकिन globally aware)। **Final layer forced global है** ताकि model का output हमेशा full context से informed हो, सिर्फ़ window से नहीं।

ये क्यों काम करता है:
- Local layers word-by-word grammar, immediate dependencies handle करते हैं — most work।
- Global layers long-range structure handle करते हैं (cross-paragraph connections, document-level facts)।
- उन्हें mix करना small-window model की speed और memory full-attention वाले की reach के साथ देता है।

ये pattern Gemma 2 में start हुआ और Gemma 4 में matured हो गया।

### 3.2 Per-Layer Embeddings (PLE) — सिर्फ़ E2B / E4B पर

एक normal transformer imagine करो: एक big embedding table, per token एक बार looked up, residual stream में feed। Vector को फिर 35-42 layers के लिए "मैं कौन सा token हूं" remember करना है।

Gemma का PLE **per layer एक separate small embedding table** add करता है। हर decoder layer को अपना quick-lookup hint मिलता है "ये कौन सा token है," fresh, हर layer। Tables big हैं (इसलिए E2B की *file* 5.1B है जबकि active math 2.3B है) लेकिन ये **lookups हैं, matmuls नहीं** — runtime पर almost free।

Net effect: same compute के लिए better parameter efficiency। E2B benchmarks पर 4B-class model की तरह punch करता है। Phones और laptops layer-by-layer hint stream कर सकते हैं as they go।

### 3.3 Global Layers — Proportional RoPE + Unified K/V

Global layers (long-reach वाले) में, Gemma 4 दो memory-savers करता है:

- **Unified K और V**: KV cache में per token keys और values को दो separate tensors store करने के बजाय, वो एक single tensor share करते हैं। Global-layer cache के लिए **आधी memory**।
- **Proportional RoPE (p-RoPE)**: एक re-scaled RoPE designed for the long ranges इन layers को cover करनी हैं। Vanilla RoPE से cleaner long-context extension।

यही 31B को 256K context run करने देता है GPU की HBM wreck किए बिना। Local layers को ये नहीं चाहिए क्योंकि window small है।

---

## 4. Hands-on — Load और Run

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
    dtype="auto",          # modern GPU पर bfloat16 pick करता है
    device_map="auto",     # visible GPUs के across spread करता है
)
print(f"loaded on: {model.device}, dtype: {model.dtype}")

# 2. Prompt
messages = [
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "Write a short joke about saving RAM."},
]

# 3. Chat template apply करो (system role, tags, etc. handle करता है)
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,        # reasoning workloads के लिए ON करो
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# 4. Generate (timing के साथ)
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

# 5. Decode और parse
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))

print(f"\n[generated {gen_len} tokens in {elapsed:.2f}s = "
      f"{gen_len/elapsed:.1f} tok/s]")
```

`dtype="auto"` Ampere+ GPUs (A100, H100, RTX 30/40/50xx) पर **bfloat16** pick करता है और older hardware पर **float16**। `device_map="auto"` अगर एक GPU पर fit न हो तो model shard करता है।

Sample output:

```
Why did the developer keep cutting his model in half?
Because every byte saved is RAM well-earned.

[generated 24 tokens in 0.31s = 77.4 tok/s]
```

---

## 5. dtype Guide — कब क्या Use करें

Specifically Gemma 4 के लिए। Numbers सिर्फ़ weight memory हैं; activations/KV cache ~20-30% add करते हैं।

| Variant | dtype | Memory (weights) | कहां run होता है | Notes |
|---------|-------|------------------|---------------|-------|
| **E2B** | bf16 | ~5 GB | phone, laptop CPU/GPU | edge के लिए default |
| **E2B** | int4 (Q4_K_M) | ~1.5 GB | phone | Ollama / llama.cpp / MLX |
| **E4B** | bf16 | ~8 GB | RTX 4060+, M-series Mac | default |
| **E4B** | int4 | ~3 GB | most laptops | comfortable |
| **26B-A4B** | bf16 | ~50 GB | 1× H100 80 GB | active 3.8B → fast inference |
| **26B-A4B** | fp8 | ~26 GB | 1× H100, 1× L40S | best throughput-per-$ |
| **26B-A4B** | int4 | ~14 GB | RTX 4090 24 GB | yes, fits! |
| **31B Dense** | bf16 | ~62 GB | 1× H100 80 GB | quality default |
| **31B Dense** | fp8 | ~31 GB | 1× H100 / 2× RTX 4090 | 2026 में server default |
| **31B Dense** | int4 | ~16 GB | RTX 4090 / Mac M3 Max | laptop-class quality |

Quick rules:

- **Training / fine-tuning**: bf16 हमेशा। Quality > memory।
- **Cloud serving**: fp8 (H100/B200) या bf16। Throughput के लिए fp8 pick करो।
- **Workstation पर single-user inference**: fit हो तो bf16, else int4 GGUF।
- **Phone / embedded**: int4 Q4_K_M या Q4_0। `llama.cpp`, `Ollama`, या `MLX` use करो।

एक specific dtype force करो:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             dtype=torch.bfloat16,
                                             device_map="auto")
```

Load के बाद verify:

```python
print(model.dtype)                                    # torch.bfloat16
print(next(model.parameters()).dtype)                 # torch.bfloat16
print(model.get_memory_footprint() / 1e9, "GB")
```

---

## 6. Tokens-per-second — Measured Throughput

ये realistic, single-stream (batch=1) numbers हैं public benchmarks और recent vLLM tests से। Multi-user (batched) throughput much higher है; chapter 17 देखो।

### E-tier (Laptop / Edge)

| Variant | Hardware | dtype | tok/s (single user) | Notes |
|---------|----------|-------|---------------------|-------|
| E2B | M3 Max (MLX) | int4 | 100-140 | Mac पर very fast |
| E2B | RTX 4070 Ti | bf16 | 200-260 | |
| E2B | RTX 4090 | bf16 | 280-340 | |
| E2B | iPhone 16 Pro / Pixel 9 (CoreML/MediaPipe) | int4 | 30-60 | mobile |
| E4B | M3 Max | int4 | 50-80 | |
| E4B | RTX 4090 | bf16 | 130-180 | |

### Mid (Workstation / Single GPU)

| Variant | Hardware | dtype | tok/s (single user) | Notes |
|---------|----------|-------|---------------------|-------|
| 26B-A4B | RTX 4090 (24 GB) | int4 | 70-110 | active 3.8B → very fast |
| 26B-A4B | H100 80 GB | bf16 | 120-150 | |
| 26B-A4B | H100 80 GB | fp8 | 200-260 | |
| 31B Dense | RTX 4090 (24 GB) | int4 | 35-50 | tight fit |
| 31B Dense | H100 80 GB | bf16 | 50-70 | |
| 31B Dense | H100 80 GB | fp8 | 90-130 | |
| 31B Dense | B200 (192 GB) | fp8 | 180-230 | |

### MoE Dense से Faster क्यों Feel होता है

26B-A4B के पास 31B से *ज़्यादा total params* हैं लेकिन ये per user **2-3× faster** है। Reason इस guide के chapter 6 और chapter 9 में है: हर token सिर्फ़ 3.8B params worth compute activate करता है, even though full 25.2B HBM में loaded है। Memory bandwidth (chapter 17 §6) decode bottleneck है — per token कम read = faster output।

### Time-to-first-token (TTFT) 4K Prompt पर

| Variant | Hardware | dtype | TTFT (4K prompt) |
|---------|----------|-------|------------------|
| E4B | RTX 4090 | bf16 | ~200 ms |
| 26B-A4B | H100 fp8 | fp8 | ~150 ms |
| 31B Dense | H100 fp8 | fp8 | ~280 ms |
| 31B Dense | B200 fp8 | fp8 | ~140 ms |

Interactive chat के लिए, TTFT < 500 ms instant feel होता है। Ghost-text code completion के लिए, < 200 ms bar है।

### एक Quick Benchmark जो आप खुद Run कर सकते हो

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

5 बार run करो और median लो (warmup matters, especially first launch)।

---

## 7. Multimodal — Images, Video, Audio

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

**Cost vs detail के लिए image budget choose करो।** `mm_tokens_per_image` set करो (या processor kwarg, version पर depend):

| Budget | इसके लिए use करो |
|--------|---------|
| **70** | classification, video frames, "photo में क्या है?" |
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

### Video (Frame-sampled, ≤ 60 sec at 1 fps)

```python
# जब आप एक path pass करते हो तो transformers' processor video automatically handle करता है
messages = [{"role":"user","content":[
    {"type":"video","video":"clip.mp4"},
    {"type":"text","text":"Summarize what happens in this clip."},
]}]
```

**Modality order matter करता है** — अपने prompt में text से **पहले** images / audio / video रखो। अगर आप text first रखो तो performance measurably drops करती है।

---

## 8. Thinking Mode — Math, Code, Hard Reasoning के लिए

System prompt के start पर `<|think|>` control token से triggered।

```python
messages = [
    {"role":"system","content":"You are a careful math tutor."},
    {"role":"user","content":"If 7 cats catch 7 mice in 7 minutes, "
                              "how many cats catch 100 mice in 50 minutes?"},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,        # <-- ये <|think|> insert करता है
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096,
                     temperature=1.0, top_p=0.95, top_k=64)
parsed = processor.parse_response(processor.decode(out[0], skip_special_tokens=False))
# parsed['thought']  -> internal reasoning
# parsed['answer']   -> user-facing reply
```

Model emit करेगा:

```
<|channel|>thought
[internal reasoning, often 100-2000 tokens]
<|channel|>
[final answer]
```

`processor.parse_response` आपके लिए इन्हें split करता है।

**Multi-turn rule:** जब conversation continue कर रहे हो, **prior turns से `thought` blocks drop करो** next user message भेजने से पहले। उन्हें history में keep करना quality degrade करता है। बस user-visible answers keep करो।

E2B / E4B के लिए thinking disabled के साथ, model normally answer generate करता है। 26B-A4B / 31B के लिए thinking disabled के साथ, ये अभी भी empty `<|channel|>thought` tags emit करता है — that's expected; बस उन्हें parse out करो।

---

## 9. Function Calling (Native, कोई Harness नहीं चाहिए)

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
# Generate, parse — model एक structured tool_call emit करता है।
# आप tool execute करते हो, result append करते हो, फिर call करते हो।
```

Native tool support means आपको separate function-calling fine-tune की need नहीं। 31B + 26B-A4B multi-step agentic tool use well handle करते हैं; E-tier simple single-tool calls के लिए acceptable है।

---

## 10. Real Benchmarks (Official Model Card से)

Sizes के across headline numbers। **bold = late 2026 तक उस size class पर किसी भी open model के लिए state-of-the-art**।

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

Gemma 3-27B → Gemma 4-31B पर AIME का leap **20% से 89%** तक है — ये capability में एक different model family है, incremental release नहीं। Thinking mode + better data responsible हैं।

---

## 11. Best Practices (Google's Official Guidance)

### Sampling
- `temperature=1.0`, `top_p=0.95`, `top_k=64` — इन्हें use करो। ये सारे sizes के across tuned defaults हैं।

### System Prompts
- Native role support means बस normally `system` role use करो।
- Thinking mode के लिए, `enable_thinking=True` के via automatically `<|think|>` prepend करो।

### Multi-turn
- **Next turn से पहले history से thought blocks strip करो**।
- Chat templates other formatting handle करते हैं; processor को tokens manage करने दो।

### Multimodal
- Order: **image → audio → video → text**।
- Video: ≤ 60 seconds at 1 fps। Audio: ≤ 30 seconds।
- Smallest image-token budget pick करो जो right answer देता है।

### Tools
- `apply_chat_template` में native `tools=` argument use करो।
- Agentic tasks के लिए, 26B-A4B और 31B work करते हैं; E-tier multi-tool reasoning पर hit-or-miss है।

### Long Context
- 31B / 26B-A4B 256K natively handle करते हैं। 256K से beyond supported नहीं।
- Full 256K context पर memory KV cache से dominated है; vLLM में 50% reduction के लिए fp8 KV use करो।

---

## 12. Production Serving

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

# 26B-A4B 4× H100 पर expert parallelism के साथ
vllm serve google/gemma-4-26B-A4B-it \
    --quantization fp8 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len 65536
```

Laptop / Mac के लिए:

```bash
# Ollama (Mac, Linux, Windows)
ollama run gemma-4:31b-q4_K_M

# llama.cpp directly
./llama-cli -m gemma-4-31b-q4_K_M.gguf -n 200 -p "Hello"

# MLX (Mac)
mlx_lm.generate --model mlx-community/gemma-4-31B-it-4bit \
                --prompt "Hello" --max-tokens 200
```

Apple Silicon के लिए, **MLX 4-bit के साथ** currently fastest local option है — same quantization पर Ollama से M3/M4 पर 2-3× faster।

---

## 13. Common Gotchas

- **`enable_thinking=True` output length blow up करता है।** `max_new_tokens=4096+` set करो या आप model को mid-thought cut करोगे।
- **Modality order**: text-first measurably worse है। हमेशा images / audio / video पहले।
- **History में Thought blocks** multi-turn quality degrade करते हैं। Next turn से पहले strip करो।
- **MLX vs llama.cpp 4-bit same format नहीं हैं।** Mix मत करो।
- **Image-token budget overrides**: `transformers` version पर depending, kwarg `mm_tokens_per_image` है या processor config के via done; model card check करो।
- **PLE memory**: tight RAM पर E-tier deploy करते समय, disk file runtime memory से bigger है — 5.1 GB E2B file पर panic मत करो।
- **E-tier पर Function calling**: works, लेकिन >2 tool steps के लिए brittle है। Serious agents के लिए 26B-A4B+ use करो।
- **Gemma 3 prompts cleanly transfer नहीं करते।** Gemma 4 का native system role है, different chat tokens। Upgrade के बाद prompts re-test करो।

---

## 14. End-to-end Real Example

A complete script: 31B load करो, thinking on के साथ multimodal coding question पूछो, time करो, result save करो।

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

# 3. Generate (TTFT timing के साथ)
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

# 4. Parse और report
raw = processor.decode(out[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(raw)

print("\n=== THINKING ===\n", parsed.get("thought", "")[:800])
print("\n=== ANSWER ===\n",  parsed.get("answer",  "")[:2000])
print(f"\n[{new_toks} tokens in {elapsed:.1f}s "
      f"= {new_toks/elapsed:.1f} tok/s]")
```

Run करो:

```bash
python gemma4_demo.py login_form.png
```

H100 fp8 पर आप ~80-120 tok/s देखोगे, first ~1000-2000 tokens `thought` channel में और final JSX `answer` में।

---

## 15. Cheat Sheet

- **Size से pick करो**: E2B (phone), E4B (laptop), 26B-A4B (cheap server), 31B (best quality)।
- **MoE rule**: 26B-A4B *per user* 31B से faster है लेकिन ज़्यादा memory चाहिए।
- **Sampling**: हमेशा `temperature=1.0, top_p=0.95, top_k=64`।
- **Image budget**: 70 / 140 / 280 / 560 / 1120 — 280 से start करो, OCR के लिए raise करो, video के लिए lower करो।
- **Modality order**: image → audio → video → text।
- **Thinking**: math/code के लिए system prompt में `<|think|>`; history से thought strip करो।
- **dtype**: bf16 quality, fp8 throughput, int4 edge।
- **Tokens/sec**: H100 पर 31B fp8 ≈ 100; 26B-A4B fp8 ≈ 230; 4090 पर E4B bf16 ≈ 150।
- **Context**: 128K (E-tier), 256K (26B-A4B / 31B)। Beyond मत जाओ।
- **Native everything**: system role, function calling, thinking, multimodal।
- **License**: Apache 2.0 — commercial use OK।

---

## 16. और गहराई से

- **Official model card** at `huggingface.co/google/gemma-4-31B-it` — एक बार पढ़ो।
- **Gemma technical reports** (जब published हो) — Google DeepMind blog।
- **Gemma launch blog** at `developers.googleblog.com` — design choices पर context।
- **vLLM Gemma 4 integration** — production serving recipes।
- **MLX Gemma examples** at `github.com/ml-explore/mlx-examples` — Apple Silicon।
- **`llama.cpp` Gemma support** — CPU और edge inference।
- **Ollama library** at `ollama.com/library/gemma-4` — locally run करने का easiest तरीका।

ये deep-dive **[16-frontier-models-2026.md](./16-frontier-models-2026.md)** को complement करता है (जो Gemma 4 को Qwen 3.6 और दूसरे frontier families के against compare करता है)। Production में Gemma 4 serve करने के लिए, **[17-production-inference.md](./17-production-inference.md)** देखो। Evaluation के लिए, **[19-evaluation.md](./19-evaluation.md)**। Fine-tuning के लिए, **[20-fine-tuning-recipes.md](./20-fine-tuning-recipes.md)** — वहां की हर recipe Gemma 4 पर काम करती है।

Curriculum का end।
