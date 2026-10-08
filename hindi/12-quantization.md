# 12 · Quantization — Models को Small और Fast बनाओ

> **TL;DR** Quantization fp16/bf16 weights को smaller integers (या low-bit floats) से replace करता है ताकि वो less memory लें और HBM से faster load हों। चूंकि LLM **inference** memory-bound है, ये available सबसे bigger single inference win है। **2026 में practical defaults हैं: laptop / CPU के लिए GGUF Q4_K_M, GPU के लिए AWQ-int4 या GPTQ-int4, top-end H100/B200 deployments के लिए FP8, और bleeding edge के लिए NVFP4 या BitNet 1.58-bit।**

## 1. Mental Model

एक fp16 weight 2 bytes है। एक int4 weight 0.5 bytes है — **4× less memory, 4× less HBM bandwidth** उसे read करने के लिए। चूंकि decode memory-bound है, इसका मतलब **4× faster generation** तक, plus आप 4× A100s की ज़रूरत के बजाय 24 GB GPU पर 70B model fit कर सकते हो।

Catch: int4 में सिर्फ़ 16 representable values हैं। इसे काम करने के लिए, हम float weights `[-w_max, +w_max]` को scaling factor के साथ `[-8, 7]` में map करते हैं:

```
w_int = round(w_float / scale)         # इसे store करो
w_recovered = w_int * scale            # runtime पर इसे use करो
```

अगर हम scale carefully choose करें और tiny precision loss accept करें, model almost identically behave करता है। Modern algorithms exactly यही करते हैं।

---

## 2. दो Big Distinctions

### Weight-only vs Activation+weight Quantization

- **Weight-only**: सिर्फ़ model weights int4/int8 में। Activations bf16 में रहती हैं। Math weights को on the fly dequantize करने के बाद bf16 में run होता है। *Most common.*
- **Activation+weight (W8A8 / W4A8 / W4A4)**: activations भी low-bit। Math int8/int4 में run होता है। More speedup, more quality risk।

### Post-training (PTQ) vs Quantization-aware Training (QAT)

- **PTQ**: trained model लो, fact के बाद small calibration dataset के साथ quantize करो। Cheap, fast, 2026 में default।
- **QAT**: training के दौरान quantization simulate करो ताकि model robust होना सीखे। More expensive, BitNet-style models द्वारा used जो extreme bit widths पर quantize करते हैं।

---

## 3. Symmetric vs Asymmetric, Per-tensor vs Per-channel

**Symmetric**: range `[-r, +r]`, zero-point = 0, simpler। **Asymmetric**: range `[a, b]`, zero-point shifts। Activations के लिए better जो centered नहीं हैं (post-ReLU)। LLM weights usually near-symmetric हैं, इसलिए symmetric quantization fine है।

**Per-tensor**: पूरे weight tensor के लिए एक scale। **Per-channel** (per-output-row): per output dimension एक scale। **Group**: input dim में `g` weights के group per एक scale (e.g., `g=128`)। **Group-wise** int4 weights के लिए 2026 standard है — उन weight rows की variance capture करता है जिनकी magnitudes wildly different हैं।

```
W: (out, in) — fp16 weight matrix
group_size g: e.g., 128
scales: (out, in/g)        # one fp16 scale per group
zeros:  (out, in/g)        # if asymmetric

w_int4[i, j] = clamp(round(W[i, j] / scales[i, j // g] + zeros[i, j // g]), 0, 15)
```

Storage: `W_int4` per element 4 bits है, `scales` और `zeros` ~1% overhead add करते हैं। fp16 vs net savings: ~3.7×।

---

## 4. 2026 Algorithm Shortlist

### GPTQ (Frantar et al. 2022)

- **Second-order information** (sequential row updates के via approximate Hessian) use करते हुए one-shot post-training quantization।
- ~2023 से GPUs पर weight-only int4 quality के लिए gold standard।
- Calibration single GPU पर ~hours लेता है।
- Use cases: int4 weights के साथ vLLM/SGLang/TensorRT-LLM के via serving।

### AWQ (Lin et al. 2023, **A**ctivation-aware **W**eight **Q**uantization)

- Notice करता है कि "important" weights वो हैं जिन्हें **large activations** से multiply किया जाता है। Per-channel weights को scale करता है उन rows को rounding error से protect करने के लिए।
- Often int4 पर GPTQ से slightly better quality, compute करने के लिए faster (~10 min vs hours)।
- GPU पर open models serve करने के लिए 2026 default।

### GGUF + GGML (`llama.cpp`)

- **CPU, laptop GPU, Apple Silicon** के लिए optimized file format और quantization recipes।
- Names जैसे `Q4_K_M` (k-quants medium), `Q5_K_S`, `Q6_K`, `IQ3_XXS` (i-quants, even more compressed)।
- Quality ladder: `Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q4_0 > Q3_K > IQ3 > Q2_K`।
- **Q4_K_M laptop inference के लिए workhorse है** — fp16 से ~50% smaller negligible quality loss के साथ।
- `llama.cpp`, `Ollama`, `LM Studio` के साथ run करो।

### bitsandbytes (`bnb`) — NF4 (QLoRA)

- 4-bit "NormalFloat" — normally-distributed weights के लिए tuned non-linear quantization।
- GPTQ से simpler; quality **fine-tuning** के लिए close enough लेकिन inference के लिए AWQ/GPTQ से slightly worse।
- 24 GB GPUs पर parameter-efficient fine-tuning के लिए famously **QLoRA** द्वारा used।

### FP8 (Hopper / H100, Blackwell / B100/B200)

- Weights और activations दोनों fp8 (E4M3 या E5M2) में। Tensor Cores natively fp8 matmul run करते हैं।
- short calibration के बाद usually no quality drop के साथ fp16 पर ~2× speedup।
- Top-tier deployments के लिए pretraining-and-inference choice. Frameworks: TensorRT-LLM, vLLM (`--quantization fp8`), MS-AMP।
- DeepSeek-V3 famously fp8 (mixed-precision recipe के साथ) में **trained** था।

### NVFP4 / FP4 (Blackwell, 2025-2026)

- 4-bit floating point (E2M1) B100/B200 पर hardware support के साथ।
- int4 से better range, dramatic speedups।
- Currently production stacks (TensorRT-LLM, vLLM) में emerging; late 2026 तक likely dominate।

### BitNet b1.58 (1.58-bit)

- Extreme: ternary weights `{-1, 0, +1}` (`log₂(3) ≈ 1.58 bits`)।
- ternary-aware optimization (QAT) के साथ **training from scratch** चाहिए।
- Microsoft का BitNet b1.58 (2024) और follow-up work दिखाते हैं कि 7B-class models 1.58 bits पर fp16 quality match करते हैं।
- 2026 follow-ups (`BitNet a4.8`, `MS-Models`) 8-bit activations + 1.58-bit weights तक push करते हैं, kernels के साथ जो CPU पर fp16 inference को beat करते हैं।

### Mixed Quantization (e.g., int4 weights + int8 activations + fp16 LM head)

- LM head और embedding often higher precision में रखे जाते हैं; बाकी सब int4। Practice में common।
- DeepSeek-V3-style: ज़्यादातर network fp8 पर, sensitive bits bf16 पर।

---

## 5. Basic Int8 Quantization Recipe (ताकि आप बाकी समझ सको)

```python
import torch

def quantize_int8_per_channel(w):
    # w: (out, in)
    scale = w.abs().amax(dim=1) / 127     # one per output row
    w_int = torch.round(w / scale.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    return w_int, scale                   # store these

def dequantize(w_int, scale):
    return w_int.float() * scale.unsqueeze(1)

W = torch.randn(2048, 4096) * 0.05
W_q, s = quantize_int8_per_channel(W)
W_rec = dequantize(W_q, s)
print((W - W_rec).abs().mean().item())    # ~tiny
```

Inference के लिए, आप पूरा matrix एक साथ dequantize नहीं करते — आप इसे matmul kernel में fuse करते हो:

```
y = (x @ W_q.float()) * scale_per_row     # equivalently per-output channel
```

cuBLAS / cuBLASLt और Triton के पास int8 matmul kernels (`gemm_int8`, `cublasLtMatmul`) हैं जो ये hardware में करते हैं। int4 के लिए, kernels custom हैं (`exllama-v2`, `marlin`, `flute`)।

---

## 6. AWQ Code में (Sketch)

AWQ insight: weights को `s` से scale करो और inputs को `1/s` से (mathematically a no-op), फिर scaled weights को quantize करो। Per channel `s` को **most-activated channels पर quantization error minimize करने** के लिए pick करो:

```
y = (x / s) @ (W * s)   # mathematically same as x @ W
```

`s_c` को activation magnitudes के basis पर हर input channel के लिए choose करके, आप उन channels को "protect" करते हो जो matter करते हैं और rounding loss को बाकी bear करने देते हो।

Practice में, आप **`autoawq`** library use करते हो:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2.5-7B-Instruct'
quant_path = 'qwen2.5-7b-awq'
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}

model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True)
tok = AutoTokenizer.from_pretrained(model_path)
model.quantize(tok, quant_config=quant_config)
model.save_quantized(quant_path)
```

फिर vLLM के साथ load करो: `vllm serve qwen2.5-7b-awq --quantization awq_marlin`।

---

## 7. GPTQ Code में (Sketch)

Same idea, different algorithm. `auto-gptq` use करो:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
qcfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)
model = AutoGPTQForCausalLM.from_pretrained(model_path, qcfg)

# calibration data pass करो — ~512 samples enough हैं
calib = [tok(text, return_tensors='pt') for text in calibration_texts[:512]]
model.quantize(calib)
model.save_quantized('qwen-7b-gptq')
```

Best quality के लिए calibration data आपकी serving distribution match करना चाहिए (chat? code? math?)।

---

## 8. CPU / Apple Silicon के लिए GGUF

```bash
# HF model को GGUF में convert करो (llama.cpp tooling use करता है)
python convert_hf_to_gguf.py Qwen/Qwen2.5-7B-Instruct --outfile qwen.gguf

# Q4_K_M पर quantize करो
./llama-quantize qwen.gguf qwen-q4km.gguf Q4_K_M

# run
./llama-cli -m qwen-q4km.gguf -p "Hello" -n 200
```

`Ollama` इसे wrap करता है; `pip install ollama` फिर `ollama run qwen2.5:7b-instruct-q4_K_M`। Most consumer-LLM apps (LM Studio, Jan, etc.) इस pipeline को use करते हैं।

---

## 9. FP8 Inference (H100 और beyond)

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct-FP8 --quantization fp8
```

अगर आपका model FP8 में published नहीं था, आप इसे on-the-fly quantize कर सकते हो:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8
```

`kv-cache-dtype fp8` weight quantization से independent है — ये किसी भी model के लिए on हो सकता है। **FP8 weights + FP8 KV cache** H100 पर highest-throughput config है।

---

## 10. Quality vs Size — क्या Expect करें

Llama / Qwen-class 7B models के लिए fp16 vs approximate average benchmark drop (आपका mileage vary हो सकता है):

| Format | Bits/weight | Memory | Quality drop |
|--------|-------------|--------|--------------|
| fp16 / bf16 | 16 | 1.0× | 0 |
| fp8 | 8 | 0.5× | ~0% |
| int8 (W8A8) | 8 | 0.5× | < 0.5% |
| GPTQ-int4 g128 | ~4.5 | 0.27× | 0.5-1.5% |
| AWQ-int4 g128 | ~4.5 | 0.27× | 0.5-1% |
| GGUF Q5_K_M | ~5.5 | 0.34× | 0.3% |
| GGUF Q4_K_M | ~4.7 | 0.29× | 0.5-1% |
| GGUF IQ3_XXS | ~3.0 | 0.20× | 1-3% |
| GGUF Q2_K | ~2.6 | 0.18× | 3-7% |
| BitNet 1.58 | 1.58 | 0.1× | (fp16 के समान, लेकिन retraining चाहिए) |

ज़्यादातर use cases के लिए, **Q4_K_M / AWQ-int4 sweet spot है** — model 4× less memory में fits <1% quality cost पर। 4 bits से नीचे, quality wobble करने लगती है unless model इसके लिए trained था।

---

## 11. Weights *नहीं* को Quantize करना

- **KV cache quantization** (chapter 9): fp8 / int8 / int4। Weight quantization से independent। Long context के लिए critical।
- **Activation quantization**: W8A8/W4A4 के लिए चाहिए। Careful per-token scaling चाहिए।
- **Optimizer state quantization**: 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`) optimizer state memory 4× cut करता है। QLoRA fine-tuning के लिए standard।
- **Gradient quantization**: PT में rarer; कुछ distributed-training schemes में used।

---

## 12. QLoRA: Quantization + LoRA = Consumer GPUs पर Giants Fine-tune करो

QLoRA (Dettmers et al. 2023) base model को NF4 में freeze करता है और bf16 में small **LoRA** adapter matrices train करता है। एक single 80 GB H100 पर 70B model fine-tune करने देता है (या 16 GB consumer GPU पर 7B)।

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', quantization_config=bnb)

lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'],
                  lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora)
```

फिर एक normal training loop। End में सिर्फ़ LoRA weights save करो (कुछ hundred MB)।

QLoRA के 2025-2026 successors:
- **DoRA** (decomposed LoRA) — same param count पर slightly better quality।
- **QLoRA + Spectrum / GaLore / VeRA** — even more memory-efficient।

---

## 13. End-to-end Example: Single GPU पर Qwen2.5-7B AWQ Ship करो

```bash
pip install autoawq vllm

# Quantize
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
m = AutoAWQForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
m.quantize(t, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'})
m.save_quantized('./qwen2.5-7b-awq'); t.save_pretrained('./qwen2.5-7b-awq')
"

# Serve
vllm serve ./qwen2.5-7b-awq --quantization awq_marlin --max-model-len 32768

# Use
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"./qwen2.5-7b-awq","messages":[{"role":"user","content":"hi"}]}'
```

Memory: weights + KV cache के लिए ~5 GB। Throughput: H100 पर ~8000 tokens/sec। Quality: chat के लिए fp16 से indistinguishable।

---

## 14. 2026 Cheat Sheet

- **GPU serving:** AWQ-int4 (`awq_marlin` kernel) या GPTQ-int4। दोनों vLLM के via।
- **CPU / laptop / Mac:** GGUF Q4_K_M `llama.cpp` या Ollama के साथ।
- **High-end production (H100/B200):** FP8 weights + FP8 KV cache, vLLM या TensorRT-LLM के via।
- **Long context:** KV cache quantize करो (`--kv-cache-dtype fp8`)।
- **1 GPU पर fine-tuning:** QLoRA (NF4 + LoRA bf16)।
- **Train-from-scratch ultra-low-bit:** BitNet b1.58 — niche लेकिन real।
- **हमेशा** अपने benchmarks पर **measure करो**; quantization quality task-dependent है।

---

## और गहराई से

- Frantar et al. 2022 — **GPTQ** paper।
- Lin et al. 2023 — **AWQ** paper।
- Dettmers et al. 2023 — **QLoRA**।
- Microsoft 2024 — **BitNet b1.58: The Era of 1-bit LLMs.**
- `llama.cpp` README का "Quantization formats" table — GGUF के लिए best quick reference।
- vLLM और TensorRT-LLM quantization docs — most up-to-date practical guides।

Next: **[13-mixture-of-experts.md](./13-mixture-of-experts.md)** — sparse models जो bigger होते हैं slower हुए बिना।
