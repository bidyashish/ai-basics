# 12 · Quantization — Make Models Small and Fast

> **TL;DR** Quantization replaces fp16/bf16 weights with smaller integers (or low-bit floats) so they take less memory and load faster from HBM. Since LLM **inference** is memory-bound, this is the biggest single inference win available. **In 2026 the practical defaults are: GGUF Q4_K_M for laptop / CPU, AWQ-int4 or GPTQ-int4 for GPU, FP8 for top-end H100/B200 deployments, and NVFP4 or BitNet 1.58-bit for the bleeding edge.**

## 1. The mental model

A fp16 weight is 2 bytes. An int4 weight is 0.5 bytes — **4× less memory, 4× less HBM bandwidth** to read it. Since decode is memory-bound, this means up to **4× faster generation**, plus you can fit a 70B model on a 24 GB GPU instead of needing 4× A100s.

The catch: int4 has only 16 representable values. To make this work, we map the float weights `[-w_max, +w_max]` to `[-8, 7]` with a scaling factor:

```
w_int = round(w_float / scale)         # store this
w_recovered = w_int * scale            # use this at runtime
```

If we choose the scale carefully and accept tiny precision loss, the model behaves almost identically. Modern algorithms do exactly this.

---

## 2. Two big distinctions

### Weight-only vs activation+weight quantization

- **Weight-only**: only the model weights are int4/int8. Activations stay in bf16. Math runs in bf16 after dequantizing weights on the fly. *Most common.*
- **Activation+weight (W8A8 / W4A8 / W4A4)**: activations are also low-bit. Math runs in int8/int4. More speedup, more quality risk.

### Post-training (PTQ) vs quantization-aware training (QAT)

- **PTQ**: take a trained model, quantize after the fact with a small calibration dataset. Cheap, fast, the default in 2026.
- **QAT**: simulate quantization during training so the model learns to be robust. More expensive, used by BitNet-style models that quantize to extreme bit widths.

---

## 3. Symmetric vs asymmetric, per-tensor vs per-channel

**Symmetric**: range `[-r, +r]`, zero-point = 0, simpler. **Asymmetric**: range `[a, b]`, zero-point shifts. Better for activations that aren't centered (post-ReLU). LLM weights are usually near-symmetric, so symmetric quantization is fine.

**Per-tensor**: one scale for the whole weight tensor. **Per-channel** (per-output-row): one scale per output dimension. **Group**: one scale per group of `g` weights in the input dim (e.g., `g=128`). **Group-wise** is the 2026 standard for int4 weights — captures the variance of weight rows that have wildly different magnitudes.

```
W: (out, in) — fp16 weight matrix
group_size g: e.g., 128
scales: (out, in/g)        # one fp16 scale per group
zeros:  (out, in/g)        # if asymmetric

w_int4[i, j] = clamp(round(W[i, j] / scales[i, j // g] + zeros[i, j // g]), 0, 15)
```

Storage: `W_int4` is 4 bits per element, `scales` and `zeros` add ~1% overhead. Net savings vs fp16: ~3.7×.

---

## 4. The 2026 algorithm shortlist

### GPTQ (Frantar et al. 2022)

- One-shot post-training quantization using **second-order information** (approximate Hessian via sequential row updates).
- Gold standard for weight-only int4 quality on GPUs since ~2023.
- Calibration takes ~hours on a single GPU.
- Use cases: serving via vLLM/SGLang/TensorRT-LLM with int4 weights.

### AWQ (Lin et al. 2023, **A**ctivation-aware **W**eight **Q**uantization)

- Notices that "important" weights are those multiplied by **large activations**. Per-channel scales the weights to protect those rows from rounding error.
- Often slightly better quality than GPTQ at int4, faster to compute (~10 min vs hours).
- The 2026 default for serving open models on GPU.

### GGUF + GGML (`llama.cpp`)

- File format and quantization recipes optimized for **CPU, laptop GPU, Apple Silicon**.
- Names like `Q4_K_M` (k-quants medium), `Q5_K_S`, `Q6_K`, `IQ3_XXS` (i-quants, even more compressed).
- Quality ladder: `Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q4_0 > Q3_K > IQ3 > Q2_K`.
- **Q4_K_M is the workhorse for laptop inference** — ~50% smaller than fp16 with negligible quality loss.
- Run with `llama.cpp`, `Ollama`, `LM Studio`.

### bitsandbytes (`bnb`) — NF4 (QLoRA)

- 4-bit "NormalFloat" — non-linear quantization tuned for normally-distributed weights.
- Simpler than GPTQ; quality close enough for **fine-tuning** but slightly worse for inference than AWQ/GPTQ.
- Famously used by **QLoRA** for parameter-efficient fine-tuning on 24 GB GPUs.

### FP8 (Hopper / H100, Blackwell / B100/B200)

- Weights and activations both in fp8 (E4M3 or E5M2). Tensor Cores natively run fp8 matmul.
- ~2× speedup over fp16 with usually no quality drop after a short calibration.
- The pretraining-and-inference choice for top-tier deployments. Frameworks: TensorRT-LLM, vLLM (`--quantization fp8`), MS-AMP.
- DeepSeek-V3 was famously **trained** in fp8 (with mixed-precision recipe).

### NVFP4 / FP4 (Blackwell, 2025-2026)

- 4-bit floating point (E2M1) with hardware support on B100/B200.
- Better range than int4, dramatic speedups.
- Currently emerging in production stacks (TensorRT-LLM, vLLM); will likely dominate by late 2026.

### BitNet b1.58 (1.58-bit)

- Extreme: ternary weights `{-1, 0, +1}` (`log₂(3) ≈ 1.58 bits`).
- Requires **training from scratch** with ternary-aware optimization (QAT).
- Microsoft's BitNet b1.58 (2024) and follow-up work show 7B-class models match fp16 quality at 1.58 bits.
- 2026 follow-ups (`BitNet a4.8`, `MS-Models`) push to 8-bit activations + 1.58-bit weights, with kernels that beat fp16 inference on CPU.

### Mixed quantization (e.g., int4 weights + int8 activations + fp16 LM head)

- The LM head and the embedding are often kept in higher precision; everything else int4. Common in practice.
- DeepSeek-V3-style: most of the network at fp8, sensitive bits at bf16.

---

## 5. The basic int8 quantization recipe (so you understand the rest)

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

For inference, you don't dequantize the whole matrix at once — you fuse it into the matmul kernel:

```
y = (x @ W_q.float()) * scale_per_row     # equivalently per-output channel
```

cuBLAS / cuBLASLt and Triton have int8 matmul kernels (`gemm_int8`, `cublasLtMatmul`) that do this in hardware. For int4, kernels are custom (`exllama-v2`, `marlin`, `flute`).

---

## 6. AWQ in code (sketch)

The AWQ insight: scale weights by `s` and inputs by `1/s` (mathematically a no-op), then quantize the scaled weights. Pick `s` per channel to **minimize quantization error on the most-activated channels**:

```
y = (x / s) @ (W * s)   # mathematically same as x @ W
```

By choosing `s_c` for each input channel based on activation magnitudes, you "protect" the channels that matter and let the others bear the rounding loss.

In practice, you use the **`autoawq`** library:

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

Then load with vLLM: `vllm serve qwen2.5-7b-awq --quantization awq_marlin`.

---

## 7. GPTQ in code (sketch)

Same idea, different algorithm. Use `auto-gptq`:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
qcfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)
model = AutoGPTQForCausalLM.from_pretrained(model_path, qcfg)

# pass calibration data — ~512 samples is enough
calib = [tok(text, return_tensors='pt') for text in calibration_texts[:512]]
model.quantize(calib)
model.save_quantized('qwen-7b-gptq')
```

Calibration data should match your serving distribution (chat? code? math?) for best quality.

---

## 8. GGUF for CPU / Apple Silicon

```bash
# convert HF model to GGUF (uses llama.cpp tooling)
python convert_hf_to_gguf.py Qwen/Qwen2.5-7B-Instruct --outfile qwen.gguf

# quantize to Q4_K_M
./llama-quantize qwen.gguf qwen-q4km.gguf Q4_K_M

# run
./llama-cli -m qwen-q4km.gguf -p "Hello" -n 200
```

`Ollama` wraps this; `pip install ollama` then `ollama run qwen2.5:7b-instruct-q4_K_M`. Most consumer-LLM apps (LM Studio, Jan, etc.) use this pipeline.

---

## 9. FP8 inference (H100 and beyond)

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct-FP8 --quantization fp8
```

If your model wasn't published in FP8, you can quantize it on-the-fly:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8
```

The `kv-cache-dtype fp8` is independent of weight quantization — it can be on for any model. **FP8 weights + FP8 KV cache** is the highest-throughput config on H100.

---

## 10. Quality vs size — what to expect

Approximate average benchmark drop vs fp16, for Llama / Qwen-class 7B models (your mileage varies):

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
| BitNet 1.58 | 1.58 | 0.1× | (similar to fp16, but requires retraining) |

For most use cases, **Q4_K_M / AWQ-int4 is the sweet spot** — model fits in 4× less memory at <1% quality cost. Below 4 bits, quality starts to wobble unless the model was trained for it.

---

## 11. Quantizing what's *not* the weights

- **KV cache quantization** (chapter 9): fp8 / int8 / int4. Independent of weight quantization. Critical for long context.
- **Activation quantization**: needed for W8A8/W4A4. Requires careful per-token scaling.
- **Optimizer state quantization**: 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`) cuts optimizer state memory 4×. Standard for QLoRA fine-tuning.
- **Gradient quantization**: rarer in PT; used in some distributed-training schemes.

---

## 12. QLoRA: quantization + LoRA = fine-tune giants on consumer GPUs

QLoRA (Dettmers et al. 2023) freezes the base model in NF4 and trains small **LoRA** adapter matrices in bf16. Lets you fine-tune a 70B model on a single 80 GB H100 (or a 7B on a 16 GB consumer GPU).

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

Then a normal training loop. Saves only the LoRA weights (a few hundred MB) at the end.

QLoRA's 2025-2026 successors:
- **DoRA** (decomposed LoRA) — slightly better quality at the same param count.
- **QLoRA + Spectrum / GaLore / VeRA** — even more memory-efficient.

---

## 13. End-to-end example: ship a Qwen2.5-7B AWQ on a single GPU

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

Memory: ~5 GB for weights + KV cache. Throughput: ~8000 tokens/sec on an H100. Quality: indistinguishable from fp16 for chat.

---

## 14. The 2026 cheat sheet

- **GPU serving:** AWQ-int4 (`awq_marlin` kernel) or GPTQ-int4. Both via vLLM.
- **CPU / laptop / Mac:** GGUF Q4_K_M with `llama.cpp` or Ollama.
- **High-end production (H100/B200):** FP8 weights + FP8 KV cache, via vLLM or TensorRT-LLM.
- **Long context:** quantize the KV cache (`--kv-cache-dtype fp8`).
- **Fine-tuning on 1 GPU:** QLoRA (NF4 + LoRA bf16).
- **Train-from-scratch ultra-low-bit:** BitNet b1.58 — niche but real.
- **Always measure** on *your* benchmarks; quantization quality is task-dependent.

---

## Going deeper

- Frantar et al. 2022 — **GPTQ** paper.
- Lin et al. 2023 — **AWQ** paper.
- Dettmers et al. 2023 — **QLoRA**.
- Microsoft 2024 — **BitNet b1.58: The Era of 1-bit LLMs.**
- The `llama.cpp` README's "Quantization formats" table — best quick reference for GGUF.
- vLLM and TensorRT-LLM quantization docs — most up-to-date practical guides.

Next: **[13-mixture-of-experts.md](./13-mixture-of-experts.md)** — sparse models that get bigger without getting slower.
