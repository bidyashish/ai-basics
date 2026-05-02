# 03 · GPU Computing — Performance के लिए Optimize करो

> **TL;DR** GPU एक चीज़ में fast है: हज़ारों numbers पर एक साथ same simple math करना। Modern deep learning ज़्यादातर matrix multiplications है, जो उस pattern में perfectly fit होती हैं। Good performance के लिए, GPU की compute units को busy रखो — उन्हें big matrices feed करके, low precision में math करके, और slow memory पर wait न करके।

## 1. CPU vs GPU एक paragraph में

CPU में strong cores की small number (8-128) होती है जो branchy, sequential code के लिए optimized हैं। GPU में हज़ारों weak cores हैं जो एक साथ कई data elements पर same instruction run करने के लिए designed हैं — **SIMT** (Single Instruction, Multiple Threads)। "इस multiply-add को billion बार करो" के लिए, GPU CPU को breakfast में खा जाता है। "इस JSON को parse करो और decide करो क्या करना है" के लिए, CPU जीतता है।

A modern training-class GPU (H100, A100, RTX 4090) में है:

- **Compute units** ("Streaming Multiprocessors", या SMs) — उनमें से 100+।
- **CUDA cores** हर SM के अंदर — general fp32/fp64 math के लिए।
- **Tensor Cores** हर SM के अंदर — special hardware for **matrix multiply-accumulate** at low precision (fp16, bf16, fp8, int8)। Matmul के लिए CUDA cores से vastly faster।
- **HBM** (high-bandwidth memory) — modern GPUs पर 24-80 GB।
- **L2 cache** — SMs के across share होने वाले tens of MB।
- **Shared memory / L1** — small, very fast, per-SM scratchpad।
- **Registers** — fastest, per-thread।

जब आप PyTorch में `a @ b` लिखते हो, hood के नीचे:

1. PyTorch एक CUDA kernel dispatch करता है (एक function जो GPU पर run होता है)।
2. cuBLAS या Triton/Flash kernel actual matmul करता है, Tensor Cores call करते हुए।
3. Results HBM में वापस लिखे जाते हैं।

---

## 2. Memorize करने worth Numbers

आपको पूरा spec sheet memorize नहीं करना, लेकिन order of magnitude जानना reasoning में help करता है।

| Metric | A100 (40 GB) | H100 (80 GB) | RTX 4090 |
|--------|--------------|--------------|----------|
| FP16/BF16 TFLOPS (Tensor Cores) | 312 | ~1000 | 165 |
| FP8 TFLOPS | n/a | ~2000 | n/a |
| HBM bandwidth | 1.6 TB/s | 3.4 TB/s | 1.0 TB/s (GDDR6X) |
| Memory | 40-80 GB | 80 GB | 24 GB |

Takeaway:
- एक H100 roughly **a quadrillion (10¹⁵) ops per second** bf16 में करता है।
- Memory fast है लेकिन compute faster है। हर weight को HBM से compute units में read करना often bottleneck है।

---

## 3. Compute-bound vs Memory-bound

ये deep learning performance में single most important mental model है।

Kernel **compute-bound** है जब उसकी bottleneck math करना है।
Kernel **memory-bound** है जब उसकी bottleneck memory read/write करना है।

**Arithmetic intensity** = प्रति byte memory move के FLOPs done।

अगर arithmetic intensity high है (big matrix पर matmul → ~`N` FLOPs per byte), आप Tensor Cores को saturate कर सकते हो। अगर low है (elementwise add → 2 FLOPs per byte), तो आप memory bandwidth पर bottleneck होते हो और Tensor Cores idle बैठते हैं।

> **Rule of thumb:** matmul compute-bound है जब matrices big और dense हों। Activations, normalizations, और elementwise ops memory-bound हैं।

इसलिए:
- Bigger batch sizes *per token* faster हैं: आप loaded weights reuse करते हो।
- Kernel fusion (कई small ops को एक में combine करना) big wins करता है — आप HBM round-trips avoid करते हो।
- Flash Attention 2-3× speedup है इसलिए नहीं कि वो math change करता है, बल्कि इसलिए कि वो full attention matrix को HBM में कभी नहीं लिखता।

---

## 4. Memory Hierarchy

```
Registers   →   Shared memory / L1   →   L2   →   HBM   →   System RAM (PCIe)
~0 cycles       ~30 cycles                ~250        ~500     ~slow
KB              ~100 KB per SM          ~50 MB       80 GB    100s GB
```

हर step right ज़्यादा capacity पर slower access है। Good kernels HBM से एक बार shared memory में read करते हैं, वहां जितना possible हो math करते हैं, फिर वापस write करते हैं। Bad kernels HBM से ping-pong करते हैं।

इसलिए frameworks operations को **fuse** करने के लिए इतना fight करते हैं: एक fused `softmax(x @ W) * dropout` kernel `x` को एक बार read करता है और एक बार write करता है, तीन separate trips के बजाय।

---

## 5. Precision: छोटा = Faster (और छोटा)

Tensor Cores lower precision पर faster run होते हैं। Memory bandwidth भी कम होती है (4 bytes per fp32 → 2 per bf16 → 1 per int8 → 0.5 per int4)। तो:

| Format | Bits | Range | Use case |
|--------|------|-------|----------|
| fp32 | 32 | huge | Master weights, reductions, optimizer state |
| tf32 | 19 | fp32 range | A100+ default for fp32 matmul; slightly less precise |
| fp16 | 16 | small (±65k) | Inference, training with loss scaling |
| bf16 | 16 | fp32 range | **Modern training का default** |
| fp8 | 8 | small | Cutting-edge training (H100), inference |
| int8 | 8 | -128 to 127 | Quantized inference |
| int4 | 4 | -8 to 7 | Heavy quantized inference (chapter 12) |

**bf16 training का sweet spot है।** ये rarely overflow करता है और barely underflow। fp16 को careful loss scaling चाहिए।

Inference के लिए, **int4** weight quantization हर जगह है — ये memory को fp32 vs 8× cut करता है minor quality loss के साथ जब सही से किया जाए।

---

## 6. CUDA 30 seconds में

आप usually deep learning के लिए hand से CUDA नहीं लिखते। लेकिन vocabulary आपको profiles और papers पढ़ने में help करती है।

- **Kernel** — एक function जो GPU पर run होता है।
- **Grid** — launch shape: कितने thread blocks run करने हैं।
- **Block** — threads का group जो memory share करते हैं।
- **Warp** — 32 threads जो lockstep में execute होते हैं।
- **Thread** — एक worker जो scalar work करता है।

जब `a @ b` run होता है, matmul kernel launch हो सकता है grid `(M/128, N/128)` और block `(256,)` के साथ — मतलब हर block result का 128×128 tile compute करता है, 256 threads cooperating के साथ।

अगर आप कभी अपने khud के kernels लिखना चाहो, today का pragmatic path **Triton** है (Python-like, JIT-compiled, raw CUDA से much friendlier)। Flash Attention v2 originally Triton में लिखा गया था।

---

## 7. PyTorch Performance Practicalities

### Data efficiently move करो

```python
# host memory को एक बार pin करो, फिर non_blocking transfers use करो
x = x.pin_memory()
x = x.to('cuda', non_blocking=True)
```

`DataLoader` पर `pin_memory=True` और `.to()` पर `non_blocking=True` copy को compute के साथ overlap करते हैं।

### Async kernel launches

CUDA calls तुरंत return करती हैं; काम asynchronously GPU पर run होता है। PyTorch आपके लिए synchronization handle करता है। कुछ time करने के लिए, CUDA events use करो:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... आपका कोड ...
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end), 'ms')
```

GPU code को `time.time()` से time मत करो — आप launch latency measure करोगे, kernel नहीं।

### `torch.compile` use करो

PyTorch 2.x में, `model = torch.compile(model)` automatically graph-level fusion करता है। आप अक्सर बिना उंगली उठाए 1.3-2× पाते हो।

### Sync points avoid करो

ये CPU को GPU की wait करने पर force करते हैं और execution serialize करते हैं:
- `tensor.item()` (scalar Python value)
- `tensor.cpu()` / `.numpy()`
- `print(tensor)` (पहले `.cpu()` call करता है)
- `if condition_on_tensor: ...`

Hot loop के अंदर, इनसे बचो। हर N steps log करो, हर step नहीं।

### Channels-last और Contiguous Layouts

कुछ operators certain layouts पर fastest हैं। Tensor Cores पर convs के लिए, channels-last (`x.to(memory_format=torch.channels_last)`) help करता है। LLMs के लिए, आप mostly इसकी चिंता नहीं करते — matmuls काफ़ी big हैं।

---

## 8. Profiling

जब कुछ slow हो, **measure करो guess नहीं**।

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
) as prof:
    for step in range(5):
        train_step()
        prof.step()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
```

फिर Chrome trace में export करो:

```python
prof.export_chrome_trace('trace.json')
# chrome://tracing खोलो और इसे load करो
```

Chrome trace timeline पर हर kernel दिखाता है। दो patterns देखने हैं:

- **Long gaps with no GPU work** → CPU bottleneck. Likely आपका data loader या Python overhead है।
- **Many tiny kernels close together** → unfused ops। `torch.compile` या rewrite करके bigger ops use करो।

LLM-specific profiling के लिए, **NVIDIA Nsight Systems (`nsys`)** heavy artillery है।

---

## 9. LLMs के लिए Big Wins

Modern LLM training में ज़्यादातर performance gains छोटे set of techniques से आते हैं:

1. **bf16 / fp8 mixed precision** — 2-4× speedup, half memory।
2. **Flash Attention** — O(T²) memory attention को tiled, memory-aware kernel से replace करता है। Sequence length में linear memory, ~2-3× faster।
3. **Activation checkpointing** — store करने के बजाय backward के दौरान activations recompute करो। Memory के लिए compute trade करता है; bigger models train करने देता है।
4. **Gradient accumulation** — effective batch size = micro-batch × num accumulation steps × num GPUs। Optimal batch size hit करने के लिए often essential।
5. **Fused optimizers** — `torch.optim.AdamW(fused=True)` पूरा optimizer step एक kernel में run करता है।
6. **`torch.compile`** — automatic kernel fusion।
7. **Tensor / pipeline / sequence parallelism** — really big models के लिए।
8. **ZeRO / FSDP** — per-GPU memory कम करने के लिए optimizer state, gradients, parameters को GPUs के across shard करो।

हम इन techniques को बाद के chapters में unpack करते हैं।

---

## 10. Common Bottlenecks और क्या करें

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| GPU util < 50% | Data loader too slow या Python overhead | More `num_workers`, simpler dataset, prefetch |
| GPU util 100% लेकिन slow | Memory-bound ops | `torch.compile` से fuse, bf16 use करो |
| OOM | Activations या KV cache | Activation checkpointing, smaller batch, FSDP |
| Slow first iteration | Kernel autotuning, cuDNN benchmarking | Expected, ignore (या एक बार `torch.backends.cudnn.benchmark=True` set करो) |
| कुछ steps के बाद `nan` | fp16 overflow | bf16 use करो, या loss scaling add करो |
| Single GPU saturated, और available | No data parallelism | DDP/FSDP use करो |

---

## 11. एक Worked Example: Forward Pass अपना time कैसे spend करता है

H100 पर एक typical 7B-parameter LLM forward pass के लिए, roughly:

- ~70% time: attention और FFN में matmuls (compute-bound, Tensor Core territory)।
- ~15%: attention softmax + masking (memory-bound, जो Flash Attention attack करता है)।
- ~10%: layer norms, residuals, embeddings (memory-bound)।
- ~5%: kernel-launch overhead, Python।

**Inference** training से much more memory-bound है। Batch size 1 और KV cache के साथ, हर generated token gigabytes weights HBM से read करता है tiny amount of compute के लिए। Memory bandwidth, flops नहीं, bottleneck है। इसलिए **quantization** (chapter 12) single biggest inference win है — आप literally fewer bytes read करते हो।

---

## 12. Mental Shortcut

जब आप wonder करो "ये slow क्यों है?" या "क्या optimize करना चाहिए?", बस ये पूछो:

1. **क्या GPU busy है?** (`nvidia-smi`, profiler.) अगर नहीं, तो पहले data/CPU fix करो।
2. **क्या matrices Tensor Cores use करने के लिए काफी big हैं?** (Big = प्रत्येक side पर thousands.) अगर नहीं, batch size बढ़ाओ।
3. **क्या मैं सही precision में हूं?** Training के लिए bf16 use करो, inference के लिए int4/8 अगर quality रहती है।
4. **क्या मैं too many tiny kernels कर रहा हूं?** `torch.compile` या fused versions use करो।
5. **क्या मैं memory-bound हूं?** Fuse करो, quantize करो, या Flash Attention use करो।

ये पांच-question checklist deep learning में ज़्यादातर performance puzzles solve करता है।

---

## और गहराई से

- *Programming Massively Parallel Processors* (Hwu/Kirk/El Hajj) — textbook।
- Horace He का blog post "Making Deep Learning Go Brrrr From First Principles" — इस topic पर best short read।
- `https://triton-lang.org` पर Triton tutorials — जब आप eventually अपने khud के kernels लिखना चाहो।
- LLM hardware पर Tim Dettmers का blog — pragmatic GPU choice advice।

Next: **[04-data.md](./04-data.md)** — data के बिना, ये सारा hardware कुछ नहीं करता।
