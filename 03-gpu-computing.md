# 03 · GPU Computing — Optimize for Performance

> **TL;DR** A GPU is fast at one thing: doing the same simple math on thousands of numbers at once. Modern deep learning is mostly matrix multiplications, which fit that pattern perfectly. To get good performance, keep the GPU's compute units busy by feeding them big matrices, doing math in low precision, and not waiting on slow memory.

## 1. CPU vs GPU in one paragraph

A CPU has a small number of strong cores (8-128) optimized for branchy, sequential code. A GPU has thousands of weak cores designed to run the same instruction across many data elements at once — **SIMT** (Single Instruction, Multiple Threads). For "do this multiply-add a billion times," the GPU eats the CPU for breakfast. For "parse this JSON and decide what to do," the CPU wins.

A modern training-class GPU (H100, A100, RTX 4090) has:

- **Compute units** ("Streaming Multiprocessors", or SMs) — 100+ of them.
- **CUDA cores** inside each SM — for general fp32/fp64 math.
- **Tensor Cores** inside each SM — special hardware for **matrix multiply-accumulate** at low precision (fp16, bf16, fp8, int8). Vastly faster than CUDA cores for matmul.
- **HBM** (high-bandwidth memory) — 24-80 GB on modern GPUs.
- **L2 cache** — tens of MB shared across SMs.
- **Shared memory / L1** — small, very fast, per-SM scratchpad.
- **Registers** — fastest, per-thread.

When you write `a @ b` in PyTorch, under the hood:

1. PyTorch dispatches a CUDA kernel (a function that runs on the GPU).
2. cuBLAS or a Triton/Flash kernel does the actual matmul, calling Tensor Cores.
3. Results are written back to HBM.

---

## 2. Key numbers to memorize

You don't need to memorize the whole spec sheet, but knowing the order of magnitude helps you reason.

| Metric | A100 (40 GB) | H100 (80 GB) | RTX 4090 |
|--------|--------------|--------------|----------|
| FP16/BF16 TFLOPS (Tensor Cores) | 312 | ~1000 | 165 |
| FP8 TFLOPS | n/a | ~2000 | n/a |
| HBM bandwidth | 1.6 TB/s | 3.4 TB/s | 1.0 TB/s (GDDR6X) |
| Memory | 40-80 GB | 80 GB | 24 GB |

The takeaway:
- One H100 does roughly **a quadrillion (10¹⁵) ops per second** in bf16.
- Memory is fast but compute is faster. Reading every weight from HBM into compute units is often the bottleneck.

---

## 3. Compute-bound vs memory-bound

This is the single most important mental model in GPU performance.

A kernel is **compute-bound** when its bottleneck is doing math.
A kernel is **memory-bound** when its bottleneck is reading/writing memory.

**Arithmetic intensity** = FLOPs done per byte of memory moved.

If arithmetic intensity is high (matmul on a big matrix → ~`N` FLOPs per byte), you can saturate the Tensor Cores. If it's low (elementwise add → 2 FLOPs per byte), you bottleneck on memory bandwidth and the Tensor Cores sit idle.

> **Rule of thumb:** matmul is compute-bound when matrices are big and dense. Activations, normalizations, and elementwise ops are memory-bound.

This is why:
- Bigger batch sizes are faster *per token*: you reuse loaded weights.
- Kernel fusion (combining many small ops into one) wins big — you avoid round-trips to HBM.
- Flash Attention is a 2-3× speedup not because it changes the math, but because it never writes the full attention matrix to HBM.

---

## 4. The memory hierarchy

```
Registers   →   Shared memory / L1   →   L2   →   HBM   →   System RAM (PCIe)
~0 cycles       ~30 cycles                ~250        ~500     ~slow
KB              ~100 KB per SM          ~50 MB       80 GB    100s GB
```

Each step right is more capacity but slower access. Good kernels read once from HBM into shared memory, do as much math there as possible, then write back. Bad kernels ping-pong to HBM.

This is why frameworks fight so hard to **fuse** operations: a fused `softmax(x @ W) * dropout` kernel reads `x` once and writes once, instead of three separate trips.

---

## 5. Precision: smaller = faster (and smaller)

Tensor Cores run faster at lower precision. Memory bandwidth is also reduced (4 bytes per fp32 → 2 per bf16 → 1 per int8 → 0.5 per int4). So:

| Format | Bits | Range | Use case |
|--------|------|-------|----------|
| fp32 | 32 | huge | Master weights, reductions, optimizer state |
| tf32 | 19 | fp32 range | A100+ default for fp32 matmul; slightly less precise |
| fp16 | 16 | small (±65k) | Inference, training with loss scaling |
| bf16 | 16 | fp32 range | **Default for modern training** |
| fp8 | 8 | small | Cutting-edge training (H100), inference |
| int8 | 8 | -128 to 127 | Quantized inference |
| int4 | 4 | -8 to 7 | Heavy quantized inference (chapter 12) |

**bf16 is the sweet spot for training.** It rarely overflows and barely underflows. fp16 needs careful loss scaling.

For inference, **int4** weight quantization is everywhere — it cuts memory 8× vs fp32 with only minor quality loss when done right.

---

## 6. CUDA in 30 seconds

You usually don't write CUDA by hand for deep learning. But the vocabulary helps you read profiles and papers.

- **Kernel** — a function that runs on the GPU.
- **Grid** — the launch shape: how many blocks of threads to run.
- **Block** — a group of threads that share memory.
- **Warp** — 32 threads that execute in lockstep.
- **Thread** — one worker that does scalar work.

When `a @ b` runs, the matmul kernel might be launched with grid `(M/128, N/128)` and block `(256,)` — meaning each block computes a 128×128 tile of the result, with 256 threads cooperating.

If you ever want to write your own kernels, today's pragmatic path is **Triton** (Python-like, JIT-compiled, much friendlier than raw CUDA). Flash Attention v2 was originally written in Triton.

---

## 7. PyTorch performance practicalities

### Move data efficiently

```python
# pin host memory once, then use non_blocking transfers
x = x.pin_memory()
x = x.to('cuda', non_blocking=True)
```

`pin_memory=True` on `DataLoader` and `non_blocking=True` on `.to()` overlap copy with compute.

### Async kernel launches

CUDA calls return immediately; the work runs asynchronously on the GPU. PyTorch handles synchronization for you. To time something, use CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... your code ...
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end), 'ms')
```

Don't time GPU code with `time.time()` — you'll measure the launch latency, not the kernel.

### Use `torch.compile`

In PyTorch 2.x, `model = torch.compile(model)` does graph-level fusion automatically. You often get 1.3-2× without lifting a finger.

### Avoid sync points

These force the CPU to wait for the GPU and serialize execution:
- `tensor.item()` (scalar Python value)
- `tensor.cpu()` / `.numpy()`
- `print(tensor)` (calls `.cpu()` first)
- `if condition_on_tensor: ...`

Inside a hot loop, avoid these. Log every N steps, not every step.

### Channels-last and contiguous layouts

Some operators are fastest on certain layouts. For convs on Tensor Cores, channels-last (`x.to(memory_format=torch.channels_last)`) helps. For LLMs, you mostly don't worry about this — the matmuls are big enough.

---

## 8. Profiling

When something is slow, **measure don't guess**.

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

Then export to Chrome trace:

```python
prof.export_chrome_trace('trace.json')
# open chrome://tracing and load it
```

The Chrome trace shows every kernel on a timeline. Two patterns to look for:

- **Long gaps with no GPU work** → CPU bottleneck. Likely your data loader or Python overhead.
- **Many tiny kernels close together** → unfused ops. `torch.compile` or rewrite to use bigger ops.

For LLM-specific profiling, **NVIDIA Nsight Systems (`nsys`)** is the heavy artillery.

---

## 9. The big wins for LLMs

Most performance gains in modern LLM training come from a small set of techniques:

1. **bf16 / fp8 mixed precision** — 2-4× speedup, half the memory.
2. **Flash Attention** — replaces the O(T²) memory attention with a tiled, memory-aware kernel. Linear memory in sequence length, ~2-3× faster.
3. **Activation checkpointing** — recompute activations during backward instead of storing them. Trades compute for memory; lets you train bigger models.
4. **Gradient accumulation** — effective batch size = micro-batch × num accumulation steps × num GPUs. Often essential to hit the optimal batch size.
5. **Fused optimizers** — `torch.optim.AdamW(fused=True)` runs the entire optimizer step in one kernel.
6. **`torch.compile`** — automatic kernel fusion.
7. **Tensor / pipeline / sequence parallelism** — for really big models.
8. **ZeRO / FSDP** — shard optimizer state, gradients, parameters across GPUs to reduce per-GPU memory.

We unpack each of these in later chapters.

---

## 10. Common bottlenecks and what to do

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| GPU util < 50% | Data loader too slow or Python overhead | More `num_workers`, simpler dataset, prefetch |
| GPU util 100% but slow | Memory-bound ops | Fuse with `torch.compile`, use bf16 |
| OOM | Activations or KV cache | Activation checkpointing, smaller batch, FSDP |
| Slow first iteration | Kernel autotuning, cuDNN benchmarking | Expected, ignore (or set `torch.backends.cudnn.benchmark=True` once) |
| `nan` after a few steps | fp16 overflow | Use bf16, or add loss scaling |
| Single GPU saturated, more available | No data parallelism | Use DDP/FSDP |

---

## 11. A worked example: how a forward pass spends its time

For a typical 7B-parameter LLM forward pass on an H100, roughly:

- ~70% of time: matmuls in attention and FFN (compute-bound, Tensor Core territory).
- ~15%: attention softmax + masking (memory-bound, what Flash Attention attacks).
- ~10%: layer norms, residuals, embeddings (memory-bound).
- ~5%: kernel-launch overhead, Python.

**Inference** is much more memory-bound than training. With batch size 1 and a KV cache, every generated token reads gigabytes of weights from HBM to do a tiny amount of compute. Memory bandwidth, not flops, is the bottleneck. This is why **quantization** (chapter 12) is the single biggest inference win — you literally read fewer bytes.

---

## 12. The mental shortcut

When you wonder "why is this slow?" or "what should I optimize?", just ask:

1. **Is the GPU busy?** (`nvidia-smi`, profiler.) If not, fix data/CPU first.
2. **Are matrices big enough to use Tensor Cores?** (Big = thousands on each side.) If not, increase batch size.
3. **Am I in the right precision?** Use bf16 for training, int4/8 for inference if quality holds.
4. **Am I doing too many tiny kernels?** Use `torch.compile` or fused versions.
5. **Am I memory-bound?** Fuse, quantize, or use Flash Attention.

That five-question checklist solves most performance puzzles in deep learning.

---

## Going deeper

- *Programming Massively Parallel Processors* (Hwu/Kirk/El Hajj) — the textbook.
- Horace He's blog post "Making Deep Learning Go Brrrr From First Principles" — the best short read on this topic.
- The Triton tutorials at `https://triton-lang.org` — when you eventually want to write your own kernels.
- Tim Dettmers' blog on LLM hardware — pragmatic GPU choice advice.

Next: **[04-data.md](./04-data.md)** — without data, all this hardware does nothing.
