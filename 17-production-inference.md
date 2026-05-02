# 17 · Production Inference — Serving 10,000 Requests

> **TL;DR** "Serve 10,000 requests" can mean three different problems: **10k QPS sustained**, **10k concurrent**, or **10k total over a day**. Each demands a different design. The 2026 stack is: **vLLM** or **SGLang** as the engine, **fp8 weights + fp8 KV** on H100/B200, **continuous batching with PagedAttention**, **prefix caching** to share system prompts, **speculative decoding** for 2× decode, **prefill/decode disaggregation** at very high scale, **prefix-aware sticky load balancing**, and **Kubernetes / Ray Serve** for autoscaling. With Qwen 3.6-27B FP8 on 8× H100, you can sustain ~3,500 chat QPS per node; ten thousand QPS = roughly 3 nodes, plus 1-2 for headroom. **Capacity isn't magic — it's arithmetic.**

---

## 1. Three definitions of "10,000 requests"

Always pin this down before building.

| Interpretation | What it really means | Hardware ballpark (Qwen3.6-27B fp8) |
|----------------|----------------------|--------------------------------------|
| **10k QPS sustained** | 10,000 chat completions per second, indefinitely | 3-5 H100 nodes (8× each) running vLLM in parallel |
| **10k concurrent** | 10k chats open at once, each user typing slowly | 1-3 nodes — concurrency is "in-flight," batched together |
| **10k requests total / day** | Bursty, low overall load | 1 GPU is plenty; focus on cold-start and reliability |

Most production deployments mix all three: bursts up to N concurrent during traffic peaks, M QPS sustained, K total per day. Plan for **peak QPS** and pay for **average QPS**.

---

## 2. The right unit of cost: tokens, not requests

A "request" is meaningless until you know how many tokens it processes.

For a chat workload, a typical request is:
- **Prompt**: ~500-2000 tokens (system + history + new user message).
- **Completion**: 100-500 tokens (avg) for chat; 1000-5000 for reasoning.

So 10,000 chat requests/sec ≈ **10-30M tokens/sec** of work. That number — total tokens/sec — is what GPUs serve, not "requests."

Throughput rules of thumb (Qwen 3.6-27B FP8 on **one 8×H100 node**):

- **Prefill**: ~150,000-250,000 tokens/sec.
- **Decode**: ~3,000-6,000 tokens/sec aggregate (across all in-flight users).

Decode is ~30-50× slower per token because it's memory-bound (chapter 9). The split of your traffic between prompt and completion drives capacity.

### The capacity formula

```
required_GPUs ≈ (avg_QPS × avg_prompt_tokens / prefill_tokens_per_sec_per_GPU)
              + (avg_QPS × avg_completion_tokens / decode_tokens_per_sec_per_GPU)
              + headroom (×1.3-1.5)
```

Plug in your numbers and you have an order-of-magnitude estimate that's usually within 30% of reality. The remaining 30% comes from queueing, retries, and tail-latency requirements.

---

## 3. SLOs you actually need

You can't optimize what you don't measure. Set these per route:

| SLO | What it is | Target for chat | Target for agents |
|-----|-----------|-----------------|---------------------|
| **TTFT** (time-to-first-token) | from request enter → first decoded token out | P95 < 500 ms | P95 < 2 s |
| **TPOT** (time-per-output-token) | streaming token interval | P95 < 50 ms (~20 tok/s) | P95 < 80 ms |
| **End-to-end latency** | request → final byte | P95 < 8 s for 200-tok reply | P95 < 30 s for 1000-tok reply |
| **Availability** | non-5xx rate | 99.9% | 99.5% |
| **Error budget** | how much you can spend per month | ~43 min/month at 99.9% | ~3.6 hr at 99.5% |

TTFT is dominated by **prompt prefill** plus **queue wait**. TPOT is dominated by **decode batch size** (bigger batch → slower per-user TPOT, higher aggregate throughput). The TTFT/TPOT trade-off is the single most important capacity dial.

---

## 4. The stack you'll actually run in 2026

### Engine layer (the GPU does the work)

| Engine | When to pick | Strengths |
|--------|--------------|-----------|
| **vLLM** (Berkeley + community) | Default. Battle-tested, OpenAI-compatible, huge model coverage | PagedAttention, continuous batching, prefix cache, speculative, MoE, multimodal, FP8 |
| **SGLang** (LMSYS) | Chat with heavy prefix sharing, structured outputs, tool calling | RadixAttention, fastest prefix-cache hit-rate; native constraint decoding |
| **TensorRT-LLM** (NVIDIA) | Max throughput on NVIDIA, willing to do build steps | Lowest TTFT, best fp8 / NVFP4 kernels |
| **TGI** (HuggingFace) | Drop-in for HF stack | Slightly slower than vLLM but simpler ops |
| **DeepSeek SGL / DeepEP**| Frontier MoE serving | Best MoE expert-parallelism kernels |

For a **green-field 10k-QPS chat service in 2026, default to vLLM or SGLang**. Use TensorRT-LLM if you've already got the build/deploy infra.

### Frontend layer (routing, queueing, batching across nodes)

- **Ray Serve** — multi-node, autoscaling, KV-aware routing. Ray's `vLLMReplica` integrates directly.
- **KServe** — Kubernetes-native, multi-tenant, A/B traffic split.
- **NVIDIA NIM / Triton** — production NVIDIA stack with multi-model rollouts.
- **Custom**: a small FastAPI / Go service in front of vLLM nodes can be enough when traffic is uniform.

### Orchestration

- **Kubernetes** with **GPU node pools** and **HPA** (Horizontal Pod Autoscaler) keyed on QPS or queue depth.
- **Cluster autoscaler** to add nodes during peaks (slow — minutes — so over-provision the floor).

### Model lifecycle

- **HuggingFace Hub** as the source of truth for weights.
- **Object store** (S3/GCS) for quantized variants.
- **Image registry** with the model + engine baked in (faster cold start than pulling from HF on boot).

---

## 5. The big inference levers (in order of impact)

### Lever 1 — Quantize the model

Switching `bf16 → FP8` weights (H100/B200) typically yields:

- ~2× throughput.
- ~50% memory savings (lets you fit more concurrent users).
- < 0.5% quality drop after a brief calibration.

```bash
# vLLM with FP8 weights and FP8 KV cache
vllm serve Qwen/Qwen3.6-27B \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

`--gpu-memory-utilization 0.9` lets vLLM use 90% of HBM for weights + KV cache, reserving 10% for activations and headroom. Push higher only if you've measured you don't OOM under burst load.

For laptop / CPU serving, switch to **GGUF Q4_K_M** with `llama.cpp` (chapter 12).

### Lever 2 — KV cache management

The KV cache is *the* memory hog at long context.

- **`--kv-cache-dtype fp8`** — 2× memory cut, free.
- **`--kv-cache-dtype fp8_e4m3` / `fp8_e5m2`** — vLLM fp8 KV variants; `e4m3` more accurate, `e5m2` more range.
- **PagedAttention** — automatic; vLLM splits the cache into pages of 16 tokens for waste-free packing. Memory utilization goes from ~60% to ~96%.
- **CPU/SSD KV offload** (vLLM 0.7+, SGLang) — evict the inactive part of the cache to CPU RAM or NVMe. Unlocks 1M-token serving.
- **KV cache compression** (`H2O`, `SnapKV`, `KIVI`) — drop low-importance tokens. Research-grade; quality varies.

### Lever 3 — Prefix caching

In production chat, the **system prompt is identical across millions of requests**. Without prefix caching, you re-prefill that 1000-token system prompt for every new conversation. With prefix caching, you compute it **once** and every subsequent request that shares the prefix gets it for free.

- **vLLM**: enable with `--enable-prefix-caching`. Hash-based, exact-match prefixes.
- **SGLang RadixAttention**: builds a radix tree over all live prefixes; sharing extends to *any* common prefix, not just the system message — perfect for few-shot prompts and multi-turn chat.

Prefix caching often gives **2-5× higher effective throughput** for chat workloads with long, shared prompts. It's the single biggest free win in the stack.

### Lever 4 — Continuous (in-flight) batching

vLLM and SGLang already do this. Each step:

1. Take the next ready batch (mix of fresh prefill + ongoing decode).
2. Run one forward pass.
3. Free finished sequences, schedule new ones.

Tunables:
- **`--max-num-seqs`** — concurrency cap. Higher = better throughput, worse TPOT.
- **`--max-num-batched-tokens`** — total tokens per step. Caps prefill burst.
- **Chunked prefill** (`--enable-chunked-prefill`) — split a giant prompt across decode steps so other users don't starve. **Strongly recommended in production.**

### Lever 5 — Speculative decoding

The decode phase is memory-bound (chapter 9). A small **draft model** proposes 4-8 tokens; the big model verifies them in one forward pass.

vLLM with a draft model:
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"model":"Qwen/Qwen3.6-1.7B","num_speculative_tokens":5}'
```

Or use the model's own MTP head (Qwen 3.6, DeepSeek-V3):
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Or **EAGLE-3**: a tiny adapter trained on hidden states; 2-3× speedup on chat workloads:
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"method":"eagle3","model":"path/to/eagle3-adapter"}'
```

Caveat: speculative decoding helps decode only. If your traffic is prefill-heavy, focus on chunked prefill instead.

### Lever 6 — Prefill/Decode disaggregation (at scale)

A frontier 2025-2026 trick used by DeepSeek's serving stack and now in vLLM (`--enable-disagg`):

- Run **prefill nodes** (compute-bound, big TP, fp8) and **decode nodes** (memory-bound, small TP, prefix cache) on separate GPUs.
- Move the freshly computed KV cache from prefill node → decode node over RDMA / NVLink.
- TTFT and TPOT each get optimized independently.

Worth the complexity above ~2,000 QPS or for very long contexts. Below that, don't bother.

### Lever 7 — Tensor / expert parallelism

For models that don't fit on one GPU:

- **TP (Tensor Parallel)**: shard each weight across `N` GPUs. Add latency for the all-reduce in each layer; gain memory.
- **PP (Pipeline Parallel)**: split layers across `N` GPUs. Worse for latency; useful for very deep models.
- **EP (Expert Parallel)**: shard MoE experts. Necessary for large MoEs (DeepSeek-V3, Gemma 4 26B-A4B).

Pick the smallest TP that fits the model + KV cache. Bigger TP than necessary always loses throughput per GPU.

For a 27B fp8 model: TP=2 fits comfortably on 2× H100. For 35B-A3B with 256 experts: EP=4 + TP=2.

### Lever 8 — Choose the right model

- **Two smaller models at the same cost** often beat one bigger model. Route easy queries to a 4B; route hard ones to a 27B.
- **Mixture of Experts** at very high QPS — same memory budget but cheaper per token.
- **Dense smaller** for predictable latency.

Multi-model routing is a 2026 default for serious deployments.

---

## 6. Memory bandwidth — HBM3, HBM4, and what users actually feel

Before you spec a fleet, understand the single most counter-intuitive truth of LLM serving: **for inference, GPU memory bandwidth (HBM) matters more than chip TFLOPS**. A faster chip with the same memory will not give you faster tokens. Most teams pay for the wrong number.

### Why decode is bandwidth-bound, not compute-bound

In the **decode** phase (the part the user actually waits on, token by token), the GPU performs *almost no compute per byte loaded*. For a single sequence each new token requires reading **every weight** of the model from HBM into Tensor Cores, doing a tiny matmul against a single-row activation, and writing back a single-row output. The Tensor Cores sit underutilized; the bottleneck is the HBM pipe.

The rough upper bound on **per-user decode tokens/sec** is exactly:

```
tokens_per_sec_per_user  ≤  HBM_bandwidth (bytes/s)  /  model_size (bytes)
```

That's it. No FLOPs term. Adding more compute helps prefill (compute-bound) and helps batched decode (when bandwidth amortizes across many users), but it does *not* help one user generate faster.

### HBM generations and the actual numbers

| GPU | Memory | HBM gen | Bandwidth | BF16 TFLOPS | FP8 TFLOPS |
|-----|--------|---------|-----------|-------------|------------|
| A100 (2020) | 40-80 GB | HBM2e | 1.6-2.0 TB/s | 312 | — |
| H100 (2022) | 80 GB | HBM3 | 3.35 TB/s | ~1000 | ~2000 |
| H200 (2024) | 141 GB | HBM3e | 4.8 TB/s | ~1000 | ~2000 |
| B200 (2024-2025) | 192 GB | HBM3e | 8.0 TB/s | ~2250 | ~4500 |
| MI300X (AMD, 2024) | 192 GB | HBM3 | 5.3 TB/s | ~1300 | ~2600 |
| GB300 / B300 (2025-2026) | 288 GB | HBM3e | 8.0 TB/s | similar | similar |
| Rubin (Nvidia, late 2026) | 288 GB | **HBM4** | 13+ TB/s expected | — | — |
| Rubin Ultra (2027) | 1 TB+ | **HBM4** | 20+ TB/s expected | — | — |

Notice what jumped between H100 → H200: the FLOPs are nearly identical, but bandwidth went up 43%. **For inference, H200 is roughly 40% faster per user than H100 — entirely because of HBM3e.** That's a real-world result you can verify on any vLLM benchmark. The chip didn't get smarter; the pipe got fatter.

The same story plays out from H200 → B200 (~67% bandwidth jump) and from B200 → Rubin/HBM4 (~60-100% bandwidth jump).

### Concrete tokens/sec ceilings

Take a few real models and compute the single-user ceiling. We use bf16 weights for clarity; halve the model size for fp8.

```
Llama-3.1-8B (bf16, ≈ 16 GB):
  H100 (3.4 TB/s):  16 / 3400 = 4.7 ms/tok  → ~213 tok/s ceiling
  H200 (4.8 TB/s):  3.3 ms/tok  → ~303 tok/s
  B200 (8.0 TB/s):  2.0 ms/tok  → ~500 tok/s
  HBM4 (~13 TB/s):  1.2 ms/tok  → ~810 tok/s

Qwen 3.6-27B (fp8, ≈ 27 GB):
  H100:  27 / 3400 = 7.9 ms/tok  → ~127 tok/s
  H200:  5.6 ms/tok  → ~178 tok/s
  B200:  3.4 ms/tok  → ~296 tok/s
  HBM4:  2.1 ms/tok  → ~480 tok/s

Llama-3.1-70B (fp8, ≈ 70 GB):
  H100 (won't fit one card; assume TP=2 → effective 6.7 TB/s aggregate, but with overhead): ~80 tok/s realistic
  B200 (fits one card, 8 TB/s): ~110 tok/s
  HBM4: ~190 tok/s
```

In practice you'll see **60-80% of the theoretical ceiling** because of attention reads, KV cache traffic, kernel launch overhead, and Python glue. But the ratios between GPUs hold almost exactly.

### What a coder feels in tokens/sec

The user *feels* this number directly when watching tokens stream into their editor or chat:

| Tokens/sec at user | Subjective feel | Effect on workflow |
|--------------------|-----------------|--------------------|
| 10 tok/s | "Painfully slow" | Will switch tabs, attention lost |
| 25 tok/s | "Reading along" | Tolerable for short answers only |
| 50 tok/s | "Comfortable" | Good for chat, slow for code |
| 80 tok/s | "Native feel" | Cursor / IDE assistant sweet spot |
| 150 tok/s | "Instant" | Long answers feel like Google search |
| 300+ tok/s | "Faster than I read" | Reasoning chains, long agents |

A coder using a 27B model is the canonical case. On H100 you serve them at ~120 tok/s — borderline native. On B200 it's ~250 tok/s — clearly native. On HBM4 hardware it's ~400 tok/s — they stop noticing the model entirely. **The hardware refresh you pay for changes whether your product feels broken or magical.**

For **reasoning models** the effect is even bigger. A `<think>...</think>` block can be 1,000-5,000 tokens. At 50 tok/s that's a 20-100 second wait before the user sees the *real* answer. At 300 tok/s it's 3-15 seconds. Same model, same quality — radically different product.

For **agents** (Claude Code, Cursor agent, Devin-style), each step in the loop is a full inference. A 30-step task at 50 tok/s vs 300 tok/s is the difference between 5 minutes and 50 seconds.

### Why batching changes the picture

Everything above was for **a single user**. When you batch B users in one decode step, you read the weights once and use them for B sequences. So:

```
aggregate_tokens_per_sec  ≈  HBM_bandwidth / model_size  ×  effective_batch_size
```

Once batch fills up, you become compute-bound, and *then* TFLOPs matter. This is the regime your throughput numbers live in.

The catch: per-user TPOT actually gets worse with bigger batch. So batching trades **revenue** (aggregate tokens) against **UX** (per-user speed). Most products pick a `max_num_seqs` that hits TPOT targets.

### What to do with this

1. **Pick GPUs by $/(TB/s) and $/GB**, not $/TFLOP, for inference fleets. For training, the calculus reverses — there compute matters most.
2. **For solo-user latency** (small batch): run on the highest-bandwidth GPU you can afford. H200/B200 are the 2026 sweet spots.
3. **For batched throughput**: lots of medium-bandwidth GPUs may beat fewer high-bandwidth ones, depending on price/perf.
4. **Quantization is a bandwidth multiplier.** Going fp16 → fp8 halves bytes-per-weight, doubling effective HBM bandwidth for the same physical GPU. fp8 → int4 doubles it again. Quantization isn't only about memory size; it's about how fast tokens stream out.
5. **Watch the HBM4 wave.** Rubin-class GPUs land late 2026 / early 2027 with ~2× H200 bandwidth. Plan refresh cycles around it; on inference workloads it's a step change, not an incremental bump.
6. **Tell your users tokens/sec, not TFLOPS.** When deciding between two providers, the only number that matters to them is how fast their tokens arrive.

A working slogan for inference engineers in 2026: **"compute is rented, bandwidth is destiny."**

---

## 7. A concrete reference design — 10,000 chat QPS on Qwen 3.6-27B

Assumptions:
- avg prompt 1500 tokens (heavy system prompt, prefix-cached after first hit).
- avg completion 300 tokens.
- TTFT P95 target: 500 ms.
- TPOT P95: 50 ms.
- Region: single AWS region with `p5.48xlarge` (8× H100 80 GB) nodes.

### Sizing

Per 8×H100 node (Qwen 3.6-27B fp8, TP=4, two replicas per node):

- Prefill throughput ~250k tokens/sec (with chunked prefill).
- Decode throughput ~5k tokens/sec aggregate at `max_num_seqs=512`.
- With prefix caching (~80% hit rate on system prompt) → effective prefill ~3k tokens/req new.

Required compute / sec:
- New prefill: 10k QPS × 3,000 effective new tokens = 30M tok/s prefill → **~120 nodes**? No — most prompts share the system prefix and the *active* prefill load drops dramatically. With realistic 80% prefix-cache hit, **active prefill ~6M tok/s → ~24-30 nodes** for prefill.
- Decode: 10k QPS × 300 tokens = 3M tok/s decode → **~600 nodes** at 5k/node.

Both numbers say decode is the binding constraint by an order of magnitude. Mitigations:
- **Speculative decoding** (MTP head): 2× → 1.5M tok/s effective decode → 300 nodes. Still big.
- Reality check: this assumes everyone gets a fresh 27B response. Most production workloads route a large fraction of traffic to **smaller models** (a 4B routed for simple queries) and reserve the 27B for hard prompts. With a 70/30 split:
  - 70% to a Qwen 3.6-4B-class model (~30k tok/s decode/node) → 70k tok/s decode load → ~3 nodes.
  - 30% to 27B (~5k × 2x spec → 10k tok/s decode/node) → 900k tok/s decode load → 90 nodes.

Even with routing, "10k QPS of full-quality flagship-model chat completions in real time" is **a hundred-GPU operation**. That number tracks with what frontier API providers actually run.

If your 10k-QPS goal is more like "10k mostly-light chat per second with occasional reasoning calls," **3-8 nodes** is plenty. Be honest about your real distribution.

### Topology

```
         ┌────────────┐
         │  Frontend  │  Edge LB (e.g., Cloudflare / ALB)
         └─────┬──────┘
               │
       ┌───────▼──────────┐
       │ Router / Gateway │   prefix-aware sticky LB,
       │   (Ray Serve /   │   request classifier (small/big),
       │   custom Go svc) │   rate limiting, auth, retries
       └───┬──────────┬───┘
           │          │
   ┌───────▼──┐  ┌────▼─────┐
   │ Small    │  │ Flagship │
   │ tier     │  │ tier     │
   │ 4B fp8   │  │ 27B fp8  │
   │ N nodes  │  │ M nodes  │
   └──┬───────┘  └─┬────────┘
      │            │
      ▼            ▼
   vLLM/SGLang   vLLM/SGLang
   replicas      replicas
   per node      per node
```

Key design choices:

- **Two model tiers** with a fast classifier (a 0.5B model or even logistic regression on prompt features) routing each request.
- **Prefix-aware routing** — send requests with the same system prompt to the same node so its prefix cache stays hot. SGLang has this; for vLLM, use a sticky hash on `system_prompt_hash`.
- **Static replicas as the floor**, autoscale on top. Cold-starting a replica is 60-180 seconds; you cannot autoscale it on QPS alone.

### Memory layout per node

For 27B fp8 on 8×H100 with TP=4, 2 replicas/node:
- Weights: ~28 GB / replica (sharded ~7 GB / GPU).
- KV cache: budget ~30 GB / replica.
- Activations + headroom: ~10 GB.
- Total per replica: ~70 GB / 4 GPUs = ~17.5 GB / GPU on 80 GB cards. Plenty of room for higher concurrency.

### Configuration

```bash
# vLLM, two replicas per 8×H100 node
vllm serve Qwen/Qwen3.6-27B \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 --port 8000 --served-model-name qwen3.6-27b
```

Run two `vllm serve` processes per node, pinned to GPUs 0-3 and 4-7 respectively (`CUDA_VISIBLE_DEVICES`). Each replica is independent, so a crash isolates blast radius.

### Routing layer (sketch)

```python
# Ray Serve — prefix-aware routing
from ray import serve
import hashlib

@serve.deployment(num_replicas=10)
class Router:
    def __init__(self):
        self.small = serve.get_app_handle("qwen-4b")
        self.big   = serve.get_app_handle("qwen-27b")

    async def __call__(self, req):
        prompt = req["messages"]
        # 1. classify
        is_hard = self.classify(prompt)            # tiny model or rule
        backend = self.big if is_hard else self.small
        # 2. prefix-aware sticky key
        sys = prompt[0]["content"] if prompt[0]["role"] == "system" else ""
        sticky = hashlib.sha256(sys.encode()).hexdigest()[:16]
        # 3. forward
        return await backend.options(stream=True).remote(req, sticky_key=sticky)
```

---

## 8. Observability — what to log and watch

Every serving system needs these dashboards:

### Per-request

- `ttft_ms` — TTFT histogram, P50/P95/P99.
- `tpot_ms` — per-token decode latency.
- `total_tokens_in`, `total_tokens_out`.
- `route_decision` — which model tier handled it.
- `prefix_cache_hit` — boolean per request.
- `error` — categorized.

### Per-engine (per vLLM replica)

- `gpu_utilization` (NVML).
- `gpu_memory_used`.
- `kv_cache_usage_pct` — if it crosses 95% you'll start dropping requests.
- `num_running_seqs` — current concurrency.
- `num_waiting_seqs` — queue depth.
- `tokens_per_sec_prefill`, `tokens_per_sec_decode`.
- `prefix_cache_hit_rate`.

### Cluster

- QPS by route, by status.
- End-to-end latency.
- Node health, OOM events, GPU thermal events.
- Cost-per-million-tokens.

Stack: **Prometheus + Grafana** for metrics, **OpenTelemetry** for traces, **Loki** for logs. vLLM exposes Prometheus metrics out of the box (`/metrics`).

### Alerts that earn their keep

- TTFT P95 > target for 5 min.
- KV-cache usage > 95% on any replica.
- Queue depth > N for 30 sec.
- GPU temperature > 88 °C.
- 5xx rate > 1%.
- Cost-per-token > 1.5× baseline (drift detector).

---

## 9. Reliability and graceful degradation

10k QPS = something fails every minute. Designs that survive:

- **Multiple replicas per region** with anti-affinity scheduling.
- **Multi-region deploy** with active/active DNS or anycast.
- **Health checks** that hit `/v1/models` *and* a smoke completion (not just a TCP ping — vLLM can be alive but stuck).
- **Backpressure**: when queue depth > threshold, return 429 with `Retry-After`. Don't accept work you can't do.
- **Timeouts** at every layer: client → LB (30s) → router (60s) → engine (60s). If TTFT > timeout, the token-stream still flows for an existing connection, but new ones go elsewhere.
- **Circuit breakers**: stop routing to a hot replica for 30s if its 5xx rate spikes.
- **Idempotent retries** with `request_id` headers (vLLM doesn't have built-in idempotency — caller must own it).
- **Graceful degradation tiers**: under heavy load, drop optional features (no streaming → buffered, smaller models for free tier, no thinking-mode).

---

## 10. Autoscaling

GPU autoscaling has two cruel limits:
1. **Cold start ~ 60-180 s** (image pull + weight load + warmup).
2. **GPU nodes are scarce** — your cloud provider may not have capacity right now.

Implications:

- **Set a generous floor** (`min_replicas`) sized for your typical hour-of-day load, not the trough.
- **Scale up early** on leading indicators (queue depth, TTFT trending up) — not only on QPS.
- **Use spot capacity for headroom**, on-demand for the floor.
- **Pre-warm** new replicas with synthetic traffic before they take real load.
- **Scale down slowly** (5-10 min cooldown) to avoid flapping.

Ray Serve, KServe, and vLLM Production Stack all have first-class autoscaling that respects these patterns.

---

## 11. Cost and how to think about it

Rough 2026 unit economics (varies wildly):

- 1 H100 80 GB on-demand ≈ $2.5-4/hour.
- 1 8×H100 node ≈ $20-30/hour.
- For Qwen 3.6-27B fp8, ~5,000 decode tokens/sec/node → ~18M tokens/hour → ~$1.50 per million output tokens at full utilization.

The economics are dominated by **utilization**. A 50% utilized fleet at twice the cost is the same as a 100% utilized one. **Autoscale aggressively, batch hard, share prefix cache.**

Cost levers, ranked:

1. Quantization (fp8 → 2× throughput) — 50% cost cut.
2. Prefix caching — 30-70% cut for chat with system prompts.
3. Speculative decoding — 30-50% cut on decode.
4. Smaller-model routing — 40-80% cut on the routed traffic.
5. Spot capacity — 50-70% cut for elastic fleet.
6. Multi-tenancy on the same engine — 10-20% cut by amortizing fixed overhead.

Stack them and a credible production deployment lands at **~5× cheaper than naive bf16 single-tier**.

---

## 12. Token caching — don't pay for the same context twice

A correctly designed product never charges itself (or its users) twice for the same tokens. Caching shows up at three layers, and a serious deployment uses all three.

### The three cache layers

1. **Provider-side cached input** (when calling an external API). Anthropic, OpenAI, Google, DeepSeek, and others give big discounts for input tokens that match a previously-sent prefix.
2. **Engine-side prefix cache / KV reuse** (when self-hosting). vLLM `--enable-prefix-caching` and SGLang RadixAttention turn prompt-prefix matches into free hits — no compute, no billing in your own infra.
3. **Application-level response cache** (above the model). For deterministic, repeat queries (FAQ, structured extraction, classification), store the *output* and skip the model entirely.

### Provider cached-input pricing (2026, public list rates)

These shift, but the orders of magnitude are stable.

| Provider | Cached-read discount | Cache-write premium | Default TTL |
|----------|---------------------|---------------------|-------------|
| Anthropic Claude | **−90%** (10% of input price) | +25% on first write | 5 min (extendable to 1 hr) |
| OpenAI | **−50%** (cached input half-price) | none | ~5-10 min sliding |
| Google Gemini | **−75% to −80%** (context caching) | small storage fee | 1 hr default, configurable |
| DeepSeek | **−90%** | none | 24 hr (KV-cache on disk) |
| Mistral / Cohere | **−50%** | none | varies |

A **cache-write** is the first time you send a prefix; the provider stores the KV cache for it. Subsequent requests that hit the same prefix pay the cached-read rate. Most providers identify cache by hash of the prefix bytes, so any single-byte difference busts the cache.

### Structuring prompts for maximum cache hits

The prompt layout is the biggest lever. Stable content first, dynamic content last. The cache is a prefix tree — the moment a byte differs, the rest of the prompt is "new" and gets billed at full rate.

```python
# WRONG — dynamic timestamp at the start blows the cache every request
messages = [
    {"role": "system",
     "content": f"Today is {datetime.utcnow()}. " + system_prompt}
]

# RIGHT — stable system prompt first, dynamic content last
messages = [
    {"role": "system", "content": stable_system_prompt},     # cached
    {"role": "user",   "content": "<docs>" + retrieved_docs + "</docs>"},
    *prior_turns,                                              # mostly cached
    {"role": "user",   "content": current_user_msg},           # new, billed
]
```

A cleaner mental model: imagine your prompt as **frozen** + **liquid**. Frozen at the front (system prompt, tool schemas, few-shot examples, retrieved documents that persist across turns), liquid at the back (the user's latest message). Make the frozen layer as long as possible — it's the part that gets cached.

### Anthropic-style explicit cache control

Anthropic's API requires you to mark cache breakpoints — useful for telling the system "everything before this is reusable":

```python
client.messages.create(
    model="claude-opus-4-7",
    system=[
        {"type": "text",
         "text": stable_system_prompt + tools_schema + few_shot_examples,
         "cache_control": {"type": "ephemeral"}},   # marks the breakpoint
    ],
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": retrieved_docs,
             "cache_control": {"type": "ephemeral"}},   # second cacheable block
            {"type": "text", "text": user_message},     # not cached
        ]}
    ],
)
```

Up to 4 cache breakpoints are allowed; the API caches each prefix segment independently. You only pay full price for the suffix after the last cached match.

### OpenAI-compatible: cache hits are automatic

OpenAI and most OpenAI-compatible APIs (including vLLM's) cache automatically on hash of the prefix — **you don't ask, you just structure prompts so prefixes repeat verbatim**. The response includes a `prompt_tokens_details.cached_tokens` field telling you how many were cached so you can verify hit rate.

```python
resp = openai.chat.completions.create(model=..., messages=...)
print(resp.usage.prompt_tokens_details.cached_tokens)  # check this metric
```

### Self-hosted: the same cache, free

When you serve your own model, prefix caching is purely an HBM tenancy game. With vLLM:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --enable-prefix-caching \
    --enable-prefix-caching-hash sha256
```

All identical prefixes across in-flight or recent requests reuse the same KV blocks. Hit rates of 70-95% on stable system prompts are normal. **Self-hosted prefix-cache hits cost you nothing — zero compute, zero billing, zero latency for the cached bytes.**

### Application-level response caching

For requests that always produce the same output, cache the *output*:

```python
import hashlib, redis

r = redis.Redis()
def cached_completion(model, messages, ttl=3600):
    key = "llm:" + hashlib.sha256(
        (model + json.dumps(messages, sort_keys=True)).encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    out = call_llm(model, messages)
    r.setex(key, ttl, json.dumps(out))
    return out
```

Best targets: structured extraction, classification, deterministic single-turn FAQ bots, normalization tasks. Don't cache anything personalized (user's name, account state) without namespacing the key by user.

### Things that quietly break caching (and your budget)

- **Timestamps, request IDs, or session IDs in the system prompt.** Even one byte busts every subsequent token.
- **Re-ordering messages or tools** between requests.
- **Mixed `temperature` / `top_p` / `seed`** with API-level response caching (different sampling = different output).
- **Different models in routing.** A 4B-tier and a 27B-tier do not share caches.
- **Whitespace differences from inconsistent serializers.**
- **Partial JSON arrays for tool definitions.** Pin a canonical order, or you get 100% miss.

A cheap monitoring habit: log `cached_tokens / prompt_tokens` per route and alert if it drops below your baseline by more than 20%. That's almost always a regression in prompt structure.

### Realistic savings

For a typical chat product (1500-token system + 500-token rolling history + 200-token new user message), about **80% of input tokens are cacheable**. With Anthropic's 90% cached-read discount, this turns into **~70% off the input bill**. With self-hosted prefix-cache + fp8, the inputs are essentially free; only output tokens and the *new* prefix segment cost you.

This is the single largest cost lever beyond quantization. Build for it from day one.

---

## 13. Pricing strategy and per-user-type token economics

Once caching is in place, pricing becomes a question of: how many tokens does each user type *actually* consume per day, and how much margin can you extract while still being competitive? Here is the calibration table to start from.

### Token consumption per archetype (2026 calibration)

These are realistic averages from public usage data and instrumented production deployments. Your numbers will vary, but the *ratios* between archetypes are surprisingly stable.

| Archetype | Avg input/req | Avg output/req | Reqs/active hour | Active hrs/day | Daily input tok | Daily output tok | Monthly tokens (in+out) |
|-----------|---------------|----------------|------------------|----------------|-----------------|------------------|-------------------------|
| **Casual chat user** (occasional ChatGPT) | 800 | 250 | 4 | 0.5 | 1.6k | 500 | ~60 k |
| **Power chat user** (writer, researcher) | 4 k | 1 k | 20 | 4 | 320 k | 80 k | ~12 M |
| **Coder, autocomplete** (Tab-style) | 3 k (file ctx) | 80 (one line) | 60 | 6 | ~1 M | 30 k | ~30 M |
| **Coder, IDE chat** (Cursor chat, Claude.ai code) | 8 k | 600 | 15 | 4 | ~480 k | 36 k | ~15 M |
| **Coding agent** (Claude Code, Cursor agent, Devin) | 50 k (long ctx) | 3 k | 5 (per task) | 6 | ~1.5 M | 90 k | ~50 M |
| **Code review agent** (per PR) | 25 k | 1.5 k | 2 | 8 | 50 k | 3 k | ~1.5 M |
| **Bot user / API consumer** (mid-volume) | 2 k | 400 | 60 (continuous) | 24 | ~3 M | 580 k | ~110 M |
| **Autonomous agent** (research, ops, 24/7) | 10 k | 1.5 k | 30 | 24 | ~7 M | 1 M | ~250 M |
| **Customer support chatbot** (per ticket) | 1.2 k | 250 | varies | n/a | by # tickets | by # tickets | scales with traffic |
| **Voice assistant** (transcribe + reply) | 800 | 200 | 30 | 1 | 24 k | 6 k | ~900 k |
| **RAG / knowledge search** | 6 k (retrieved) | 400 | 8 | 2 | 96 k | 6.4 k | ~3 M |
| **Translation / batch summarization** | 5 k | 5 k | 100 (batch) | 0.5 | 250 k | 250 k | ~15 M |
| **Marketing / content tool** | 2 k | 800 | 5 | 1 | 10 k | 4 k | ~400 k |
| **Tutor / education app** | 1 k | 600 | 30 | 2 | 30 k | 18 k | ~1.5 M |
| **Reasoning research user** (heavy `<think>`) | 3 k | 5 k (mostly thinking) | 10 | 2 | 60 k | 100 k | ~5 M |

A few things to notice:
- **Output tokens are typically 5-10× more expensive than input tokens** to produce (decode is memory-bound). Charge accordingly.
- **Coding agents are 20-50× more expensive per user than chat users.** They burn long context per step and run dozens of steps per task.
- **Bots running 24/7 dwarf even agents** — they're effectively many users compressed into one account.
- **Reasoning users push output tokens up 5-10×** because most of those tokens are thinking, not visible answer.

### Cost-per-user math

Take a representative API rate (varies by model, this is a 2026 mid-tier estimate):

```
input  (uncached): $3.00 / M tokens
input  (cached):   $0.30 / M tokens   ← 90% off
output:            $15.00 / M tokens
```

Now compute monthly cost per user, assuming 80% of input is cacheable (well-structured prompts):

| Archetype | Monthly in (M) | Monthly out (M) | Effective in $ | Out $ | **Total / user / month** |
|-----------|---------------|-----------------|----------------|-------|--------------------------|
| Casual chat | 0.05 | 0.01 | $0.03 | $0.15 | **$0.18** |
| Power chat | 9.6 | 2.4 | $5.18 | $36 | **$41** |
| Coder autocomplete | 30 | 0.9 | $16.20 | $13.50 | **$30** |
| Coder IDE chat | 14.4 | 1.1 | $7.78 | $16.20 | **$24** |
| Coding agent | 45 | 2.7 | $24.30 | $40.50 | **$65** |
| Code review agent | 1.5 | 0.09 | $0.81 | $1.35 | **$2.16** |
| Bot user | ~90 | 17 | $48.60 | $261 | **$310** |
| Autonomous agent | ~210 | 30 | $113 | $450 | **$563** |
| Voice assistant | 0.72 | 0.18 | $0.39 | $2.70 | **$3.10** |
| RAG search | 2.9 | 0.19 | $1.55 | $2.85 | **$4.40** |
| Reasoning research | 1.8 | 3 | $0.97 | $45 | **$46** |

Two observations:
1. **The cost gap between "casual chat" and "coding agent" is ~360×.** A pricing plan that treats all users the same will lose money on agents and overcharge casual users.
2. **Output dominates the bill** for almost every category. This is why "fast tokens" hardware (HBM bandwidth, §6) directly equates to better margin — you serve more output tokens per GPU-hour.

### Pricing strategy patterns

There is no single "right" pricing model, but four patterns dominate 2026 LLM products:

1. **Freemium + small-model free tier.** Casual users on a 1-3B model (subsidised, near-free in cost), upgrades behind paywall. Examples: ChatGPT free, Claude.ai free, Gemini free.
2. **Per-month plans with token quotas.** Tiers like $20 → 20M tokens, $100 → 200M tokens. Plan covers ~80% of users; heavy users overflow into per-token billing or get politely throttled. Most consumer LLM products.
3. **Per-token PAYG (API).** Cost-plus-margin per million tokens, separate input/output rates, separate cached input rate. The Anthropic / OpenAI / DeepSeek model. Required for any serious developer audience.
4. **Tiered model pricing.** Small model unlimited (or generous), big model gated by request count or token volume. Lets coding-agent-class users self-select into the right plan.

Most successful products combine 2 and 4: a flat plan with a token cap, with the small/big tier auto-selected for the user.

### Margin math

```
Revenue per million output tokens:  $15
Cost per million output tokens:     $1.50  (well-utilized fp8 27B, on-prem)
Gross margin:                        90%

Revenue per million cached input:   $0.30
Cost per million cached input:      ~$0.05  (almost free with prefix cache)
Gross margin:                        ~83%
```

After fixed costs (idle replicas, support, R&D, edge infra, payment fees), real net margin lands around **50-70%** for a self-hosted product, **20-40%** if you're reselling someone else's API.

### Caps: the agent-loop bankruptcy problem

A poorly-designed agent can recurse for 100,000 tokens of `<think>`, retry-loop another 50,000 on a tool failure, and chew up a year of free credits in 90 minutes. Real failure mode that has hit many companies. Always implement:

- **Per-request `max_tokens` cap** (10-32k for chat; 100-200k for explicit agent endpoints).
- **Per-conversation token budget** (e.g., 200k cumulative across one chat session before warning).
- **Per-user daily quota** with warning at 80%, hard cap at 100%.
- **Spend alerts** at $10, $50, $100 per user (configurable).
- **Soft kill** — switch the user to a smaller model when they cross a threshold.
- **Hard kill** — return 429 with a clear `"budget_exceeded"` body.
- **Idle-loop detector** — agents stuck in a tool-retry loop (same tool call N times) trip a circuit breaker.

Make these defaults, not opt-in.

### Anti-pricing-mistakes checklist

- Don't price input and output the same. Output is 5-10× more expensive.
- Don't ignore cache-hit rate. A product without caching has 5-10× higher unit costs than one with.
- Don't let agents run unbounded. Measure 99th-percentile token usage; build for that, not the median.
- Don't price reasoning tokens like normal output — they can be 5× more numerous.
- Don't resell flat. Always have a usage-based component for whales.
- Don't surprise users with bills. Show running token counters in the UI.

### A worked example: pricing a coding tool

You build a Cursor-style assistant. Target user: the **coder IDE chat** archetype (~$24/user/month at API cost).

- Build cost: ~$24 raw at API rates with full caching.
- Add infra, support, R&D overhead: ~$32 effective fully loaded cost.
- Target gross margin 60% → list price $80/month.
- Market reality: competitors are $20/month for Cursor-Pro-tier offerings.
- **Solution**: route 70% of traffic to a cheaper small model (drop blended cost to ~$10), make the big model a "max-tier" feature gated to power users at $40/month with metered overage.

This is exactly the multi-tier strategy every successful 2026 coding tool has converged on. **Hardware bandwidth, caching, smart routing, and user-archetype pricing all stack into the same equation: how much margin you keep per million tokens served.**

---

## 14. Security & abuse

10k QPS to a public API attracts attention.

- **Authentication**: API keys per tenant; OAuth for orgs.
- **Rate limiting**: per-key QPS, tokens/min, monthly quota.
- **Input filtering**: max prompt length, ban known-bad prompts (jailbreaks, exfiltration).
- **Output filtering**: PII/safety classifier on streaming output (small fast model).
- **Logging**: request hashes only, never plaintext prompts/outputs into long-term storage without explicit consent.
- **Tenant isolation**: KV cache must not leak across tenants. vLLM's prefix cache hashes the prompt — if two tenants accidentally send the same plaintext, they share. For strict isolation, namespace the prefix cache key with tenant ID.
- **DoS protection**: reasoning models can be made to produce 100k tokens; cap `max_tokens` per request.
- **Prompt injection** in tool-calling agents: validate tool arguments, sandbox tool execution.

---

## 15. Testing the rig before traffic

Before flipping the DNS, run:

1. **Load test** at 1.5× target QPS for 30 min with realistic prompt distributions. Measure TTFT/TPOT/error rate. (Tools: `k6`, `vegeta`, `genai-perf`.)
2. **Soak test** at 1× target for 12+ hours — catches memory leaks, cache fragmentation.
3. **Chaos test** — kill one replica during load. Verify request succeeds via retry; alert fires.
4. **Long-context test** — a few requests at 100k tokens. Verify they don't starve everyone else.
5. **Burst test** — 0 → 1.5× target QPS in 10 seconds. Watch queue depth and TTFT.
6. **Tenant isolation** — two simulated tenants with different system prompts, verify no leakage in prefix cache hashing.
7. **Cache regression test** — measure `cached_tokens / prompt_tokens` for canonical traffic and alert if it drifts more than 20%.
8. **Cost-cap test** — simulate a runaway agent loop and verify the spend cap fires within seconds, not after a $1,000 bill.

Do all of these in a staging environment that mirrors prod. Skipping this step is how you learn the hard way that vLLM's `max_num_seqs` was set too low.

---

## 16. One-page checklist

Before you call this "production":

- [ ] Engine: vLLM or SGLang with FP8 weights + KV.
- [ ] Prefix caching enabled on engine **and** prompt structure designed for cache hits.
- [ ] Chunked prefill enabled.
- [ ] Speculative decoding (MTP / EAGLE) for decode-heavy traffic.
- [ ] `max_num_seqs` and `max_num_batched_tokens` tuned to your SLOs.
- [ ] Two-tier model routing (small/big) with a classifier.
- [ ] Prefix-aware sticky load balancing.
- [ ] Hardware chosen by HBM bandwidth and capacity, not just FLOPs.
- [ ] Health checks that exercise a real completion.
- [ ] Prometheus metrics + Grafana dashboards (TTFT, TPOT, queue depth, KV usage, cache hit rate).
- [ ] Alerts on TTFT P95, queue depth, 5xx rate, KV usage, cache-hit drift.
- [ ] Autoscaler with sensible floor, scale-up on leading indicator.
- [ ] Per-tenant rate limits and quotas.
- [ ] Per-user spend caps and idle-agent-loop detection.
- [ ] Per-request `request_id` header for traceability.
- [ ] Multi-region or at minimum multi-zone deployment.
- [ ] Soak + load + chaos + cache-regression + cost-cap tested.
- [ ] On-call runbook: how to roll back, how to drain a node, how to bump quotas.

If any item is "no" or "I think so," you don't yet have a 10k-QPS-grade system.

---

## 17. The 2026 cheat sheet

- **vLLM is the safe default. SGLang for prefix-heavy workloads. TRT-LLM for max throughput.**
- **FP8 weights + FP8 KV** on H100/B200.
- **HBM bandwidth, not FLOPs**, sets per-user tokens/sec. Buy the GPU with the fattest pipe.
- **Prefix caching** is the biggest free win for chat — design prompts as frozen-then-liquid.
- **Speculative decoding** (MTP / EAGLE-3) is the second biggest.
- **Route easy queries to a smaller model.** Two tiers > one big model at scale.
- **Prefill/decode disaggregation** above ~2k QPS or for very long contexts.
- **Autoscale on queue depth**, not just QPS.
- **Cold start is 1-3 minutes** — keep a generous floor.
- **Observability before optimization.** You can't fix what you can't see.
- **Price per archetype**, not per "user." Coding agents cost 360× a casual chat user.
- **Output tokens are 5-10× more expensive than cached input.** Bill them differently.
- **Cap everything** — per-request, per-conversation, per-day, per-spend. Agent loops will find any uncapped budget.
- **10k QPS of frontier-quality chat ≈ 50-150 GPUs**, which is an order-of-magnitude reality check most teams under-estimate.

---

## Going deeper

- **vLLM docs** — `docs.vllm.ai`, the canonical reference. Read the "Production" section.
- **SGLang docs** — `sglang.ai/docs`, especially RadixAttention.
- **NVIDIA NIM / TensorRT-LLM cookbook** — for NVIDIA-native deployments.
- **DeepSeek inference paper (2024)** — the disaggregated prefill/decode architecture explained.
- **Ray Serve LLM tutorial** — multi-replica vLLM with autoscaling on K8s.
- **Anthropic, OpenAI, Google cache-control docs** — the canonical references for provider-side caching pricing and TTLs.
- **NVIDIA HBM3e / HBM4 hardware briefs** and Tim Dettmers' GPU buying guide — to make hardware decisions on memory rather than marketing.
- **k6 + genai-perf** — load testing toolkits actually designed for streaming token APIs.
- **The OpenAI / Anthropic public API SLOs and incident postmortems** — the most honest accounts of what production LLM serving looks like at scale.

That's the end of the curriculum. If you've read chapters 00 → 17, you can now pretrain a small LM, build it from scratch, fine-tune it, quantize it, and serve it to ten thousand users without bankrupting yourself. **Now go ship something.**
