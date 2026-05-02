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

## 6. A concrete reference design — 10,000 chat QPS on Qwen 3.6-27B

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

## 7. Observability — what to log and watch

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

## 8. Reliability and graceful degradation

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

## 9. Autoscaling

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

## 10. Cost and how to think about it

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

## 11. Security & abuse

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

## 12. Testing the rig before traffic

Before flipping the DNS, run:

1. **Load test** at 1.5× target QPS for 30 min with realistic prompt distributions. Measure TTFT/TPOT/error rate. (Tools: `k6`, `vegeta`, `genai-perf`.)
2. **Soak test** at 1× target for 12+ hours — catches memory leaks, cache fragmentation.
3. **Chaos test** — kill one replica during load. Verify request succeeds via retry; alert fires.
4. **Long-context test** — a few requests at 100k tokens. Verify they don't starve everyone else.
5. **Burst test** — 0 → 1.5× target QPS in 10 seconds. Watch queue depth and TTFT.
6. **Tenant isolation** — two simulated tenants with different system prompts, verify no leakage in prefix cache hashing.

Do all of these in a staging environment that mirrors prod. Skipping this step is how you learn the hard way that vLLM's `max_num_seqs` was set too low.

---

## 13. One-page checklist

Before you call this "production":

- [ ] Engine: vLLM or SGLang with FP8 weights + KV.
- [ ] Prefix caching enabled.
- [ ] Chunked prefill enabled.
- [ ] Speculative decoding (MTP / EAGLE) for decode-heavy traffic.
- [ ] `max_num_seqs` and `max_num_batched_tokens` tuned to your SLOs.
- [ ] Two-tier model routing (small/big) with a classifier.
- [ ] Prefix-aware sticky load balancing.
- [ ] Health checks that exercise a real completion.
- [ ] Prometheus metrics + Grafana dashboards (TTFT, TPOT, queue depth, KV usage).
- [ ] Alerts on TTFT P95, queue depth, 5xx rate, KV usage.
- [ ] Autoscaler with sensible floor, scale-up on leading indicator.
- [ ] Per-tenant rate limits and quotas.
- [ ] Per-request `request_id` header for traceability.
- [ ] Multi-region or at minimum multi-zone deployment.
- [ ] Soak + load + chaos tested.
- [ ] On-call runbook: how to roll back, how to drain a node, how to bump quotas.

If any item is "no" or "I think so," you don't yet have a 10k-QPS-grade system.

---

## 14. The 2026 cheat sheet

- **vLLM is the safe default. SGLang for prefix-heavy workloads. TRT-LLM for max throughput.**
- **FP8 weights + FP8 KV** on H100/B200.
- **Prefix caching** is the biggest free win for chat.
- **Speculative decoding** (MTP / EAGLE-3) is the second biggest.
- **Route easy queries to a smaller model.** Two tiers > one big model at scale.
- **Prefill/decode disaggregation** above ~2k QPS or for very long contexts.
- **Autoscale on queue depth**, not just QPS.
- **Cold start is 1-3 minutes** — keep a generous floor.
- **Observability before optimization.** You can't fix what you can't see.
- **10k QPS of frontier-quality chat ≈ 50-150 GPUs**, which is an order-of-magnitude reality check most teams under-estimate.

---

## Going deeper

- **vLLM docs** — `docs.vllm.ai`, the canonical reference. Read the "Production" section.
- **SGLang docs** — `sglang.ai/docs`, especially RadixAttention.
- **NVIDIA NIM / TensorRT-LLM cookbook** — for NVIDIA-native deployments.
- **DeepSeek inference paper (2024)** — the disaggregated prefill/decode architecture explained.
- **Ray Serve LLM tutorial** — multi-replica vLLM with autoscaling on K8s.
- **k6 + genai-perf** — load testing toolkits actually designed for streaming token APIs.
- **The OpenAI / Anthropic public API SLOs and incident postmortems** — the most honest accounts of what production LLM serving looks like at scale.

That's the end of the curriculum. If you've read chapters 00 → 17, you can now pretrain a small LM, build it from scratch, fine-tune it, quantize it, and serve it to ten thousand users. **Now go ship something.**
