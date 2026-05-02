# 17 · Production Inference — 10,000 Requests Serve करना

> **TL;DR** "10,000 requests serve करो" का मतलब तीन different problems हो सकते हैं: **10k QPS sustained**, **10k concurrent**, या **10k total एक day में**। हर एक एक different design demand करता है। 2026 stack है: engine के रूप में **vLLM** या **SGLang**, H100/B200 पर **fp8 weights + fp8 KV**, **PagedAttention के साथ continuous batching**, system prompts share करने के लिए **prefix caching**, 2× decode के लिए **speculative decoding**, very high scale पर **prefill/decode disaggregation**, **prefix-aware sticky load balancing**, और autoscaling के लिए **Kubernetes / Ray Serve**। 8× H100 पर Qwen 3.6-27B FP8 के साथ, आप ~3,500 chat QPS per node sustain कर सकते हो; ten thousand QPS = roughly 3 nodes, plus 1-2 headroom के लिए। **Capacity magic नहीं — arithmetic है।**

---

## 1. "10,000 Requests" की तीन Definitions

Building से पहले हमेशा इसे pin down करो।

| Interpretation | Actually इसका मतलब क्या | Hardware ballpark (Qwen3.6-27B fp8) |
|----------------|----------------------|--------------------------------------|
| **10k QPS sustained** | 10,000 chat completions per second, indefinitely | 3-5 H100 nodes (8× each) vLLM parallel चला रहे |
| **10k concurrent** | 10k chats एक साथ open, हर user slowly type करते हुए | 1-3 nodes — concurrency "in-flight" है, batched together |
| **10k requests total / day** | Bursty, low overall load | 1 GPU काफी; cold-start और reliability पर focus |

ज़्यादातर production deployments तीनों mix करते हैं: traffic peaks के दौरान N concurrent तक bursts, M QPS sustained, K total per day। **Peak QPS के लिए plan करो** और **average QPS के लिए pay करो**।

---

## 2. सही Cost Unit: Tokens, Requests नहीं

एक "request" तब तक meaningless है जब तक आप नहीं जानते कि वो कितने tokens process करता है।

Chat workload के लिए, typical request है:
- **Prompt**: ~500-2000 tokens (system + history + new user message)।
- **Completion**: chat के लिए 100-500 tokens (avg); reasoning के लिए 1000-5000।

तो 10,000 chat requests/sec ≈ work के **10-30M tokens/sec**। वो number — total tokens/sec — वो है जो GPUs serve करते हैं, "requests" नहीं।

Throughput rules of thumb (Qwen 3.6-27B FP8 **एक 8×H100 node** पर):

- **Prefill**: ~150,000-250,000 tokens/sec।
- **Decode**: ~3,000-6,000 tokens/sec aggregate (सारे in-flight users के across)।

Decode per token ~30-50× slower है क्योंकि ये memory-bound है (chapter 9)। आपके traffic का prompt और completion के बीच split capacity drive करता है।

### Capacity Formula

```
required_GPUs ≈ (avg_QPS × avg_prompt_tokens / prefill_tokens_per_sec_per_GPU)
              + (avg_QPS × avg_completion_tokens / decode_tokens_per_sec_per_GPU)
              + headroom (×1.3-1.5)
```

अपने numbers plug in करो और आपके पास एक order-of-magnitude estimate है जो usually reality के 30% के अंदर है। Remaining 30% queueing, retries, और tail-latency requirements से आता है।

---

## 3. SLOs जो आपको Actually चाहिए

जो आप measure नहीं कर सकते उसे optimize नहीं कर सकते। ये per route set करो:

| SLO | क्या है | Chat के लिए target | Agents के लिए target |
|-----|-----------|-----------------|---------------------|
| **TTFT** (time-to-first-token) | request enter → first decoded token out तक | P95 < 500 ms | P95 < 2 s |
| **TPOT** (time-per-output-token) | streaming token interval | P95 < 50 ms (~20 tok/s) | P95 < 80 ms |
| **End-to-end latency** | request → final byte | 200-tok reply के लिए P95 < 8 s | 1000-tok reply के लिए P95 < 30 s |
| **Availability** | non-5xx rate | 99.9% | 99.5% |
| **Error budget** | per month आप कितना spend कर सकते हो | 99.9% पर ~43 min/month | 99.5% पर ~3.6 hr |

TTFT **prompt prefill** plus **queue wait** से dominated है। TPOT **decode batch size** से dominated है (bigger batch → slower per-user TPOT, higher aggregate throughput)। TTFT/TPOT trade-off most important capacity dial है।

---

## 4. 2026 में Actually चलने वाला Stack

### Engine Layer (GPU work करता है)

| Engine | कब pick करें | Strengths |
|--------|--------------|-----------|
| **vLLM** (Berkeley + community) | Default. Battle-tested, OpenAI-compatible, huge model coverage | PagedAttention, continuous batching, prefix cache, speculative, MoE, multimodal, FP8 |
| **SGLang** (LMSYS) | Heavy prefix sharing वाले chat, structured outputs, tool calling | RadixAttention, fastest prefix-cache hit-rate; native constraint decoding |
| **TensorRT-LLM** (NVIDIA) | NVIDIA पर max throughput, build steps करने को willing | Lowest TTFT, best fp8 / NVFP4 kernels |
| **TGI** (HuggingFace) | HF stack के लिए drop-in | vLLM से slightly slower लेकिन simpler ops |
| **DeepSeek SGL / DeepEP**| Frontier MoE serving | Best MoE expert-parallelism kernels |

**2026 में green-field 10k-QPS chat service के लिए, vLLM या SGLang को default करो**। TensorRT-LLM तब use करो अगर आपके पास already build/deploy infra है।

### Frontend Layer (routing, queueing, nodes के across batching)

- **Ray Serve** — multi-node, autoscaling, KV-aware routing। Ray का `vLLMReplica` directly integrate करता है।
- **KServe** — Kubernetes-native, multi-tenant, A/B traffic split।
- **NVIDIA NIM / Triton** — multi-model rollouts के साथ production NVIDIA stack।
- **Custom**: vLLM nodes के सामने एक small FastAPI / Go service काफ़ी हो सकता है जब traffic uniform हो।

### Orchestration

- **Kubernetes** **GPU node pools** और QPS या queue depth पर keyed **HPA** (Horizontal Pod Autoscaler) के साथ।
- Peaks के दौरान nodes add करने के लिए **Cluster autoscaler** (slow — minutes — तो floor over-provision करो)।

### Model Lifecycle

- **HuggingFace Hub** weights के लिए source of truth।
- Quantized variants के लिए **Object store** (S3/GCS)।
- **Image registry** model + engine baked in के साथ (boot पर HF से pull करने से faster cold start)।

---

## 5. Big Inference Levers (impact के order में)

### Lever 1 — Model को Quantize करो

`bf16 → FP8` weights (H100/B200) पर switch करना typically yields:

- ~2× throughput।
- ~50% memory savings (more concurrent users fit करने देता है)।
- Brief calibration के बाद < 0.5% quality drop।

```bash
# vLLM FP8 weights और FP8 KV cache के साथ
vllm serve Qwen/Qwen3.6-27B \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

`--gpu-memory-utilization 0.9` vLLM को weights + KV cache के लिए HBM का 90% use करने देता है, activations और headroom के लिए 10% reserve करते हुए। Higher push सिर्फ़ तभी करो अगर आपने measure कर लिया हो कि burst load के तहत OOM नहीं होते।

Laptop / CPU serving के लिए, `llama.cpp` के साथ **GGUF Q4_K_M** पर switch करो (chapter 12)।

### Lever 2 — KV Cache Management

KV cache long context पर *the* memory hog है।

- **`--kv-cache-dtype fp8`** — 2× memory cut, free।
- **`--kv-cache-dtype fp8_e4m3` / `fp8_e5m2`** — vLLM fp8 KV variants; `e4m3` more accurate, `e5m2` more range।
- **PagedAttention** — automatic; vLLM cache को 16 tokens के pages में split करता है waste-free packing के लिए। Memory utilization ~60% से ~96% तक जाती है।
- **CPU/SSD KV offload** (vLLM 0.7+, SGLang) — cache के inactive part को CPU RAM या NVMe पर evict करो। 1M-token serving unlock करता है।
- **KV cache compression** (`H2O`, `SnapKV`, `KIVI`) — low-importance tokens drop करो। Research-grade; quality varies।

### Lever 3 — Prefix Caching

Production chat में, **system prompt millions of requests के across identical है**। Prefix caching के बिना, हर new conversation के लिए आप वो 1000-token system prompt re-prefill करते हो। Prefix caching के साथ, आप उसे **एक बार** compute करते हो और हर subsequent request जो prefix share करता है उसे free पाता है।

- **vLLM**: `--enable-prefix-caching` से enable करो। Hash-based, exact-match prefixes।
- **SGLang RadixAttention**: सारे live prefixes के ऊपर एक radix tree build करता है; sharing *किसी भी* common prefix तक extend होती है, सिर्फ़ system message नहीं — few-shot prompts और multi-turn chat के लिए perfect।

Prefix caching often long, shared prompts वाले chat workloads के लिए **2-5× higher effective throughput** देती है। ये stack में single biggest free win है।

### Lever 4 — Continuous (in-flight) Batching

vLLM और SGLang ये already करते हैं। हर step:

1. Next ready batch लो (fresh prefill + ongoing decode का mix)।
2. एक forward pass run करो।
3. Finished sequences free करो, नए schedule करो।

Tunables:
- **`--max-num-seqs`** — concurrency cap। Higher = better throughput, worse TPOT।
- **`--max-num-batched-tokens`** — per step total tokens। Prefill burst cap करता है।
- **Chunked prefill** (`--enable-chunked-prefill`) — एक giant prompt को decode steps के across split करो ताकि दूसरे users starve न हों। **Production में strongly recommended।**

### Lever 5 — Speculative Decoding

Decode phase memory-bound है (chapter 9)। एक small **draft model** 4-8 tokens propose करता है; big model उन्हें एक forward pass में verify करता है।

vLLM एक draft model के साथ:
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"model":"Qwen/Qwen3.6-1.7B","num_speculative_tokens":5}'
```

या model का खुद का MTP head use करो (Qwen 3.6, DeepSeek-V3):
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

या **EAGLE-3**: एक tiny adapter जो hidden states पर trained है; chat workloads पर 2-3× speedup:
```bash
vllm serve Qwen/Qwen3.6-27B \
    --speculative-config '{"method":"eagle3","model":"path/to/eagle3-adapter"}'
```

Caveat: speculative decoding सिर्फ़ decode को help करता है। अगर आपका traffic prefill-heavy है, instead chunked prefill पर focus करो।

### Lever 6 — Prefill/Decode Disaggregation (scale पर)

DeepSeek के serving stack द्वारा used और अब vLLM (`--enable-disagg`) में एक frontier 2025-2026 trick:

- **Prefill nodes** (compute-bound, big TP, fp8) और **decode nodes** (memory-bound, small TP, prefix cache) को separate GPUs पर run करो।
- Freshly computed KV cache को prefill node → decode node पर RDMA / NVLink के through move करो।
- TTFT और TPOT हर एक independently optimized होते हैं।

~2,000 QPS से ऊपर या very long contexts के लिए complexity worth है। उसके नीचे, bother मत करो।

### Lever 7 — Tensor / Expert Parallelism

ऐसे models के लिए जो एक GPU पर fit नहीं होते:

- **TP (Tensor Parallel)**: हर weight को `N` GPUs के across shard करो। हर layer में all-reduce के लिए latency add होती है; memory gain।
- **PP (Pipeline Parallel)**: layers को `N` GPUs के across split करो। Latency के लिए worse; very deep models के लिए useful।
- **EP (Expert Parallel)**: MoE experts shard करो। Large MoEs (DeepSeek-V3, Gemma 4 26B-A4B) के लिए necessary।

Smallest TP pick करो जो model + KV cache fit करे। Necessary से bigger TP हमेशा per GPU throughput खोता है।

27B fp8 model के लिए: TP=2 2× H100 पर comfortably fit होता है। 256 experts वाले 35B-A3B के लिए: EP=4 + TP=2।

### Lever 8 — सही Model Choose करो

- **Same cost पर दो smaller models** often एक bigger model को beat करते हैं। Easy queries 4B पर route करो; hard वाले 27B पर route करो।
- Very high QPS पर **Mixture of Experts** — same memory budget लेकिन per token cheaper।
- Predictable latency के लिए **Dense smaller**।

Multi-model routing 2026 में serious deployments के लिए default है।

---

## 6. एक Concrete Reference Design — Qwen 3.6-27B पर 10,000 chat QPS

Assumptions:
- avg prompt 1500 tokens (heavy system prompt, first hit के बाद prefix-cached)।
- avg completion 300 tokens।
- TTFT P95 target: 500 ms।
- TPOT P95: 50 ms।
- Region: single AWS region with `p5.48xlarge` (8× H100 80 GB) nodes।

### Sizing

Per 8×H100 node (Qwen 3.6-27B fp8, TP=4, two replicas per node):

- Prefill throughput ~250k tokens/sec (chunked prefill के साथ)।
- Decode throughput ~5k tokens/sec aggregate `max_num_seqs=512` पर।
- Prefix caching के साथ (~80% hit rate system prompt पर) → effective prefill ~3k tokens/req new।

Required compute / sec:
- New prefill: 10k QPS × 3,000 effective new tokens = 30M tok/s prefill → **~120 nodes**? नहीं — most prompts system prefix share करते हैं और *active* prefill load dramatically drops। Realistic 80% prefix-cache hit के साथ, **active prefill ~6M tok/s → prefill के लिए ~24-30 nodes**।
- Decode: 10k QPS × 300 tokens = 3M tok/s decode → 5k/node पर **~600 nodes**।

दोनों numbers कहते हैं decode by an order of magnitude binding constraint है। Mitigations:
- **Speculative decoding** (MTP head): 2× → 1.5M tok/s effective decode → 300 nodes। अभी भी big।
- Reality check: ये assume करता है कि सबको fresh 27B response मिलता है। ज़्यादातर production workloads traffic का large fraction **smaller models** (simple queries के लिए routed 4B) पर route करते हैं और 27B को hard prompts के लिए reserve करते हैं। 70/30 split के साथ:
  - 70% Qwen 3.6-4B-class model पर (~30k tok/s decode/node) → 70k tok/s decode load → ~3 nodes।
  - 30% 27B पर (~5k × 2x spec → 10k tok/s decode/node) → 900k tok/s decode load → 90 nodes।

Routing के साथ भी, "real time में 10k QPS full-quality flagship-model chat completions" **एक hundred-GPU operation** है। वो number frontier API providers actually क्या run करते हैं उससे tracks करता है।

अगर आपका 10k-QPS goal "10k mostly-light chat per second occasional reasoning calls के साथ" जैसा ज़्यादा है, तो **3-8 nodes plenty हैं**। अपने real distribution के बारे में honest रहो।

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

- एक fast classifier (एक 0.5B model या even prompt features पर logistic regression) के साथ हर request route करते हुए **दो model tiers**।
- **Prefix-aware routing** — same system prompt वाले requests को same node पर भेजो ताकि उसका prefix cache hot रहे। SGLang के पास ये है; vLLM के लिए, `system_prompt_hash` पर sticky hash use करो।
- Floor के रूप में **Static replicas**, top पर autoscale। एक replica cold-starting 60-180 seconds है; आप उसे केवल QPS पर autoscale नहीं कर सकते।

### Per Node Memory Layout

8×H100 पर 27B fp8 के लिए TP=4, 2 replicas/node के साथ:
- Weights: ~28 GB / replica (~7 GB / GPU sharded)।
- KV cache: budget ~30 GB / replica।
- Activations + headroom: ~10 GB।
- Total per replica: ~70 GB / 4 GPUs = 80 GB cards पर ~17.5 GB / GPU। Higher concurrency के लिए plenty room।

### Configuration

```bash
# vLLM, per 8×H100 node दो replicas
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

Per node दो `vllm serve` processes चलाओ, GPUs 0-3 और 4-7 respectively पर pinned (`CUDA_VISIBLE_DEVICES`)। हर replica independent है, तो एक crash blast radius isolate करता है।

### Routing Layer (Sketch)

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
        is_hard = self.classify(prompt)            # tiny model या rule
        backend = self.big if is_hard else self.small
        # 2. prefix-aware sticky key
        sys = prompt[0]["content"] if prompt[0]["role"] == "system" else ""
        sticky = hashlib.sha256(sys.encode()).hexdigest()[:16]
        # 3. forward
        return await backend.options(stream=True).remote(req, sticky_key=sticky)
```

---

## 7. Observability — क्या Log और Watch करें

हर serving system को इन dashboards की need होती है:

### Per-request

- `ttft_ms` — TTFT histogram, P50/P95/P99।
- `tpot_ms` — per-token decode latency।
- `total_tokens_in`, `total_tokens_out`।
- `route_decision` — कौन सा model tier ने handle किया।
- `prefix_cache_hit` — per request boolean।
- `error` — categorized।

### Per-engine (Per vLLM Replica)

- `gpu_utilization` (NVML)।
- `gpu_memory_used`।
- `kv_cache_usage_pct` — अगर ये 95% cross करे आप requests drop करना start करोगे।
- `num_running_seqs` — current concurrency।
- `num_waiting_seqs` — queue depth।
- `tokens_per_sec_prefill`, `tokens_per_sec_decode`।
- `prefix_cache_hit_rate`।

### Cluster

- QPS by route, by status।
- End-to-end latency।
- Node health, OOM events, GPU thermal events।
- Cost-per-million-tokens।

Stack: metrics के लिए **Prometheus + Grafana**, traces के लिए **OpenTelemetry**, logs के लिए **Loki**। vLLM box के out Prometheus metrics expose करता है (`/metrics`)।

### Earn-their-keep Alerts

- 5 min के लिए TTFT P95 > target।
- किसी replica पर KV-cache usage > 95%।
- 30 sec के लिए Queue depth > N।
- GPU temperature > 88 °C।
- 5xx rate > 1%।
- Cost-per-token > 1.5× baseline (drift detector)।

---

## 8. Reliability और Graceful Degradation

10k QPS = हर minute कुछ fail होता है। Designs जो survive करते हैं:

- **Anti-affinity scheduling** के साथ per region multiple replicas।
- Active/active DNS या anycast के साथ **Multi-region deploy**।
- **Health checks** जो `/v1/models` *और* एक smoke completion hit करें (सिर्फ़ TCP ping नहीं — vLLM alive हो सकता है लेकिन stuck)।
- **Backpressure**: जब queue depth > threshold, `Retry-After` के साथ 429 return करो। Work accept मत करो जो आप कर नहीं सकते।
- हर layer पर **Timeouts**: client → LB (30s) → router (60s) → engine (60s)। अगर TTFT > timeout, token-stream existing connection के लिए अभी भी flow होती है, लेकिन new ones कहीं और जाते हैं।
- **Circuit breakers**: 30s के लिए hot replica पर routing stop करो अगर इसकी 5xx rate spike करे।
- `request_id` headers के साथ **Idempotent retries** (vLLM के पास built-in idempotency नहीं — caller को own करना है)।
- **Graceful degradation tiers**: heavy load के तहत, optional features drop करो (streaming नहीं → buffered, free tier के लिए smaller models, no thinking-mode)।

---

## 9. Autoscaling

GPU autoscaling के दो cruel limits हैं:
1. **Cold start ~ 60-180 s** (image pull + weight load + warmup)।
2. **GPU nodes scarce हैं** — आपका cloud provider के पास abhi capacity नहीं हो सकती।

Implications:

- **एक generous floor set करो** (`min_replicas`) typical hour-of-day load के लिए sized, trough के लिए नहीं।
- Leading indicators (queue depth, TTFT trending up) पर **early scale up करो** — सिर्फ़ QPS पर नहीं।
- Headroom के लिए **spot capacity use करो**, floor के लिए on-demand।
- Real load लेने से पहले synthetic traffic से नए replicas **pre-warm करो**।
- Flapping avoid करने के लिए **slowly scale down करो** (5-10 min cooldown)।

Ray Serve, KServe, और vLLM Production Stack सब इन patterns को respect करने वाले first-class autoscaling रखते हैं।

---

## 10. Cost और इसके बारे में कैसे सोचें

Rough 2026 unit economics (wildly varies):

- 1 H100 80 GB on-demand ≈ $2.5-4/hour।
- 1 8×H100 node ≈ $20-30/hour।
- Qwen 3.6-27B fp8 के लिए, ~5,000 decode tokens/sec/node → ~18M tokens/hour → ~$1.50 per million output tokens at full utilization।

Economics **utilization** से dominated है। 50% utilized fleet at twice the cost 100% utilized वाले के same है। **Aggressively autoscale करो, hard batch करो, prefix cache share करो।**

Cost levers, ranked:

1. Quantization (fp8 → 2× throughput) — 50% cost cut।
2. Prefix caching — system prompts वाले chat के लिए 30-70% cut।
3. Speculative decoding — decode पर 30-50% cut।
4. Smaller-model routing — routed traffic पर 40-80% cut।
5. Spot capacity — elastic fleet के लिए 50-70% cut।
6. Same engine पर Multi-tenancy — fixed overhead amortize करके 10-20% cut।

उन्हें stack करो और एक credible production deployment **naive bf16 single-tier से ~5× cheaper** lands है।

---

## 11. Security & Abuse

Public API पर 10k QPS attention attract करता है।

- **Authentication**: per tenant API keys; orgs के लिए OAuth।
- **Rate limiting**: per-key QPS, tokens/min, monthly quota।
- **Input filtering**: max prompt length, known-bad prompts ban (jailbreaks, exfiltration)।
- **Output filtering**: streaming output पर PII/safety classifier (small fast model)।
- **Logging**: सिर्फ़ request hashes, never explicit consent के बिना plaintext prompts/outputs को long-term storage में।
- **Tenant isolation**: KV cache tenants के across leak नहीं होना चाहिए। vLLM का prefix cache prompt hash करता है — अगर दो tenants accidentally same plaintext भेजते हैं, वो share करते हैं। Strict isolation के लिए, prefix cache key को tenant ID के साथ namespace करो।
- **DoS protection**: reasoning models से 100k tokens produce कराया जा सकता है; per request `max_tokens` cap करो।
- Tool-calling agents में **Prompt injection**: tool arguments validate करो, tool execution sandbox करो।

---

## 12. Traffic से पहले Rig Test करना

DNS flip करने से पहले, run करो:

1. Realistic prompt distributions के साथ 30 min के लिए target QPS के 1.5× पर **Load test**। TTFT/TPOT/error rate measure करो। (Tools: `k6`, `vegeta`, `genai-perf`)।
2. 12+ hours के लिए 1× target पर **Soak test** — memory leaks, cache fragmentation catch करता है।
3. **Chaos test** — load के दौरान एक replica kill करो। Verify request retry के via succeed करता है; alert fires।
4. **Long-context test** — 100k tokens पर कुछ requests। Verify वो everyone को starve नहीं करते।
5. **Burst test** — 10 seconds में 0 → target QPS के 1.5×। Queue depth और TTFT देखो।
6. **Tenant isolation** — different system prompts के साथ दो simulated tenants, prefix cache hashing में no leakage verify करो।

ये सब एक staging environment में करो जो prod को mirror करे। ये step skip करना वो है जिससे आप hard way से सीखते हो कि vLLM का `max_num_seqs` too low set था।

---

## 13. एक One-page Checklist

इसे "production" call करने से पहले:

- [ ] Engine: vLLM या SGLang FP8 weights + KV के साथ।
- [ ] Prefix caching enabled।
- [ ] Chunked prefill enabled।
- [ ] Decode-heavy traffic के लिए Speculative decoding (MTP / EAGLE)।
- [ ] अपने SLOs के लिए `max_num_seqs` और `max_num_batched_tokens` tuned।
- [ ] एक classifier के साथ Two-tier model routing (small/big)।
- [ ] Prefix-aware sticky load balancing।
- [ ] Health checks जो एक real completion exercise करें।
- [ ] Prometheus metrics + Grafana dashboards (TTFT, TPOT, queue depth, KV usage)।
- [ ] TTFT P95, queue depth, 5xx rate, KV usage पर Alerts।
- [ ] Sensible floor के साथ Autoscaler, leading indicator पर scale-up।
- [ ] Per-tenant rate limits और quotas।
- [ ] Traceability के लिए Per-request `request_id` header।
- [ ] Multi-region या at minimum multi-zone deployment।
- [ ] Soak + load + chaos tested।
- [ ] On-call runbook: कैसे roll back करें, कैसे एक node drain करें, कैसे quotas bump करें।

अगर कोई item "no" या "I think so" है, आपके पास अभी 10k-QPS-grade system नहीं है।

---

## 14. 2026 Cheat Sheet

- **vLLM safe default है। Prefix-heavy workloads के लिए SGLang। Max throughput के लिए TRT-LLM।**
- H100/B200 पर **FP8 weights + FP8 KV**।
- Chat के लिए **Prefix caching** biggest free win है।
- **Speculative decoding** (MTP / EAGLE-3) second biggest है।
- **Easy queries एक smaller model पर route करो।** Scale पर एक big model से two tiers > अच्छा है।
- ~2k QPS से ऊपर या very long contexts के लिए **Prefill/decode disaggregation**।
- सिर्फ़ QPS नहीं, **Queue depth पर autoscale करो**।
- **Cold start 1-3 minutes है** — एक generous floor रखो।
- **Optimization से पहले Observability।** जो आप देख नहीं सकते उसे fix नहीं कर सकते।
- **Frontier-quality chat के 10k QPS ≈ 50-150 GPUs**, जो एक order-of-magnitude reality check है जिसे ज़्यादातर teams under-estimate करती हैं।

---

## और गहराई से

- **vLLM docs** — `docs.vllm.ai`, canonical reference. "Production" section पढ़ो।
- **SGLang docs** — `sglang.ai/docs`, especially RadixAttention।
- **NVIDIA NIM / TensorRT-LLM cookbook** — NVIDIA-native deployments के लिए।
- **DeepSeek inference paper (2024)** — disaggregated prefill/decode architecture explained।
- **Ray Serve LLM tutorial** — K8s पर autoscaling के साथ multi-replica vLLM।
- **k6 + genai-perf** — load testing toolkits actually streaming token APIs के लिए designed।
- **OpenAI / Anthropic public API SLOs और incident postmortems** — production LLM serving scale पर कैसा दिखता है उसके most honest accounts।

That's curriculum का end. अगर आपने chapters 00 → 17 पढ़े, आप अब एक small LM pretrain, scratch से build, fine-tune, quantize, और दस हज़ार users को serve कर सकते हो। **अब कुछ ship करो।**
