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

## 6. Memory Bandwidth — HBM3, HBM4, और User Actually क्या Feel करते हैं

Fleet spec करने से पहले, LLM serving की एक सबसे counter-intuitive सच्चाई समझो: **inference के लिए, GPU memory bandwidth (HBM) chip TFLOPS से ज़्यादा matter करती है**। एक faster chip same memory के साथ आपको faster tokens नहीं देगा। ज़्यादातर teams ग़लत number के लिए pay करती हैं।

### Decode Bandwidth-bound क्यों है, Compute-bound नहीं

**Decode** phase में (वो part जिसका user actually wait करता है, token by token), GPU per byte loaded *almost कोई compute नहीं* करता है। एक single sequence के लिए हर new token को model के **हर weight** को HBM से Tensor Cores में read करना है, single-row activation के against एक tiny matmul करना है, और single-row output वापस write करना है। Tensor Cores underutilized बैठते हैं; bottleneck HBM pipe है।

**Per-user decode tokens/sec** का rough upper bound exactly है:

```
tokens_per_sec_per_user  ≤  HBM_bandwidth (bytes/s)  /  model_size (bytes)
```

बस। कोई FLOPs term नहीं। More compute add करना prefill (compute-bound) में help करता है और batched decode में help करता है (जब bandwidth कई users के across amortize करती है), लेकिन ये *एक* user को faster generate करने में help नहीं करता।

### HBM Generations और Actual Numbers

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

देखो H100 → H200 के बीच क्या jump हुआ: FLOPs nearly identical हैं, लेकिन bandwidth 43% ऊपर गई। **Inference के लिए, H200 H100 से per user roughly 40% faster है — entirely HBM3e की वजह से।** ये एक real-world result है जिसे आप किसी भी vLLM benchmark पर verify कर सकते हो। Chip smarter नहीं हुआ; pipe fatter हुआ।

Same story H200 → B200 (~67% bandwidth jump) से और B200 → Rubin/HBM4 (~60-100% bandwidth jump) से play out होती है।

### Concrete Tokens/sec Ceilings

कुछ real models लो और single-user ceiling compute करो। हम clarity के लिए bf16 weights use करते हैं; fp8 के लिए model size आधा करो।

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
  H100 (एक card पर fit नहीं होगा; assume TP=2 → effective 6.7 TB/s aggregate, लेकिन overhead के साथ): ~80 tok/s realistic
  B200 (एक card पर fits, 8 TB/s): ~110 tok/s
  HBM4: ~190 tok/s
```

Practice में आप **theoretical ceiling का 60-80%** देखोगे attention reads, KV cache traffic, kernel launch overhead, और Python glue की वजह से। लेकिन GPUs के बीच ratios almost exactly hold करते हैं।

### एक Coder क्या Feel करता है Tokens/sec में

User इस number को directly *feel* करता है जब वो tokens को अपने editor या chat में stream होते देखता है:

| User पर Tokens/sec | Subjective feel | Workflow पर effect |
|--------------------|-----------------|--------------------|
| 10 tok/s | "Painfully slow" | Tabs switch करेंगे, attention lost |
| 25 tok/s | "Reading along" | सिर्फ़ short answers के लिए tolerable |
| 50 tok/s | "Comfortable" | Chat के लिए good, code के लिए slow |
| 80 tok/s | "Native feel" | Cursor / IDE assistant sweet spot |
| 150 tok/s | "Instant" | Long answers Google search जैसे feel |
| 300+ tok/s | "Faster than I read" | Reasoning chains, long agents |

27B model use करने वाला coder canonical case है। H100 पर आप उन्हें ~120 tok/s पर serve करते हो — borderline native। B200 पर ये ~250 tok/s है — clearly native। HBM4 hardware पर ये ~400 tok/s है — वो model को entirely notice करना बंद कर देते हैं। **जिस hardware refresh के लिए आप pay करते हो वो ये change करता है कि आपका product broken feel होगा या magical।**

**Reasoning models** के लिए effect और भी bigger है। एक `<think>...</think>` block 1,000-5,000 tokens हो सकता है। 50 tok/s पर वो user को *real* answer देखने से पहले 20-100 second wait है। 300 tok/s पर ये 3-15 seconds है। Same model, same quality — radically different product।

**Agents** (Claude Code, Cursor agent, Devin-style) के लिए, loop में हर step एक full inference है। 30-step task 50 tok/s पर vs 300 tok/s पर 5 minutes और 50 seconds का difference है।

### Batching Picture को क्यों Change करती है

ऊपर सब कुछ **एक single user** के लिए था। जब आप एक decode step में B users batch करते हो, weights एक बार read होते हैं और B sequences के लिए use होते हैं। तो:

```
aggregate_tokens_per_sec  ≈  HBM_bandwidth / model_size  ×  effective_batch_size
```

एक बार batch fill हो जाए, आप compute-bound बन जाते हो, और *फिर* TFLOPs matter करते हैं। ये वो regime है जिसमें आपके throughput numbers रहते हैं।

Catch: per-user TPOT bigger batch के साथ actually worse होता है। तो batching **revenue** (aggregate tokens) को **UX** (per-user speed) के against trade करता है। ज़्यादातर products एक `max_num_seqs` pick करते हैं जो TPOT targets hit करे।

### इसके साथ क्या करें

1. Inference fleets के लिए **GPUs को $/(TB/s) और $/GB से pick करो**, $/TFLOP से नहीं। Training के लिए calculus reverses — वहां compute most matter करता है।
2. **Solo-user latency** (small batch) के लिए: आप जो highest-bandwidth GPU afford कर सकते हो उस पर run करो। H200/B200 2026 sweet spots हैं।
3. **Batched throughput** के लिए: lots of medium-bandwidth GPUs price/perf के depending fewer high-bandwidth वालों को beat कर सकते हैं।
4. **Quantization एक bandwidth multiplier है।** fp16 → fp8 जाना bytes-per-weight आधा करता है, same physical GPU के लिए effective HBM bandwidth double करता है। fp8 → int4 इसे फिर से double करता है। Quantization सिर्फ़ memory size के बारे में नहीं है; ये है कि tokens कितनी fast stream out होते हैं।
5. **HBM4 wave को watch करो।** Rubin-class GPUs late 2026 / early 2027 में land होते हैं ~2× H200 bandwidth के साथ। Refresh cycles इसके आसपास plan करो; inference workloads पर ये एक step change है, incremental bump नहीं।
6. **अपने users को tokens/sec बताओ, TFLOPS नहीं।** दो providers के बीच decide करते समय, उनके लिए सिर्फ़ वो number matter करता है कि उनके tokens कितनी fast आते हैं।

2026 में inference engineers के लिए एक working slogan: **"compute rented है, bandwidth destiny है।"**

---

## 7. एक Concrete Reference Design — Qwen 3.6-27B पर 10,000 chat QPS

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

## 8. Observability — क्या Log और Watch करें

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

## 9. Reliability और Graceful Degradation

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

## 10. Autoscaling

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

## 11. Cost और इसके बारे में कैसे सोचें

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

## 12. Token Caching — एक ही Context के लिए दो बार Pay मत करो

एक correctly designed product कभी भी same tokens के लिए ख़ुद को (या अपने users को) दो बार charge नहीं करता। Caching तीन layers पर show होती है, और एक serious deployment तीनों use करती है।

### तीन Cache Layers

1. **Provider-side cached input** (जब external API call कर रहे हो)। Anthropic, OpenAI, Google, DeepSeek, और दूसरे previously-sent prefix से match होने वाले input tokens के लिए big discounts देते हैं।
2. **Engine-side prefix cache / KV reuse** (जब self-hosting)। vLLM `--enable-prefix-caching` और SGLang RadixAttention prompt-prefix matches को free hits में बदलते हैं — अपनी infra में no compute, no billing।
3. **Application-level response cache** (model के ऊपर)। Deterministic, repeat queries (FAQ, structured extraction, classification) के लिए, *output* store करो और model entirely skip करो।

### Provider Cached-input Pricing (2026, public list rates)

ये shift करते हैं, लेकिन orders of magnitude stable हैं।

| Provider | Cached-read discount | Cache-write premium | Default TTL |
|----------|---------------------|---------------------|-------------|
| Anthropic Claude | **−90%** (input price का 10%) | First write पर +25% | 5 min (1 hr तक extendable) |
| OpenAI | **−50%** (cached input half-price) | none | ~5-10 min sliding |
| Google Gemini | **−75% to −80%** (context caching) | small storage fee | 1 hr default, configurable |
| DeepSeek | **−90%** | none | 24 hr (KV-cache on disk) |
| Mistral / Cohere | **−50%** | none | varies |

एक **cache-write** पहली बार है जब आप एक prefix भेजते हो; provider उसके लिए KV cache store करता है। Subsequent requests जो same prefix hit करते हैं वो cached-read rate pay करते हैं। Most providers prefix bytes के hash से cache identify करते हैं, तो single-byte difference cache को break करता है।

### Maximum Cache Hits के लिए Prompts Structure करना

Prompt layout biggest lever है। Stable content first, dynamic content last। Cache एक prefix tree है — moment एक byte differ हो, prompt का बाकी हिस्सा "new" है और full rate पर billed होता है।

```python
# WRONG — start पर dynamic timestamp हर request पर cache को blow करता है
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

एक cleaner mental model: अपने prompt को **frozen** + **liquid** की तरह imagine करो। Front पर frozen (system prompt, tool schemas, few-shot examples, retrieved documents जो turns के across persist करते हैं), पीछे liquid (user का latest message)। Frozen layer को as long as possible बनाओ — वही part है जो cache होता है।

### Anthropic-style Explicit Cache Control

Anthropic का API आपको cache breakpoints mark करना require करता है — system को बताने के लिए useful है "इसके पहले सब कुछ reusable है":

```python
client.messages.create(
    model="claude-opus-4-7",
    system=[
        {"type": "text",
         "text": stable_system_prompt + tools_schema + few_shot_examples,
         "cache_control": {"type": "ephemeral"}},   # breakpoint mark करता है
    ],
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": retrieved_docs,
             "cache_control": {"type": "ephemeral"}},   # second cacheable block
            {"type": "text", "text": user_message},     # cached नहीं
        ]}
    ],
)
```

4 cache breakpoints तक allowed हैं; API हर prefix segment को independently cache करता है। आप last cached match के बाद सिर्फ़ suffix के लिए full price pay करते हो।

### OpenAI-compatible: Cache Hits Automatic हैं

OpenAI और most OpenAI-compatible APIs (vLLM के सहित) prefix के hash पर automatically cache करते हैं — **आप ask नहीं करते, आप बस prompts को structure करते हो ताकि prefixes verbatim repeat हों**। Response में एक `prompt_tokens_details.cached_tokens` field include होता है जो आपको बताता है कि कितने cached थे ताकि आप hit rate verify कर सको।

```python
resp = openai.chat.completions.create(model=..., messages=...)
print(resp.usage.prompt_tokens_details.cached_tokens)  # इस metric को check करो
```

### Self-hosted: Same Cache, Free

जब आप अपना खुद का model serve करते हो, prefix caching purely एक HBM tenancy game है। vLLM के साथ:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --enable-prefix-caching \
    --enable-prefix-caching-hash sha256
```

In-flight या recent requests के across सारे identical prefixes same KV blocks reuse करते हैं। Stable system prompts पर 70-95% hit rates normal हैं। **Self-hosted prefix-cache hits आपको कुछ नहीं cost करते — zero compute, zero billing, cached bytes के लिए zero latency।**

### Application-level Response Caching

ऐसे requests के लिए जो हमेशा same output produce करते हैं, *output* को cache करो:

```python
import hashlib, redis, json

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

Best targets: structured extraction, classification, deterministic single-turn FAQ bots, normalization tasks। User के name, account state जैसे personalized कुछ भी cache मत करो user से key namespace किए बिना।

### चीज़ें जो Quietly Caching (और आपका Budget) Break करती हैं

- **System prompt में timestamps, request IDs, या session IDs।** एक byte भी हर subsequent token को bust करता है।
- Requests के बीच **messages या tools को re-order करना**।
- **Mixed `temperature` / `top_p` / `seed`** API-level response caching के साथ (different sampling = different output)।
- Routing में **Different models।** एक 4B-tier और एक 27B-tier caches share नहीं करते।
- **Inconsistent serializers से whitespace differences।**
- **Tool definitions के लिए partial JSON arrays।** एक canonical order pin करो, या आपको 100% miss मिलेगा।

A cheap monitoring habit: per route `cached_tokens / prompt_tokens` log करो और alert करो अगर ये आपके baseline से 20% से ज़्यादा drop करे। ये almost always prompt structure में regression है।

### Realistic Savings

Typical chat product के लिए (1500-token system + 500-token rolling history + 200-token new user message), about **80% input tokens cacheable हैं**। Anthropic के 90% cached-read discount के साथ, ये **input bill पर ~70% off** बनता है। Self-hosted prefix-cache + fp8 के साथ, inputs essentially free हैं; सिर्फ़ output tokens और *new* prefix segment आपको cost करते हैं।

ये quantization के beyond single largest cost lever है। Day one से इसके लिए build करो।

---

## 13. Pricing Strategy और Per-user-type Token Economics

एक बार caching place में हो जाए, pricing एक question बनती है: हर user type *actually* कितने tokens per day consume करता है, और आप margin कितनी extract कर सकते हो जबकि competitive भी रहो? यहां calibration table है जिससे शुरू करो।

### Per Archetype Token Consumption (2026 calibration)

ये public usage data और instrumented production deployments से realistic averages हैं। आपके numbers vary होंगे, लेकिन archetypes के बीच *ratios* surprisingly stable हैं।

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

कुछ चीज़ें notice करो:
- **Output tokens typically input tokens से 5-10× ज़्यादा expensive हैं** produce करने के लिए (decode memory-bound है)। Accordingly charge करो।
- **Coding agents chat users से per user 20-50× ज़्यादा expensive हैं।** वो per step long context burn करते हैं और per task dozens of steps run करते हैं।
- **24/7 चलने वाले Bots agents को भी dwarf करते हैं** — वो effectively एक account में compressed कई users हैं।
- **Reasoning users output tokens को 5-10× तक push करते हैं** क्योंकि उनमें से ज़्यादातर tokens thinking हैं, visible answer नहीं।

### Cost-per-user Math

एक representative API rate लो (model से vary होता है, ये एक 2026 mid-tier estimate है):

```
input  (uncached): $3.00 / M tokens
input  (cached):   $0.30 / M tokens   ← 90% off
output:            $15.00 / M tokens
```

अब per user monthly cost compute करो, मानते हुए कि 80% input cacheable है (well-structured prompts):

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
1. **"Casual chat" और "coding agent" के बीच cost gap ~360× है।** एक pricing plan जो सारे users को same treat करे coding agents पर पैसे खोएगा और casual users को overcharge करेगा।
2. **Almost हर category के लिए Output bill को dominate करता है।** इसलिए "fast tokens" hardware (HBM bandwidth, §6) directly better margin के बराबर है — आप per GPU-hour ज़्यादा output tokens serve करते हो।

### Pricing Strategy Patterns

कोई single "right" pricing model नहीं है, लेकिन 2026 LLM products में चार patterns dominate करते हैं:

1. **Freemium + small-model free tier।** Casual users एक 1-3B model पर (subsidised, near-free in cost), upgrades paywall के पीछे। Examples: ChatGPT free, Claude.ai free, Gemini free।
2. **Token quotas वाले Per-month plans।** Tiers like $20 → 20M tokens, $100 → 200M tokens। Plan ~80% users को cover करता है; heavy users per-token billing में overflow होते हैं या politely throttled। ज़्यादातर consumer LLM products।
3. **Per-token PAYG (API)।** Per million tokens cost-plus-margin, separate input/output rates, separate cached input rate। Anthropic / OpenAI / DeepSeek model। किसी भी serious developer audience के लिए required।
4. **Tiered model pricing।** Small model unlimited (या generous), big model request count या token volume द्वारा gated। Coding-agent-class users को सही plan में self-select करने देता है।

Most successful products 2 और 4 combine करते हैं: एक token cap के साथ flat plan, small/big tier user के लिए auto-selected।

### Margin Math

```
Per million output tokens revenue:  $15
Per million output tokens cost:     $1.50  (well-utilized fp8 27B, on-prem)
Gross margin:                        90%

Per million cached input revenue:   $0.30
Per million cached input cost:      ~$0.05  (prefix cache के साथ almost free)
Gross margin:                        ~83%
```

Fixed costs (idle replicas, support, R&D, edge infra, payment fees) के बाद, real net margin self-hosted product के लिए **50-70%** के आसपास lands है, **20-40%** अगर आप किसी और का API resell कर रहे हो।

### Caps: Agent-loop Bankruptcy Problem

एक poorly-designed agent 100,000 tokens `<think>` के लिए recurse कर सकता है, tool failure पर 50,000 retry-loop और chew up कर सकता है, और 90 minutes में एक year के free credits खा सकता है। Real failure mode जिसने कई companies को hit किया है। हमेशा implement करो:

- **Per-request `max_tokens` cap** (chat के लिए 10-32k; explicit agent endpoints के लिए 100-200k)।
- **Per-conversation token budget** (e.g., warning से पहले एक chat session पर 200k cumulative)।
- **Per-user daily quota** 80% पर warning, 100% पर hard cap के साथ।
- **Spend alerts** $10, $50, $100 per user पर (configurable)।
- **Soft kill** — जब user threshold cross करे तो उसे smaller model पर switch करो।
- **Hard kill** — clear `"budget_exceeded"` body के साथ 429 return करो।
- **Idle-loop detector** — tool-retry loop में stuck agents (same tool call N times) circuit breaker trip करते हैं।

इन्हें defaults बनाओ, opt-in नहीं।

### Anti-pricing-mistakes Checklist

- Input और output को same price मत करो। Output 5-10× ज़्यादा expensive है।
- Cache-hit rate को ignore मत करो। Caching के बिना product के unit costs एक caching वाले से 5-10× higher हैं।
- Agents को unbounded run मत करने दो। 99th-percentile token usage measure करो; उसके लिए build करो, median के लिए नहीं।
- Reasoning tokens को normal output की तरह price मत करो — वो 5× ज़्यादा numerous हो सकते हैं।
- Flat resell मत करो। हमेशा whales के लिए usage-based component रखो।
- Users को bills से surprise मत करो। UI में running token counters दिखाओ।

### A Worked Example: एक Coding Tool Pricing करना

आप एक Cursor-style assistant build करते हो। Target user: **coder IDE chat** archetype (~$24/user/month at API cost)।

- Build cost: full caching के साथ API rates पर ~$24 raw।
- Infra, support, R&D overhead add करो: ~$32 effective fully loaded cost।
- Target gross margin 60% → list price $80/month।
- Market reality: competitors $20/month हैं Cursor-Pro-tier offerings के लिए।
- **Solution**: 70% traffic को cheaper small model पर route करो (blended cost ~$10 तक drop), big model को $40/month पर metered overage के साथ power users को gated "max-tier" feature बनाओ।

ये exactly वो multi-tier strategy है जिस पर हर successful 2026 coding tool converged है। **Hardware bandwidth, caching, smart routing, और user-archetype pricing सब same equation में stack होते हैं: per million tokens served आप कितनी margin keep करते हो।**

---

## 14. Security & Abuse

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

## 15. Traffic से पहले Rig Test करना

DNS flip करने से पहले, run करो:

1. Realistic prompt distributions के साथ 30 min के लिए target QPS के 1.5× पर **Load test**। TTFT/TPOT/error rate measure करो। (Tools: `k6`, `vegeta`, `genai-perf`)।
2. 12+ hours के लिए 1× target पर **Soak test** — memory leaks, cache fragmentation catch करता है।
3. **Chaos test** — load के दौरान एक replica kill करो। Verify request retry के via succeed करता है; alert fires।
4. **Long-context test** — 100k tokens पर कुछ requests। Verify वो everyone को starve नहीं करते।
5. **Burst test** — 10 seconds में 0 → target QPS के 1.5×। Queue depth और TTFT देखो।
6. **Tenant isolation** — different system prompts के साथ दो simulated tenants, prefix cache hashing में no leakage verify करो।
7. **Cache regression test** — canonical traffic के लिए `cached_tokens / prompt_tokens` measure करो और alert करो अगर ये 20% से ज़्यादा drift करे।
8. **Cost-cap test** — एक runaway agent loop simulate करो और verify करो कि spend cap seconds में fire करे, $1,000 bill के बाद नहीं।

ये सब एक staging environment में करो जो prod को mirror करे। ये step skip करना वो है जिससे आप hard way से सीखते हो कि vLLM का `max_num_seqs` too low set था।

---

## 16. एक One-page Checklist

इसे "production" call करने से पहले:

- [ ] Engine: vLLM या SGLang FP8 weights + KV के साथ।
- [ ] Engine पर Prefix caching enabled **और** cache hits के लिए designed prompt structure।
- [ ] Chunked prefill enabled।
- [ ] Decode-heavy traffic के लिए Speculative decoding (MTP / EAGLE)।
- [ ] अपने SLOs के लिए `max_num_seqs` और `max_num_batched_tokens` tuned।
- [ ] एक classifier के साथ Two-tier model routing (small/big)।
- [ ] Prefix-aware sticky load balancing।
- [ ] Hardware HBM bandwidth और capacity से chosen, सिर्फ़ FLOPs से नहीं।
- [ ] Health checks जो एक real completion exercise करें।
- [ ] Prometheus metrics + Grafana dashboards (TTFT, TPOT, queue depth, KV usage, cache hit rate)।
- [ ] TTFT P95, queue depth, 5xx rate, KV usage, cache-hit drift पर Alerts।
- [ ] Sensible floor के साथ Autoscaler, leading indicator पर scale-up।
- [ ] Per-tenant rate limits और quotas।
- [ ] Per-user spend caps और idle-agent-loop detection।
- [ ] Traceability के लिए Per-request `request_id` header।
- [ ] Multi-region या at minimum multi-zone deployment।
- [ ] Soak + load + chaos + cache-regression + cost-cap tested।
- [ ] On-call runbook: कैसे roll back करें, कैसे एक node drain करें, कैसे quotas bump करें।

अगर कोई item "no" या "I think so" है, आपके पास अभी 10k-QPS-grade system नहीं है।

---

## 17. 2026 Cheat Sheet

- **vLLM safe default है। Prefix-heavy workloads के लिए SGLang। Max throughput के लिए TRT-LLM।**
- H100/B200 पर **FP8 weights + FP8 KV**।
- **HBM bandwidth, FLOPs नहीं**, per-user tokens/sec set करता है। सबसे fattest pipe वाला GPU buy करो।
- Chat के लिए **Prefix caching** biggest free win है — prompts को frozen-then-liquid के रूप में design करो।
- **Speculative decoding** (MTP / EAGLE-3) second biggest है।
- **Easy queries एक smaller model पर route करो।** Scale पर एक big model से two tiers > अच्छा है।
- ~2k QPS से ऊपर या very long contexts के लिए **Prefill/decode disaggregation**।
- सिर्फ़ QPS नहीं, **Queue depth पर autoscale करो**।
- **Cold start 1-3 minutes है** — एक generous floor रखो।
- **Optimization से पहले Observability।** जो आप देख नहीं सकते उसे fix नहीं कर सकते।
- "user" से नहीं, **archetype से Price करो**। Coding agents एक casual chat user से 360× cost करते हैं।
- **Output tokens cached input से 5-10× ज़्यादा expensive हैं।** उन्हें differently bill करो।
- **हर चीज़ Cap करो** — per-request, per-conversation, per-day, per-spend। Agent loops कोई भी uncapped budget find कर लेंगे।
- **Frontier-quality chat के 10k QPS ≈ 50-150 GPUs**, जो एक order-of-magnitude reality check है जिसे ज़्यादातर teams under-estimate करती हैं।

---

## और गहराई से

- **vLLM docs** — `docs.vllm.ai`, canonical reference. "Production" section पढ़ो।
- **SGLang docs** — `sglang.ai/docs`, especially RadixAttention।
- **NVIDIA NIM / TensorRT-LLM cookbook** — NVIDIA-native deployments के लिए।
- **DeepSeek inference paper (2024)** — disaggregated prefill/decode architecture explained।
- **Ray Serve LLM tutorial** — K8s पर autoscaling के साथ multi-replica vLLM।
- **Anthropic, OpenAI, Google cache-control docs** — provider-side caching pricing और TTLs के लिए canonical references।
- **NVIDIA HBM3e / HBM4 hardware briefs** और Tim Dettmers की GPU buying guide — marketing के बजाय memory पर hardware decisions लेने के लिए।
- **k6 + genai-perf** — load testing toolkits actually streaming token APIs के लिए designed।
- **OpenAI / Anthropic public API SLOs और incident postmortems** — production LLM serving scale पर कैसा दिखता है उसके most honest accounts।

That's curriculum का end. अगर आपने chapters 00 → 17 पढ़े, आप अब एक small LM pretrain, scratch से build, fine-tune, quantize, और दस हज़ार users को serve कर सकते हो खुद को bankrupt किए बिना। **अब कुछ ship करो।**
