# 05 · Model Scale — Size, Data, और Compute

> **TL;DR** Model **quality** smoothly तीन चीज़ों से scale करती है: parameter count `N`, training tokens `D`, और compute `C ≈ 6 N D`। **Chinchilla** ने कहा था कि `D ≈ 20 × N` *training cost के लिए* compute-optimal है। 2026 में कोई Chinchilla-optimal train नहीं करता — small models deliberately **over-trained** हैं (Llama 3.1-8B ने 15T tokens देखे, Qwen2.5-7B ने 18T) क्योंकि cheap inference cheap training से ज़्यादा matter करता है। साथ ही, **test-time compute** (long chain-of-thought) एक दूसरा axis खोलता है: एक small model जो ज़्यादा देर तक सोचता है, अक्सर एक big वाले को beat करता है जो तुरंत answer देता है।

## 1. Scaling-law Mental Model

2020 में, Kaplan et al. ने दिखाया कि LLM loss power law की तरह गिरती है:

```
loss(N, D) ≈ A / Nᵅ + B / Dᵝ + irreducible_floor
```

हर axis (params `N`, tokens `D`) diminishing returns देता है, लेकिन predictably। **आप small experiments से big experiments तक extrapolate कर सकते हो।** ये LLM training के बारे में single most important practical fact है।

लगभग हर team real training run इस तरह start करती है:

1. 5-10 small models (10M-1B params) different `(N, D)` combinations पर train करो।
2. Scaling law fit करो।
3. Big run के loss को predict करो।
4. Big run पर millions of dollars spend करो और ±0.05 nats सही होओ।

अगर आपका big model prediction से बहुत off land करता है, आपके code में bug है।

---

## 2. Chinchilla: DeepMind ने Map कैसे Redraw किया

Kaplan की 2020 advice थी "models को mostly bigger बनाओ।" Hoffmann et al. (DeepMind, 2022, "Chinchilla") ने ज़्यादा careful study की और पाया:

> Fixed compute budget के लिए, **smaller train करो, more tokens पर**। Compute-optimal ratio roughly **20 tokens per parameter** है।

तो Chinchilla-70B (1.4 T tokens) ने Gopher-280B (300 B tokens) को far less compute पर beat किया।

Compute `C ≈ 6 × N × D` है (forward + backward × ≈ 2 FLOPs प्रत्येक per param per token)। Budget `C` के लिए:

```
N* ≈ sqrt(C / 6 / 20)         # optimal params
D* ≈ 20 × N*                  # optimal tokens
```

Quick numbers:

| Compute (FLOPs) | Optimal N | Optimal D |
|-----------------|-----------|-----------|
| 1e21 | ~1.3 B | 26 B |
| 1e22 | ~4 B | 80 B |
| 1e23 | ~13 B | 260 B |
| 1e24 | ~40 B | 800 B |
| 1e25 | ~130 B | 2.6 T |

ये **training-optimal** frontier है। ये loss-per-FLOP को minimize करता है।

---

## 3. कोई Chinchilla-optimal क्यों नहीं रहा

Chinchilla **सिर्फ़ training cost** को optimize करता है। लेकिन:

- Small open-weights models training के बाद **billions of times serve** होते हैं।
- Inference cost ≈ `2 × N` FLOPs per generated token।
- Bigger N = forever per inference more expensive।

तो आप smaller model पर extra training compute spend करना prefer करोगे जो forever faster run होगा। ये है **over-training**, और हर modern open-weights small model ये करता है:

| Model | Params N | Training tokens D | Tokens / param |
|-------|----------|-------------------|----------------|
| Chinchilla 2022 | 70 B | 1.4 T | 20 |
| Llama 2-7B | 7 B | 2 T | ~290 |
| Llama 3-8B | 8 B | 15 T | ~1900 |
| Llama 3.1-8B | 8 B | 15 T | ~1900 |
| Qwen2.5-7B | 7 B | 18 T | ~2600 |
| Qwen3-4B | 4 B | 36 T | ~9000 |
| SmolLM2-1.7B | 1.7 B | 11 T | ~6500 |

Chinchilla से loss-per-FLOP बदतर है — लेकिन **loss-per-inference-FLOP** *much* better है। Beyond-Chinchilla paper (Sardana et al. 2023) ने इसे formalize किया: inference-aware budget के लिए optimal `D/N` easily 10-100× हो सकता है।

> **2026 small models के लिए practical rule:** जितने high-quality tokens पर train कर सकते हो उतने पर train करो, generally **500-2000+ tokens per parameter**।

---

## 4. Compute Formula — और इसे कैसे Use करें

```
C ≈ 6 × N × D                 # total training FLOPs
```

Examples:
- 1B model को 100B tokens पर train करो: `6 × 1e9 × 1e11 = 6e20` FLOPs।
- एक H100 ~1e15 BF16 FLOPs/s sustained (~30% of peak) करता है।
- → 6e20 / 1e15 = 6e5 seconds = ~7 GPU-days।
- 8 H100s पर good DDP के साथ: ~1 day।

7B model 2T tokens पर: `6 × 7e9 × 2e12 = 8.4e22` FLOPs। ~8.4e22 / 1e15 ≈ 1000 GPU-days। 256 H100s पर, ~4 days। ये roughly वो है जो Llama 2-7B-class model train करने में लगता है।

ये back-of-envelope ज़्यादातर experiments plan करने के लिए काफ़ी है। **`C = 6ND` memorize करो।**

### Activations के बारे में क्या?

Memory और compute same नहीं हैं। `6ND` formula compute के लिए है। Memory dominated है:

- **Parameters**: `2N` bytes (bf16)।
- **Gradients**: `2N` bytes।
- **Optimizer state (AdamW)**: `4N + 4N = 8N` bytes (fp32 momentum + variance)।
- **Activations**: `B × T × layers × D` पर depend; activation checkpointing के साथ, much less।

7B AdamW model को सिर्फ़ params + grads + opt state के लिए ~`12N = 84 GB` चाहिए — इसलिए FSDP / ZeRO sharding (chapter 14)।

---

## 5. N कहां से आता है?

ज़्यादातर params दो जगह रहते हैं:

- **Embeddings + LM head**: `2 × V × D` (दो अगर tied नहीं)।
- **Per layer**: 4 attention projections (`4 × D²`) + FFN (`2 × D × F` या SwiGLU के लिए `3 × D × F`)। SwiGLU के साथ `F = 4 D × 2/3 ≈ 2.67 D`, ये roughly per layer `8 D²` है।
- Plus norms, biases (negligible)।

एक rough model:

```
N ≈ V × D × 2 + L × 12 × D²
```

Llama-3-8B के लिए: `V = 128k`, `D = 4096`, `L = 32`, `H = 32`, `F = 14336`। Plug in → ~7.5 B। Close enough.

ये formula आपको बताता है कि **depth doubling params doubles करता है; width doubling params quadruples करती है**। Most modern models depth/width इस तरह pick करते हैं कि `D ≈ 64 × sqrt(L)` roughly. यहां mistakes hurt करती हैं।

---

## 6. Emergence — और Modern Revision

2022 के आसपास papers ने claim किया कि **emergent abilities** suddenly कुछ scale (e.g., 70 B) के बाद दिखती हैं। बाद में analysis (Schaeffer et al., "Are Emergent Abilities a Mirage?") ने दिखाया कि कई "emergent" curves **discrete metrics** (exact-match accuracy) का artifact हैं। Continuously measure करने पर (e.g., correct answer का log-prob), gains smooth हैं।

Practical implication: **एक sudden phase transition पर bet मत लगाओ।** अगर आपका scaling curve कहता है "ये small model MMLU पर 30% पाता है," 10× scale करने से 40-50% smoothly मिलेगा, 90% का leap नहीं।

बची हुई "true" emergence *complex behavior* के level पर है (multi-step reasoning, tool use), और इसे **right data पर training** से unlock किया जा सकता है (chain-of-thought, tool-use traces) sheer scale से ज़्यादा।

---

## 7. Test-time Compute: दूसरा Axis

2025 में OpenAI के o1 और DeepSeek R1 ने ये obvious बना दिया: **एक model जो ज़्यादा देर सोचता है, एक much larger model को outperform कर सकता है जो एक बार सोचता है**।

कैसे: model को chain-of-thought (CoT) traces के साथ train करो; inference पर, उसे answer से पहले hundreds या thousands "thinking" tokens generate करने दो। ये per query ज़्यादा inference compute cost करता है लेकिन और training compute नहीं।

Reasoning-model papers (DeepSeek-R1, Tülu 3 with verifiable rewards, OpenR1) दिखाते हैं कि **एक 7B reasoning model math/code पर एक 70B non-reasoning model को beat कर सकता है**। तो 2026 में:

- Chat / writing के लिए → bigger model, normal decoding।
- Math / code / analysis के लिए → smaller reasoning model with long thinking।
- Choice अब per-task है, per-deployment नहीं।

Practically, इसका मतलब scale अब सिर्फ़ `N`, `D`, `C_train` नहीं है। एक चौथा axis `C_test` (per query generated tokens) है। Test-time compute के लिए scaling laws अब exist करते हैं (OpenAI का "scaling reasoning" report, DeepMind का "Scaling Inference-Time Compute")।

---

## 8. अपने Project के लिए N, D, C Pick करना

कुछ quick prescriptions:

### "मैं कुछ small सीखना / replicate करना चाहता हूं"
- **N = 100M-500M, D = 10B-30B**। Single GPU node पर hours में train होता है, perfectly capable readable English produce करने का। Reference: nanoGPT, SmolLM, MicroLlama।

### "मुझे fine-tuning के लिए useful base model चाहिए, single-node budget"
- **N = 1B-3B, D = 100-300B tokens।** FineWeb-Edu + StarCoder2 use करो। 8× H100 पर 1-2 weeks में train होता है। Result 2023 के 7B models के साथ rivals।

### "मुझे 2026 में competitive chat model चाहिए"
- **N = 4B-14B, D = 4-15T tokens।** Cluster-scale (64+ H100) चाहिए। Pretraining के बाद, SFT + DPO/KTO करो। Qwen2.5/Llama-3.1 territory match करो।

### "मुझे reasoning model चाहिए"
- एक strong base से start करो। Verifiable rewards के साथ RL apply करो (math/code के लिए RLHF replaced by RLVR)। 7-32B params often काफी हैं।

### "मुझे frontier model चाहिए"
- मत करो। Open वाले use करो। Frontier को 100M+ GPU-hours और 100+ engineers चाहिए। अगर आपको really must करना है, अपने compute provider से बात करो; आपको चाहिए होगा।

---

## 9. Width vs Depth vs Heads

Fixed N के अंदर, आप layers `L`, model dim `D`, heads `H`, और FFN dim `F` choose करते हो। Common modern shapes:

| Model | L | D | H | F | N |
|-------|---|---|---|---|---|
| Llama-3.2-1B | 16 | 2048 | 32 | 8192 | 1.2 B |
| Llama-3-8B | 32 | 4096 | 32 | 14336 | 8 B |
| Qwen2.5-7B | 28 | 3584 | 28 | 18944 | 7 B |
| Qwen2.5-14B | 48 | 5120 | 40 | 13824 | 14 B |
| DeepSeek-V3-base | 61 | 7168 | 128 | 18432* (MoE) | 671 B (37 B active) |

*DeepSeek-V3 shared + routed experts के साथ MoE use करता है, chapter 13 देखो।

Empirically hold up करने वाले rules of thumb:

- **`head_dim = D / H = 64 या 128`**। 64 से नीचे मत जाओ; Tensor Cores उसे चाहते हैं।
- **Slightly deeper-than-wide** small N पर slightly wider-than-deep को beat करता है (better learning dynamics)।
- **`F ≈ 2.67 × D` for SwiGLU** FFN compute को 4× ReLU FFN के बराबर रखता है।
- **Vocab size 128k+** multilingual / strong tokenizer के लिए; smaller vocabs (32k) speed के लिए।

---

## 10. Data scaling vs Model scaling: कौन सा lever pull करें?

आपके पास compute का एक extra unit है। उसे spend करो:

- More data पर? (Train longer.)
- Bigger model पर? (Same data, more params.)

Chinchilla curve कहता है optimum के पास इसे roughly equally split करो। लेकिन दो real-world wrinkles:

- **Inference**: more-data को better choice बनाता है (§3 देखो)।
- **Data wall**: high-quality tokens finite हैं। अगर आप already best 10T tokens use कर चुके हो, data doubling lower-quality data use करना means → diminishing returns. Currently front-runners 15-30 T pretraining tokens पर इस wall को hit करते हैं।

2026 में, **synthetic data** (Cosmopedia, Phi, Llama-Nemotron) data wall का answer है। आप strong teacher model use करके clean, on-distribution data generate करते हो। ये math, code, और structured reasoning के लिए astonishingly well काम करता है।

---

## 11. Worked Example: 1B model planning

Goal: एक useful 1 B base model।

- `D ≈ 1.5T tokens` pick करो (over-train ~1500×)।
- Compute ≈ `6 × 1e9 × 1.5e12 = 9e21 FLOPs`।
- 8× H100 पर ~1e15 flops/s sustained → ~6e6 / 86400 ≈ ~10 days। Doable।
- Architecture: `L=22, D=2048, H=16, head_dim=128, F=5632 (SwiGLU)`। ≈1.1 B yields।
- Tokenizer: 32 k या 50 k BPE; tied embeddings; embedding `V × D = 50e3 × 2048 ≈ 100 M` params share करता है (significant)।
- Optimizer state: AdamW fp32 को `8N = 8 GB` चाहिए (हर H100 80 GB fits)।
- Mix: 70% FineWeb-Edu, 20% StarCoder2-filtered, 5% FineMath, 5% Cosmopedia।
- High-quality + math-heavy mix के साथ last 100 B tokens anneal करो।

Run, evaluate (chapter 14), iterate. ये एक real, achievable 2026 weekend-side-project specification है।

---

## 12. tl;dr-of-the-tl;dr

- **Loss predictable है।** Small experiments fit करो, extrapolate करो।
- **Chinchilla-optimal मत बनो — over-train करो।** Inference cost training cost से बड़ा है।
- **`6 N D` FLOPs।** ये memorize करो।
- **Test-time compute real है।** एक small reasoning model often एक big plain वाले को beat करता है।
- **Data quality > quantity > model size**, roughly leverage के order में 2026 में।

---

## और गहराई से

- Hoffmann et al. 2022 — "Training Compute-Optimal LLMs" (Chinchilla)। Optimum समझने के लिए एक बार पढ़ो।
- Sardana et al. 2023 — "Beyond Chinchilla-Optimal," inference-aware version।
- Hoffmann et al. 2025 — modern architectures के लिए updated scaling-law fits।
- DeepSeek-V3 technical report — MoE के साथ modern compute-budget arithmetic।
- Llama 3, Qwen 2.5, Qwen3 technical reports — real teams द्वारा use किए गए real recipes।

Next: **[06-tokenization-embeddings.md](./06-tokenization-embeddings.md)** — text को numbers में बदलना।
