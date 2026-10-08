# 05 · Model Scale — Size, Data, and Compute

> **TL;DR** Model **quality** scales smoothly with three things: parameter count `N`, training tokens `D`, and compute `C ≈ 6 N D`. **Chinchilla** said `D ≈ 20 × N` is compute-optimal *for training cost*. In 2026 nobody trains Chinchilla-optimal anymore — small models are deliberately **over-trained** (Llama 3.1-8B saw 15 T tokens, Qwen2.5-7B saw 18 T) because cheap inference matters more than cheap training. Plus, **test-time compute** (long chain-of-thought) opens a second axis: a small model thinking longer often beats a big one answering immediately.

## 1. The scaling-law mental model

In 2020, Kaplan et al. showed that LLM loss falls like a power law:

```
loss(N, D) ≈ A / Nᵅ + B / Dᵝ + irreducible_floor
```

Each axis (params `N`, tokens `D`) gives diminishing returns, but predictably. **You can extrapolate from small experiments to large ones.** This is the single most important practical fact about LLM training.

Pretty much every team starts a real training run by:

1. Training 5-10 small models (10M-1B params) on different `(N, D)` combinations.
2. Fitting the scaling law.
3. Predicting the loss of the big run.
4. Spending millions of dollars on the big run and being right ±0.05 nats.

If your big model lands far off the prediction, your code has a bug.

---

## 2. Chinchilla: how DeepMind redrew the map

Kaplan's 2020 advice was "make models bigger, mostly." Hoffmann et al. (DeepMind, 2022, "Chinchilla") did a more careful study and found:

> For a fixed compute budget, **train smaller, on more tokens**. The compute-optimal ratio is roughly **20 tokens per parameter**.

So Chinchilla-70B (1.4 T tokens) beat Gopher-280B (300 B tokens) at far less compute.

The compute is `C ≈ 6 × N × D` (forward + backward × ≈ 2 FLOPs each per param per token). For a budget `C`:

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

This is the **training-optimal** frontier. It minimizes loss-per-FLOP.

---

## 3. Why nobody is Chinchilla-optimal anymore

Chinchilla optimizes **training cost only**. But:

- Small open-weights models are **served billions of times** after training.
- Inference cost ≈ `2 × N` FLOPs per generated token.
- Bigger N = more expensive per inference forever.

So you'd rather spend extra training compute on a smaller model that runs faster forever. This is **over-training**, and every modern open-weights small model does it:

| Model | Params N | Training tokens D | Tokens / param |
|-------|----------|-------------------|----------------|
| Chinchilla 2022 | 70 B | 1.4 T | 20 |
| Llama 2-7B | 7 B | 2 T | ~290 |
| Llama 3-8B | 8 B | 15 T | ~1900 |
| Llama 3.1-8B | 8 B | 15 T | ~1900 |
| Qwen2.5-7B | 7 B | 18 T | ~2600 |
| Qwen3-4B | 4 B | 36 T | ~9000 |
| SmolLM2-1.7B | 1.7 B | 11 T | ~6500 |

Loss-per-FLOP is worse than Chinchilla — but **loss-per-inference-FLOP** is *much* better. The Beyond-Chinchilla paper (Sardana et al. 2023) formalized this: the optimal `D/N` for an inference-aware budget can easily be 10-100×.

> **Practical rule for 2026 small models:** train on as many high-quality tokens as you can afford, generally **500-2000+ tokens per parameter**.

---

## 4. The compute formula — and how to use it

```
C ≈ 6 × N × D                 # total training FLOPs
```

Examples:
- Train a 1 B model on 100 B tokens: `6 × 1e9 × 1e11 = 6e20` FLOPs.
- An H100 does ~1e15 BF16 FLOPs/s sustained (~30% of peak).
- → 6e20 / 1e15 = 6e5 seconds = ~7 GPU-days.
- On 8 H100s with good DDP: ~1 day.

For a 7 B model on 2 T tokens: `6 × 7e9 × 2e12 = 8.4e22` FLOPs. ~8.4e22 / 1e15 ≈ 1000 GPU-days. On 256 H100s, ~4 days. This is roughly what it takes to train a Llama 2-7B-class model.

This back-of-envelope is enough to plan most experiments. **Memorize `C = 6ND`.**

### What about activations?

Memory and compute aren't the same. The `6ND` formula is for compute. Memory is dominated by:

- **Parameters**: `2N` bytes (bf16).
- **Gradients**: `2N` bytes.
- **Optimizer state (AdamW)**: `4N + 4N = 8N` bytes (fp32 momentum + variance).
- **Activations**: depends on `B × T × layers × D`; with activation checkpointing, much less.

A 7 B AdamW model needs ~`12N = 84 GB` just for params + grads + opt state — hence FSDP / ZeRO sharding (chapter 14).

---

## 5. Where does N come from?

Most params live in two places:

- **Embeddings + LM head**: `2 × V × D` (two if not tied).
- **Per layer**: 4 attention projections (`4 × D²`) + FFN (`2 × D × F` or `3 × D × F` for SwiGLU). For SwiGLU with `F = 4 D × 2/3 ≈ 2.67 D`, that's roughly `8 D²` per layer.
- Plus norms, biases (negligible).

A rough model:

```
N ≈ V × D × 2 + L × 12 × D²
```

For Llama-3-8B: `V = 128k`, `D = 4096`, `L = 32`, `H = 32`, `F = 14336`. Plug in → ~7.5 B. Close enough.

This formula tells you that **doubling depth doubles params; doubling width quadruples params**. Most modern models pick depth/width such that `D ≈ 64 × sqrt(L)` roughly. Mistakes here hurt.

---

## 6. Emergence — and the modern revision

Around 2022 papers claimed **emergent abilities** appear suddenly past some scale (e.g., 70 B). Later analysis (Schaeffer et al., "Are Emergent Abilities a Mirage?") showed many "emergent" curves are an artifact of **discrete metrics** (exact-match accuracy). When measured continuously (e.g., log-prob of correct answer), the gains are smooth.

Practical implication: **don't bet on a sudden phase transition.** If your scaling curve says "this small model gets 30% on MMLU," scaling 10× will give 40-50% smoothly, not a leap to 90%.

The remaining "true" emergence is at the level of *complex behavior* (multi-step reasoning, tool use), and it can be unlocked by **training on the right data** (chain-of-thought, tool-use traces) more than by sheer scale.

---

## 7. Test-time compute: the second axis

In 2025 OpenAI's o1 and DeepSeek R1 made it obvious: **a model that thinks longer can outperform a much larger model that thinks once**.

How: train the model with chain-of-thought (CoT) traces; at inference, let it generate hundreds or thousands of "thinking" tokens before its answer. This costs more inference compute per query but no more training compute.

Reasoning-model papers (DeepSeek-R1, Tülu 3 with verifiable rewards, OpenR1) show that **a 7 B reasoning model can beat a 70 B non-reasoning model on math/code**. So in 2026:

- For chat / writing → bigger model, normal decoding.
- For math / code / analysis → smaller reasoning model with long thinking.
- The choice is now per-task, not per-deployment.

Practically, this means scale isn't only `N`, `D`, `C_train` anymore. There's a fourth axis `C_test` (tokens generated per query). Scaling laws for test-time compute exist now (OpenAI's "scaling reasoning" report, DeepMind's "Scaling Inference-Time Compute").

---

## 8. Picking N, D, C for your project

Some quick prescriptions:

### "I want to learn / replicate something small"
- **N = 100M-500M, D = 10B-30B**. Trains in hours on a single GPU node, perfectly capable of producing readable English. Reference: nanoGPT, SmolLM, MicroLlama.

### "I want a useful base model for fine-tuning, single-node budget"
- **N = 1B-3B, D = 100-300B tokens.** Use FineWeb-Edu + StarCoder2. Trains in 1-2 weeks on 8× H100. Result rivals 2023's 7B models.

### "I want a competitive chat model in 2026"
- **N = 4B-14B, D = 4-15T tokens.** Need cluster-scale (64+ H100). After pretraining, do SFT + DPO/KTO. Match Qwen2.5/Llama-3.1 territory.

### "I want a reasoning model"
- Start from a strong base. Apply RL with verifiable rewards (RLHF replaced by RLVR for math/code). 7-32B params often enough.

### "I want a frontier model"
- Don't. Use the open ones. The frontier requires 100M+ GPU-hours and 100+ engineers. If you really must, talk to your compute provider; you'll need it.

---

## 9. Width vs depth vs heads

Within a fixed N, you choose layers `L`, model dim `D`, heads `H`, and FFN dim `F`. Common modern shapes:

| Model | L | D | H | F | N |
|-------|---|---|---|---|---|
| Llama-3.2-1B | 16 | 2048 | 32 | 8192 | 1.2 B |
| Llama-3-8B | 32 | 4096 | 32 | 14336 | 8 B |
| Qwen2.5-7B | 28 | 3584 | 28 | 18944 | 7 B |
| Qwen2.5-14B | 48 | 5120 | 40 | 13824 | 14 B |
| DeepSeek-V3-base | 61 | 7168 | 128 | 18432* (MoE) | 671 B (37 B active) |

*DeepSeek-V3 uses MoE with shared + routed experts, see chapter 13.

Rules of thumb that hold up empirically:

- **`head_dim = D / H = 64 or 128`**. Don't go below 64; Tensor Cores want it.
- **Slightly deeper-than-wide** beats slightly wider-than-deep at small N (better learning dynamics).
- **`F ≈ 2.67 × D` for SwiGLU** keeps the FFN compute equal to a 4× ReLU FFN.
- **Vocab size 128k+** for multilingual / strong tokenizer; smaller vocabs (32k) for speed.

---

## 10. Data scaling vs model scaling: which lever to pull?

You have one extra unit of compute. Spend it on:

- More data? (Train longer.)
- Bigger model? (Same data, more params.)

The Chinchilla curve says split it roughly equally near the optimum. But two real-world wrinkles:

- **Inference**: makes more-data the better choice (see §3).
- **Data wall**: high-quality tokens are finite. If you've already used the best 10T tokens, doubling data means using lower-quality data → diminishing returns. Currently the front-runners hit this wall around 15-30 T pretraining tokens.

In 2026, **synthetic data** (Cosmopedia, Phi, Llama-Nemotron) is the answer to the data wall. You generate clean, on-distribution data using a strong teacher model. This works astonishingly well for math, code, and structured reasoning.

---

## 11. A worked example: planning a 1B model

Goal: a useful 1 B base model.

- Pick `D ≈ 1.5T tokens` (over-train ~1500×).
- Compute ≈ `6 × 1e9 × 1.5e12 = 9e21 FLOPs`.
- On 8× H100 at ~1e15 flops/s sustained → ~6e6 / 86400 ≈ ~10 days. Doable.
- Architecture: `L=22, D=2048, H=16, head_dim=128, F=5632 (SwiGLU)`. Yields ≈1.1 B.
- Tokenizer: 32 k or 50 k BPE; tied embeddings; embedding shares `V × D = 50e3 × 2048 ≈ 100 M` params (significant).
- Optimizer state: AdamW fp32 needs `8N = 8 GB` (fits each H100 80 GB).
- Mix: 70% FineWeb-Edu, 20% StarCoder2-filtered, 5% FineMath, 5% Cosmopedia.
- Anneal last 100 B tokens with high-quality + math-heavy mix.

Run, evaluate (chapter 14), iterate. This is a real, achievable 2026 weekend-side-project specification.

---

## 12. The tl;dr-of-the-tldr

- **Loss is predictable.** Fit small experiments, extrapolate.
- **Don't be Chinchilla-optimal — over-train.** Inference cost outweighs training cost.
- **`6 N D` FLOPs.** Memorize this.
- **Test-time compute is real.** A small reasoning model often beats a big plain one.
- **Data quality > quantity > model size**, in roughly that order of leverage in 2026.

---

## Going deeper

- Hoffmann et al. 2022 — "Training Compute-Optimal LLMs" (Chinchilla). Read once to understand the optimum.
- Sardana et al. 2023 — "Beyond Chinchilla-Optimal," the inference-aware version.
- Hoffmann et al. 2025 — updated scaling-law fits for modern architectures.
- DeepSeek-V3 technical report — modern compute-budget arithmetic with MoE.
- Llama 3, Qwen 2.5, Qwen3 technical reports — real recipes used by real teams.

Next: **[06-tokenization-embeddings.md](./06-tokenization-embeddings.md)** — turning text into numbers.
