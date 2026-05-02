# 25 · Model Releases, Versioning, and Looking Inside the Model

> **TL;DR** Model versions follow a four-tier ladder: **major (X → X+1) = fresh pretrain**, often with a new architecture and tokenizer; **minor (X → X.5) = continual pretrain** on top of the existing checkpoint with new/refreshed data; **point (X.5 → X.6/X.7) = post-training only** (SFT/DPO/GRPO refresh, no pretraining); **patch (X.7 → X.7.1) = targeted bug/safety fix**. Each tier focuses on a different stage of the data pipeline and uses a different set of tools. Inside the model, you can read its mind: **forward hooks** capture activations, **TransformerLens / NNsight** make every layer and head inspectable, **logit-lens** projects intermediate activations through the LM head to see "what does the model think the answer is at layer 17?", and **activation patching** copies activations between forward passes to isolate which neurons cause which behavior. **Ablating a single head, layer, or weight** is the most reliable single-line technique to find what matters in a model.

---

## 1. The release ladder

A real production model family doesn't ship one model — it ships a *cadence*. Each version step costs different money, takes different time, and changes different parts of the pipeline. Treating them all the same is the most expensive mistake.

| Tier | Version step | What changes | What does NOT change | Compute cost | Calendar time |
|------|--------------|---------------|----------------------|--------------|---------------|
| **Major** | X → X+1 | architecture, tokenizer, **fresh pretrain from random init** | brand identity | $$$$ (10⁵-10⁷ GPU-h) | months |
| **Minor** | X → X.5 | **continual pretrain** on top of the checkpoint, new data mix, more tokens, sometimes added modality / longer context | architecture, tokenizer | $$ (~5-20% of major) | weeks |
| **Mid-train / anneal release** | X.5 → X.5b | a **second pretrain phase** on extra-clean / domain-specific data; same arch and weights as starting point | tokenizer | $ (~1-5% of major) | days-weeks |
| **Point** | X.5 → X.6 / X.7 | **post-training only** — new SFT mix, new DPO/GRPO data, updated chat template | base model weights at initialization of post-training | $$ (orders of magnitude cheaper than pretrain) | days |
| **Patch** | X.7 → X.7.1 | targeted SFT on safety/bug examples, prompt template fix | almost everything | $ | hours-days |

You'll see this exact ladder in every public family:

- **Anthropic Claude:** 3 → 3.5 → 3.7 → 4 → 4.5 → 4.6 → 4.7 — the major bumps are new pretrains, the .5 / .6 / .7 are post-training-heavy refreshes (with occasional mid-train uplifts).
- **OpenAI GPT:** 4 → 4o → 4.1 → 5 → 5.5 — `4 → 4o` was a new pretrain with native multimodality; `4 → 4.1` was continual pretrain + post-training; `4.1 → 5` was a fresh pretrain.
- **Meta Llama:** 3 → 3.1 → 3.2 → 3.3 → 4 — `3.1` extended context via continual pretrain; `3.2` added vision via continual pretrain on visual data; `3.3` was post-training; `4` was a new pretrain with new architecture.
- **Qwen:** 2 → 2.5 → 3 → 3.5 → 3.6 — most majors are fresh pretrains; `3.6` was a hybrid-attention overhaul (chapter 16).
- **DeepSeek:** V2 → V2.5 → V3 → V3.1 → V3.2 — `V3` introduced MLA + MoE fine-grained experts and was a fresh pretrain.
- **Gemma:** 2 → 2.5 → 3 → 3n → 4 — `3n` was a new architecture for edge; `4` was a fresh pretrain on a new design (chapter 24).

Some other release patterns layered on top:

- **Distillation children**: a tiny model distilled from the big one, released alongside the major (Llama-3-8B distilled from 70B; Phi family from much larger teachers).
- **Long-context variants**: a `-1M` or `-256K` variant produced by extra continual pretrain + YaRN tuning. Same base, different context.
- **Reasoning variants**: a `-Reasoner` or `-Thinking` variant trained with GRPO on top of the base, released as a sibling.
- **Multimodal extensions**: vision/audio added via continual pretrain on multimodal data.

When you plan a release, **decide which tier you're in first** — the tooling, calendar, and risk profile are completely different.

---

## 2. Major release: fresh pretrain from scratch

You do this when:

- The architecture changes meaningfully (MoE introduced, hybrid attention, new positional encoding, new tokenizer).
- The data mix has a fundamental rewrite (new domain dominance, multilingual focus, much higher quality threshold).
- The compute budget allows it (these are 10⁵-10⁷ GPU-hour runs, $1M-$1B+).

**The pretrain pipeline starts from random init.** Every weight is reset; nothing carries forward. This is the only way to *truly* incorporate a new architecture.

Stages and their focus:

| Stage | What you focus on | Tools |
|-------|------------------|-------|
| **Architecture design** | layer count, hidden dim, FFN ratio, attention type, RoPE base | paper search, scaling-law fit |
| **Tokenizer** | vocab size, BPE merges, multilingual coverage | `tokenizers` (HuggingFace) |
| **Data mix** | shards by domain, quality filtering, dedup at scale, decontamination | `datatrove`, `mosaicml-streaming`, MinHash, classifier filters |
| **Optimizer & schedule** | AdamW vs Muon, warmup, cosine, anneal phase | `torchtitan`, custom |
| **Distributed training** | FSDP2 + TP + EP + PP, fault tolerance, checkpointing | `torchtitan`, `Megatron-LM`, `NeMo`, `DeepSpeed`, `Levanter` |
| **Mid-flight monitoring** | loss curves, grad norm, eval at checkpoints | W&B, TensorBoard |
| **Mid-train annealing** | last 5-10% of tokens on curated mix | same trainer, just config flip |
| **Post-training** | SFT, DPO, GRPO | `trl`, alignment-handbook |
| **Evaluation** | full benchmark suite, internal evals | `lm-evaluation-harness`, `inspect-ai` |

Calendar: 2-6 months from architecture-design lock to release. Most of this is the actual pretrain run.

---

## 3. Minor release: continual pretrain

A minor release **starts from the existing major's checkpoint** and continues training on new data. The architecture, tokenizer, and weights all carry forward. You're effectively *more* training, not new training.

When to do it:

- You have **fresh data** that wasn't in the original mix (new web crawl, new domain, new language).
- You want to **extend context** (8K → 128K via YaRN + continual pretrain on long docs).
- You want to **add a modality** (text → text+vision via continual pretrain on multimodal data).
- You want to **specialize** (general → coding-focused, medical-focused).
- You hit data wall and want to push tokens-per-param higher.

What carries forward, what doesn't:

```
Carries forward:
   weights, optimizer state (sometimes), tokenizer, architecture

Resets (or starts fresh):
   learning-rate schedule (small warmup, lower peak LR)
   data mix (potentially different from base)
   evaluation cadence
```

Critical decisions:

- **LR**: continual pretrain LR is 10-30% of original peak LR (e.g., 3e-4 base → 5e-5 to 1e-4 continual).
- **Tokens budget**: 5-30% of the original pretrain token count.
- **Anneal phase**: very common. Most "X.5" releases are essentially "X + a long anneal phase on better data".
- **Risk: catastrophic forgetting.** If your new data is too narrow, the model forgets general capabilities. Mix in 30-60% general data even when targeting a specialty.

A common 2026 mid-train recipe:

```python
# pseudocode for continual pretrain config diff vs base
config = base_config.copy()
config.lr_peak = 5e-5                   # 1/6 of base 3e-4
config.lr_warmup_steps = 1000           # short warmup
config.lr_schedule = "linear_decay"     # decay to ~0 by end
config.total_tokens = 200_000_000_000   # ~5% of original 4T
config.data_mix = {                     # heavy specialty + general
    "math": 0.30,
    "code": 0.20,
    "general_web_filtered": 0.30,
    "instruction_traces": 0.20,
}
config.start_from_checkpoint = "base/step_1500000.pt"
```

This gives you a "X.5" — same model, more capable, no architecture surgery.

---

## 4. Point release: post-training only

The cheapest, most frequent kind. You take the base model checkpoint and run SFT + DPO/GRPO on it. **No pretraining.** Costs single-digit thousands of GPU-hours. You can ship a point release every 2-4 weeks if you have the data.

Where the focus is:

| Sub-stage | What you focus on | Tools |
|-----------|------------------|-------|
| **SFT data curation** | quality, diversity, refusal coverage, format conformity | `datatrove`, manual review, judge-filtering |
| **SFT** | LoRA / QLoRA / full, hyperparameters | `trl`, `axolotl`, `unsloth`, `alignment-handbook` |
| **Preference data** | UltraFeedback, HelpSteer3, your own | curation pipelines |
| **DPO / KTO / SimPO** | beta sweep, pair quality | `trl.DPOTrainer` |
| **GRPO / RLVR** (reasoning) | verifier quality, reward shaping | `trl.GRPOTrainer`, OpenR1 |
| **Constitutional revisions** | Anthropic-style self-critique | scripted |
| **Eval gauntlet** | win-rate vs prior version, regression on golden set | `promptfoo`, LangSmith, internal harness |

This is where most "we made the model 5% smarter on coding" wins come from in production. Don't overlook point releases.

---

## 5. Patch release

Targeted, surgical fixes:

- A specific safety failure mode → 200-500 SFT examples teaching the better behavior.
- A formatting bug (model outputs broken JSON in a corner case) → SFT on corrected examples.
- A refusal regression (model declines harmless requests) → SFT on permissive examples.
- A regional/cultural bias spotted post-launch.

Patches use the **same pipeline as point releases**, just with smaller, more targeted data. Often a LoRA on top of the existing weights merged in. Calendar: hours to days.

---

## 6. The data pipeline per release tier

Because the focus shifts at each tier, the data pipeline does too. Here's what changes:

| Pipeline component | Major | Minor | Point | Patch |
|---------------------|-------|-------|-------|-------|
| **Raw crawl / web data** | full reset, multi-TB | refresh new data | — | — |
| **Filtering classifiers** | retrained from scratch | reused or fine-tuned | — | — |
| **Tokenization** | new tokenizer trained | same tokenizer | same | same |
| **Sharding** | full re-shard, billions of files | append new shards | — | — |
| **Deduplication** | global MinHash on whole corpus | dedup new vs old | — | — |
| **Mid-train mix** | designed | replaces final-stage mix | — | — |
| **SFT data** | whole new mix curated | extended | refreshed | targeted additions |
| **Preference pairs** | from scratch | augmented | refreshed | targeted |
| **Eval suite** | full reset, including new benchmarks | extended | extended | spot checks |

The tooling stack mirrors this:

```
Tier              Primary data tooling
─────────────────────────────────────────────────────────
Major          datatrove, mosaicml-streaming, Spark for dedup,
               Ray Data, fastText classifiers, n-gram decontam
Minor          add: contextual filters, anneal data assembly,
               long-context document selection
Point          alignment-handbook data prep, judge-based filtering,
               preference-data labeling pipelines
Patch          targeted SFT collection, often hand-curated
```

---

## 7. The tool stack — by role

The tool you reach for depends on the role you're playing right now. Each role has its own canonical 2026 stack.

### Pretraining engineer

- **`torchtitan`** — PyTorch's reference pretraining framework. FSDP2, TP, PP, activation checkpointing, FP8.
- **Megatron-LM / Megatron-Core** (NVIDIA) — battle-tested at frontier scale.
- **NeMo** (NVIDIA) — production framework, end-to-end including data and post-training.
- **Levanter** (Stanford) — JAX-based, reproducible, hashable runs.
- **`llm.c`** (Karpathy) — pure C/CUDA reference, educational gold.
- **OLMo trainer** (AI2) — fully open recipe.

### Post-training (fine-tune / SFT / DPO / GRPO) engineer

- **`trl`** (HuggingFace) — `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`. Canonical.
- **`peft`** — LoRA / QLoRA / DoRA / VeRA implementations.
- **`alignment-handbook`** (HuggingFace) — production-grade scripts.
- **`axolotl`** — popular YAML-driven trainer.
- **`unsloth`** — 2-3× faster QLoRA on consumer GPUs.
- **`OpenRLHF`**, **`OpenR1`** — RL/RLVR pipelines.

### Inference / serving engineer

- **vLLM** — production default.
- **SGLang** — best for prefix-cache-heavy workloads.
- **TensorRT-LLM** — best NVIDIA latency.
- **TGI** — HuggingFace's serving stack.
- **`llama.cpp` / Ollama** — CPU + edge.
- **MLX** — Apple Silicon.

### Research / interpretability engineer

- **TransformerLens** (Neel Nanda et al.) — *the* standard for mechanistic interpretability. Hooks every activation; `model.run_with_cache(...)`.
- **NNsight** (David Bau lab) — context-managed forward passes; works on huge models via `nnsight-server`.
- **`captum`** (Meta) — feature attribution, integrated gradients, layer attributions.
- **`sae_lens`** — sparse autoencoder training & inference for feature discovery.
- **`circuitsvis`** — interactive attention/circuit visualization in Jupyter.
- **`inspectus`** — attention pattern visualization.
- **Goodfire / Transluce / Anthropic's Reach** — commercial / hosted SAE / interpretability tools.

### Observability / monitoring engineer

- **W&B (Weights & Biases)** — research and training run tracking. Default.
- **TensorBoard** — local, simple.
- **MLflow** — experiment tracking.
- **Aim** — open-source W&B alternative.
- **LangSmith / Langfuse / Braintrust / Helicone / Phoenix** — *production* LLM observability (traces, judge runs, cost).
- **Prometheus + Grafana + OpenTelemetry** — infra-side metrics for serving (chapter 17).
- **`wandb-weave`** (W&B) — newer LLM-specific traces.

### Testing / eval engineer

- **`lm-evaluation-harness`** — capability benchmarks.
- **`promptfoo`** — prompt comparison + CI.
- **`inspect-ai`** (UK AISI) — capability + safety eval framework.
- **`deepeval`** — Python-friendly LLM eval.
- **`RAGAS`** — RAG-specific.
- **`pytest`** — your existing Python test runner; combine with golden-set eval.
- **`Garak`** (NVIDIA) — vulnerability scanning.
- **Arena-Hard-Auto / WildBench** — pairwise judge-based eval.

A typical 2026 product team has 2-4 of each category in active use.

---

## 8. Looking inside the model — the mental model

A trained transformer has two kinds of state:

- **Weights** — the parameters you saved to disk. Frozen at inference.
- **Activations** — the values that *flow through* the network during a forward pass. Different for every input.

To debug:

- Inspect **weights** when you ask "what is the model itself like?" (e.g., norm of a layer's projection, attention pattern templates baked in, neuron directions).
- Inspect **activations** when you ask "what is the model doing on this specific input?" (e.g., what does layer 17 think the next token is, which heads attend where).

Both are accessible. The harness for getting at them is the difference between guessing and knowing.

---

## 9. Forward hooks — the simplest possible debugger

Native PyTorch. Three lines:

```python
activations = {}
def hook(name):
    def fn(module, inputs, output):
        activations[name] = output.detach().cpu()
    return fn

# attach
handle = model.layers[17].self_attn.register_forward_hook(hook("layer17_attn"))

# run
model(input_ids)

# inspect
print(activations["layer17_attn"].shape)        # (B, T, D)
print(activations["layer17_attn"].std())

# detach
handle.remove()
```

You can hook **any** `nn.Module` — every linear, every attention, every FFN. Output of `register_forward_hook` is a handle you call `.remove()` on when done; without that, you leak GPU memory.

For a quick layer-by-layer sanity check:

```python
acts = {}
hooks = []
for i, layer in enumerate(model.layers):
    hooks.append(layer.register_forward_hook(
        lambda m, _, o, i=i: acts.setdefault(f"l{i}", o.detach().std().item())
    ))
model(input_ids)
for h in hooks: h.remove()
print({k: round(v, 3) for k, v in acts.items()})
# {'l0': 0.21, 'l1': 0.34, ..., 'l31': 1.05}
```

If one layer's output std is wildly different from its neighbors, something's off — that's how you find a broken layer in seconds.

---

## 10. TransformerLens — the proper interpretability tool

For real interpretability work, `transformer_lens` is the standard. Loads any HuggingFace model, exposes every internal as a named activation, lets you hook anything.

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokens = model.to_tokens("The cat sat on the")

logits, cache = model.run_with_cache(tokens)

# every activation is named and accessible:
cache["blocks.10.attn.hook_pattern"].shape       # (B, H, T, T) — attention pattern
cache["blocks.10.hook_resid_post"].shape         # (B, T, D) — residual after block 10
cache["blocks.10.attn.hook_q"].shape             # (B, T, H, d) — Q vectors
```

`run_with_cache` runs the forward and stores **everything** internally. After it, you can inspect any layer, any head, any neuron. This is the workhorse of the field.

---

## 11. Logit Lens — what does the model think *now*?

The trick: instead of only running the LM head on the final layer's output, run it on **every** intermediate layer's residual stream. You get a per-layer prediction of "what would the next token be if I stopped here?"

```python
# logit lens in 5 lines
final_norm = model.ln_f                  # the final norm before head
unembed    = model.unembed.W_U           # (D, V) tied or untied LM head

for layer_idx in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer_idx}.hook_resid_post"]    # (B, T, D)
    logits_at_layer = final_norm(resid) @ unembed           # (B, T, V)
    top = logits_at_layer[0, -1].argmax().item()
    print(layer_idx, model.to_string(top))
```

You'll see something like:

```
0  the          (random)
1  the          (still random)
...
8  is           (model is starting to converge)
12 mat
16 mat
20 mat          (final answer locked in)
24 mat
```

The layer where the model "decides" gives you a hint about which layers do the work for that input. Apply across many inputs: you find that some prompts decide at layer 12, some at layer 28 — gives you intuition for what depths handle.

**Tuned lens** (Belrose et al. 2023) is the supervised upgrade: instead of using the final norm + unembed at every layer, train a small per-layer linear probe to predict next-token from each layer. More accurate, especially in the middle. `tuned-lens` package implements it.

---

## 12. Ablation: change one thing to find out what it does

The mechanistic interpretability mantra: **to know what a component does, ablate it and see what breaks.**

### Ablate one attention head

```python
def zero_head(module, inputs, output, head_idx):
    # output: (B, T, H, d). Zero head_idx.
    output[:, :, head_idx, :] = 0
    return output

handle = model.blocks[10].attn.register_forward_hook(
    lambda m, i, o: zero_head(m, i, o, head_idx=3)
)
loss_with_ablation = compute_loss(model, eval_set)
handle.remove()
```

Run this for every head in every layer. The heads whose ablation hurts most on a specific task are the ones that *do* that task. Anthropic's Indirect Object Identification (IOI) circuit was discovered exactly this way.

### Ablate one layer

```python
def skip_layer(module, inputs, output):
    # Replace the block's output with its input — equivalent to skipping.
    return inputs[0]

handle = model.blocks[15].register_forward_hook(skip_layer)
# evaluate; remove
```

### Ablate one neuron in the FFN

```python
def zero_neuron(module, inputs, output, idx):
    output[..., idx] = 0     # MLP intermediate is (B, T, F); zero col idx
    return output

handle = model.blocks[10].mlp.gate_proj.register_forward_hook(
    lambda m, i, o: zero_neuron(m, i, o, idx=2048)
)
```

### Ablate one weight (a literal scalar)

```python
# Keep a backup, modify, run, restore.
W = model.blocks[10].mlp.gate_proj.weight
backup = W[5, 1024].clone()
W.data[5, 1024] = 0.0
# evaluate
W.data[5, 1024] = backup    # restore
```

This sounds extreme, but for surgical questions ("does *this exact* weight matter?") it's the cleanest experiment.

### Mean ablation vs zero ablation

Setting things to zero is convenient but also unrealistic — the model never sees pure-zero activations during training. **Mean ablation** is more honest: replace the activation with its mean across the dataset.

```python
mean_acts = []
for batch in dataset:
    out = model.run_with_cache(batch.tokens)[1]["blocks.10.hook_z"]
    mean_acts.append(out.mean(dim=(0,1)))
mean_z = torch.stack(mean_acts).mean(dim=0)   # (H, d)

def mean_ablate(m, i, o):
    o[:, :, 3] = mean_z[3]    # replace head 3 with the dataset mean
    return o
```

Mean ablation results are usually closer to "what would the model do without this component" than zero ablation.

---

## 13. Activation patching — the most powerful technique

Instead of zeroing out, **copy an activation from one input to another**. If the second input now behaves like the first did at that point, you've localized the behavior.

A canonical example: "patch the attention output of layer 8, head 5, on the indirect-object position from a clean run into a corrupted run." If the corrupted run now gets the right answer, you've shown that head 5 in layer 8 carries the indirect-object information.

```python
# pseudo-code, TransformerLens makes this much cleaner
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, _         = model(corrupt_tokens)

def patch(module, inputs, output, layer, head, pos, clean_value):
    output[:, pos, head, :] = clean_value
    return output

handle = model.blocks[8].attn.register_forward_hook(
    lambda m, i, o: patch(m, i, o, layer=8, head=5, pos=4,
                           clean_value=clean_cache["blocks.8.attn.hook_z"][0, 4, 5])
)
patched_logits = model(corrupt_tokens)
# compare patched_logits to clean_logits — if close, head 5 was the culprit.
```

`transformer_lens.patching` provides these as one-liners (`patching.get_act_patch_attn_head_out_all_pos(...)`).

---

## 14. Sparse Autoencoders (SAEs) — the 2024-2026 revolution

Activations in a transformer are *dense* — every neuron fires on many things, every "thing" the model knows is spread across many neurons. **Sparse Autoencoders** train a wider, sparse layer to decompose the dense activations into individual interpretable features.

The recipe:
1. Run forward passes on a large dataset, capture the activations at some layer.
2. Train an SAE: encoder (D → many * D, e.g., 8×D) + ReLU sparsity + decoder (back to D), with reconstruction + sparsity loss.
3. The SAE features are interpretable directions: each one tends to fire on a specific semantic concept.

This is how Anthropic's "Scaling Monosemanticity" work and Goodfire's product-grade interpretability operate.

Tools:
- **`sae_lens`** (Joseph Bloom et al.) — open-source SAE training and analysis.
- **Goodfire** — hosted SAE-based feature search and intervention.
- **Anthropic's Reach (internal)** — hosted with public APIs partial.

Once you have SAE features, you can:
- **Search** for the feature that fires on "code that handles SQL injections."
- **Steer** by adding the feature direction to the residual stream — making the model behave as if it just thought about that concept.
- **Intervene at deployment** to suppress unwanted features (safety) or amplify desired ones.

A 2026-grade interpretability product looks like: SAEs trained at multiple layers, browsable feature catalog, deploy-time intervention API. **This is the new frontier of model debugging.**

---

## 15. Putting it together — debugging a real failure

Suppose your model occasionally outputs `</think>` in the middle of a normal chat — a leakage from thinking-mode tokens. You want to find why.

1. **Reproduce.** Find 50 prompts that trigger it.
2. **Forward-pass hook every layer's residual norm.** See if the norm explodes at a specific layer when the bad token comes out. Often yes.
3. **TransformerLens cache + logit lens.** Project residual at every layer through unembed. See where `</think>` first becomes the top prediction. Say it's layer 22.
4. **Ablate each head in layer 22.** Find the 1-2 heads whose ablation removes the leak.
5. **Inspect those heads' attention patterns.** They probably attend to specific positions or token types. Visualize with `circuitsvis`.
6. **Confirm with patching.** Patch in the head's output from a clean run on the buggy input — does the leak go away?
7. **Fix.** Either: SFT a few hundred examples avoiding the leak, OR (the fancy fix) train an SAE on layer 22 and steer the relevant feature down at inference, OR (the surgical fix) zero a specific weight in the head's output projection.

This is how a real interpretability team of 2026 actually debugs production models. You don't need a special library; what you need is the *mental model* — weights vs activations, hooks, ablate-and-test.

---

## 16. The complete workflow for shipping a new version

Putting the whole chapter together:

```
[research / data / arch decision]
   │
   ▼
[major release]      —— fresh pretrain, full data pipeline
   │
   ▼
[evaluation gauntlet] —— lm-eval-harness, internal evals, golden sets
   │
   ▼
[mid-train / minor]   —— continual pretrain on better data, maybe add modality
   │
   ▼
[post-training (point release)] —— SFT + DPO/GRPO, alignment-handbook
   │
   ▼
[red-team / safety patches]    —— targeted SFT, content classifiers
   │
   ▼
[shipping (point or patch)]
   │
   ▼
[observability + golden set in CI]  —— LangSmith / Braintrust / promptfoo
   │
   ▼
[interpretability investigation]    —— for failures: hooks, logit lens, SAEs
   │
   ▼
[next iteration]  ── back to top of the cycle
```

Each arrow has its own tooling, calendar, and risk profile. **Pretraining decisions are slow and expensive; post-training is fast and cheap. Match your problem to the right tier.**

---

## 17. Cheat sheet

- **Major (X → X+1) = fresh pretrain.** Months. New arch / tokenizer allowed.
- **Minor (X → X.5) = continual pretrain.** Weeks. Same arch and weights.
- **Point (X.5 → X.6 / X.7) = post-training only.** Days. SFT/DPO/GRPO.
- **Patch (X.7 → X.7.1) = targeted post-training.** Hours-days.
- **Catastrophic forgetting** is the biggest risk in continual pretrain. Mix in 30-60% general data.
- **Tooling per role**: `torchtitan`/Megatron for pretrain, `trl`/`peft`/`axolotl`/`unsloth` for post-train, vLLM/SGLang for serving, TransformerLens/NNsight for research, W&B/Langfuse for observability, lm-eval/promptfoo/inspect-ai for testing.
- **PyTorch forward hooks** are the simplest way to see inside a model. Three lines.
- **TransformerLens `run_with_cache`** captures every activation, named and addressable.
- **Logit lens** projects intermediate residuals through the LM head — see what the model thinks at each layer.
- **Ablation**: zero out a head/layer/neuron/weight, see what breaks. The single most reliable debugging technique.
- **Mean ablation > zero ablation** for honest results.
- **Activation patching** localizes behavior by copying activations between forward passes.
- **Sparse Autoencoders (SAEs)** decompose dense activations into interpretable features — the 2026 frontier.

---

## 18. Going deeper

- **Anthropic's "Scaling Monosemanticity"** and **"Toy Models of Superposition"** — the canonical SAE / mechanistic interp papers.
- **TransformerLens docs** at `transformerlensorg.github.io/TransformerLens/` — read the tutorials.
- **NNsight docs** — for the modern, scalable variant.
- **Belrose et al. 2023** — Tuned Lens.
- **Anthropic / OpenAI / DeepMind interpretability research blogs** — best public technical writeups.
- **`sae_lens` GitHub** — open-source SAE training, public reproduction of Anthropic's work.
- **Llama 3 / 3.1 / 3.2 / 3.3 / 4 model cards** — read in order to see a real release ladder play out in public.
- **DeepSeek-V2 → V3 → V3.1 release notes** — clearest example of major-vs-minor decisions explained.
- **Karpathy's `llm.c`** — for end-to-end pretraining you can read.
- **`torchtitan`** — the cleanest open-source pretraining framework in 2026.
- **MATS / SERI MATS programs and Apollo Research blogs** — community-driven interpretability research.

That's the full release-engineering and debug-the-internals chapter. Together with chapter 14 (training) and chapter 19 (eval), this completes the production-LLM lifecycle: from random init to deployed model to debugging the failures users find.
